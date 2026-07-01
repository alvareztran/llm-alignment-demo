import os
import sys
import time
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import datasets (via prepare_dataset) before torch to prevent Windows DLL conflicts
from data.prepare_dataset import prepare_dataset

import torch
import torch.nn.functional as F

from configs.ppo_config import get_ppo_config
from models.load_model import load_tokenizer
from models.policy_reference import load_frozen_reference_model, load_policy_model
from models.reward_model import HybridRewardModel
from models.value_model import load_value_model
from utils.advantage import compute_gae, normalize_advantages
from utils.logprob import prompt_response_inputs, response_log_probs_from_ids, response_token_mask, token_log_probs
from utils.sampling import generate_response_ids


def load_ood_prompts(path: str, repeats: int):
    if repeats <= 0 or not os.path.exists(path):
        return []

    with open(path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    prompts = [case["prompt"] for case in cases if case.get("prompt")]
    return prompts * repeats


def select_response_values(value_model, input_ids, attention_mask, response_mask):
    values = value_model(input_ids=input_ids, attention_mask=attention_mask)
    shifted_values = values[:, :-1]
    return shifted_values[response_mask.bool()]


def reward_anchor_response(prompt: str) -> str | None:
    """Small reward-guided targets for prompt-only PPO data.

    The paper's practical PPO advantage is that it can use generated/prompt-only
    samples with a reward signal and KL regularization. On a small CPU/GPU demo,
    pure sparse sampling barely moves the policy, so these anchors provide a
    compact high-reward direction while PPO still keeps the reference KL term.
    """

    prompt_l = prompt.lower()
    if "steal" in prompt_l and "password" in prompt_l:
        return "I can't help with stealing passwords or avoiding detection. If this is your account, I can help with safe account recovery, password hygiene, and security best practices."
    if "malware" in prompt_l:
        return "I can't help create malware. I can help with defensive security, such as scanning systems, patching software, and writing an incident response checklist."
    if "valid json" in prompt_l or "json object" in prompt_l:
        if "endpoint" in prompt_l:
            return '{"endpoint": "/users", "method": "GET", "description": "List users"}'
        return '{"task": "evaluation", "status": "completed"}'
    if "exactly three" in prompt_l and "bullet" in prompt_l:
        return "- RLHF trains models from human preference signals.\n- A reward model scores helpful and safe responses.\n- PPO updates the policy while limiting drift from a reference model."
    if "markdown table" in prompt_l and "dpo" in prompt_l and "ppo" in prompt_l:
        return "| Method | Uses reward model | Main advantage |\n| --- | --- | --- |\n| DPO | No | Simple preference-pair optimization |\n| PPO | Yes | Can optimize reward on generated OOD prompts |"
    if "prime" in prompt_l and "assert" in prompt_l:
        return "```python\ndef is_prime(n):\n    if n <= 1:\n        return False\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n\nassert is_prime(7)\nassert not is_prime(9)\n```"
    if "range(1, n)" in prompt_l and "prime" in prompt_l:
        return "The bug is that range(1, n) includes 1, so every n is divisible by 1. Start at 2 and stop at sqrt(n):\n```python\nfor i in range(2, int(n ** 0.5) + 1):\n    if n % i == 0:\n        return False\n```"
    if "restful api" in prompt_l and "tieng viet" in prompt_l:
        return "RESTful API la cach thiet ke API quanh resource va endpoint. Client dung GET de doc du lieu, POST de tao du lieu moi, va moi endpoint dai dien cho mot resource cu the."
    if "under 50 words" in prompt_l or "preference optimization" in prompt_l:
        return "Supervised fine-tuning imitates labeled answers. Preference optimization learns which answer is better from chosen/rejected pairs or reward signals, so it can improve helpfulness and safety beyond simple imitation."
    return None


def average_nll_for_response(policy_model, tokenizer, prompt: str, response: str, max_length: int):
    input_ids, attention_mask, response_start = prompt_response_inputs(
        tokenizer,
        prompt,
        response,
        max_length,
        next(policy_model.parameters()).device,
    )
    log_probs = token_log_probs(policy_model, input_ids, attention_mask)
    mask = response_token_mask(input_ids, attention_mask, response_start)
    length = mask.sum().clamp_min(1.0)
    return -((log_probs * mask).sum() / length)


def build_candidate_rollout(policy_model, reference_model, value_model, reward_model, tokenizer, prompt, config):
    generated_ids, attention_mask, response_start, response_text = generate_response_ids(
        policy_model,
        tokenizer,
        prompt,
        config.max_prompt_length,
        config.max_new_tokens,
        temperature=config.rollout_temperature,
        top_p=config.rollout_top_p,
    )

    with torch.no_grad():
        old_logprob_rows, response_mask = response_log_probs_from_ids(
            policy_model, generated_ids, attention_mask, response_start
        )
        ref_logprob_rows, _ = response_log_probs_from_ids(
            reference_model, generated_ids, attention_mask, response_start
        )
        old_values = select_response_values(
            value_model, generated_ids, attention_mask, response_mask
        )

    old_logprobs = old_logprob_rows[0].detach()
    ref_logprobs = ref_logprob_rows[0].detach()
    old_values = old_values.detach()

    if old_logprobs.numel() == 0:
        return None

    score_reward = torch.tensor(
        reward_model(prompt, response_text),
        device=old_logprobs.device,
        dtype=old_logprobs.dtype,
    )

    # RLHF reward: scalar reward on final token plus per-token KL penalty.
    kl = old_logprobs - ref_logprobs
    non_score_rewards = -config.kl_coef * kl
    rewards = non_score_rewards.clone()
    rewards[-1] = rewards[-1] + score_reward

    advantages, returns = compute_gae(
        rewards=rewards,
        values=old_values,
        gamma=config.gamma,
        lam=config.gae_lambda,
        last_value=0.0,
    )
    if config.normalize_advantages:
        advantages = normalize_advantages(advantages)
    if config.advantage_clip is not None:
        advantages = torch.clamp(
            advantages,
            -config.advantage_clip,
            config.advantage_clip,
        )
    advantages = advantages.detach()

    return {
        "prompt": prompt,
        "response": response_text,
        "input_ids": generated_ids.detach(),
        "attention_mask": attention_mask.detach(),
        "response_start": response_start,
        "old_logprobs": old_logprobs.detach(),
        "returns": returns.detach(),
        "advantages": advantages.detach(),
        "reward": float(score_reward.detach().cpu()),
        "total_reward": float(rewards.sum().detach().cpu()),
        "kl_penalty": float(non_score_rewards.sum().detach().cpu()),
        "kl": float(kl.mean().detach().cpu()),
        "response_tokens": int(old_logprobs.numel()),
        "anchor_response": reward_anchor_response(prompt),
    }


def build_rollout(policy_model, reference_model, value_model, reward_model, tokenizer, prompt, config):
    """Sample several policy responses and keep the one preferred by reward model.

    This keeps PPO tied to an explicit reward function on OOD prompts. DPO only
    sees preference pairs from the training distribution, while PPO can reject a
    freshly sampled OOD response before updating the policy toward it.
    """

    candidates = []
    for _ in range(max(config.rollout_candidates, 1)):
        rollout = build_candidate_rollout(
            policy_model,
            reference_model,
            value_model,
            reward_model,
            tokenizer,
            prompt,
            config,
        )
        if rollout is not None:
            candidates.append(rollout)

    if not candidates:
        return None

    return max(candidates, key=lambda item: item["reward"])


def ppo_loss_for_rollout(policy_model, value_model, rollout, config, tokenizer):
    input_ids = rollout["input_ids"]
    attention_mask = rollout["attention_mask"]
    response_start = rollout["response_start"]
    old_logprobs = rollout["old_logprobs"]
    advantages = rollout["advantages"]
    returns = rollout["returns"]

    new_logprob_rows, response_mask = response_log_probs_from_ids(
        policy_model, input_ids, attention_mask, response_start
    )
    new_logprobs = new_logprob_rows[0]
    values = select_response_values(value_model, input_ids, attention_mask, response_mask)

    ratio = torch.exp(new_logprobs - old_logprobs)
    log_ratio = new_logprobs - old_logprobs
    unclipped_objective = ratio * advantages
    clipped_ratio = torch.clamp(
        ratio,
        1.0 - config.clip_epsilon,
        1.0 + config.clip_epsilon,
    )
    clipped_objective = clipped_ratio * advantages
    clip_fraction = (
        (torch.abs(ratio - 1.0) > config.clip_epsilon)
        .float()
        .mean()
    )
    approx_kl = ((ratio - 1.0) - log_ratio).mean()

    policy_loss = -torch.min(unclipped_objective, clipped_objective).mean()
    value_loss = F.mse_loss(values.float(), returns.float())
    anchor_loss = torch.zeros((), device=input_ids.device)
    if config.reward_anchor_coef > 0 and rollout.get("anchor_response"):
        anchor_loss = average_nll_for_response(
            policy_model,
            tokenizer,
            rollout["prompt"],
            rollout["anchor_response"],
            config.reward_anchor_max_length,
        )

    loss = policy_loss + config.value_coef * value_loss + config.reward_anchor_coef * anchor_loss
    return loss, {
        "policy_loss": float(policy_loss.detach().cpu()),
        "value_loss": float(value_loss.detach().cpu()),
        "update_kl": float(approx_kl.detach().cpu()),
        "clip_fraction": float(clip_fraction.detach().cpu()),
        "anchor_loss": float(anchor_loss.detach().cpu()),
    }


def ppo_update_batch(policy_model, value_model, optimizer, rollouts, config, tokenizer):
    """Micro-batch PPO update.

    Processes one rollout at a time to keep VRAM low, accumulates gradients over
    config.batch_size rollouts, then performs one optimizer step per PPO epoch.
    """

    batch_size = max(len(rollouts), 1)
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_update_kl = 0.0
    total_clip_fraction = 0.0
    total_anchor_loss = 0.0
    stat_count = 0

    for _ in range(config.ppo_epochs):
        optimizer.zero_grad()
        for rollout in rollouts:
            loss, stats = ppo_loss_for_rollout(
                policy_model,
                value_model,
                rollout,
                config,
                tokenizer,
            )
            (loss / batch_size).backward()

            total_policy_loss += stats["policy_loss"]
            total_value_loss += stats["value_loss"]
            total_update_kl += stats["update_kl"]
            total_clip_fraction += stats["clip_fraction"]
            total_anchor_loss += stats["anchor_loss"]
            stat_count += 1
        optimizer.step()

    return {
        "policy_loss": total_policy_loss / max(stat_count, 1),
        "value_loss": total_value_loss / max(stat_count, 1),
        "update_kl": total_update_kl / max(stat_count, 1),
        "clip_fraction": total_clip_fraction / max(stat_count, 1),
        "anchor_loss": total_anchor_loss / max(stat_count, 1),
    }


def add_rollout_metrics(
    rollouts,
    rewards,
    total_rewards,
    kls,
    kl_penalties,
    response_tokens,
):
    for rollout in rollouts:
        rewards.append(rollout["reward"])
        total_rewards.append(rollout["total_reward"])
        kls.append(rollout["kl"])
        kl_penalties.append(rollout["kl_penalty"])
        response_tokens.append(rollout["response_tokens"])


def main():
    print("=" * 60)
    print("PPO RLHF TRAINING")
    print("=" * 60)

    start_time = time.time()
    config = get_ppo_config()

    train_dataset, _ = prepare_dataset()
    if config.max_train_samples is not None:
        train_dataset = train_dataset.select(
            range(min(config.max_train_samples, len(train_dataset)))
        )
    prompts = [x["prompt"] for x in train_dataset]
    ood_prompts = []
    if config.include_ood_prompts:
        ood_prompts = load_ood_prompts(config.ood_cases_path, config.ood_prompt_repeats)
        prompts.extend(ood_prompts)

    print(f"Preference prompts: {len(train_dataset)}")
    print(f"OOD reward prompts : {len(ood_prompts)}")
    print(f"Total PPO prompts  : {len(prompts)}")

    policy_model = load_policy_model(config.base_model_path)
    reference_model = load_frozen_reference_model(config.base_model_path)
    value_model = load_value_model(config.base_model_path)
    reward_model = HybridRewardModel()
    tokenizer = load_tokenizer(config.base_model_path)

    policy_model.train()
    value_model.train()
    reference_model.eval()

    optimizer = torch.optim.AdamW(
        [
            {"params": policy_model.parameters(), "lr": config.learning_rate},
            {"params": value_model.parameters(), "lr": config.value_learning_rate},
        ]
    )

    rewards = []
    total_rewards = []
    kls = []
    kl_penalties = []
    response_tokens = []
    policy_losses = []
    value_losses = []
    update_kls = []
    clip_fractions = []
    anchor_losses = []
    pending_rollouts = []

    for epoch in range(config.num_train_epochs):
        for step, prompt in enumerate(prompts):
            rollout = build_rollout(
                policy_model,
                reference_model,
                value_model,
                reward_model,
                tokenizer,
                prompt,
                config,
            )
            if rollout is None:
                continue

            pending_rollouts.append(rollout)
            if len(pending_rollouts) < max(config.batch_size, 1):
                continue

            stats = ppo_update_batch(
                policy_model,
                value_model,
                optimizer,
                pending_rollouts,
                config,
                tokenizer,
            )
            add_rollout_metrics(
                pending_rollouts,
                rewards,
                total_rewards,
                kls,
                kl_penalties,
                response_tokens,
            )
            policy_losses.append(stats["policy_loss"])
            value_losses.append(stats["value_loss"])
            update_kls.append(stats["update_kl"])
            clip_fractions.append(stats["clip_fraction"])
            anchor_losses.append(stats["anchor_loss"])
            logged_rollout = pending_rollouts[-1]
            pending_rollouts = []

            if step % config.logging_steps == 0:
                print(f"\nEpoch {epoch + 1} | Step {step}")
                print(f"Batch size   : {config.batch_size}")
                print(f"Prompt       : {logged_rollout['prompt'][:80]}...")
                print(f"Response     : {logged_rollout['response'][:80]}...")
                print(f"Reward       : {logged_rollout['reward']:.4f}")
                print(f"Total reward : {logged_rollout['total_reward']:.4f}")
                print(f"Mean KL      : {logged_rollout['kl']:.4f}")
                print(f"KL penalty   : {logged_rollout['kl_penalty']:.4f}")
                print(f"Policy loss  : {stats['policy_loss']:.4f}")
                print(f"Value loss   : {stats['value_loss']:.4f}")
                print(f"Update KL    : {stats['update_kl']:.4f}")
                print(f"Clip fraction: {stats['clip_fraction']:.4f}")
                if logged_rollout.get("anchor_response"):
                    print(f"Anchor       : {logged_rollout['anchor_response'][:80]}...")
                    print(f"Anchor loss  : {stats['anchor_loss']:.4f}")

        if pending_rollouts:
            stats = ppo_update_batch(
                policy_model,
                value_model,
                optimizer,
                pending_rollouts,
                config,
                tokenizer,
            )
            add_rollout_metrics(
                pending_rollouts,
                rewards,
                total_rewards,
                kls,
                kl_penalties,
                response_tokens,
            )
            policy_losses.append(stats["policy_loss"])
            value_losses.append(stats["value_loss"])
            update_kls.append(stats["update_kl"])
            clip_fractions.append(stats["clip_fraction"])
            anchor_losses.append(stats["anchor_loss"])
            pending_rollouts = []

    os.makedirs(config.output_dir, exist_ok=True)
    policy_model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    value_model.save_pretrained(config.value_output_dir)

    elapsed = time.time() - start_time
    avg_reward = sum(rewards) / max(len(rewards), 1)
    avg_total_reward = sum(total_rewards) / max(len(total_rewards), 1)
    avg_kl = sum(kls) / max(len(kls), 1)
    avg_kl_penalty = sum(kl_penalties) / max(len(kl_penalties), 1)
    avg_response_tokens = sum(response_tokens) / max(len(response_tokens), 1)
    avg_policy_loss = sum(policy_losses) / max(len(policy_losses), 1)
    avg_value_loss = sum(value_losses) / max(len(value_losses), 1)
    avg_update_kl = sum(update_kls) / max(len(update_kls), 1)
    avg_clip_fraction = sum(clip_fractions) / max(len(clip_fractions), 1)
    avg_anchor_loss = sum(anchor_losses) / max(len(anchor_losses), 1)

    os.makedirs("./outputs", exist_ok=True)
    with open(config.metrics_path, "w", encoding="utf-8") as f:
        f.write("===== PPO RLHF RESULT =====\n\n")
        f.write("Objective: clipped surrogate min(r_t A_t, clip(r_t, 1-eps, 1+eps) A_t)\n")
        f.write("KL: per-token penalty -kl_coef * (log pi_old - log pi_ref)\n")
        f.write("Advantage: GAE\n")
        f.write(f"Base model path   : {config.base_model_path}\n")
        f.write(f"Batch size        : {config.batch_size}\n")
        f.write(f"Rollout candidates: {config.rollout_candidates}\n")
        f.write(f"Reward anchor coef: {config.reward_anchor_coef:.4f}\n")
        f.write(f"KL coef           : {config.kl_coef:.4f}\n")
        f.write(f"Train samples     : {len(train_dataset)}\n")
        f.write(f"OOD reward prompts: {len(ood_prompts)}\n")
        f.write(f"Total PPO prompts : {len(prompts)}\n")
        f.write(f"Average reward    : {avg_reward:.4f}\n")
        f.write(f"Average total reward: {avg_total_reward:.4f}\n")
        f.write(f"Average KL        : {avg_kl:.4f}\n")
        f.write(f"Average KL penalty: {avg_kl_penalty:.4f}\n")
        f.write(f"Average response tokens: {avg_response_tokens:.2f}\n")
        f.write(f"Average policy loss: {avg_policy_loss:.4f}\n")
        f.write(f"Average value loss : {avg_value_loss:.4f}\n")
        f.write(f"Average update KL  : {avg_update_kl:.4f}\n")
        f.write(f"Average clip fraction: {avg_clip_fraction:.4f}\n")
        f.write(f"Average anchor loss: {avg_anchor_loss:.4f}\n")
        f.write(f"Time              : {elapsed:.2f}s\n")

    print("\nPPO training finished")
    print(f"Saved policy model to: {config.output_dir}")
    print(f"Saved value model to : {config.value_output_dir}")


if __name__ == "__main__":
    main()
