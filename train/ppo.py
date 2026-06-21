import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from configs.ppo_config import get_ppo_config
from data.prepare_dataset import prepare_dataset
from models.load_model import load_tokenizer
from models.policy_reference import load_frozen_reference_model, load_policy_model
from models.reward_model import RuleBasedRewardModel
from models.value_model import load_value_model
from utils.advantage import compute_gae, normalize_advantages
from utils.logprob import response_log_probs_from_ids
from utils.sampling import generate_response_ids


def select_response_values(value_model, input_ids, attention_mask, response_mask):
    values = value_model(input_ids=input_ids, attention_mask=attention_mask)
    shifted_values = values[:, :-1]
    return shifted_values[response_mask.bool()]


def build_rollout(policy_model, reference_model, value_model, reward_model, tokenizer, prompt, config):
    generated_ids, attention_mask, response_start, response_text = generate_response_ids(
        policy_model,
        tokenizer,
        prompt,
        config.max_prompt_length,
        config.max_new_tokens,
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
    rewards = -config.kl_coef * kl
    rewards[-1] = rewards[-1] + score_reward

    advantages, returns = compute_gae(
        rewards=rewards,
        values=old_values,
        gamma=config.gamma,
        lam=config.gae_lambda,
        last_value=0.0,
    )
    advantages = normalize_advantages(advantages).detach()

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
        "kl": float(kl.mean().detach().cpu()),
    }


def ppo_update(policy_model, value_model, optimizer, rollout, config):
    input_ids = rollout["input_ids"]
    attention_mask = rollout["attention_mask"]
    response_start = rollout["response_start"]
    old_logprobs = rollout["old_logprobs"]
    advantages = rollout["advantages"]
    returns = rollout["returns"]

    total_policy_loss = 0.0
    total_value_loss = 0.0

    for _ in range(config.ppo_epochs):
        new_logprob_rows, response_mask = response_log_probs_from_ids(
            policy_model, input_ids, attention_mask, response_start
        )
        new_logprobs = new_logprob_rows[0]
        values = select_response_values(value_model, input_ids, attention_mask, response_mask)

        ratio = torch.exp(new_logprobs - old_logprobs)
        unclipped_objective = ratio * advantages
        clipped_ratio = torch.clamp(
            ratio,
            1.0 - config.clip_epsilon,
            1.0 + config.clip_epsilon,
        )
        clipped_objective = clipped_ratio * advantages

        policy_loss = -torch.min(unclipped_objective, clipped_objective).mean()
        value_loss = F.mse_loss(values.float(), returns.float())
        loss = policy_loss + config.value_coef * value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_policy_loss += float(policy_loss.detach().cpu())
        total_value_loss += float(value_loss.detach().cpu())

    return {
        "policy_loss": total_policy_loss / config.ppo_epochs,
        "value_loss": total_value_loss / config.ppo_epochs,
    }


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

    policy_model = load_policy_model()
    reference_model = load_frozen_reference_model()
    value_model = load_value_model()
    reward_model = RuleBasedRewardModel()
    tokenizer = load_tokenizer()

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
    kls = []
    policy_losses = []
    value_losses = []

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

            stats = ppo_update(policy_model, value_model, optimizer, rollout, config)
            rewards.append(rollout["reward"])
            kls.append(rollout["kl"])
            policy_losses.append(stats["policy_loss"])
            value_losses.append(stats["value_loss"])

            if step % config.logging_steps == 0:
                print(f"\nEpoch {epoch + 1} | Step {step}")
                print(f"Prompt       : {prompt[:80]}...")
                print(f"Response     : {rollout['response'][:80]}...")
                print(f"Reward       : {rollout['reward']:.4f}")
                print(f"Mean KL      : {rollout['kl']:.4f}")
                print(f"Policy loss  : {stats['policy_loss']:.4f}")
                print(f"Value loss   : {stats['value_loss']:.4f}")

    os.makedirs(config.output_dir, exist_ok=True)
    policy_model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    value_model.save_pretrained(config.value_output_dir)

    elapsed = time.time() - start_time
    avg_reward = sum(rewards) / max(len(rewards), 1)
    avg_kl = sum(kls) / max(len(kls), 1)
    avg_policy_loss = sum(policy_losses) / max(len(policy_losses), 1)
    avg_value_loss = sum(value_losses) / max(len(value_losses), 1)

    os.makedirs("./outputs", exist_ok=True)
    with open(config.metrics_path, "w", encoding="utf-8") as f:
        f.write("===== PPO RLHF RESULT =====\n\n")
        f.write("Objective: clipped surrogate min(r_t A_t, clip(r_t, 1-eps, 1+eps) A_t)\n")
        f.write("KL: per-token penalty -kl_coef * (log pi_old - log pi_ref)\n")
        f.write("Advantage: GAE\n")
        f.write(f"Train samples     : {len(train_dataset)}\n")
        f.write(f"Average reward    : {avg_reward:.4f}\n")
        f.write(f"Average KL        : {avg_kl:.4f}\n")
        f.write(f"Average policy loss: {avg_policy_loss:.4f}\n")
        f.write(f"Average value loss : {avg_value_loss:.4f}\n")
        f.write(f"Time              : {elapsed:.2f}s\n")

    print("\nPPO training finished")
    print(f"Saved policy model to: {config.output_dir}")
    print(f"Saved value model to : {config.value_output_dir}")


if __name__ == "__main__":
    main()
