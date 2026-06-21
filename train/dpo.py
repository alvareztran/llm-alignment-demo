import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from configs.dpo_config import get_dpo_config
from data.prepare_dataset import prepare_dataset
from models.load_model import load_tokenizer
from models.policy_reference import load_frozen_reference_model, load_policy_model
from utils.logprob import response_log_prob


def dpo_loss(policy_model, reference_model, tokenizer, example, config):
    prompt = example["prompt"]
    chosen = example["chosen"]
    rejected = example["rejected"]

    policy_chosen_logp = response_log_prob(
        policy_model, tokenizer, prompt, chosen, config.max_length
    )
    policy_rejected_logp = response_log_prob(
        policy_model, tokenizer, prompt, rejected, config.max_length
    )

    with torch.no_grad():
        ref_chosen_logp = response_log_prob(
            reference_model, tokenizer, prompt, chosen, config.max_length
        )
        ref_rejected_logp = response_log_prob(
            reference_model, tokenizer, prompt, rejected, config.max_length
        )

    policy_log_ratio = policy_chosen_logp - policy_rejected_logp
    reference_log_ratio = ref_chosen_logp - ref_rejected_logp
    logits = config.beta * (policy_log_ratio - reference_log_ratio)
    return -F.logsigmoid(logits)


def main():
    print("=" * 60)
    print("DPO TRAINING")
    print("=" * 60)

    start_time = time.time()
    config = get_dpo_config()

    train_dataset, eval_dataset = prepare_dataset()
    if config.max_train_samples is not None:
        train_dataset = train_dataset.select(
            range(min(config.max_train_samples, len(train_dataset)))
        )

    policy_model = load_policy_model()
    reference_model = load_frozen_reference_model()
    tokenizer = load_tokenizer()

    policy_model.train()
    reference_model.eval()

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=config.learning_rate)
    losses = []

    for epoch in range(config.num_train_epochs):
        for step, example in enumerate(train_dataset):
            loss = dpo_loss(policy_model, reference_model, tokenizer, example, config)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_value = float(loss.detach().cpu())
            losses.append(loss_value)

            if step % config.logging_steps == 0:
                print(f"Epoch {epoch + 1} | Step {step} | DPO loss {loss_value:.4f}")

    os.makedirs(config.output_dir, exist_ok=True)
    policy_model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    elapsed = time.time() - start_time
    avg_loss = sum(losses) / max(len(losses), 1)

    os.makedirs("./outputs", exist_ok=True)
    with open(config.metrics_path, "w", encoding="utf-8") as f:
        f.write("===== DPO RESULT =====\n\n")
        f.write("Objective: -log sigmoid(beta * ((log pi_w - log pi_l) - (log ref_w - log ref_l)))\n")
        f.write(f"Train samples: {len(train_dataset)}\n")
        f.write(f"Eval samples : {len(eval_dataset)}\n")
        f.write(f"Average loss : {avg_loss:.4f}\n")
        f.write(f"Time         : {elapsed:.2f}s\n")

    print("\nDPO training finished")
    print(f"Saved policy model to: {config.output_dir}")
    print(f"Average loss: {avg_loss:.4f}")


if __name__ == "__main__":
    main()
