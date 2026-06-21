import os
import sys
import time
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from configs.sft_config import get_sft_config
from data.prepare_dataset import prepare_dataset
from models.load_model import load_model, load_tokenizer
from utils.logprob import prompt_response_inputs


def build_sft_batch(tokenizer, prompt, response, max_length, device):
    """Build labels for SFT, masking prompt tokens with -100."""
    input_ids, attention_mask, response_start = prompt_response_inputs(
        tokenizer,
        prompt,
        response,
        max_length,
        device,
    )
    labels = input_ids.clone()
    labels[:, :response_start] = -100

    if response_start >= labels.shape[1]:
        return None

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "response_tokens": labels.shape[1] - response_start,
    }


def main():
    print("=" * 60)
    print("MINI SFT TRAINING")
    print("=" * 60)

    start_time = time.time()
    config = get_sft_config()

    train_dataset, eval_dataset = prepare_dataset()
    if config.max_train_samples is not None:
        train_dataset = train_dataset.select(
            range(min(config.max_train_samples, len(train_dataset)))
        )

    model = load_model()
    tokenizer = load_tokenizer()
    device = next(model.parameters()).device
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    losses = []
    response_token_counts = []

    for epoch in range(config.num_train_epochs):
        for step, example in enumerate(train_dataset):
            batch = build_sft_batch(
                tokenizer,
                example["prompt"],
                example["chosen"],
                config.max_length,
                device,
            )
            if batch is None:
                continue

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_value = float(loss.detach().cpu())
            losses.append(loss_value)
            response_token_counts.append(batch["response_tokens"])

            if step % config.logging_steps == 0:
                print(f"Epoch {epoch + 1} | Step {step} | SFT loss {loss_value:.4f}")

    os.makedirs(config.output_dir, exist_ok=True)
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    elapsed = time.time() - start_time
    avg_loss = sum(losses) / max(len(losses), 1)
    avg_response_tokens = sum(response_token_counts) / max(len(response_token_counts), 1)

    os.makedirs("./outputs", exist_ok=True)
    with open(config.metrics_path, "w", encoding="utf-8") as f:
        f.write("===== MINI SFT RESULT =====\n\n")
        f.write("Objective: causal LM cross-entropy on chosen response tokens only\n")
        f.write("Prompt labels are masked with -100\n")
        f.write(f"Train samples: {len(train_dataset)}\n")
        f.write(f"Eval samples : {len(eval_dataset)}\n")
        f.write(f"Average loss : {avg_loss:.4f}\n")
        f.write(f"Average response tokens: {avg_response_tokens:.2f}\n")
        f.write(f"Time         : {elapsed:.2f}s\n")

    print("\nMini SFT training finished")
    print(f"Saved SFT model to: {config.output_dir}")
    print(f"Average loss: {avg_loss:.4f}")


if __name__ == "__main__":
    main()
