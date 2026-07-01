import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import datasets (via prepare_dataset) before torch to prevent Windows DLL conflicts
from data.prepare_dataset import prepare_dataset

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification

from configs.reward_model_config import get_reward_model_config
from models.load_model import load_tokenizer, get_dtype


def main():
    print("=" * 60)
    print("REWARD MODEL TRAINING")
    print("=" * 60)

    start_time = time.time()
    config = get_reward_model_config()

    train_dataset, eval_dataset = prepare_dataset()
    if config.max_train_samples is not None:
        train_dataset = train_dataset.select(
            range(min(config.max_train_samples, len(train_dataset)))
        )

    # Load SFT model but for sequence classification with 1 label
    print("Loading base SFT model for sequence classification...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model_path,
        num_labels=1,
        torch_dtype=get_dtype(),
        device_map="auto" if device == "cuda" else None
    )
    
    tokenizer = load_tokenizer(config.base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure model pad token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    losses = []

    for epoch in range(config.num_train_epochs):
        for step, example in enumerate(train_dataset):
            prompt = example["prompt"]
            chosen = example["chosen"]
            rejected = example["rejected"]

            # Tokenize chosen and rejected pairs
            chosen_inputs = tokenizer(
                prompt + chosen,
                return_tensors="pt",
                truncation=True,
                max_length=config.max_length
            ).to(device)

            rejected_inputs = tokenizer(
                prompt + rejected,
                return_tensors="pt",
                truncation=True,
                max_length=config.max_length
            ).to(device)

            # Get scores from classification model
            chosen_outputs = model(**chosen_inputs)
            rejected_outputs = model(**rejected_inputs)

            # Logits has shape [1, 1] representing scalar reward
            r_w = chosen_outputs.logits[0, 0]
            r_l = rejected_outputs.logits[0, 0]

            # Bradley-Terry preference loss
            loss = -F.logsigmoid(r_w - r_l)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_value = float(loss.detach().cpu())
            losses.append(loss_value)

            if step % config.logging_steps == 0:
                print(f"Epoch {epoch + 1} | Step {step} | Loss {loss_value:.4f} | r_w {float(r_w):.4f} | r_l {float(r_l):.4f}")

    os.makedirs(config.output_dir, exist_ok=True)
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    elapsed = time.time() - start_time
    avg_loss = sum(losses) / max(len(losses), 1)

    os.makedirs("./outputs", exist_ok=True)
    with open(config.metrics_path, "w", encoding="utf-8") as f:
        f.write("===== REWARD MODEL TRAINING RESULT =====\n\n")
        f.write("Objective: -log sigmoid(r(x, y_w) - r(x, y_l))\n")
        f.write(f"Base model path: {config.base_model_path}\n")
        f.write(f"Train samples: {len(train_dataset)}\n")
        f.write(f"Average loss : {avg_loss:.4f}\n")
        f.write(f"Time         : {elapsed:.2f}s\n")

    print("\nReward Model training finished")
    print(f"Saved reward model to: {config.output_dir}")
    print(f"Average loss: {avg_loss:.4f}")


if __name__ == "__main__":
    main()
