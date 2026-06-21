import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation.generate import generate_response
from models.load_model import MODEL_NAME, load_tokenizer

MODEL_PATHS = {
    "base": MODEL_NAME,
    "dpo": "./outputs/dpo_model",
    "ppo": "./outputs/ppo_model",
}


def model_exists(path: str) -> bool:
    return os.path.isdir(path) or path == MODEL_NAME


def load_model_for_compare(path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype)
    model.to(device)
    return model


def compare_models(prompt: str):
    tokenizer = load_tokenizer()
    results = {"prompt": prompt}

    print("=" * 80)
    print("PROMPT")
    print("=" * 80)
    print(prompt)

    for name, path in MODEL_PATHS.items():
        if not model_exists(path):
            output = f"[Skipped: model not found at {path}]"
        else:
            model = load_model_for_compare(path)
            output = generate_response(model, tokenizer, prompt)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        results[name] = output

        print("\n" + "=" * 80)
        print(f"{name.upper()} MODEL")
        print("=" * 80)
        print(output)

    return results


def save_result(result, path="./outputs/compare_result.txt"):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write("===== PROMPT =====\n")
        f.write(result["prompt"] + "\n\n")

        f.write("===== BASE =====\n")
        f.write(result["base"] + "\n\n")

        f.write("===== DPO =====\n")
        f.write(result["dpo"] + "\n\n")

        f.write("===== PPO =====\n")
        f.write(result["ppo"] + "\n\n")

    print(f"\nSaved to {path}")


if __name__ == "__main__":
    test_prompt = (
        "Human: Explain what Artificial Intelligence is.\n"
        "Assistant:"
    )

    result = compare_models(test_prompt)
    save_result(result)
