import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "HuggingFaceTB/SmolLM-135M-Instruct"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_dtype():
    if DEVICE == "cuda":
        return torch.bfloat16
    return torch.float32


def resolve_model_path(model_name_or_path=None):
    if not model_name_or_path:
        return MODEL_NAME
    if os.path.isdir(model_name_or_path):
        return model_name_or_path
    if os.path.isabs(model_name_or_path) or model_name_or_path.startswith("."):
        return MODEL_NAME
    return model_name_or_path


def load_tokenizer(model_name_or_path=None):
    resolved_model = resolve_model_path(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(resolved_model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    return tokenizer


def load_model(model_name_or_path=None):
    resolved_model = resolve_model_path(model_name_or_path)

    print("=" * 60)
    print("Loading model...")
    print("=" * 60)

    model = AutoModelForCausalLM.from_pretrained(
        resolved_model,
        torch_dtype=get_dtype(),
        device_map="auto" if DEVICE == "cuda" else None,
    )

    print(f"Model : {resolved_model}")
    print(f"Device: {DEVICE}")

    if DEVICE == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    print()

    return model


def load_reference_model(model_name_or_path=None):
    resolved_model = resolve_model_path(model_name_or_path)

    print("=" * 60)
    print("Loading reference model...")
    print("=" * 60)

    ref_model = AutoModelForCausalLM.from_pretrained(
        resolved_model,
        torch_dtype=get_dtype(),
        device_map="auto" if DEVICE == "cuda" else None,
    )
    print(f"Reference model: {resolved_model}")

    for param in ref_model.parameters():
        param.requires_grad = False

    return ref_model


def load_everything(model_name_or_path=None):
    tokenizer = load_tokenizer(model_name_or_path)
    model = load_model(model_name_or_path)
    ref_model = load_reference_model(model_name_or_path)

    return model, ref_model, tokenizer


if __name__ == "__main__":
    model, ref_model, tokenizer = load_everything()

    print("=" * 60)
    print("Everything loaded successfully!")
    print("=" * 60)
