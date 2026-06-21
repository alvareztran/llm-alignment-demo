import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "HuggingFaceTB/SmolLM-135M-Instruct"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_dtype():
    if DEVICE == "cuda":
        return torch.bfloat16
    return torch.float32


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    return tokenizer


def load_model():

    print("=" * 60)
    print("Loading model...")
    print("=" * 60)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=get_dtype(),
        device_map="auto" if DEVICE == "cuda" else None,
    )

    print(f"Model : {MODEL_NAME}")
    print(f"Device: {DEVICE}")

    if DEVICE == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    print()

    return model


def load_reference_model():

    print("=" * 60)
    print("Loading reference model...")
    print("=" * 60)

    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=get_dtype(),
        device_map="auto" if DEVICE == "cuda" else None,
    )

    # freeze ref model (quan trọng cho PPO)
    for param in ref_model.parameters():
        param.requires_grad = False

    return ref_model


def load_everything():
    tokenizer = load_tokenizer()
    model = load_model()
    ref_model = load_reference_model()

    return model, ref_model, tokenizer


if __name__ == "__main__":
    model, ref_model, tokenizer = load_everything()

    print("=" * 60)
    print("Everything loaded successfully!")
    print("=" * 60)