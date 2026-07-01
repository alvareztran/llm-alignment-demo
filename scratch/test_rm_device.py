import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from transformers import AutoModelForSequenceClassification
from models.load_model import DEVICE, get_dtype

print("Device:", DEVICE)
print("Dtype:", get_dtype())
print("Loading with device_map...")
model = AutoModelForSequenceClassification.from_pretrained(
    "./outputs/sft_model",
    num_labels=1,
    torch_dtype=get_dtype(),
    device_map="auto" if DEVICE == "cuda" else None
)
print("Successfully loaded model on GPU/CPU!")
