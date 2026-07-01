import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

print("Importing...")
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification
from configs.reward_model_config import get_reward_model_config
from data.prepare_dataset import prepare_dataset
from models.load_model import load_tokenizer, get_dtype

print("Main starts")
config = get_reward_model_config()
train_dataset, eval_dataset = prepare_dataset()
train_dataset = train_dataset.select(range(5))

print("Loading model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForSequenceClassification.from_pretrained(
    config.base_model_path,
    num_labels=1,
    torch_dtype=get_dtype(),
    device_map="auto" if device == "cuda" else None
)
print("Model loaded successfully!")
