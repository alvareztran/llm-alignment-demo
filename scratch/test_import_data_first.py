import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

print("Step 1: import data.prepare_dataset")
from data.prepare_dataset import prepare_dataset

print("Step 2: import torch")
import torch

print("Step 3: import F")
import torch.nn.functional as F

print("Step 4: import transformers classification")
from transformers import AutoModelForSequenceClassification

print("Step 5: import configs")
from configs.reward_model_config import get_reward_model_config

print("Step 6: import load_model")
from models.load_model import load_tokenizer, get_dtype

print("Step 7: import models.reward_model")
import models.reward_model

print("Step 8: import train.reward_model")
import train.reward_model

print("Done!")
