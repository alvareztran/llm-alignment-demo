import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from models.load_model import DEVICE, get_dtype
from ppo.reward import compute_reward


class LearnedRewardModel:
    """Loads the trained Sequence Classification model and computes raw scalar reward."""

    def __init__(self, model_dir="./outputs/reward_model"):
        if not os.path.exists(model_dir):
            raise FileNotFoundError(
                f"Reward model not found at '{model_dir}'. "
                "Please run train/reward_model.py first."
            )
        print(f"Loading learned reward model from {model_dir} on {DEVICE}...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            torch_dtype=get_dtype()
        ).to(DEVICE)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

    @torch.no_grad()
    def get_score(self, prompt: str, response: str) -> float:
        text = prompt + response
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        logits = self.model(**inputs).logits
        return float(logits[0, 0].cpu())


class HybridRewardModel:
    """Combines learned semantic reward model score with strict rule-based formatting bonuses."""

    def __init__(self, model_dir="./outputs/reward_model"):
        self.learned_rm = LearnedRewardModel(model_dir)

    def __call__(self, prompt: str, response: str) -> float:
        # Get learned semantic preference score (logits are typically in [-3.0, 3.0])
        learned_score = self.learned_rm.get_score(prompt, response)
        # Get rule-based formatting bonuses/penalties
        rule_score = compute_reward(prompt, response)
        # Combine them
        return learned_score + rule_score

