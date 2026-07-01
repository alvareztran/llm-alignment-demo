import sys
from pathlib import Path
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data.prepare_dataset import prepare_dataset
from evaluation.preference_margin_plot import compute_model_margins, load_model, summarize
from models.load_model import load_tokenizer

def main():
    _, eval_dataset = prepare_dataset()
    examples = [
        {
            "prompt": x["prompt"],
            "chosen": x["chosen"],
            "rejected": x["rejected"],
        }
        for x in eval_dataset.select(range(20))
    ]
    
    tokenizer = load_tokenizer("./outputs/sft_model")
    
    MODEL_PATHS = {
        "base": "./outputs/sft_model",
        "dpo": "./outputs/dpo_model",
        "ppo": "./outputs/ppo_model",
    }
    
    model_rows = {}
    for model_name, model_path in MODEL_PATHS.items():
        print(f"Loading {model_name}...")
        model = load_model(model_path, "cpu")
        _, rows = compute_model_margins(model, tokenizer, examples, 512)
        model_rows[model_name] = rows
        del model
        
    summary = summarize(model_rows)
    print("\nRESULTS ON HH-RLHF TEST DATASET:")
    for model_name, stats in summary.items():
        print(f"Model: {model_name.upper()}")
        print(f"  Win rate: {stats['win_rate']:.3f} ({stats['wins']}/{stats['total']})")
        print(f"  Avg margin: {stats['avg_margin']:.4f}")

if __name__ == "__main__":
    main()
