import sys
from pathlib import Path
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation.plot_probabilities import load_examples, average_response_log_prob
from models.load_model import load_tokenizer
from transformers import AutoModelForCausalLM

def main():
    prompts, responses, splits = load_examples(8)
    tokenizer = load_tokenizer("./outputs/sft_model")
    
    model = AutoModelForCausalLM.from_pretrained("./outputs/ppo_model")
    model.to("cpu")
    model.eval()
    
    # Calculate null logprobs for calibration
    null_prompt = "Human: \nAssistant:"
    null_logps = []
    for response in responses:
        null_logp = average_response_log_prob(torch, model, tokenizer, null_prompt, response, 512)
        null_logps.append(null_logp)
    null_logps = np.array(null_logps)
    
    # Temperature values to test
    for temp in [1.0, 0.2, 0.1, 0.05, 0.03]:
        matrix = np.zeros((8, 8))
        for i, prompt in enumerate(prompts):
            logps = []
            for j, response in enumerate(responses):
                logp = average_response_log_prob(torch, model, tokenizer, prompt, response, 512)
                calibrated_logp = logp - null_logps[j]
                logps.append(calibrated_logp)
            logps = np.array(logps)
            
            # Apply temperature scaling
            scaled_logps = logps / temp
            probs = np.exp(scaled_logps - np.max(scaled_logps))
            matrix[i] = probs / probs.sum()
            
        diag = np.diagonal(matrix)
        print(f"Temp {temp} | Avg Diag Prob: {diag.mean():.4f} | Diag values: {list(np.round(diag, 2))}")

if __name__ == "__main__":
    main()
