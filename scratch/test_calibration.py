import sys
from pathlib import Path
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation.plot_probabilities import load_examples, average_response_log_prob, resolve_device
from models.load_model import load_tokenizer
from transformers import AutoModelForCausalLM

def main():
    prompts, responses, splits = load_examples(8)
    tokenizer = load_tokenizer("./outputs/sft_model")
    
    # Load DPO model
    model = AutoModelForCausalLM.from_pretrained("./outputs/dpo_model")
    model.to("cpu")
    model.eval()
    
    # Method 1: Raw average logprob (Current)
    matrix_raw = np.zeros((8, 8))
    for i, prompt in enumerate(prompts):
        logps = []
        for response in responses:
            logp = average_response_log_prob(torch, model, tokenizer, prompt, response, 512)
            logps.append(logp)
        logps = np.array(logps)
        probs = np.exp(logps - np.max(logps))
        matrix_raw[i] = probs / probs.sum()
        
    # Method 2: Calibrated average logprob
    null_prompt = "Human: \nAssistant:"
    null_logps = []
    for response in responses:
        null_logp = average_response_log_prob(torch, model, tokenizer, null_prompt, response, 512)
        null_logps.append(null_logp)
    null_logps = np.array(null_logps)
    
    matrix_calib = np.zeros((8, 8))
    for i, prompt in enumerate(prompts):
        logps = []
        for j, response in enumerate(responses):
            logp = average_response_log_prob(torch, model, tokenizer, prompt, response, 512)
            calibrated_logp = logp - null_logps[j]
            logps.append(calibrated_logp)
        logps = np.array(logps)
        # Apply temperature scaling to make it sharper if needed
        probs = np.exp(logps - np.max(logps))
        matrix_calib[i] = probs / probs.sum()

    print("DIAGONAL PROBABILITIES (RAW METHOD):")
    print(np.diagonal(matrix_raw))
    print("Average raw diagonal prob:", np.diagonal(matrix_raw).mean())
    
    print("\nDIAGONAL PROBABILITIES (CALIBRATED METHOD):")
    print(np.diagonal(matrix_calib))
    print("Average calibrated diagonal prob:", np.diagonal(matrix_calib).mean())

if __name__ == "__main__":
    main()
