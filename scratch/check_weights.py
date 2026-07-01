import torch
from transformers import AutoModelForCausalLM

sft = AutoModelForCausalLM.from_pretrained("./outputs/sft_model")
dpo = AutoModelForCausalLM.from_pretrained("./outputs/dpo_model")
ppo = AutoModelForCausalLM.from_pretrained("./outputs/ppo_model")

# Get first parameter
sft_p = next(sft.parameters())
dpo_p = next(dpo.parameters())
ppo_p = next(ppo.parameters())

print("SFT parameter sum:", float(sft_p.sum()))
print("DPO parameter sum:", float(dpo_p.sum()))
print("PPO parameter sum:", float(ppo_p.sum()))

# Compare element-wise
print("SFT vs DPO equal?", torch.equal(sft_p, dpo_p))
print("SFT vs PPO equal?", torch.equal(sft_p, ppo_p))

diff_dpo = (sft_p - dpo_p).abs().max()
diff_ppo = (sft_p - ppo_p).abs().max()
print("SFT vs DPO max absolute diff:", float(diff_dpo))
print("SFT vs PPO max absolute diff:", float(diff_ppo))
