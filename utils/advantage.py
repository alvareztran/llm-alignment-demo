import torch


def compute_gae(rewards, values, gamma=1.0, lam=0.95, last_value=0.0):
    """Generalized Advantage Estimation for one complete sampled response."""
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros((), device=rewards.device, dtype=rewards.dtype)
    next_value = torch.as_tensor(last_value, device=rewards.device, dtype=rewards.dtype)

    for t in reversed(range(rewards.shape[0])):
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
        next_value = values[t]

    returns = advantages + values
    return advantages, returns


def normalize_advantages(advantages, eps=1e-8):
    if advantages.numel() <= 1:
        return advantages
    return (advantages - advantages.mean()) / (advantages.std(unbiased=False) + eps)
