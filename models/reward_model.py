from ppo.reward import compute_reward


class RuleBasedRewardModel:
    """Explicit scalar reward function r(x, y) used by PPO."""

    def __call__(self, prompt: str, response: str) -> float:
        return float(compute_reward(prompt, response))
