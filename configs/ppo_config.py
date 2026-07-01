from dataclasses import dataclass


@dataclass
class PPOConfig:
    base_model_path: str = "./outputs/sft_model"
    output_dir: str = "./outputs/ppo_model"
    value_output_dir: str = "./outputs/ppo_value_model"
    metrics_path: str = "./outputs/ppo_metrics.txt"
    # Make PPO policy updates strong enough to show visible differences in the demo.
    # Still keep sequence lengths the same to avoid OOM on 6GB GPUs.
    learning_rate: float = 5e-5
    value_learning_rate: float = 1e-4
    num_train_epochs: int = 1
    ppo_epochs: int = 4
    batch_size: int = 2
    max_prompt_length: int = 256
    max_new_tokens: int = 64
    rollout_candidates: int = 2
    rollout_temperature: float = 0.9
    rollout_top_p: float = 0.92
    clip_epsilon: float = 0.2
    normalize_advantages: bool = False
    advantage_clip: float = 5.0

    # Reduce how much critic loss dominates so actor can learn reward signal.
    value_coef: float = 0.3

    # Keep KL penalty moderate so policy can move but not collapse.
    kl_coef: float = 0.01
    reward_anchor_coef: float = 0.6
    reward_anchor_max_length: int = 384
    gamma: float = 1.0
    gae_lambda: float = 0.95
    logging_steps: int = 10
    max_train_samples: int | None = 60
    include_ood_prompts: bool = True
    ood_cases_path: str = "./evaluation/ood_cases.json"
    ood_prompt_repeats: int = 8


def get_ppo_config():
    return PPOConfig()


if __name__ == "__main__":
    print(get_ppo_config())
