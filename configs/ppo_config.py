from dataclasses import dataclass


@dataclass
class PPOConfig:
    output_dir: str = "./outputs/ppo_model"
    value_output_dir: str = "./outputs/ppo_value_model"
    metrics_path: str = "./outputs/ppo_metrics.txt"
    learning_rate: float = 1e-6
    value_learning_rate: float = 1e-5
    num_train_epochs: int = 1
    ppo_epochs: int = 2
    batch_size: int = 1
    max_prompt_length: int = 256
    max_new_tokens: int = 128
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    kl_coef: float = 0.05
    gamma: float = 1.0
    gae_lambda: float = 0.95
    logging_steps: int = 10
    max_train_samples: int | None = 30


def get_ppo_config():
    return PPOConfig()


if __name__ == "__main__":
    print(get_ppo_config())
