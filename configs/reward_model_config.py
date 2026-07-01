from dataclasses import dataclass


@dataclass
class RewardModelConfig:
    base_model_path: str = "./outputs/sft_model"
    output_dir: str = "./outputs/reward_model"
    metrics_path: str = "./outputs/reward_model_metrics.txt"
    learning_rate: float = 2e-5
    num_train_epochs: int = 2
    max_length: int = 512
    max_train_samples: int | None = 150
    logging_steps: int = 10


def get_reward_model_config():
    return RewardModelConfig()
