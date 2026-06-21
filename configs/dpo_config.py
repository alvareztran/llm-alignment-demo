from dataclasses import dataclass


@dataclass
class DPOConfig:
    output_dir: str = "./outputs/dpo_model"
    metrics_path: str = "./outputs/dpo_metrics.txt"
    learning_rate: float = 5e-6
    beta: float = 0.1
    num_train_epochs: int = 1
    train_batch_size: int = 2
    eval_batch_size: int = 2
    max_length: int = 512
    max_prompt_length: int = 256
    logging_steps: int = 10
    max_train_samples: int | None = 50


def get_dpo_config():
    return DPOConfig()


if __name__ == "__main__":
    print(get_dpo_config())
