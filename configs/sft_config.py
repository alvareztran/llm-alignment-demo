from dataclasses import dataclass


@dataclass
class SFTConfig:
    output_dir: str = "./outputs/sft_model"
    metrics_path: str = "./outputs/sft_metrics.txt"
    learning_rate: float = 2e-5
    num_train_epochs: int = 1
    max_length: int = 512
    logging_steps: int = 10
    max_train_samples: int | None = 100


def get_sft_config():
    return SFTConfig()


if __name__ == "__main__":
    print(get_sft_config())
