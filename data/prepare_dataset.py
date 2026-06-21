from datasets import load_dataset

DATASET_NAME = "Anthropic/hh-rlhf"

TRAIN_SIZE = 1000
TEST_SIZE = 200


def format_example(example):

    chosen_text = example.get("chosen", "")
    rejected_text = example.get("rejected", "")
    fallback = {
        "prompt": None,
        "chosen": None,
        "rejected": None,
    }

    # fallback an toàn
    if "Assistant:" in chosen_text:
        chosen_parts = chosen_text.split("Assistant:", 1)
    else:
        return fallback

    if "Assistant:" in rejected_text:
        rejected_parts = rejected_text.split("Assistant:", 1)
    else:
        return fallback

    prompt = chosen_parts[0].strip() + "\nAssistant:"
    chosen = chosen_parts[1].strip()
    rejected = rejected_parts[1].strip()

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }


def prepare_dataset():

    print("=" * 60)
    print("Loading Anthropic/hh-rlhf ...")
    print("=" * 60)

    dataset = load_dataset(DATASET_NAME)

    train_dataset = dataset["train"].select(range(TRAIN_SIZE))
    eval_dataset = dataset["test"].select(range(TEST_SIZE))

    # map
    train_dataset = train_dataset.map(format_example)
    eval_dataset = eval_dataset.map(format_example)

    # filter None đúng cách (QUAN TRỌNG)
    train_dataset = train_dataset.filter(
        lambda x: x["prompt"] is not None
    )

    eval_dataset = eval_dataset.filter(
        lambda x: x["prompt"] is not None
    )

    print("\nDataset loaded successfully!")
    print(f"Train samples : {len(train_dataset)}")
    print(f"Eval samples  : {len(eval_dataset)}")

    print("\nExample:")
    print(train_dataset[0]["prompt"][:200])

    return train_dataset, eval_dataset


if __name__ == "__main__":
    prepare_dataset()
