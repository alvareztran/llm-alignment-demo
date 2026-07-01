import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def normalize_rows(matrix):
    row_sums = matrix.sum(axis=1, keepdims=True)
    return matrix / np.clip(row_sums, a_min=1e-12, a_max=None)


def split_index(num_examples):
    return max(1, int(np.ceil(num_examples * 0.55)))


def sample_labels(num_examples):
    split = split_index(num_examples)
    return [
        f"ID {i + 1}" if i < split else f"OOD {i + 1 - split}"
        for i in range(num_examples)
    ]


def response_labels(num_examples):
    split = split_index(num_examples)
    return [
        f"ID target {i + 1}" if i < split else f"OOD safe {i + 1 - split}"
        for i in range(num_examples)
    ]


def build_preference_matrix(num_examples):
    matrix = np.zeros((num_examples, num_examples), dtype=np.float64)
    np.fill_diagonal(matrix, 1.0)
    return matrix


def build_dpo_matrix(num_examples):
    matrix = np.full((num_examples, num_examples), 0.01, dtype=np.float64)
    split = split_index(num_examples)

    for i in range(num_examples):
        if i < split:
            # In-distribution: direct preference pairs are covered, so DPO stays confident.
            matrix[i, i] = 0.80
            if i + 1 < num_examples:
                matrix[i, i + 1] = 0.10
        else:
            # OOD: DPO overfits the nearest seen preference pattern and assigns
            # high mass to a misleading response instead of the safe target.
            lure_col = max(0, split - 1)
            nearby_seen_col = max(0, split - 2)
            matrix[i, i] = 0.18
            matrix[i, lure_col] = 0.46
            matrix[i, nearby_seen_col] = 0.20

    return normalize_rows(matrix)


def build_ppo_matrix(num_examples):
    matrix = np.full((num_examples, num_examples), 0.01, dtype=np.float64)
    split = split_index(num_examples)

    for i in range(num_examples):
        # Reward-guided behavior keeps probability on the safe target even when
        # the prompt is outside the direct preference-pair coverage.
        matrix[i, i] = 0.68 if i < split else 0.74
        matrix[i, (i + 1) % num_examples] = 0.08
        matrix[i, (i - 1) % num_examples] = 0.06

    return normalize_rows(matrix)


def summarize_behavior(preference_matrix, dpo_matrix, ppo_matrix):
    split = split_index(preference_matrix.shape[0])
    target_cols = np.arange(preference_matrix.shape[0])
    dpo_target = dpo_matrix[np.arange(dpo_matrix.shape[0]), target_cols]
    ppo_target = ppo_matrix[np.arange(ppo_matrix.shape[0]), target_cols]
    return {
        "id_rows": split,
        "ood_rows": int(preference_matrix.shape[0] - split),
        "dpo_id_target_probability": round(float(dpo_target[:split].mean()), 3),
        "dpo_ood_target_probability": round(float(dpo_target[split:].mean()), 3) if split < len(dpo_target) else None,
        "ppo_id_target_probability": round(float(ppo_target[:split].mean()), 3),
        "ppo_ood_target_probability": round(float(ppo_target[split:].mean()), 3) if split < len(ppo_target) else None,
    }


def save_summary(path, summary):
    if not path:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved toy probability summary to {path}")


def plot_heatmaps(preference_matrix, dpo_matrix, ppo_matrix, output_path):
    labels_x = response_labels(preference_matrix.shape[1])
    labels_y = sample_labels(preference_matrix.shape[0])
    split = split_index(preference_matrix.shape[0])

    fig, axes = plt.subplots(1, 3, figsize=(19, max(5.5, preference_matrix.shape[0] * 0.72)))
    fig.suptitle(
        "Toy OOD demo: DPO can follow a misleading seen pattern; PPO follows the reward target",
        fontsize=14,
        y=1.03,
    )
    fig.tight_layout(pad=4.0)
    fig.subplots_adjust(bottom=0.28)

    images = [
        (preference_matrix, "1. Covered preference pairs", "Blues", False),
        (dpo_matrix, "2. DPO-like: brittle on OOD", "magma", True),
        (ppo_matrix, "3. PPO-like: reward-safe on OOD", "viridis", True),
    ]

    for ax, (matrix, title, cmap, colorbar) in zip(axes, images):
        im = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_title(title, pad=12)
        ax.set_xlabel("Candidate response")
        ax.set_ylabel("Prompt / sample")
        ax.set_xticks(range(len(labels_x)))
        ax.set_xticklabels(labels_x, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(labels_y)))
        ax.set_yticklabels(labels_y, fontsize=9)
        if 0 < split < matrix.shape[0]:
            ax.axhline(split - 0.5, color="white", linewidth=2.0)
            ax.text(
                -0.48,
                split - 0.5,
                "OOD starts",
                va="bottom",
                ha="left",
                fontsize=8,
                color="white",
                bbox={"facecolor": "black", "alpha": 0.55, "pad": 2, "edgecolor": "none"},
            )

        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = matrix[row, col]
                text_color = "white" if value > 0.45 else "black"
                ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=7, color=text_color)

        if colorbar:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    note = (
        "Interpretation: this is a controlled visualization, not a benchmark. "
        "Rows above the divider are covered by preference data.\n"
        "On OOD rows, DPO-like behavior is pulled toward a familiar ID target, while PPO-like behavior "
        "stays concentrated on the OOD safe reward target."
    )
    fig.text(0.5, 0.02, note, ha="center", va="bottom", fontsize=10, wrap=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Create a lightweight paper-style toy heatmap without loading LLM checkpoints."
    )
    parser.add_argument("--num-examples", type=int, default=8, help="Number of toy prompt/response pairs.")
    parser.add_argument("--output", type=str, default="./outputs/probability_heatmaps.png", help="Output image path.")
    parser.add_argument("--output-json", type=str, default="./outputs/probability_heatmaps_summary.json", help="Output summary JSON path.")
    args = parser.parse_args()

    if args.num_examples < 4:
        raise ValueError("--num-examples must be at least 4 to show both in-distribution and OOD rows.")

    preference_matrix = build_preference_matrix(args.num_examples)
    dpo_matrix = build_dpo_matrix(args.num_examples)
    ppo_matrix = build_ppo_matrix(args.num_examples)
    summary = summarize_behavior(preference_matrix, dpo_matrix, ppo_matrix)

    print("Building OOD-focused toy probability heatmaps...")
    plot_heatmaps(preference_matrix, dpo_matrix, ppo_matrix, args.output)
    save_summary(args.output_json, summary)
    print(f"Saved toy probability heatmap to {args.output}")


if __name__ == "__main__":
    main()
