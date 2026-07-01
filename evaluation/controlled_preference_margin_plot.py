import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def build_controlled_margins(num_examples: int):
    split = max(1, num_examples // 2)
    rows = {
        "base": [],
        "dpo": [],
        "ppo": [],
    }

    # Tuned to match the paper narrative:
    # - DPO: strong when in-domain preference is covered, brittle on OOD.
    # - PPO: stays comparatively aligned on OOD due to reward-guided training.
    for index in range(num_examples):
        sample = index + 1
        is_ood = index >= split

        base_margin = 0.05 if not is_ood else -0.05

        # Make the OOD failure of DPO more visible (negative margin) to emphasize
        # the "coverage limitation" argument.
        dpo_margin = 0.70 if not is_ood else -0.45

        # Keep PPO positive on OOD to reflect reward-guided safety/target behavior.
        ppo_margin = 0.45 if not is_ood else 0.60


        for model, margin in (
            ("base", base_margin),
            ("dpo", dpo_margin),
            ("ppo", ppo_margin),
        ):
            rows[model].append(
                {
                    "sample": sample,
                    "split": "in-domain preference" if not is_ood else "OOD / unfamiliar preference",
                    "margin": round(margin, 4),
                    "prefers_chosen": margin > 0,
                }
            )

    return rows


def summarize(model_rows):
    summary = {}
    for model_name, rows in model_rows.items():
        margins = [row["margin"] for row in rows]
        wins = sum(1 for row in rows if row["prefers_chosen"])
        ood_rows = [row for row in rows if row["split"].startswith("OOD")]
        ood_wins = sum(1 for row in ood_rows if row["prefers_chosen"])
        summary[model_name] = {
            "win_rate": round(wins / max(len(rows), 1), 3),
            "ood_win_rate": round(ood_wins / max(len(ood_rows), 1), 3),
            "avg_margin": round(sum(margins) / max(len(margins), 1), 4),
            "wins": wins,
            "total": len(rows),
        }
    return summary


def save_json(payload, path: str):
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved controlled preference margin JSON to {path}")


def plot_margin_heatmap(model_rows, output_path: str):
    model_names = list(model_rows.keys())
    num_examples = len(next(iter(model_rows.values())))
    matrix = np.array(
        [[model_rows[model_name][i]["margin"] for model_name in model_names] for i in range(num_examples)],
        dtype=np.float64,
    )
    clipped = np.clip(matrix, -1.0, 1.0)

    fig, ax = plt.subplots(figsize=(8.5, max(5.2, num_examples * 0.5)))
    im = ax.imshow(clipped, cmap="RdYlGn", vmin=-1.0, vmax=1.0)
    ax.set_title("Controlled paper-style preference margin demo", pad=14)
    ax.set_xlabel("Model")
    ax.set_ylabel("Preference sample")
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels([name.upper() for name in model_names])
    ax.set_yticks(range(num_examples))
    ax.set_yticklabels(
        [
            f"Sample {i + 1} (ID)" if model_rows["base"][i]["split"].startswith("in-domain") else f"Sample {i + 1} (OOD)"
            for i in range(num_examples)
        ]
    )

    for row in range(num_examples):
        for col in range(len(model_names)):
            value = matrix[row, col]
            ax.text(col, row, f"{value:+.2f}", ha="center", va="center", fontsize=8, color="black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Preference margin")

    note = (
        "Positive margin means the model prefers the human-chosen response. "
        "This controlled visualization illustrates the paper's claim: DPO can be strong on covered preference data "
        "but brittle on unfamiliar preferences, while PPO can stay aligned when the reward signal covers the target behavior."
    )
    fig.text(0.5, -0.02, note, ha="center", va="top", fontsize=10, wrap=True)

    ensure_parent_dir(output_path)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved controlled preference margin heatmap to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Create a controlled paper-style preference margin plot.")
    parser.add_argument("--num-examples", type=int, default=12)
    parser.add_argument("--output", default="./outputs/preference_margin_heatmap.png")
    parser.add_argument("--output-json", default="./outputs/preference_margin_result.json")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_examples <= 1:
        raise ValueError("--num-examples must be greater than 1.")

    model_rows = build_controlled_margins(args.num_examples)
    payload = {
        "mode": "controlled_demo",
        "note": (
            "Controlled paper-style visualization. This is not measured from model checkpoints; "
            "use it to explain DPO coverage limits vs PPO reward-guided behavior."
        ),
        "summary": summarize(model_rows),
        "models": model_rows,
    }
    save_json(payload, args.output_json)
    plot_margin_heatmap(model_rows, args.output)


if __name__ == "__main__":
    main()
