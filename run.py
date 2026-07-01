import sys
import subprocess


def run_command(args, report_failure=True):
    result = subprocess.run([sys.executable, *args], check=False)
    if report_failure and result.returncode != 0:
        print(f"\nCommand failed with exit code {result.returncode}:")
        print(f"{sys.executable} {' '.join(args)}")
    return result.returncode


def ask_yes_no(message):
    answer = input(f"{message} [y/N]: ").strip().lower()
    return answer in {"y", "yes"}


def run_real_probability_plot():
    print("\n" + "=" * 60)
    print("DPO VS PPO REAL PROBABILITY HEATMAP")
    print("=" * 60)
    print("This does not train again. It loads trained DPO and PPO checkpoints from outputs/.")
    print("It scores each candidate response against each prompt and plots row-normalized probabilities.")

    num_examples = ask_positive_int("Number of prompt-response examples [8]: ", 8, min_value=2)
    if num_examples is None:
        return

    output_path = "./outputs/probability_heatmaps_real.png"
    exit_code = run_command(
        [
            "evaluation/plot_probabilities.py",
            "--num-examples",
            num_examples,
            "--output",
            output_path,
            "--output-json",
            "./outputs/probability_heatmaps_real_summary.json",
            "--device",
            "cpu",
        ]
    )
    if exit_code == 0:
        print(f"\nSaved real probability heatmap to: {output_path}")
        print("./outputs/probability_heatmaps_real_summary.json")


def run_compare():
    print("\n" + "=" * 60)
    print("COMPARE EXISTING MODELS")
    print("=" * 60)
    print("This does not train again. It only loads saved checkpoints from outputs/.")
    prompt = input("Enter prompt for compare: ").strip()

    if prompt:
        run_command(["evaluation/compare.py", "--prompt", prompt])
    else:
        run_command(["evaluation/compare.py"])


def ask_positive_int(prompt, default_value, min_value=1):
    raw_value = input(prompt).strip()
    if not raw_value:
        return str(default_value)
    if raw_value.isdigit() and int(raw_value) >= min_value:
        return raw_value
    if min_value == 1:
        print("Invalid number of examples. Please enter a positive integer.")
    else:
        print(f"Invalid number of examples. Please enter an integer >= {min_value}.")
    return None


def run_toy_probability_plot():
    print("\n" + "=" * 60)
    print("PAPER-STYLE TOY PROBABILITY HEATMAP DEMO")
    print("=" * 60)
    print("This does not train again and does not load model checkpoints.")
    print("It creates a controlled toy visualization for the paper discussion.")

    num_examples = ask_positive_int("Number of examples [8]: ", 8, min_value=4)
    if num_examples is None:
        return

    output_path = "./outputs/probability_heatmaps.png"
    exit_code = run_command(
        [
            "evaluation/plot_toy_probabilities.py",
            "--num-examples",
            num_examples,
            "--output",
            output_path,
            "--output-json",
            "./outputs/probability_heatmaps_summary.json",
        ]
    )
    if exit_code == 0:
        print(f"\nSaved probability heatmap to: {output_path}")
        print("./outputs/probability_heatmaps_summary.json")


def run_reward_model_training():
    print("\n" + "=" * 60)
    print("RUNNING REWARD MODEL TRAINING")
    print("=" * 60)
    print("Warning: this will overwrite outputs/reward_model and outputs/reward_model_metrics.txt.")
    run_command(["train/reward_model.py"])


def run_ood_benchmark():
    print("\n" + "=" * 60)
    print("OOD BENCHMARK")
    print("=" * 60)
    print("This does not train again. It runs a small rule-based OOD benchmark.")
    exit_code = run_command(
        [
            "evaluation/ood_benchmark.py",
            "--output-json",
            "./outputs/ood_benchmark_result.json",
            "--output-html",
            "./outputs/ood_benchmark_report.html",
        ]
    )
    if exit_code == 0:
        print("\nSaved OOD benchmark outputs:")
        print("./outputs/ood_benchmark_result.json")
        print("./outputs/ood_benchmark_report.html")


def run_tests():
    print("\n" + "=" * 60)
    print("RUNNING SANITY TESTS")
    print("=" * 60)
    run_command(["-m", "unittest", "tests/test_sanity.py"])


def run_sft():
    print("\n" + "=" * 60)
    print("RUNNING MINI SFT TRAINING")
    print("=" * 60)
    print("Warning: this will overwrite outputs/sft_model and outputs/sft_metrics.txt.")
    run_command(["train/sft.py"])


def run_dpo():
    print("\n" + "=" * 60)
    print("RUNNING DPO TRAINING")
    print("=" * 60)
    print("Warning: this will overwrite outputs/dpo_model and outputs/dpo_metrics.txt.")
    run_command(["train/dpo.py"])


def run_ppo():
    print("\n" + "=" * 60)
    print("RUNNING PPO TRAINING")
    print("=" * 60)
    print("Warning: this will overwrite outputs/ppo_model, outputs/ppo_value_model, and outputs/ppo_metrics.txt.")
    run_command(["train/ppo.py"])


def run_training_pipeline():
    print("\n" + "=" * 60)
    print("RUNNING FULL TRAINING PIPELINE")
    print("=" * 60)
    print("This will train SFT, Reward Model, DPO, and PPO again, then run compare and OOD benchmark.")
    print("Use this only when you intentionally want to regenerate checkpoints.")

    if not ask_yes_no("Continue with full training?"):
        print("Cancelled.")
        return

    run_sft()
    run_reward_model_training()
    run_dpo()
    run_ppo()
    run_compare()
    run_ood_benchmark()


def main():
    print("=" * 60)
    print("ALIGNMENT PROJECT RUNNER")
    print("=" * 60)

    print(
        """
Choose option:

1. Plot trained DPO vs PPO probability heatmap
2. Plot toy probability heatmaps (paper-style demo, no model loading)
3. Run OOD benchmark (recommended comparison report)
4. Run sanity tests
5. Train mini SFT
6. Train Reward Model
7. Train DPO
8. Train PPO
9. Train all, then compare + benchmark

10. PAPER DEMO (toy heatmap + OOD benchmark + compare)
"""
    )

    choice = input("Enter choice: ")


    if choice == "1":
        run_real_probability_plot()
    elif choice == "2":
        run_toy_probability_plot()
    elif choice == "3":
        run_ood_benchmark()
    elif choice == "4":
        run_tests()
    elif choice == "5":
        run_sft()
    elif choice == "6":
        run_reward_model_training()
    elif choice == "7":
        run_dpo()
    elif choice == "8":
        run_ppo()
    elif choice == "9":
        run_training_pipeline()
    elif choice == "10":
        # Paper demo tailored to the narrative (DPO brittle on OOD vs PPO reward-guided robustness).
        print("\n" + "=" * 60)
        print("PAPER DEMO: TOY HEATMAP + OOD BENCHMARK + COMPARE")
        print("=" * 60)
        # 1) Toy explanation heatmaps (no model needed)
        run_toy_probability_plot()
        # 2) OOD benchmark (requires trained checkpoints)
        run_ood_benchmark()
        # 3) Optional human-readable compare for a single prompt
        run_compare()
    else:
        print("Invalid choice")



if __name__ == "__main__":
    main()
