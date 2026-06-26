import sys
import subprocess


def run_command(args):
    subprocess.run([sys.executable, *args], check=False)


def run_sft():
    print("\n" + "=" * 60)
    print("RUNNING MINI SFT TRAINING")
    print("=" * 60)
    run_command(["train/sft.py"])


def run_dpo():
    print("\n" + "=" * 60)
    print("RUNNING DPO TRAINING")
    print("=" * 60)
    run_command(["train/dpo.py"])


def run_ppo():
    print("\n" + "=" * 60)
    print("RUNNING PPO TRAINING")
    print("=" * 60)
    run_command(["train/ppo.py"])


def run_compare():
    print("\n" + "=" * 60)
    print("RUNNING COMPARISON")
    print("=" * 60)
    prompt = input("Enter prompt for compare: ").strip()

    if prompt:
        run_command(["evaluation/compare.py", "--prompt", prompt])
    else:
        run_command(["evaluation/compare.py"])


def main():
    print("=" * 60)
    print("ALIGNMENT PROJECT RUNNER")
    print("=" * 60)

    print(
        """
Choose option:

1. Train mini SFT
2. Train DPO
3. Train PPO
4. Compare models
5. Run all
"""
    )

    choice = input("Enter choice: ")

    if choice == "1":
        run_sft()
    elif choice == "2":
        run_dpo()
    elif choice == "3":
        run_ppo()
    elif choice == "4":
        run_compare()
    elif choice == "5":
        run_sft()
        run_dpo()
        run_ppo()
        run_compare()
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
