"""
run.py

Entry point chính cho toàn bộ project:

- Train DPO
- Train PPO
- Compare results
"""

import os
import sys


def run_dpo():
    print("\n" + "="*60)
    print("RUNNING DPO TRAINING")
    print("="*60)
    os.system(f'"{sys.executable}" train/dpo.py')


def run_ppo():
    print("\n" + "="*60)
    print("RUNNING PPO TRAINING")
    print("="*60)
    os.system(f'"{sys.executable}" train/ppo.py')


def run_compare():
    print("\n" + "="*60)
    print("RUNNING COMPARISON")
    print("="*60)
    os.system(f'"{sys.executable}" evaluation/compare.py')


def main():

    print("="*60)
    print("ALIGNMENT PROJECT RUNNER")
    print("="*60)

    print("""
Choose option:

1. Train DPO
2. Train PPO
3. Compare models
4. Run all
""")

    choice = input("Enter choice: ")

    if choice == "1":
        run_dpo()

    elif choice == "2":
        run_ppo()

    elif choice == "3":
        run_compare()

    elif choice == "4":
        run_dpo()
        run_ppo()
        run_compare()

    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
