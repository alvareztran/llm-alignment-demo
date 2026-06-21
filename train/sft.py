"""Optional SFT stage placeholder.

This project starts from an instruct model, so the default pipeline does not
run SFT. A paper-faithful full RLHF setup may add supervised cross-entropy
training here before DPO/PPO.
"""


def main():
    print("SFT stage is optional and is not enabled for this small-scale project.")


if __name__ == "__main__":
    main()
