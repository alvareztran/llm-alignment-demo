import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import traceback

try:
    print("Trying to import train.reward_model...")
    import train.reward_model
    print("Import successful!")
except Exception as e:
    print("Import failed with exception:")
    traceback.print_exc()
