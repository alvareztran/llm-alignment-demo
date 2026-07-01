import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import traceback
try:
    print("Importing models.reward_model...")
    import models.reward_model
    print("Success!")
except Exception as e:
    print("Failed:")
    traceback.print_exc()
