print("Importing configs.reward_model_config...")
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from configs.reward_model_config import get_reward_model_config
print("Success!", get_reward_model_config())
