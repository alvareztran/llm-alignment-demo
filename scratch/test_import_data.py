print("Importing data.prepare_dataset...")
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from data.prepare_dataset import prepare_dataset
print("Success!", prepare_dataset)
