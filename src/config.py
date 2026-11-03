import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ML_DIR = os.path.join(PROJECT_ROOT, "ml")

for _d in (DATA_DIR, ML_DIR):
    os.makedirs(_d, exist_ok=True)

# Motor States
CLASSES = ["NORMAL", "FAULT"]
NORMAL_IDX = 0
FAULT_IDX = 1

# Feature Engineering
WINDOW_SIZE = 100   # 1 second of data at 100Hz
STEP_SIZE = 50      # 50% overlap
FEATURES = ["rms", "variance", "peak", "crest_factor"]

# Training
RANDOM_SEED = 42
TEST_SIZE = 0.2