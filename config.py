import os

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
ANOMALY_DIR = os.path.join(WORKSPACE_DIR, "Anomaly-Videos-Part-3")
NORMAL_DIR = os.path.join(WORKSPACE_DIR, "Normal_Videos_for_Event_Recognition")
CACHE_DIR = os.path.join(WORKSPACE_DIR, "cached_features")
# Baseline split file (preserved): os.path.join(WORKSPACE_DIR, "dataset_split.json")
SPLIT_FILE = os.path.join(WORKSPACE_DIR, "dataset_split_expanded.json")

# Feature Extraction Settings
SAMPLE_FPS = 5
BACKBONE_NAME = "mobilenet_v3_large"
FEATURE_DIM = 960

# Temporal Windowing Settings
WINDOW_SIZE = 16  # ~3.2 seconds at 5 FPS
STRIDE = 8        # 50% overlap

# Pooling Settings
POOLING_METHOD = "topk_percentage"  # Options: "topk_percentage", "max", "topk_count"
TOPK_PERCENTAGE = 0.20
TOPK_COUNT = 3

# Training Settings
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 42
