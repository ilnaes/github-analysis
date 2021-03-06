# Base params
WANDB = False
SEED = 69420
DEBUG = False
TARGET = "language"
NUM_LABELS = 5

MAX_LEN = 256

# Split params
N_FOLDS = 1 # Colab so not too many folds

# Model
MODEL_NAME = "distilroberta-base"

# Training params
LR = 5e-5
HEAD_LR = 1e-3
BATCH_SIZE = 24
EPOCHS = 20
PATIENCE = 0
WEIGHT_DECAY = 0.01
HIGH_DROPOUT = 0.1
# N_LAST_HIDDEN = 6
HIDDEN_SIZE = 768

# Paths
# BASE_DIR = "/content/drive/MyDrive/misc/data/github_analysis/"
BASE_DIR = "./"
TRAIN_FILE = BASE_DIR + "data/clean.csv"
INDEX_FILE = BASE_DIR + "data/indices.csv"
OUTPUT_DIR = BASE_DIR + "outputs/"
MODEL_SAVE_DIR = OUTPUT_DIR
