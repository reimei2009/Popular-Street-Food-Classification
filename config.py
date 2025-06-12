import os
import random
import numpy as np
import torch
import torch.nn.functional as F

def seed_everything(seed):
    # Set fixed seed for reproducibility across libraries
    random.seed(seed)                          # Python random module seed
    np.random.seed(seed)                       # NumPy seed
    torch.manual_seed(seed)                    # PyTorch CPU seed
    torch.cuda.manual_seed(seed)               # PyTorch CUDA seed for single GPU
    torch.cuda.manual_seed_all(seed)           # PyTorch CUDA seed for all GPUs if using multi-GPU
    # Ensure deterministic behavior for CUDA convolution operations
    torch.backends.cudnn.deterministic = True
    # Disable benchmark mode to prevent nondeterministic algorithm selection
    torch.backends.cudnn.benchmark = False

# Random seed for reproducibility
SEED = 42
seed_everything(SEED)

# Device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = os.cpu_count()

# Directory configurations
MODEL_FOLDER = "models"
SAMPLE_FOLDER = "sample"
DATA_FOLDER = '/kaggle/input/popular-street-foods/popular_street_foods/dataset'
PLOT_IMAGE_PATH = 'plot_image.png'
HISTORY_PATH = "history.json"
MODEL_PATH = 'best_f1_model.pth'

# Training hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 20
IMG_SIZE = 224
CRITERION = F.cross_entropy

# Best metrics initialization
BEST_LOSS = float('inf')
BEST_ACCURACY = 0.0
BEST_F1 = 0.0
HISTORY = {
    "train": {"loss": [], "accuracy": [], "f1": []},
    "val": {"loss": [], "accuracy": [], "f1": []}
}

# Placeholder for class names (set after dataset loading)
CLASS_NAMES = None