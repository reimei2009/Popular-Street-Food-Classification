import os
import re
import shutil
from config import MODEL_FOLDER, SAMPLE_FOLDER

def setup_directories(clear=True):
    if clear:
        for folder in [MODEL_FOLDER, SAMPLE_FOLDER]:
            try:
                shutil.rmtree(folder)
            except FileNotFoundError:
                pass
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    os.makedirs(SAMPLE_FOLDER, exist_ok=True)

def get_latest_epoch(folder: str = './', prefix="model_epoch_", suffix=".pth"):
    max_epoch = 0
    for filename in os.listdir(folder):
        match = re.match(fr"{prefix}(\d+){suffix}", filename)
        if match:
            epoch_num = int(match.group(1))
            if epoch_num > max_epoch:
                max_epoch = epoch_num
    return max_epoch