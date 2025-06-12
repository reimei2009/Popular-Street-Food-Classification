from collections import defaultdict
import random
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from config import DATA_FOLDER, BATCH_SIZE, NUM_WORKERS, CLASS_NAMES

def load_datasets(train_transform, val_transform):
    # Load full dataset
    full_dataset = ImageFolder(DATA_FOLDER, transform=None)
    
    # Set class names globally
    global CLASS_NAMES
    CLASS_NAMES = full_dataset.classes
    num_classes = len(full_dataset.classes)

    # Split dataset into train and validation
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(full_dataset.samples):
        class_indices[label].append(idx)

    train_indices = []
    val_indices = []

    for label, indices in class_indices.items():
        random.shuffle(indices)
        train_indices.extend(indices[:150])
        val_indices.extend(indices[150:])

    # Create train and validation subsets
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    # Apply transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    return train_loader, val_loader, num_classes, train_dataset, val_dataset, full_dataset