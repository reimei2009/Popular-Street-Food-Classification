import torch.nn as nn
from torchvision import models
from config import DEVICE

def initialize_model(num_classes):
    # Load the pre-trained ResNet-50 model with default ImageNet weights
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Replace the final fully connected layer to match the number of target classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Move the model to the selected device (CPU or GPU)
    model = model.to(DEVICE)

    return model