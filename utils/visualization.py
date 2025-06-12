import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import confusion_matrix
from config import DEVICE, SAMPLE_FOLDER, CLASS_NAMES, DATA_FOLDER

def predict_image(model, source: str | np.ndarray, threshold=0.4, unknown_label="unknown"):
    model.eval()
    if isinstance(source, str):
        image_path = source.strip()
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"{image_path} does not exist")
        image = Image.open(image_path).convert('RGB')
    elif isinstance(source, np.ndarray):
        if source.ndim == 2:
            source = np.stack([source] * 3, axis=-1)
        elif source.ndim == 3 and source.shape[2] == 1:
            source = np.repeat(source, 3, axis=2)
        image = Image.fromarray(source.astype('uint8')).convert('RGB')
    else:
        raise TypeError("Input must be a file path (str) or image array (np.ndarray)")
    
    from data.transforms import val_transform
    input_tensor = val_transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
    
    if confidence.item() < threshold:
        return unknown_label
    else:
        return CLASS_NAMES[predicted.item()]

def visualize(model, paths, real_label, threshold=0.2):
    plt.figure(figsize=(15, 8))
    for i, image_path in enumerate(paths):
        pred = predict_image(model, image_path, threshold)
        is_correct = (pred == real_label)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.subplot(len(paths) // 5 + 1, 5, i + 1)
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"Predict: {pred}\nReal label: {real_label}", color='green' if is_correct else 'red')
    
    plt.savefig(os.path.join(SAMPLE_FOLDER, f"Predict_{real_label}.png"))
    plt.tight_layout()
    plt.show()

def visualize_predictions(model, selection, class_names):
    subfolder = os.listdir(DATA_FOLDER)
    subfolder = [os.path.join(DATA_FOLDER, x) for x in sorted(subfolder)]
    for i, class_name in enumerate(selection):
        paths = [os.path.join(sorted(subfolder)[i], filename) for filename in selection[class_name]]
        visualize(model, paths, class_names[i])

def plot_confusion_matrix(all_labels, all_preds, class_names):
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix Heatmap')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(SAMPLE_FOLDER, 'confusion_matrix.png'))
    plt.show()