import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from tqdm import tqdm

def plot_dataset_stats():
    # Load dataset statistics from CSV file
    data = pd.read_csv('/kaggle/input/popular-street-foods/popular_street_foods/dataset_stats.csv')

    # Print number of unique classes
    print("Classes:", data['class'].nunique())

    # Print total number of images across all classes
    print("Total Images:", data['image_count'].sum())

    # Plot the distribution of image counts per class
    plt.figure(figsize=(12, 6))
    sns.barplot(x='class', y='image_count', data=data.sort_values('image_count', ascending=False))
    plt.xticks(rotation=90)
    plt.title("Image Count per Food Class")
    plt.tight_layout()
    plt.grid(True)
    plt.show()

def count(data_loader):
    real_labels = []
    for images, labels in tqdm(data_loader, desc="Counting", leave=False):
        real_labels.extend(labels.cpu().numpy().tolist())
    return np.array(real_labels)

def plot_label_distribution(train_loader, val_loader, title='Stacked Label Distribution'):
    val_labels = count(val_loader)
    train_labels = count(train_loader)

    train_counts = Counter(train_labels)
    val_counts = Counter(val_labels)

    all_labels = sorted(set(train_counts.keys()) | set(val_counts.keys()))

    train_values = [train_counts.get(l, 0) for l in all_labels]
    val_values = [val_counts.get(l, 0) for l in all_labels]

    x = np.arange(len(all_labels))

    plt.figure(figsize=(8, 5))
    plt.bar(x, train_values, label='Train', color='skyblue')
    plt.bar(x, val_values, bottom=train_values, label='Validation', color='orange')

    plt.xticks(x, all_labels)
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.title(title)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()