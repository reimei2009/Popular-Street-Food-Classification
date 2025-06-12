import os
import json
import matplotlib.pyplot as plt
from config import SAMPLE_FOLDER

def plot_history(history: dict[str, dict[str, list[float]]],
                 plot_image_path: str = 'plot_image.png',
                 history_path: str = "history.json",
                 save: bool = True):
    train_history = history['train']
    val_history = history['val']
    epochs = range(1, len(train_history['loss']) + 1)
    
    history_path = os.path.join(SAMPLE_FOLDER, history_path)
    plot_image_path = os.path.join(SAMPLE_FOLDER, plot_image_path)
    
    plt.figure(figsize=(14, 5))
    
    for i, k in enumerate(train_history):
        plt.subplot(1, 3, i + 1)
        plt.plot(epochs, train_history[k], 'bo-', label=f'Train {k.capitalize()}')
        plt.plot(epochs, val_history[k], 'ro-', label=f'Val {k.capitalize()}')
        plt.xlabel('Epoch')
        plt.ylabel(k.capitalize())
        plt.title(f'{k.capitalize()} over Epochs')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    
    if save:
        plt.savefig(plot_image_path)
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)
    
    plt.show()