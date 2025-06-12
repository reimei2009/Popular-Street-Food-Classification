import os
import random
import torch
import torch.optim as optim
from tqdm import tqdm
from data.dataset import load_datasets
from data.transforms import train_transform, val_transform
from data.eda import plot_dataset_stats, plot_label_distribution
from models.model import initialize_model
from models.train import fit, evaluate
from utils.file_utils import setup_directories
from utils.metrics import plot_history
from utils.visualization import visualize_predictions, plot_confusion_matrix
from config import (DEVICE, EPOCHS, CRITERION, MODEL_PATH, MODEL_FOLDER, PLOT_IMAGE_PATH, 
                    HISTORY_PATH, HISTORY, DATA_FOLDER, BATCH_SIZE, NUM_WORKERS, CLASS_NAMES)

def main():
    # Setup directories
    setup_directories(clear=True)

    # Perform EDA
    plot_dataset_stats()

    # Load datasets
    train_loader, val_loader, num_classes, train_dataset, val_dataset, full_dataset = load_datasets(
        train_transform, val_transform)
    
    # Plot label distribution
    plot_label_distribution(train_loader, val_loader)

    # Initialize model and optimizer
    model = initialize_model(num_classes)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train model
    history = fit(model=model, optimizer=optimizer, device=DEVICE, epochs=EPOCHS,
                  train_loader=train_loader, val_loader=val_loader, criterion=CRITERION,
                  history=HISTORY)
    
    # Plot training history
    plot_history(history, plot_image_path=PLOT_IMAGE_PATH, history_path=HISTORY_PATH, save=True)

    # Load and evaluate best model
    model_path = os.path.join(MODEL_FOLDER, MODEL_PATH)
    if os.path.exists(model_path):
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(model_path, weights_only=True))
        else:
            model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')))
        model.to(DEVICE)
        val_result = evaluate(model, val_loader, DEVICE, CRITERION)
        print(f"Best model loaded with "
              f"f1 score: {val_result['f1']:.04f} | "
              f"accuracy: {val_result['accuracy']:.04f} | "
              f"Loss: {val_result['loss']:.04f}")

    # Select random images for visualization
    subfolder = os.listdir(DATA_FOLDER)
    subfolder = [os.path.join(DATA_FOLDER, x) for x in sorted(subfolder)]
    k = 10
    selection = {
        CLASS_NAMES[i]: random.sample(os.listdir(subfolder[i]), k)
        for i in range(len(subfolder))
    }
    visualize_predictions(model, selection, CLASS_NAMES)

    # Evaluate on full dataset
    test_dataset = torch.utils.data.ImageFolder(root=DATA_FOLDER, transform=val_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, 
                                              shuffle=False, num_workers=NUM_WORKERS)
    
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating", leave=False):
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_result = evaluate(model, test_loader, DEVICE, CRITERION)
    print(f"Full dataset test with "
          f"f1 score: {val_result['f1']:.04f} | "
          f"accuracy: {val_result['accuracy']:.04f} | "
          f"Loss: {val_result['loss']:.04f}")

    # Plot confusion matrix
    plot_confusion_matrix(all_labels, all_preds, CLASS_NAMES)

if __name__ == "__main__":
    main()