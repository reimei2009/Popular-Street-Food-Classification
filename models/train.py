import torch
import torch.nn as nn
import time
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from config import DEVICE, MODEL_FOLDER, BEST_LOSS, BEST_ACCURACY, BEST_F1
from utils.file_utils import get_latest_epoch
import os

@torch.no_grad()
def evaluate(model: nn.Module, data_loader: torch.utils.data.DataLoader, 
             device: torch.device, criterion):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0

    for images, labels in tqdm(data_loader, desc="Evaluating", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return {'loss': avg_loss, 'accuracy': accuracy, 'f1': f1}

def train(model: nn.Module, train_loader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Adam, device: torch.device,
          criterion):
    model.train()
    losses = []
    all_preds = []
    all_labels = []
    correct_train = 0
    total_train = 0

    loop = tqdm(train_loader, desc="Training", leave=False)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        losses.append(loss.item())
        _, predicted = torch.max(outputs, 1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        all_preds.extend(predicted.cpu())
        all_labels.extend(labels.cpu())
        accuracy = (predicted == labels).sum().item() / labels.size(0)
        f1 = f1_score(labels.cpu(), predicted.cpu(), average='macro')
        loop.set_postfix(loss=loss.item(), accuracy=accuracy, f1=f1)

    avg_loss = np.mean(losses)
    accuracy = correct_train / total_train
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return {'loss': avg_loss, 'accuracy': accuracy, 'f1': f1}

def save_best_models(model: nn.Module, 
                     val_result: dict, epoch: int, 
                     loss_save_path, acc_save_path, f1_save_path):
    global BEST_LOSS, BEST_ACCURACY, BEST_F1
    updated = False

    loss_save_path = os.path.join(MODEL_FOLDER, loss_save_path)
    acc_save_path = os.path.join(MODEL_FOLDER, acc_save_path)
    f1_save_path = os.path.join(MODEL_FOLDER, f1_save_path)

    if val_result['loss'] < BEST_LOSS:
        BEST_LOSS = val_result['loss']
        torch.save(model.state_dict(), loss_save_path)
        print(f"✅Best loss model saved at epoch {epoch + 1:02d}")
        updated = True

    if val_result['accuracy'] > BEST_ACCURACY:
        BEST_ACCURACY = val_result['accuracy']
        torch.save(model.state_dict(), acc_save_path)
        print(f"✅Best accuracy model saved at epoch {epoch + 1:02d}")
        updated = True

    if val_result['f1'] > BEST_F1:
        BEST_F1 = val_result['f1']
        torch.save(model.state_dict(), f1_save_path)
        print(f"✅Best F1 model saved at epoch {epoch + 1:02d}")
        updated = True

    return updated

def save_epoch_model(model: nn.Module, epoch: int):
    path = os.path.join(MODEL_FOLDER, f"model_epoch_{epoch + 1:02d}.pth")
    torch.save(model.state_dict(), path)
    print(f"Model for epoch {epoch + 1:02d} saved.")

def fit(model: nn.Module, optimizer: torch.optim.Adam,
        device: torch.device, epochs: int, 
        train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, 
        criterion, gamma: float = 0.1, patience = 5,
        loss_save_path = 'best_loss_model.pth', 
        acc_save_path = 'best_acc_model.pth', 
        f1_save_path = 'best_f1_model.pth',
        history: dict[str, dict[str, list[float]]] = {}):
    
    global BEST_LOSS, BEST_ACCURACY, BEST_F1

    start_epoch = get_latest_epoch(MODEL_FOLDER)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=gamma, patience=3)
    early_stop_counter = 0
    lr_no_improve_counter = 0

    for epoch in range(start_epoch, start_epoch + epochs):
        start_time = time.time()
        print('=' * 25 + f'Epoch {epoch + 1:02d}/{start_epoch + epochs:02d}' + '=' * 25)
        train_result = train(model, train_loader, optimizer, device, criterion)
        val_result = evaluate(model, val_loader, device, criterion)

        history['train']['loss'].append(train_result['loss'])
        history['train']['accuracy'].append(train_result['accuracy'])
        history['train']['f1'].append(train_result['f1'])
        history['val']['loss'].append(val_result['loss'])
        history['val']['accuracy'].append(val_result['accuracy'])
        history['val']['f1'].append(val_result['f1'])

        if save_best_models(model, val_result, epoch, loss_save_path, acc_save_path, f1_save_path):
            early_stop_counter = 0
            lr_no_improve_counter = 0
        else:
            early_stop_counter += 1
            lr_no_improve_counter += 1
            print(f'🚫🚫Stopping counter {early_stop_counter}/{patience}')
            print(f'🚫🚫Learning rate counter {lr_no_improve_counter}/{3}')

        save_epoch_model(model, epoch)
        minutes, seconds = divmod(time.time() - start_time, 60)
        print(f"Epoch time: {int(minutes):02d}:{int(seconds):02d} (mm:ss)")
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{start_epoch + epochs}] "
              f"Train Loss: {train_result['loss']:.4f} | "
              f"Val Loss: {val_result['loss']:.4f} | "
              f"Train Acc: {train_result['accuracy']:.4f} | "
              f"Val Acc: {val_result['accuracy']:.4f} | "
              f"Train F1: {train_result['f1']:.4f} | "
              f"Val F1: {val_result['f1']:.4f} | "
              f"LR: {current_lr:.6f}")

        if early_stop_counter >= patience:
            print("❌❌Early stopping triggered")
            break

        scheduler.step(val_result['f1'])

        if lr_no_improve_counter >= 3:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"🆎🆎Learning rate reduced to {current_lr:.6f} at epoch {epoch + 1:02d}")
            lr_no_improve_counter = 0

    return history