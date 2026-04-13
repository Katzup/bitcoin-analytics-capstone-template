"""
Binary classification training pipeline for directional prediction.

Trains CNN to predict up/down direction for next H days.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import config
from model import TradingCNNClassifier
from dataset import BinaryStockImageDataset


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device).squeeze()  # Shape: (batch_size,)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)  # Shape: (batch_size, num_classes)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Track predictions for metrics
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device).squeeze()

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Get predictions and probabilities
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of class 1 (up)

    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

    return metrics


def save_checkpoint(model, optimizer, epoch, metrics, filepath):
    """Save model checkpoint."""
    if filepath is None:
        return  # Skip saving if no path provided
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }, filepath)


def train_binary_classifier(
    model,
    train_loader,
    val_loader,
    num_epochs=100,
    device='cpu',
    save_path='checkpoints/best_binary_model.pt'
):
    """
    Train binary direction classifier.

    Args:
        model: TradingCNNClassifier with num_classes=2
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Maximum number of epochs
        device: Device to train on
        save_path: Path to save best model

    Returns:
        dict: Training history
    """
    # Setup
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    # Training history
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'best_epoch': 0,
        'best_val_f1': 0.0
    }

    # Early stopping
    patience_counter = 0

    print(f"Training on {device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Record history
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1'])

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"Val Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f} | F1: {val_metrics['f1']:.4f}")

        # Save best model based on F1 score
        if val_metrics['f1'] > history['best_val_f1']:
            history['best_val_f1'] = val_metrics['f1']
            history['best_epoch'] = epoch
            patience_counter = 0

            save_checkpoint(model, optimizer, epoch, val_metrics, save_path)
            print(f"✅ New best model saved (F1: {val_metrics['f1']:.4f})")
        else:
            patience_counter += 1

        # Early stopping check
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"\n⚠️ Early stopping triggered after {epoch+1} epochs")
            break

    print(f"\n🎉 Training complete!")
    print(f"Best epoch: {history['best_epoch']+1}")
    print(f"Best val F1: {history['best_val_f1']:.4f}")

    return history


def create_dataloaders(dataset, batch_size=32):
    """
    Split dataset and create dataloaders.

    Args:
        dataset: BinaryStockImageDataset instance
        batch_size: Batch size for dataloaders

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(config.TRAIN_SPLIT * total_size)
    val_size = int(config.VAL_SPLIT * total_size)
    test_size = total_size - train_size - val_size

    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Dataset splits: Train={train_size}, Val={val_size}, Test={test_size}")

    return train_loader, val_loader, test_loader


def main():
    """Test binary training pipeline."""
    print("=" * 80)
    print("Binary Classification Training Pipeline")
    print("=" * 80)

    print("\n⚠️ This script requires data to be loaded.")
    print("Example usage:")
    print("  from data_store import DataStore")
    print("  store = DataStore()")
    print("  df = store.get_bars('SPY')")
    print("  dataset = BinaryStockImageDataset(df, lookback=60, horizon=5)")
    print("  train_loader, val_loader, test_loader = create_dataloaders(dataset)")
    print("  model = TradingCNNClassifier(num_channels=2, num_classes=2)")
    print("  history = train_binary_classifier(model, train_loader, val_loader)")


if __name__ == '__main__':
    main()
