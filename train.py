"""
Training pipeline for visual trading models.

Implements training loop with validation, checkpointing, and early stopping.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
import json
from datetime import datetime

import config
from model import TradingCNN, TradingCNNClassifier, DeepTradingCNN
from dataset import StockImageDataset


def create_model(model_type='simple', num_channels=2):
    """
    Create model instance based on configuration.

    Args:
        model_type: 'simple', 'deep', or 'classifier'
        num_channels: Number of input channels

    Returns:
        PyTorch model instance
    """
    if model_type == 'simple':
        return TradingCNN(num_channels=num_channels)
    elif model_type == 'deep':
        return DeepTradingCNN(num_channels=num_channels, dropout=config.DROPOUT)
    elif model_type == 'classifier':
        return TradingCNNClassifier(num_channels=num_channels, num_classes=config.NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_dataloaders(dataset, batch_size=32):
    """
    Split dataset and create train/val/test dataloaders.

    Args:
        dataset: StockImageDataset instance
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

    return train_loader, val_loader, test_loader


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train for one epoch.

    Returns:
        float: Average training loss
    """
    model.train()
    total_loss = 0

    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """
    Validate model on validation set.

    Returns:
        float: Average validation loss
    """
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def save_checkpoint(model, optimizer, epoch, val_loss, filepath):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, filepath)


def train(model, train_loader, val_loader, num_epochs=100, device='cpu'):
    """
    Main training loop with early stopping.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Maximum number of epochs
        device: Device to train on

    Returns:
        dict: Training history
    """
    # Setup
    model = model.to(device)

    if config.LOSS_TYPE == 'mse':
        criterion = nn.MSELoss()
    elif config.LOSS_TYPE == 'crossentropy':
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss type: {config.LOSS_TYPE}")

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'best_epoch': 0,
        'best_val_loss': float('inf')
    }

    # Early stopping
    patience_counter = 0

    print(f"Training on {device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss = validate(model, val_loader, criterion, device)

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # Save best model
        if val_loss < history['best_val_loss']:
            history['best_val_loss'] = val_loss
            history['best_epoch'] = epoch
            patience_counter = 0

            # Save checkpoint
            checkpoint_path = os.path.join(
                config.CHECKPOINT_DIR,
                f'best_model_epoch_{epoch}.pt'
            )
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
            print(f"✅ New best model saved (val_loss: {val_loss:.6f})")
        else:
            patience_counter += 1

        # Early stopping check
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"\n⚠️ Early stopping triggered after {epoch+1} epochs")
            break

    print(f"\n🎉 Training complete!")
    print(f"Best epoch: {history['best_epoch']+1}")
    print(f"Best val loss: {history['best_val_loss']:.6f}")

    return history


def main():
    """Main training script."""
    print("=" * 80)
    print("Visual Trading System - Training Pipeline")
    print("=" * 80)

    # Load data (placeholder - replace with actual data loading)
    print("\n⚠️ Note: This script requires price_data to be loaded.")
    print("Please implement data loading logic based on your data source.")
    print("\nExample usage:")
    print("  import pandas as pd")
    print("  price_data = pd.read_csv('your_data.csv')")
    print("  dataset = StockImageDataset(price_data, lookback=60, horizon=5)")
    print("  train_loader, val_loader, test_loader = create_dataloaders(dataset)")
    print("  model = create_model(model_type='simple')")
    print("  history = train(model, train_loader, val_loader, device=config.DEVICE)")

    return None


if __name__ == '__main__':
    main()
