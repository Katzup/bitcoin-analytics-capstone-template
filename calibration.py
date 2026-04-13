"""
Temperature Scaling for Probability Calibration

Implements temperature scaling for PyTorch models to calibrate prediction
probabilities. This is critical for DCA allocation since the entire tilt
depends on probability quality.

Reference: "On Calibration of Modern Neural Networks" (Guo et al., 2017)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class TemperatureScaling(nn.Module):
    """
    Temperature scaling layer for probability calibration.

    Applies a learned temperature parameter T to model logits:
        calibrated_probs = softmax(logits / T)

    Fit on validation set to minimize negative log-likelihood.
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)  # Initialize T=1.5

    def forward(self, logits):
        """
        Scale logits by temperature and apply softmax.

        Args:
            logits: Model output before softmax (batch_size, num_classes)

        Returns:
            Calibrated probabilities (batch_size, num_classes)
        """
        scaled_logits = logits / self.temperature
        return torch.softmax(scaled_logits, dim=1)

    def fit(self, logits, labels, max_iter=50, lr=0.01):
        """
        Learn optimal temperature on validation set.

        Args:
            logits: Validation logits (N, num_classes)
            labels: Validation labels (N,)
            max_iter: Maximum optimization iterations
            lr: Learning rate

        Returns:
            Optimal temperature value
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        logits = logits.clone()
        labels = labels.clone()

        def closure():
            optimizer.zero_grad()
            loss = criterion(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        return self.temperature.item()


def calibrate_model(model, val_loader, device='cpu'):
    """
    Calibrate a trained PyTorch model using temperature scaling.

    Args:
        model: Trained PyTorch model
        val_loader: Validation data loader
        device: 'cpu' or 'cuda'

    Returns:
        Tuple of (calibrated_model, temperature_value)
    """
    model.eval()

    # Collect validation logits and labels
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            all_logits.append(logits)
            all_labels.append(labels)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels).squeeze()  # Ensure 1D labels

    # Fit temperature scaling
    temp_scaler = TemperatureScaling()
    temp_value = temp_scaler.fit(all_logits, all_labels)

    print(f"  Calibration complete: T = {temp_value:.4f}")

    return temp_scaler, temp_value


def get_calibrated_probabilities(model, temp_scaler, data, device='cpu'):
    """
    Get calibrated probabilities from model predictions.

    Args:
        model: Trained model
        temp_scaler: Fitted TemperatureScaling object
        data: Input data
        device: 'cpu' or 'cuda'

    Returns:
        Calibrated probabilities (N, num_classes)
    """
    model.eval()
    temp_scaler.eval()

    with torch.no_grad():
        data = data.to(device)
        logits = model(data)
        calibrated_probs = temp_scaler(logits)

    return calibrated_probs


def evaluate_calibration(probs, labels, num_bins=10):
    """
    Evaluate calibration quality using Expected Calibration Error (ECE).

    Args:
        probs: Predicted probabilities (N, num_classes)
        labels: True labels (N,)
        num_bins: Number of bins for calibration plot

    Returns:
        Dictionary with calibration metrics
    """
    # Get predicted class and confidence
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = (predictions == labels).float()

    # Bin predictions by confidence
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    bin_stats = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find predictions in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()

            # ECE contribution
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            bin_stats.append({
                'bin_lower': bin_lower.item(),
                'bin_upper': bin_upper.item(),
                'accuracy': accuracy_in_bin.item(),
                'confidence': avg_confidence_in_bin.item(),
                'count': in_bin.sum().item()
            })

    return {
        'ece': ece.item(),
        'accuracy': accuracies.mean().item(),
        'avg_confidence': confidences.mean().item(),
        'bin_stats': bin_stats
    }


def main():
    """Example usage."""
    print("Temperature Scaling Calibration Module")
    print("See calibrate_model() for usage with PyTorch models")


if __name__ == '__main__':
    main()
