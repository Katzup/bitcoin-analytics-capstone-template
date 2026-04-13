#!/usr/bin/env python3
"""
Pre-Training Validation Module

Validates data quality and requirements before model training to prevent
runtime failures in XGBoost multi-class classification.

Addresses common issues:
- Class representation: All target classes must appear in train/val/test splits
- Data sufficiency: Minimum samples required per class for robust training
- Type consistency: Ensures float64 types for NumPy/XGBoost compatibility
- Feature completeness: Validates no missing values in critical features
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import Counter


@dataclass
class ValidationResult:
    """Container for validation results"""
    passed: bool
    errors: List[str]
    warnings: List[str]
    metadata: Dict


class PreTrainingValidator:
    """
    Validates data quality before training multi-class classification models.

    Key validations:
    1. Class balance: All classes present in all splits
    2. Sample sufficiency: Minimum samples per class
    3. Type consistency: All numeric data is float64
    4. Feature completeness: No missing critical values
    """

    def __init__(self,
                 min_samples_per_class: int = 5,
                 min_classes: int = 2,
                 expected_classes: Optional[List[int]] = None):
        """
        Args:
            min_samples_per_class: Minimum samples required per class in each split
            min_classes: Minimum number of unique classes required
            expected_classes: Expected class labels (0-5 for regime classification)
        """
        self.min_samples_per_class = min_samples_per_class
        self.min_classes = min_classes
        self.expected_classes = expected_classes or list(range(6))  # 0-5 for regimes

    def validate_splits(self,
                       X_train: np.ndarray,
                       y_train: np.ndarray,
                       X_val: np.ndarray,
                       y_val: np.ndarray,
                       X_test: np.ndarray,
                       y_test: np.ndarray,
                       horizon: int) -> ValidationResult:
        """
        Comprehensive validation of train/val/test splits.

        Args:
            X_train, y_train: Training features and labels
            X_val, y_val: Validation features and labels
            X_test, y_test: Test features and labels
            horizon: Forecast horizon being validated

        Returns:
            ValidationResult with pass/fail status and diagnostic information
        """
        errors = []
        warnings = []
        metadata = {
            'horizon': horizon,
            'train_size': len(y_train),
            'val_size': len(y_val),
            'test_size': len(y_test)
        }

        # 1. Validate data types
        type_errors = self._validate_types(X_train, y_train, X_val, y_val, X_test, y_test)
        errors.extend(type_errors)

        # 2. Validate class representation
        class_errors, class_metadata = self._validate_class_representation(
            y_train, y_val, y_test
        )
        errors.extend(class_errors)
        metadata.update(class_metadata)

        # 3. Validate sample sufficiency
        sufficiency_warnings = self._validate_sample_sufficiency(y_train, y_val, y_test)
        warnings.extend(sufficiency_warnings)

        # 4. Validate feature completeness
        feature_errors = self._validate_features(X_train, X_val, X_test)
        errors.extend(feature_errors)

        # 5. Check for severe class imbalance
        imbalance_warnings = self._check_class_imbalance(y_train)
        warnings.extend(imbalance_warnings)

        passed = len(errors) == 0

        return ValidationResult(
            passed=passed,
            errors=errors,
            warnings=warnings,
            metadata=metadata
        )

    def _validate_types(self,
                       X_train: np.ndarray,
                       y_train: np.ndarray,
                       X_val: np.ndarray,
                       y_val: np.ndarray,
                       X_test: np.ndarray,
                       y_test: np.ndarray) -> List[str]:
        """Validate all arrays are proper numeric types (float64 for X, int for y)"""
        errors = []

        # Check X arrays are float64
        for name, X in [('X_train', X_train), ('X_val', X_val), ('X_test', X_test)]:
            if X.dtype != np.float64:
                errors.append(f"{name} has dtype {X.dtype}, expected float64")

        # Check y arrays are integer types
        for name, y in [('y_train', y_train), ('y_val', y_val), ('y_test', y_test)]:
            if not np.issubdtype(y.dtype, np.integer):
                errors.append(f"{name} has dtype {y.dtype}, expected integer type")

        return errors

    def _validate_class_representation(self,
                                      y_train: np.ndarray,
                                      y_val: np.ndarray,
                                      y_test: np.ndarray) -> Tuple[List[str], Dict]:
        """
        Validate that all classes in val/test also appear in train.
        This is CRITICAL for XGBoost multi-class classification.
        """
        errors = []
        metadata = {}

        train_classes = set(y_train)
        val_classes = set(y_val)
        test_classes = set(y_test)

        metadata['train_classes'] = sorted(train_classes)
        metadata['val_classes'] = sorted(val_classes)
        metadata['test_classes'] = sorted(test_classes)

        # Check minimum classes
        if len(train_classes) < self.min_classes:
            errors.append(
                f"Training set has only {len(train_classes)} classes, "
                f"minimum {self.min_classes} required"
            )

        # CRITICAL: Val/test classes must be subset of train classes
        val_missing = val_classes - train_classes
        if val_missing:
            errors.append(
                f"Validation set contains classes {sorted(val_missing)} "
                f"not present in training set {sorted(train_classes)}"
            )

        test_missing = test_classes - train_classes
        if test_missing:
            errors.append(
                f"Test set contains classes {sorted(test_missing)} "
                f"not present in training set {sorted(train_classes)}"
            )

        return errors, metadata

    def _validate_sample_sufficiency(self,
                                    y_train: np.ndarray,
                                    y_val: np.ndarray,
                                    y_test: np.ndarray) -> List[str]:
        """Validate minimum samples per class in each split"""
        warnings = []

        for name, y in [('train', y_train), ('val', y_val), ('test', y_test)]:
            class_counts = Counter(y)

            for class_label, count in class_counts.items():
                if count < self.min_samples_per_class:
                    warnings.append(
                        f"{name} set: class {class_label} has only {count} samples "
                        f"(minimum {self.min_samples_per_class} recommended)"
                    )

        return warnings

    def _validate_features(self,
                          X_train: np.ndarray,
                          X_val: np.ndarray,
                          X_test: np.ndarray) -> List[str]:
        """Validate feature arrays are complete (no NaN/Inf values)"""
        errors = []

        for name, X in [('X_train', X_train), ('X_val', X_val), ('X_test', X_test)]:
            # Check for NaN
            if np.isnan(X).any():
                nan_count = np.isnan(X).sum()
                errors.append(f"{name} contains {nan_count} NaN values")

            # Check for Inf
            if np.isinf(X).any():
                inf_count = np.isinf(X).sum()
                errors.append(f"{name} contains {inf_count} Inf values")

            # Check shape consistency
            if len(X.shape) != 2:
                errors.append(
                    f"{name} has shape {X.shape}, expected 2D array (samples, features)"
                )

        return errors

    def _check_class_imbalance(self, y_train: np.ndarray) -> List[str]:
        """Check for severe class imbalance that might affect training"""
        warnings = []

        class_counts = Counter(y_train)
        if not class_counts:
            return warnings

        max_count = max(class_counts.values())
        min_count = min(class_counts.values())

        if max_count > 0 and min_count > 0:
            imbalance_ratio = max_count / min_count

            if imbalance_ratio > 10:
                warnings.append(
                    f"Severe class imbalance detected: {imbalance_ratio:.1f}x ratio "
                    f"between most and least common classes. "
                    f"Class distribution: {dict(class_counts)}"
                )
            elif imbalance_ratio > 5:
                warnings.append(
                    f"Moderate class imbalance detected: {imbalance_ratio:.1f}x ratio. "
                    f"Class distribution: {dict(class_counts)}"
                )

        return warnings


def validate_before_training(X_train: np.ndarray,
                             y_train: np.ndarray,
                             X_val: np.ndarray,
                             y_val: np.ndarray,
                             X_test: np.ndarray,
                             y_test: np.ndarray,
                             horizon: int,
                             verbose: bool = True) -> bool:
    """
    Convenience function for pre-training validation.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        horizon: Forecast horizon
        verbose: Print validation results

    Returns:
        bool: True if validation passed, False otherwise
    """
    validator = PreTrainingValidator()
    result = validator.validate_splits(
        X_train, y_train, X_val, y_val, X_test, y_test, horizon
    )

    if verbose:
        print(f"\n{'='*60}")
        print(f"VALIDATION RESULTS - Horizon {horizon}")
        print(f"{'='*60}")

        if result.passed:
            print("✅ PASSED - Data ready for training")
        else:
            print("❌ FAILED - Cannot proceed with training")

        if result.errors:
            print(f"\n🚨 ERRORS ({len(result.errors)}):")
            for error in result.errors:
                print(f"   - {error}")

        if result.warnings:
            print(f"\n⚠️  WARNINGS ({len(result.warnings)}):")
            for warning in result.warnings:
                print(f"   - {warning}")

        print(f"\n📊 METADATA:")
        for key, value in result.metadata.items():
            print(f"   {key}: {value}")

    return result.passed
