"""
Test script to validate P0 pipeline setup without running full backtest.

Checks:
1. All modules import correctly
2. Configuration is valid
3. Dependencies are installed
4. Data structures work
"""

import sys
import os


def test_imports():
    """Test that all modules import correctly."""
    print("Testing imports...")

    try:
        import torch
        print(f"  ✅ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"  ❌ PyTorch: {e}")
        return False

    try:
        import numpy
        print(f"  ✅ NumPy {numpy.__version__}")
    except ImportError as e:
        print(f"  ❌ NumPy: {e}")
        return False

    try:
        import pandas
        print(f"  ✅ Pandas {pandas.__version__}")
    except ImportError as e:
        print(f"  ❌ Pandas: {e}")
        return False

    try:
        from pyts.image import GramianAngularField
        print(f"  ✅ pyts (GAF)")
    except ImportError as e:
        print(f"  ❌ pyts: {e}")
        return False

    try:
        import requests
        print(f"  ✅ requests {requests.__version__}")
    except ImportError as e:
        print(f"  ❌ requests: {e}")
        return False

    try:
        import config
        print(f"  ✅ config.py")
    except ImportError as e:
        print(f"  ❌ config: {e}")
        return False

    try:
        from polygon_data import PolygonDataFetcher
        print(f"  ✅ polygon_data.py")
    except ImportError as e:
        print(f"  ❌ polygon_data: {e}")
        return False

    try:
        from data_store import DataStore
        print(f"  ✅ data_store.py")
    except ImportError as e:
        print(f"  ❌ data_store: {e}")
        return False

    try:
        from dataset import BinaryStockImageDataset
        print(f"  ✅ dataset.py (BinaryStockImageDataset)")
    except ImportError as e:
        print(f"  ❌ dataset: {e}")
        return False

    try:
        from model import TradingCNNClassifier
        print(f"  ✅ model.py (TradingCNNClassifier)")
    except ImportError as e:
        print(f"  ❌ model: {e}")
        return False

    try:
        from train_binary import create_dataloaders
        print(f"  ✅ train_binary.py")
    except ImportError as e:
        print(f"  ❌ train_binary: {e}")
        return False

    try:
        from walkforward_backtest import WalkForwardBacktester
        print(f"  ✅ walkforward_backtest.py")
    except ImportError as e:
        print(f"  ❌ walkforward_backtest: {e}")
        return False

    try:
        from p0_runner import run_p0_pipeline
        print(f"  ✅ p0_runner.py")
    except ImportError as e:
        print(f"  ❌ p0_runner: {e}")
        return False

    return True


def test_config():
    """Test configuration is valid."""
    print("\nTesting configuration...")

    import config

    # Check critical parameters
    checks = [
        (config.LOOKBACK_WINDOW, "LOOKBACK_WINDOW", int, 1, 500),
        (config.PREDICTION_HORIZON, "PREDICTION_HORIZON", int, 1, 100),
        (config.NUM_CLASSES, "NUM_CLASSES", int, 2, 2),
        (config.BATCH_SIZE, "BATCH_SIZE", int, 1, 1024),
        (config.LEARNING_RATE, "LEARNING_RATE", (int, float), 0.00001, 0.1),
    ]

    for value, name, type_check, min_val, max_val in checks:
        if not isinstance(value, type_check):
            print(f"  ❌ {name} should be {type_check}, got {type(value)}")
            return False
        if not (min_val <= value <= max_val):
            print(f"  ❌ {name} should be between {min_val} and {max_val}, got {value}")
            return False
        print(f"  ✅ {name} = {value}")

    # Check device
    print(f"  ✅ DEVICE = {config.DEVICE}")

    return True


def test_data_structures():
    """Test basic data structure creation."""
    print("\nTesting data structures...")

    import pandas as pd
    import numpy as np
    from dataset import BinaryStockImageDataset
    from model import TradingCNNClassifier

    # Create synthetic data
    np.random.seed(42)
    n_days = 200
    dates = pd.date_range(start='2024-01-01', periods=n_days, freq='D')

    price_data = pd.DataFrame({
        'date': dates,
        'close': 100 + np.cumsum(np.random.randn(n_days) * 2),
        'volume': np.random.randint(1000000, 10000000, n_days)
    })

    try:
        # Test dataset creation
        dataset = BinaryStockImageDataset(price_data, lookback=60, horizon=5)
        print(f"  ✅ Dataset created: {len(dataset)} samples")

        # Test sample retrieval
        image, label = dataset[0]
        print(f"  ✅ Sample shape: image={image.shape}, label={label.shape}")

        # Test model creation
        model = TradingCNNClassifier(num_channels=2, num_classes=2)
        print(f"  ✅ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")

        # Test forward pass
        import torch
        batch = image.unsqueeze(0)  # Add batch dimension
        output = model(batch)
        print(f"  ✅ Forward pass: output shape={output.shape}")

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

    return True


def test_api_key():
    """Test Polygon API key."""
    print("\nChecking Polygon API key...")

    api_key = os.getenv('POLYGON_API_KEY')

    if api_key:
        print(f"  ✅ POLYGON_API_KEY is set (length: {len(api_key)})")
        return True
    else:
        print(f"  ⚠️ POLYGON_API_KEY not set")
        print(f"     Set with: export POLYGON_API_KEY='your_key'")
        print(f"     Or add to .env file")
        return False


def main():
    """Run all tests."""
    print("=" * 80)
    print("Visual Trading System - Setup Validation")
    print("=" * 80)

    results = {}

    # Run tests
    results['imports'] = test_imports()
    results['config'] = test_config()
    results['data_structures'] = test_data_structures()
    results['api_key'] = test_api_key()

    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name.upper()}: {status}")

    all_passed = all(results.values())

    print("=" * 80)

    if all_passed:
        print("🎉 All tests passed! System is ready.")
        print("\nNext steps:")
        print("  1. Set POLYGON_API_KEY if not already set")
        print("  2. Run: python p0_runner.py")
        return 0
    else:
        print("⚠️ Some tests failed. Please fix issues before running pipeline.")
        print("\nCommon fixes:")
        print("  - Missing dependencies: pip install -r requirements.txt")
        print("  - API key: export POLYGON_API_KEY='your_key'")
        return 1


if __name__ == '__main__':
    sys.exit(main())
