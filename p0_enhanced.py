"""
Enhanced P0 Runner with Technical Indicators and Deep CNN

Improvements over basic P0:
- Technical indicators (RSI, MACD, Bollinger Bands)
- 5-channel GAF images
- DeepTradingCNN architecture
- 50 training epochs (default)
- Better optimization

Usage:
    export POLYGON_API_KEY='your_api_key'
    python p0_enhanced.py --ticker SPY
"""

import os
import sys
from datetime import datetime, timedelta
import torch
import config
from polygon_data import PolygonDataFetcher
from data_store import DataStore
from indicators import add_technical_indicators
from dataset import EnhancedBinaryDataset
from model import DeepTradingCNNClassifier, TradingCNNClassifier
from train_binary import create_dataloaders, train_binary_classifier
from walkforward_backtest import WalkForwardBacktester


class EnhancedBacktester(WalkForwardBacktester):
    """Enhanced backtester with indicator support and deep CNN."""

    def _train_model(self, train_data, num_epochs, device):
        """Train model with enhanced dataset and deep CNN."""
        print("  Training enhanced model...")

        # Add technical indicators
        train_data_with_indicators = add_technical_indicators(train_data)

        # Create enhanced dataset
        dataset = EnhancedBinaryDataset(
            train_data_with_indicators,
            self.lookback,
            self.horizon,
            use_indicators=True
        )

        if len(dataset) == 0:
            raise ValueError("Empty dataset - insufficient training data")

        # Get number of channels from dataset
        num_channels = len(dataset.channels)

        # Create dataloaders
        train_loader, val_loader, _ = create_dataloaders(dataset, batch_size=config.BATCH_SIZE)

        # Create DeepTradingCNNClassifier model
        model = DeepTradingCNNClassifier(num_channels=num_channels, num_classes=2, dropout=config.DROPOUT)
        model_params = sum(p.numel() for p in model.parameters())
        print(f"  Model: DeepTradingCNNClassifier with {num_channels} channels ({model_params:,} params)")

        # Train
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        history = train_binary_classifier(
            model, train_loader, val_loader,
            num_epochs=num_epochs,
            device=device,
            save_path=None
        )

        sys.stdout = old_stdout

        print(f"  ✅ Training complete (Best F1: {history['best_val_f1']:.4f})")

        return model

    def _predict(self, model, test_data, device):
        """Generate predictions with enhanced dataset."""
        model.eval()
        predictions = []

        # Add indicators to test data
        test_data_with_indicators = add_technical_indicators(test_data)

        # Create dataset
        dataset = EnhancedBinaryDataset(
            test_data_with_indicators,
            self.lookback,
            self.horizon,
            use_indicators=True
        )

        if len(dataset) == 0:
            return predictions

        with torch.no_grad():
            for i in range(len(dataset)):
                image, _ = dataset[i]
                image = image.unsqueeze(0).to(device)

                outputs = model(image)
                probs = torch.softmax(outputs, dim=1)
                prob_up = probs[0, 1].item()

                predictions.append({
                    'prob_up': prob_up,
                    'signal': 1 if prob_up >= self.prediction_threshold else 0
                })

        return predictions


def run_enhanced_pipeline(
    ticker='SPY',
    days_history=730,
    train_window=252,
    test_window=21,
    num_epochs=50,
    force_refresh=False
):
    """
    Run enhanced P0 pipeline with technical indicators.

    Args:
        ticker: Stock ticker to trade
        days_history: Days of historical data to fetch
        train_window: Training window size in days
        test_window: Test window size in days
        num_epochs: Training epochs per window (default: 50)
        force_refresh: If True, re-fetch data even if exists

    Returns:
        dict: Backtest results
    """
    print("=" * 80)
    print("Visual Trading System - Enhanced P0 Pipeline")
    print("=" * 80)
    print(f"Ticker: {ticker}")
    print(f"Features: Price + Volume + RSI + MACD + Bollinger Bands")
    print(f"Model: DeepTradingCNN")
    print(f"Epochs: {num_epochs} per window")
    print(f"Device: {config.DEVICE}")
    print("=" * 80)

    # Step 1: Load or fetch data
    print("\n[Step 1/4] Loading data...")
    store = DataStore()
    existing_data = store.get_bars(ticker)

    if existing_data.empty or force_refresh:
        print(f"  Fetching from Polygon...")

        api_key = os.getenv('POLYGON_API_KEY')
        if not api_key:
            print("\n❌ ERROR: POLYGON_API_KEY environment variable not set")
            return None

        fetcher = PolygonDataFetcher(api_key)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days_history)).strftime('%Y-%m-%d')

        df = fetcher.fetch_daily_bars(ticker, start_date, end_date)

        if df.empty:
            print(f"❌ Failed to fetch data for {ticker}")
            return None

        store.save_bars(df, replace=True)
        price_data = df
    else:
        print(f"  ✅ Found {len(existing_data)} bars in database")
        price_data = existing_data

    print(f"  Data range: {price_data['date'].min()} to {price_data['date'].max()}")

    # Step 2: Add technical indicators
    print("\n[Step 2/4] Adding technical indicators...")
    price_data_enhanced = add_technical_indicators(price_data)

    indicators_added = [col for col in price_data_enhanced.columns if col not in price_data.columns]
    print(f"  ✅ Added {len(indicators_added)} indicators:")
    print(f"     {', '.join(indicators_added[:10])}{'...' if len(indicators_added) > 10 else ''}")

    # Step 3: Validate data
    print("\n[Step 3/4] Validating data...")
    min_required = config.LOOKBACK_WINDOW + config.PREDICTION_HORIZON + train_window + test_window
    if len(price_data_enhanced) < min_required:
        print(f"❌ Insufficient data. Need {min_required} bars, have {len(price_data_enhanced)}")
        return None

    print(f"  ✅ Sufficient data for backtesting")

    # Step 4: Run enhanced backtest
    print("\n[Step 4/4] Running enhanced walk-forward backtest...")
    print(f"  This will take longer due to deeper model and more features...")

    backtester = EnhancedBacktester(
        price_data=price_data_enhanced,
        lookback=config.LOOKBACK_WINDOW,
        horizon=config.PREDICTION_HORIZON,
        train_window=train_window,
        test_window=test_window,
        initial_capital=100000,
        position_size=1.0,
        transaction_cost=config.TRANSACTION_COST,
        prediction_threshold=config.PREDICTION_THRESHOLD,
        confidence_threshold=config.CONFIDENCE_THRESHOLD
    )

    results = backtester.run(num_epochs=num_epochs, device=config.DEVICE)

    if not results or results.get('num_trades', 0) == 0:
        print("⚠️ Backtest completed but no trades were generated")
        return results

    # Step 5: Save results
    print("\n[Step 5/5] Saving results...")
    if 'trades' in results and not results['trades'].empty:
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        trade_file = f"{config.RESULTS_DIR}/{ticker}_enhanced_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results['trades'].to_csv(trade_file, index=False)
        print(f"  ✅ Trades saved to {trade_file}")

    if 'equity_curve' in results and not results['equity_curve'].empty:
        equity_file = f"{config.RESULTS_DIR}/{ticker}_enhanced_equity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results['equity_curve'].to_csv(equity_file, index=False)
        print(f"  ✅ Equity curve saved to {equity_file}")

    print("\n" + "=" * 80)
    print("ENHANCED P0 PIPELINE COMPLETE")
    print("=" * 80)

    return results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Enhanced Visual Trading System P0')
    parser.add_argument('--ticker', type=str, default='SPY', help='Stock ticker (default: SPY)')
    parser.add_argument('--days', type=int, default=730, help='Days of history (default: 730)')
    parser.add_argument('--train-window', type=int, default=252, help='Training window (default: 252)')
    parser.add_argument('--test-window', type=int, default=21, help='Test window (default: 21)')
    parser.add_argument('--epochs', type=int, default=50, help='Epochs per window (default: 50)')
    parser.add_argument('--refresh', action='store_true', help='Force refresh data')

    args = parser.parse_args()

    results = run_enhanced_pipeline(
        ticker=args.ticker,
        days_history=args.days,
        train_window=args.train_window,
        test_window=args.test_window,
        num_epochs=args.epochs,
        force_refresh=args.refresh
    )

    if results is None:
        print("\n❌ Pipeline failed")
        sys.exit(1)
    else:
        print("\n✅ Pipeline succeeded")
        sys.exit(0)


if __name__ == '__main__':
    main()
