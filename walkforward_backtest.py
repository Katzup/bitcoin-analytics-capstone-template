"""
Walk-forward backtesting for binary classification trading system.

Implements realistic backtesting with train/test separation and transaction costs.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
from typing import Dict, List
import config
from dataset import BinaryStockImageDataset
from model import TradingCNNClassifier
from train_binary import train_binary_classifier, create_dataloaders


class WalkForwardBacktester:
    """
    Walk-forward backtester for binary classification trading strategy.

    Trains model on rolling window, tests on out-of-sample period,
    and simulates realistic trading with transaction costs.
    """

    def __init__(
        self,
        price_data: pd.DataFrame,
        lookback: int = 60,
        horizon: int = 5,
        train_window: int = 252,
        test_window: int = 21,
        initial_capital: float = 100000,
        position_size: float = 1.0,
        transaction_cost: float = 0.001,
        prediction_threshold: float = 0.5,
        confidence_threshold: float = 0.6,
        min_notional: float = 10.0,
        min_units: float = 1e-8
    ):
        """
        Initialize walk-forward backtester.

        Args:
            price_data: DataFrame with OHLCV data
            lookback: Window size for image generation
            horizon: Prediction horizon in days
            train_window: Training window size in days
            test_window: Test window size in days
            initial_capital: Starting capital
            position_size: Fraction of capital per trade (1.0 = full capital)
            transaction_cost: Transaction cost as fraction (0.001 = 0.1%)
            prediction_threshold: Probability threshold for long signal
            confidence_threshold: Minimum confidence to trade
            min_notional: Minimum dollar value to execute ($10 prevents dust trades)
            min_units: Minimum asset units to count as trade (1e-8 for BTC)
        """
        self.price_data = price_data.reset_index(drop=True)
        self.lookback = lookback
        self.horizon = horizon
        self.train_window = train_window
        self.test_window = test_window
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.transaction_cost = transaction_cost
        self.prediction_threshold = prediction_threshold
        self.confidence_threshold = confidence_threshold
        self.min_notional = min_notional
        self.min_units = min_units

        self.trades = []
        self.equity_curve = []

    def run(self, num_epochs=50, device='cpu'):
        """
        Run walk-forward backtest.

        Args:
            num_epochs: Epochs per training window
            device: Device for training

        Returns:
            dict: Backtest results
        """
        print("=" * 80)
        print("Walk-Forward Backtest")
        print("=" * 80)
        print(f"Data points: {len(self.price_data)}")
        print(f"Train window: {self.train_window} days")
        print(f"Test window: {self.test_window} days")
        print(f"Initial capital: ${self.initial_capital:,.2f}")
        print(f"Position size: {self.position_size*100:.1f}%")
        print(f"Transaction cost: {self.transaction_cost*100:.2f}%")
        print("=" * 80)

        # Initialize capital
        capital = self.initial_capital
        position = None  # Current position (None or dict with entry info)

        # Minimum data needed
        min_data_needed = self.lookback + self.horizon + self.train_window
        if len(self.price_data) < min_data_needed:
            print(f"❌ Insufficient data. Need {min_data_needed}, have {len(self.price_data)}")
            return {}

        # Walk-forward loop
        start_idx = 0
        test_start_idx = self.train_window

        while test_start_idx + self.test_window + self.lookback + self.horizon < len(self.price_data):
            print(f"\n{'='*80}")
            print(f"Walk {len(self.trades)//self.test_window + 1}")
            print(f"Train: {start_idx} to {test_start_idx}")
            print(f"Test: {test_start_idx} to {test_start_idx + self.test_window}")

            # Get training data
            train_data = self.price_data.iloc[start_idx:test_start_idx].reset_index(drop=True)

            # Train model
            model = self._train_model(train_data, num_epochs, device)

            # Get test data
            test_data = self.price_data.iloc[test_start_idx:test_start_idx + self.test_window + self.lookback + self.horizon].reset_index(drop=True)

            # Generate predictions for test period
            predictions = self._predict(model, test_data, device)

            # Simulate trading on test period
            for i, pred in enumerate(predictions):
                actual_idx = test_start_idx + i
                current_price = self.price_data.iloc[actual_idx]['close']
                current_date = self.price_data.iloc[actual_idx].get('date', actual_idx)

                # Check if we should enter position
                if position is None and pred['prob_up'] >= self.confidence_threshold:
                    # Calculate target position size
                    target_notional = capital * self.position_size

                    # Guardrail: Check minimum notional value
                    if abs(target_notional) < self.min_notional:
                        shares = 0.0
                    else:
                        # Enter long position (use fractional shares for high-priced assets like Bitcoin)
                        shares = target_notional / current_price

                    # Guardrail: Only execute if shares meet minimum units threshold
                    if abs(shares) > self.min_units:
                        cost = shares * current_price * (1 + self.transaction_cost)

                        if cost <= capital:
                            position = {
                                'entry_date': current_date,
                                'entry_idx': actual_idx,
                                'entry_price': current_price,
                                'shares': shares,
                                'cost': cost
                            }
                            capital -= cost
                            print(f"  📈 Long {shares:.8f} shares @ ${current_price:.2f} (prob={pred['prob_up']:.3f})")

                # Check if we should exit position
                elif position is not None:
                    holding_period = actual_idx - position['entry_idx']
                    exit_signal = False
                    exit_reason = ""

                    # Exit at horizon
                    if holding_period >= self.horizon:
                        exit_signal = True
                        exit_reason = f"horizon ({self.horizon} days)"

                    if exit_signal:
                        # Exit position
                        revenue = position['shares'] * current_price * (1 - self.transaction_cost)
                        pnl = revenue - position['cost']
                        pnl_pct = (pnl / position['cost']) * 100

                        capital += revenue

                        # Guardrail: Only record trade if shares were actually executed
                        if abs(position['shares']) > self.min_units:
                            trade = {
                                'entry_date': position['entry_date'],
                                'exit_date': current_date,
                                'entry_price': position['entry_price'],
                                'exit_price': current_price,
                                'shares': position['shares'],
                                'pnl': pnl,
                                'pnl_pct': pnl_pct,
                                'holding_period': holding_period,
                                'exit_reason': exit_reason
                            }

                            self.trades.append(trade)
                            print(f"  📉 Exit @ ${current_price:.2f} | PnL: ${pnl:+,.2f} ({pnl_pct:+.2f}%) | {exit_reason}")

                        position = None

                # Record equity
                equity = capital
                if position is not None:
                    equity += position['shares'] * current_price

                self.equity_curve.append({
                    'date': current_date,
                    'equity': equity
                })

            # Move forward
            start_idx += self.test_window
            test_start_idx += self.test_window

        # Calculate results
        results = self._calculate_results()
        self._print_results(results)

        return results

    def _train_model(self, train_data: pd.DataFrame, num_epochs: int, device: str):
        """Train model on training window."""
        print("  Training model...")

        # Create dataset
        dataset = BinaryStockImageDataset(train_data, self.lookback, self.horizon)

        if len(dataset) == 0:
            raise ValueError("Empty dataset - insufficient training data")

        # Create dataloaders
        train_loader, val_loader, _ = create_dataloaders(dataset, batch_size=config.BATCH_SIZE)

        # Create model
        model = TradingCNNClassifier(num_channels=config.NUM_CHANNELS, num_classes=2)

        # Train (reduced verbosity)
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = StringIO()  # Suppress training output

        history = train_binary_classifier(
            model, train_loader, val_loader,
            num_epochs=num_epochs,
            device=device,
            save_path=None  # Don't save during backtest
        )

        sys.stdout = old_stdout

        print(f"  ✅ Training complete (Best F1: {history['best_val_f1']:.4f})")

        return model

    def _predict(self, model, test_data: pd.DataFrame, device: str):
        """Generate predictions for test period."""
        model.eval()
        predictions = []

        # Create dataset
        dataset = BinaryStockImageDataset(test_data, self.lookback, self.horizon)

        if len(dataset) == 0:
            return predictions

        with torch.no_grad():
            for i in range(len(dataset)):
                image, _ = dataset[i]
                image = image.unsqueeze(0).to(device)  # Add batch dimension

                outputs = model(image)
                probs = torch.softmax(outputs, dim=1)
                prob_up = probs[0, 1].item()  # Probability of class 1 (up)

                predictions.append({
                    'prob_up': prob_up,
                    'signal': 1 if prob_up >= self.prediction_threshold else 0
                })

        return predictions

    def _calculate_results(self) -> Dict:
        """Calculate backtest performance metrics."""
        if not self.trades:
            equity_df = pd.DataFrame(self.equity_curve)
            final_equity = equity_df['equity'].iloc[-1] if not equity_df.empty else self.initial_capital
            return {
                'total_return': 0,
                'final_equity': final_equity,
                'num_trades': 0,
                'win_rate': 0,
                'avg_pnl': 0,
                'avg_pnl_pct': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'trades': pd.DataFrame(),
                'equity_curve': equity_df
            }

        trades_df = pd.DataFrame(self.trades)
        equity_df = pd.DataFrame(self.equity_curve)

        # Total return
        total_return = (equity_df['equity'].iloc[-1] - self.initial_capital) / self.initial_capital

        # Win rate
        wins = (trades_df['pnl'] > 0).sum()
        win_rate = wins / len(trades_df)

        # Average PnL
        avg_pnl = trades_df['pnl'].mean()
        avg_pnl_pct = trades_df['pnl_pct'].mean()

        # Sharpe ratio (simplified)
        returns = equity_df['equity'].pct_change().dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 0 else 0

        # Max drawdown
        equity_series = equity_df['equity']
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max
        max_drawdown = drawdown.min()

        return {
            'total_return': total_return,
            'final_equity': equity_df['equity'].iloc[-1],
            'num_trades': len(trades_df),
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'avg_pnl_pct': avg_pnl_pct,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'trades': trades_df,
            'equity_curve': equity_df
        }

    def _print_results(self, results: Dict):
        """Print formatted results."""
        print("\n" + "=" * 80)
        print("BACKTEST RESULTS")
        print("=" * 80)
        print(f"Final Equity:      ${results['final_equity']:,.2f}")
        print(f"Total Return:      {results['total_return']*100:+.2f}%")
        print(f"Number of Trades:  {results['num_trades']}")
        print(f"Win Rate:          {results['win_rate']*100:.2f}%")
        print(f"Avg PnL:           ${results['avg_pnl']:+,.2f} ({results['avg_pnl_pct']:+.2f}%)")
        print(f"Sharpe Ratio:      {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown:      {results['max_drawdown']*100:.2f}%")
        print("=" * 80)


def main():
    """Test walk-forward backtest."""
    print("Walk-Forward Backtest Module")
    print("See p0_runner.py for complete example")


if __name__ == '__main__':
    main()
