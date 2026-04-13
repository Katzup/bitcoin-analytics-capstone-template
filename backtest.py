"""
Backtesting framework for visual trading models.

Implements walk-forward analysis with realistic trading simulation including
transaction costs, position sizing, and risk management.
"""

import torch
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple

import config


class Trade:
    """Represents a single trade."""

    def __init__(self, entry_date, entry_price, size, direction='long'):
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.size = size
        self.direction = direction
        self.exit_date = None
        self.exit_price = None
        self.pnl = 0
        self.return_pct = 0
        self.holding_period = 0

    def close(self, exit_date, exit_price):
        """Close the trade and calculate P&L."""
        self.exit_date = exit_date
        self.exit_price = exit_price
        self.holding_period = (exit_date - self.entry_date).days

        if self.direction == 'long':
            self.pnl = (exit_price - self.entry_price) * self.size
            self.return_pct = (exit_price - self.entry_price) / self.entry_price
        else:  # short
            self.pnl = (self.entry_price - exit_price) * self.size
            self.return_pct = (self.entry_price - exit_price) / self.entry_price


class Portfolio:
    """Manages portfolio state and trades."""

    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = []
        self.closed_trades = []
        self.equity_curve = []

    def get_total_equity(self, current_prices):
        """Calculate total portfolio equity."""
        positions_value = sum(
            pos.size * current_prices.get(pos.entry_date, pos.entry_price)
            for pos in self.positions
        )
        return self.cash + positions_value

    def open_position(self, date, price, signal_strength):
        """Open new position with position sizing."""
        # Calculate position size
        position_value = self.cash * config.POSITION_SIZE
        shares = position_value / price

        # Deduct transaction costs
        cost = position_value * config.TRANSACTION_COST
        self.cash -= (position_value + cost)

        # Create trade
        trade = Trade(date, price, shares, direction='long')
        self.positions.append(trade)

        return trade

    def close_position(self, position, date, price, reason='signal'):
        """Close position and record trade."""
        # Calculate proceeds
        proceeds = position.size * price
        cost = proceeds * config.TRANSACTION_COST
        self.cash += (proceeds - cost)

        # Close trade
        position.close(date, price)
        self.positions.remove(position)
        self.closed_trades.append(position)

        return position

    def check_stops(self, date, current_price):
        """Check stop loss and take profit levels."""
        positions_to_close = []

        for pos in self.positions:
            current_return = (current_price - pos.entry_price) / pos.entry_price
            holding_days = (date - pos.entry_date).days

            # Stop loss
            if current_return <= config.STOP_LOSS:
                positions_to_close.append((pos, 'stop_loss'))

            # Take profit
            elif current_return >= config.TAKE_PROFIT:
                positions_to_close.append((pos, 'take_profit'))

            # Max holding period
            elif holding_days >= config.MAX_HOLDING_PERIOD:
                positions_to_close.append((pos, 'time_exit'))

        # Close positions
        for pos, reason in positions_to_close:
            self.close_position(pos, date, current_price, reason)


class Backtester:
    """Walk-forward backtesting engine."""

    def __init__(self, model, data, initial_capital=100000):
        """
        Initialize backtester.

        Args:
            model: Trained PyTorch model
            data: DataFrame with OHLC data
            initial_capital: Starting capital
        """
        self.model = model
        self.data = data
        self.portfolio = Portfolio(initial_capital)
        self.predictions = []

    def generate_signals(self, images_batch):
        """Generate trading signals from model predictions."""
        self.model.eval()

        with torch.no_grad():
            predictions = self.model(images_batch)

        return predictions.cpu().numpy()

    def run(self, start_date=None, end_date=None):
        """
        Run backtest simulation.

        Returns:
            dict: Backtest results and metrics
        """
        print("🚀 Starting backtest simulation...")

        # Filter data by date range
        if start_date:
            self.data = self.data[self.data.index >= start_date]
        if end_date:
            self.data = self.data[self.data.index <= end_date]

        # Walk-forward simulation
        for i in range(config.LOOKBACK_WINDOW, len(self.data) - config.PREDICTION_HORIZON):
            current_date = self.data.index[i]
            current_price = self.data.iloc[i]['close']

            # Check existing positions for stops
            self.portfolio.check_stops(current_date, current_price)

            # Generate signal from model prediction
            # Extract lookback window
            window = self.data.iloc[i-config.LOOKBACK_WINDOW:i]

            # Convert to GAF images (same as dataset.py)
            from pyts.image import GramianAngularField
            gaf = GramianAngularField(image_size=config.LOOKBACK_WINDOW, method=config.GAF_METHOD)

            price_img = gaf.transform(window['close'].values.reshape(1, -1))
            vol_img = gaf.transform(window['volume'].values.reshape(1, -1))
            img = np.stack([price_img[0], vol_img[0]], axis=0)

            # Convert to tensor and get prediction
            image_tensor = torch.FloatTensor(img).unsqueeze(0).to(config.DEVICE)

            self.model.eval()
            with torch.no_grad():
                prediction = self.model(image_tensor).item()

            # Trading logic
            if prediction > 0.02 and len(self.portfolio.positions) < config.MAX_POSITIONS:
                # Buy signal
                self.portfolio.open_position(current_date, current_price, prediction)

            elif prediction < -0.02 and self.portfolio.positions:
                # Sell signal - close all positions
                for pos in list(self.portfolio.positions):
                    self.portfolio.close_position(pos, current_date, current_price, 'signal')

            # Record equity
            equity = self.portfolio.get_total_equity({current_date: current_price})
            self.portfolio.equity_curve.append({
                'date': current_date,
                'equity': equity
            })

        # Calculate metrics
        results = self.calculate_metrics()

        print("\n✅ Backtest complete!")
        return results

    def calculate_metrics(self):
        """Calculate performance metrics."""
        trades = self.portfolio.closed_trades
        equity_curve = pd.DataFrame(self.portfolio.equity_curve)

        if len(trades) == 0:
            return {
                'total_trades': 0,
                'message': 'No trades executed'
            }

        # Basic metrics
        returns = equity_curve['equity'].pct_change().dropna()
        total_return = (equity_curve['equity'].iloc[-1] / self.portfolio.initial_capital - 1) * 100

        # Trade statistics
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]

        win_rate = len(winning_trades) / len(trades) * 100 if trades else 0

        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0

        # Risk metrics
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 0 else 0

        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100

        # Profit factor
        total_profit = sum(t.pnl for t in winning_trades)
        total_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

        return {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'final_equity': equity_curve['equity'].iloc[-1],
        }

    def print_results(self, results):
        """Print formatted backtest results."""
        print("\n" + "=" * 80)
        print("BACKTEST RESULTS")
        print("=" * 80)

        if results.get('total_trades', 0) == 0:
            print(results.get('message', 'No trades'))
            return

        print(f"\n📊 Trade Statistics:")
        print(f"  Total Trades:     {results['total_trades']}")
        print(f"  Winning Trades:   {results['winning_trades']}")
        print(f"  Losing Trades:    {results['losing_trades']}")
        print(f"  Win Rate:         {results['win_rate']:.2f}%")

        print(f"\n💰 Performance Metrics:")
        print(f"  Total Return:     {results['total_return']:.2f}%")
        print(f"  Sharpe Ratio:     {results['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown:     {results['max_drawdown']:.2f}%")
        print(f"  Profit Factor:    {results['profit_factor']:.2f}")

        print(f"\n💵 Trade P&L:")
        print(f"  Average Win:      ${results['avg_win']:.2f}")
        print(f"  Average Loss:     ${results['avg_loss']:.2f}")
        print(f"  Final Equity:     ${results['final_equity']:.2f}")

        print("=" * 80)


def main():
    """Example backtest usage."""
    print("Visual Trading System - Backtesting Framework")
    print("\n⚠️ This is a template. Implement data loading and model integration.")
    print("\nExample usage:")
    print("  from model import TradingCNN")
    print("  model = TradingCNN()")
    print("  model.load_state_dict(torch.load('best_model.pt'))")
    print("  backtester = Backtester(model, price_data)")
    print("  results = backtester.run()")
    print("  backtester.print_results(results)")


if __name__ == '__main__':
    main()
