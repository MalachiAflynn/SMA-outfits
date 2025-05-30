import argparse
import pandas as pd
import csv # Fallback if pandas is not available or fails
from collections import Counter

# Import strategy functions
from strategies.sp500_sma_enhanced import sp500_sma_enhanced_strategy
from strategies.nasdaq_sma_enhanced import nasdaq_sma_enhanced_strategy
from strategies.dji_sma_enhanced import dji_sma_enhanced_strategy

def load_ohlcv_from_csv(filepath: str) -> list[dict] | None:
    """
    Loads OHLCV data from a CSV file into a list of dictionaries.

    Expected CSV columns:
    - 'Timestamp' or 'Date': Date and time of the data point.
    - 'Open': Opening price.
    - 'High': Highest price.
    - 'Low': Lowest price.
    - 'Close': Closing price.
    - 'Volume': Trading volume.

    Column names are case-insensitive. The function will attempt to identify
    them even with slight variations (e.g., 'time' for 'timestamp').

    Args:
        filepath: Path to the CSV file.

    Returns:
        A list of OHLCV dictionaries, or None if loading fails.
        Numeric values (Open, High, Low, Close, Volume) are converted to floats
        (Volume to int if possible without loss). Timestamp is kept as a string.
    """
    ohlcv_data = []
    try:
        # Try with pandas first for robustness (handles various CSV quirks)
        df = pd.read_csv(filepath)

        # Normalize column names (convert to lowercase for matching)
        df.columns = [col.lower() for col in df.columns]

        # Identify column names - very basic matching
        col_map = {}
        for col in df.columns:
            if 'time' in col or 'date' in col: # Catches 'timestamp', 'Date', 'Time'
                col_map['timestamp'] = col
            elif 'open' in col:
                col_map['open'] = col
            elif 'high' in col:
                col_map['high'] = col
            elif 'low' in col:
                col_map['low'] = col
            elif 'close' in col: # Catches 'close', 'last'
                col_map['close'] = col
            elif 'vol' in col: # Catches 'volume'
                col_map['volume'] = col

        required_keys = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if not all(key in col_map for key in required_keys):
            print(f"Error: CSV file must contain identifiable columns for {', '.join(required_keys)}.")
            print(f"Identified columns: {col_map}")
            return None

        for _, row in df.iterrows():
            ohlcv_data.append({
                'timestamp': str(row[col_map['timestamp']]),
                'open': float(row[col_map['open']]),
                'high': float(row[col_map['high']]),
                'low': float(row[col_map['low']]),
                'close': float(row[col_map['close']]),
                'volume': int(float(row[col_map['volume']])) if pd.notna(row[col_map['volume']]) else 0 # Handle NaN
            })

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except ImportError: # Fallback to standard csv if pandas is not installed
        print("Pandas not found, attempting to load CSV with standard 'csv' module.")
        print("Ensure CSV has headers: Timestamp,Open,High,Low,Close,Volume (exact match, case-sensitive)")
        try:
            with open(filepath, mode='r', newline='') as file:
                reader = csv.DictReader(file)
                # Check for required headers (case-sensitive for standard csv DictReader)
                expected_headers = {'Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'}
                if not expected_headers.issubset(reader.fieldnames if reader.fieldnames else {}):
                    print(f"Error: CSV headers do not match expected: {expected_headers}")
                    return None

                for row in reader:
                    ohlcv_data.append({
                        'timestamp': str(row['Timestamp']),
                        'open': float(row['Open']),
                        'high': float(row['High']),
                        'low': float(row['Low']),
                        'close': float(row['Close']),
                        'volume': int(float(row['Volume']))
                    })
        except FileNotFoundError:
            print(f"Error: File not found at {filepath}")
            return None
        except Exception as e:
            print(f"Error loading CSV with standard 'csv' module: {e}")
            return None
    except Exception as e:
        print(f"Error processing CSV file with pandas: {e}")
        return None

    return ohlcv_data

def main():
    """
    Main function for the backtester.
    Parses arguments, loads data, runs the selected strategy, and prints signal counts.
    """
    parser = argparse.ArgumentParser(description="CLI Backtester for trading strategies.")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the CSV file containing OHLCV data."
    )
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        choices=['SP500', 'NASDAQ', 'DJI'],
        help="Name of the strategy to backtest."
    )
    # Future arguments could include: ATR multipliers, date ranges, etc.
    args = parser.parse_args()

    print(f"Loading data from: {args.data}")
    ohlcv_data = load_ohlcv_from_csv(args.data)

    if ohlcv_data is None:
        print("Failed to load data. Exiting.")
        return
    if not ohlcv_data:
        print("No data found in the file. Exiting.")
        return

    print(f"Loaded {len(ohlcv_data)} data points.")

    signal_dicts = []
    strategy_name_for_print = ""

    # Default ATR multipliers (can be made configurable later)
    # These are the defaults from the strategy files themselves
    common_params = {
        # 'sl_atr_multiplier': 2.0, # Example, will use strategy defaults
        # 'tp_atr_multiplier': 4.0,
        # 'trailing_sl_atr_multiplier': 1.5
    }

    if args.strategy == 'SP500':
        strategy_name_for_print = "S&P 500 Enhanced SMA Strategy"
        signal_dicts = sp500_sma_enhanced_strategy(ohlcv_data, **common_params)
    elif args.strategy == 'NASDAQ':
        strategy_name_for_print = "NASDAQ Enhanced SMA Strategy"
        signal_dicts = nasdaq_sma_enhanced_strategy(ohlcv_data, **common_params)
    elif args.strategy == 'DJI':
        strategy_name_for_print = "DJI Enhanced SMA Strategy"
        # For DJI, using its specific defaults if not overridden by common_params
        dji_params = {**common_params}
        # dji_params['sl_atr_multiplier'] = 1.5 # Example of overriding for specific strategy
        # dji_params['tp_atr_multiplier'] = 3.0
        # dji_params['trailing_sl_atr_multiplier'] = 1.25
        signal_dicts = dji_sma_enhanced_strategy(ohlcv_data, **dji_params)
    else:
        # This case should ideally not be reached due to argparse 'choices'
        print(f"Error: Unknown strategy '{args.strategy}'. Exiting.")
        return

    print(f"\nBacktesting strategy: {strategy_name_for_print} on data from: {args.data}")

    if not signal_dicts:
        print("Strategy did not generate any signals (empty list).")
        return

    # --- Portfolio Simulation & Metrics ---
    initial_capital = 10000.0
    current_capital = initial_capital
    active_position = None  # 'LONG', 'SHORT', or None
    entry_price = 0.0
    trades_executed = []
    commission_per_trade = 1.0 # Fixed commission per side (entry/exit)
    position_size = 1 # Number of shares per trade (simplified)

    for i, signal_data in enumerate(signal_dicts):
        # current_bar_ohlcv = ohlcv_data[i] # Not strictly needed if signal_data has prices
        signal = signal_data['signal']

        # Handle Exits First
        if active_position == 'LONG':
            if signal in ['EXIT_LONG_TSL', 'EXIT_LONG_TP', 'EXIT_LONG_OPPOSING_SIGNAL']:
                exit_price = signal_data['exit_price']
                pnl = (exit_price - entry_price) * position_size
                current_capital += pnl
                current_capital -= commission_per_trade # Commission for exit
                trades_executed.append({
                    'entry_price': entry_price, 'exit_price': exit_price,
                    'pnl': pnl, 'type': 'LONG', 'exit_reason': signal,
                    'timestamp_exit': signal_data['timestamp']
                })
                active_position = None
                entry_price = 0.0
        elif active_position == 'SHORT':
            if signal in ['EXIT_SHORT_TSL', 'EXIT_SHORT_TP', 'EXIT_SHORT_OPPOSING_SIGNAL']:
                exit_price = signal_data['exit_price']
                pnl = (entry_price - exit_price) * position_size
                current_capital += pnl
                current_capital -= commission_per_trade # Commission for exit
                trades_executed.append({
                    'entry_price': entry_price, 'exit_price': exit_price,
                    'pnl': pnl, 'type': 'SHORT', 'exit_reason': signal,
                    'timestamp_exit': signal_data['timestamp']
                })
                active_position = None
                entry_price = 0.0

        # Handle Entries (only if no position and not just exited)
        # The strategy logic already ensures exit and entry don't happen on the same bar if 'continue' is used after exit.
        if active_position is None:
            if signal == 'BUY':
                active_position = 'LONG'
                entry_price = signal_data['entry_price']
                current_capital -= commission_per_trade # Commission for entry
                # Record entry timestamp for the trade when it's eventually closed
                # This might require adding 'timestamp_entry' to trades_executed later
            elif signal == 'SELL':
                active_position = 'SHORT'
                entry_price = signal_data['entry_price']
                current_capital -= commission_per_trade # Commission for entry

    # --- Reporting ---
    print("\n--- Backtest Report ---")
    print(f"Strategy: {args.strategy}, Data: {args.data}")
    print(f"Initial Capital: {initial_capital:.2f}")
    print(f"Final Capital: {current_capital:.2f}")

    net_profit = current_capital - initial_capital
    print(f"Net Profit/Loss: {net_profit:.2f}")

    total_trades = len(trades_executed)
    print(f"Total Trades Executed: {total_trades}")

    if total_trades > 0:
        winning_trades = sum(1 for trade in trades_executed if trade['pnl'] > 0)
        losing_trades = sum(1 for trade in trades_executed if trade['pnl'] <= 0) # Includes break-even

        gross_profit = sum(trade['pnl'] for trade in trades_executed if trade['pnl'] > 0)
        gross_loss = sum(trade['pnl'] for trade in trades_executed if trade['pnl'] < 0) # Sum of negative PnLs

        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        profit_factor = gross_profit / abs(gross_loss) if abs(gross_loss) > 0 else float('inf')

        print(f"Winning Trades: {winning_trades}")
        print(f"Losing Trades: {losing_trades}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Gross Profit: {gross_profit:.2f}")
        print(f"Gross Loss: {abs(gross_loss):.2f}")
        print(f"Profit Factor: {profit_factor:.2f}")

        print("\n--- Sample Executed Trades (first 5) ---")
        for i, trade in enumerate(trades_executed[:5]):
            print(f"  Trade {i+1}: Type: {trade['type']}, Entry: {trade['entry_price']:.2f}, Exit: {trade['exit_price']:.2f}, PnL: {trade['pnl']:.2f}, Reason: {trade['exit_reason']}, Exit TS: {trade['timestamp_exit']}")
    else:
        print("No trades were executed to calculate further metrics.")

    # Also print original signal summary
    signal_counts = Counter(s['signal'] for s in signal_dicts if s)
    print("\n--- Signal Summary (from strategy output) ---")
    if not signal_counts:
        print("No signals were processed or all signals were None.")
    else:
        for signal_type, count in sorted(signal_counts.items()):
            print(f"{signal_type}: {count}")
    total_signals_from_strategy = sum(signal_counts.values())
    print(f"Total signals from strategy: {total_signals_from_strategy} (out of {len(signal_dicts)} bars)")


if __name__ == '__main__':
    main()
