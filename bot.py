import numpy as np # For generating sample data
import argparse # For command-line arguments

# Import necessary strategies
from strategies.sp500_sma_enhanced import sp500_sma_enhanced_strategy
from strategies.nasdaq_sma_enhanced import nasdaq_sma_enhanced_strategy
from strategies.dji_sma_enhanced import dji_sma_enhanced_strategy

def load_market_data() -> list[dict]:
    """
    Simulates loading market data as a list of OHLCV dictionaries.

    This is a placeholder function. In a real application, this function
    would fetch data from a CSV file, database, or an API.

    Returns:
        A list of sample OHLCV data dictionaries.
    """
    # TODO: Replace with actual data loading logic
    sample_ohlcv_data = []
    base_price = 150 # Initial base price for generic data
    base_volume = 2000
    num_periods = 1200 # Generate 1200 data points

    # Adjust price simulation if needed for different strategies, or keep generic
    # For DJI, base_price could be set higher if strategy selection happened before data load.
    # For now, keeping a generic data generation process.

    for i in range(num_periods):
        # Simulate price movements with some trends and randomness
        if i > 0 and i % 70 < 35 : # Create a bit of an uptrend phase
            price_change = np.random.uniform(0.1, 2.5) + (i % 35) * 0.05
        elif i > 0 and i % 70 >= 35: # Create a bit of a downtrend/correction phase
            price_change = np.random.uniform(-2.5, -0.1) - ((i % 70) - 35) * 0.05
        else: # Initial phase
            price_change = np.random.randn() * 2


        open_p = round(base_price + np.random.randn() * 0.5, 2)
        close_p = round(open_p + price_change, 2)

        # Ensure high is highest and low is lowest
        high_p = round(max(open_p, close_p, base_price + np.random.uniform(0,3)), 2)
        low_p = round(min(open_p, close_p, base_price - np.random.uniform(0,3)), 2)

        # Ensure close is within high/low
        close_p = max(low_p, min(high_p, close_p))


        # Simulate volume changes
        volume_change = np.random.randint(-200, 200) + np.sin(i / 20) * 100 # Add some cyclicality
        current_volume = max(500, base_volume + volume_change + (i//10)*10) # Gradual increase

        sample_ohlcv_data.append({
            'timestamp': f'T{i+1}', # Simplified timestamp (just period number)
            'open': open_p,
            'high': high_p,
            'low': low_p,
            'close': close_p,
            'volume': int(current_volume)
        })

        base_price = close_p # Next period's open is based on current close

    print(f"Generated {len(sample_ohlcv_data)} sample OHLCV data points for the bot.")
    return sample_ohlcv_data


if __name__ == "__main__":
    # 1. Load market data (now OHLCV)
    market_data_ohlcv = load_market_data()

    # --- Argument Parsing for Strategy Selection ---
    parser = argparse.ArgumentParser(description="Trading Bot with strategy selection.")
    parser.add_argument(
        "--strategy",
        type=str,
        choices=['SP500', 'NASDAQ', 'DJI'],
        default='SP500',
        help="Strategy to use: SP500, NASDAQ, or DJI (default: SP500)"
    )
    args = parser.parse_args()
    # --- End Argument Parsing ---

    if not market_data_ohlcv:
        print("No market data loaded. Exiting.")
    else:
        print(f"First data point: {market_data_ohlcv[0]}")
        print(f"Last data point: {market_data_ohlcv[-1]}")

        # 2. Select and execute the chosen trading strategy
        chosen_strategy_name = ""
        signal_dicts = []

        if args.strategy == 'SP500':
            chosen_strategy_name = "S&P 500 Enhanced SMA Strategy"
            print(f"\nRunning {chosen_strategy_name}...")
            signal_dicts = sp500_sma_enhanced_strategy(market_data_ohlcv)
        elif args.strategy == 'NASDAQ':
            chosen_strategy_name = "NASDAQ Enhanced SMA Strategy"
            print(f"\nRunning {chosen_strategy_name}...")
            signal_dicts = nasdaq_sma_enhanced_strategy(market_data_ohlcv)
        elif args.strategy == 'DJI':
            chosen_strategy_name = "DJI Enhanced SMA Strategy"
            print(f"\nRunning {chosen_strategy_name}...")
            signal_dicts = dji_sma_enhanced_strategy(market_data_ohlcv)
        else: # Should not be reached if choices are enforced by argparse
            print(f"Error: Unknown strategy '{args.strategy}'. Exiting.")
            exit()

        # 3. Print the generated signals (with index/timestamp for non-HOLD signals)
        print("\nGenerated signals:")
        active_signals_found = False
        if signal_dicts:
            buy_count = 0
            sell_count = 0
            hold_count = 0
            for i, result in enumerate(signal_dicts):
                signal_type = result['signal']
                timestamp = result['timestamp']
                if signal_type == 'BUY':
                    buy_count += 1
                    print(f"Period {i} ({timestamp}): {signal_type} @ {result['entry_price']:.2f}, "
                          f"Initial SL: {result['stop_loss']:.2f}, Initial TP: {result['take_profit']:.2f}")
                    active_signals_found = True
                elif signal_type == 'SELL':
                    sell_count += 1
                    print(f"Period {i} ({timestamp}): {signal_type} @ {result['entry_price']:.2f}, "
                          f"Initial SL: {result['stop_loss']:.2f}, Initial TP: {result['take_profit']:.2f}")
                    active_signals_found = True
                elif signal_type in ['EXIT_LONG_TSL', 'EXIT_SHORT_TSL', 'EXIT_LONG_TP', 'EXIT_SHORT_TP',
                                     'EXIT_LONG_OPPOSING_SIGNAL', 'EXIT_SHORT_OPPOSING_SIGNAL']:
                    active_signals_found = True
                    print(f"Period {i} ({timestamp}): {signal_type} @ {result['exit_price']:.2f}")
                else:  # HOLD
                    hold_count += 1
                    # Optionally print HOLD signals too, or keep it less verbose
                    # print(f"Period {i} ({timestamp}): {signal_type}")

            if not active_signals_found and i == len(signal_dicts) - 1: # Check only at the end of the loop
                print("No 'BUY' or 'SELL' signals generated. All signals are 'HOLD'.")

            print(f"\nSignal Summary: BUY: {buy_count}, SELL: {sell_count}, HOLD: {hold_count}")

        else: # This else corresponds to 'if signal_dicts:'
            print("No signals were generated (empty list).")

        # You could add further logic here, e.g., portfolio simulation, performance metrics etc.
        print("\nBot execution finished.")
