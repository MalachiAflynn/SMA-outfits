import matplotlib.pyplot as plt
import matplotlib.dates as mdates # Potentially for future timestamp formatting
import numpy as np # For handling None in indicators for plotting
import argparse # For command-line arguments

# Import from our existing modules
from bot import load_market_data
from strategies.sp500_sma_enhanced import sp500_sma_enhanced_strategy
from strategies.nasdaq_sma_enhanced import nasdaq_sma_enhanced_strategy
from strategies.dji_sma_enhanced import dji_sma_enhanced_strategy
# Indicator functions are now imported from indicators.py within each strategy file,
# but plot_strategy.py needs to calculate them for plotting.
# So, we import them here directly from one of the strategy files (assuming they are identical)
# or ideally from indicators.py itself if plot_strategy was a module in a package.
# For simplicity, using the ones from sp500_sma_enhanced (which imports from indicators)
# This assumes that plot_strategy.py is in the root directory, and can access strategies.indicators
from strategies.indicators import calculate_sma, calculate_rsi

def plot_signals(ohlcv_data: list[dict], signal_dicts: list[dict],
                 strategy_name: str,
                 sma_short: list[float | None], sma_medium: list[float | None], sma_long: list[float | None],
                 sma_short_period: int, sma_medium_period: int, sma_long_period: int,
                 rsi14: list[float | None]):
    """
    Plots the closing prices, SMAs, RSI, and buy/sell signals with SL/TP levels.

    Args:
        ohlcv_data: List of OHLCV data dictionaries.
        signal_dicts: List of signal dictionaries from the strategy.
        strategy_name: Name of the strategy being plotted.
        sma_short: List of short-period SMA values.
        sma_medium: List of medium-period SMA values.
        sma_long: List of long-period SMA values.
        sma_short_period: Period value for the short SMA.
        sma_medium_period: Period value for the medium SMA.
        sma_long_period: Period value for the long SMA.
        rsi14: List of 14-period RSI values.
    """
    closes = [d['close'] for d in ohlcv_data]
    timestamps = range(len(ohlcv_data))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Top Subplot: Price and SMAs
    ax1.plot(timestamps, closes, label='Close Price', color='blue', alpha=0.7)
    ax1.plot(timestamps, sma_short, label=f'SMA({sma_short_period})', color='orange', linestyle='--', alpha=0.8)
    ax1.plot(timestamps, sma_medium, label=f'SMA({sma_medium_period})', color='purple', linestyle='--', alpha=0.8)
    ax1.plot(timestamps, sma_long, label=f'SMA({sma_long_period})', color='brown', linestyle='--', alpha=0.8)

    # Plot Buy/Sell signals and SL/TP lines
    # For unique legend entries
    buy_signal_legend_added = False
    sell_signal_legend_added = False
    exit_long_tsl_legend_added = False
    exit_short_tsl_legend_added = False
    exit_long_tp_legend_added = False
    exit_short_tp_legend_added = False
    exit_long_opp_legend_added = False
    exit_short_opp_legend_added = False

    for i, sig_dict in enumerate(signal_dicts):
        signal_type = sig_dict.get('signal') # Use .get for safety, though signal should always be there

        # Determine y-coordinate for marker, preferring exit_price if available, else entry_price, else close
        y_marker_price = None
        if 'exit_price' in sig_dict:
            y_marker_price = sig_dict['exit_price']
        elif 'entry_price' in sig_dict:
            y_marker_price = sig_dict['entry_price']

        # Fallback to close price if other prices are not in dict (e.g. for HOLD or unexpected cases)
        # or if the specific price is None (though less likely for entry/exit prices)
        if y_marker_price is None and i < len(closes) and closes[i] is not None:
            y_marker_price = closes[i]

        # If still no valid price for marker, skip plotting this marker
        if y_marker_price is None:
            continue

        if signal_type == 'BUY':
            ax1.plot(timestamps[i], y_marker_price * 0.98, '^', color='green', markersize=10,
                     label='Buy Signal' if not buy_signal_legend_added else "")
            buy_signal_legend_added = True
        elif signal_type == 'SELL':
            ax1.plot(timestamps[i], y_marker_price * 1.02, 'v', color='red', markersize=10,
                     label='Sell Signal' if not sell_signal_legend_added else "")
            sell_signal_legend_added = True
        elif signal_type == 'EXIT_LONG_TSL':
            ax1.plot(timestamps[i], y_marker_price, 'x', color='blue', markersize=8,
                     label='Exit Long (TSL)' if not exit_long_tsl_legend_added else "")
            exit_long_tsl_legend_added = True
        elif signal_type == 'EXIT_SHORT_TSL':
            ax1.plot(timestamps[i], y_marker_price, 'x', color='purple', markersize=8,
                     label='Exit Short (TSL)' if not exit_short_tsl_legend_added else "")
            exit_short_tsl_legend_added = True
        elif signal_type == 'EXIT_LONG_TP':
            ax1.plot(timestamps[i], y_marker_price, 's', color='cyan', markersize=8,
                     label='Exit Long (TP)' if not exit_long_tp_legend_added else "")
            exit_long_tp_legend_added = True
        elif signal_type == 'EXIT_SHORT_TP':
            ax1.plot(timestamps[i], y_marker_price, 's', color='magenta', markersize=8,
                     label='Exit Short (TP)' if not exit_short_tp_legend_added else "")
            exit_short_tp_legend_added = True
        elif signal_type == 'EXIT_LONG_OPPOSING_SIGNAL':
            ax1.plot(timestamps[i], y_marker_price, 'o', color='darkorange', markersize=7,
                     label='Exit Long (Opposing)' if not exit_long_opp_legend_added else "")
            exit_long_opp_legend_added = True
        elif signal_type == 'EXIT_SHORT_OPPOSING_SIGNAL':
            ax1.plot(timestamps[i], y_marker_price, 'o', color='deeppink', markersize=7,
                     label='Exit Short (Opposing)' if not exit_short_opp_legend_added else "")
            exit_short_opp_legend_added = True

    ax1.set_title(f'{strategy_name} Visualization: Price, SMAs, and Signals')
    ax1.set_ylabel('Price')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # Bottom Subplot: RSI
    # Convert None to NaN for plotting RSI as matplotlib handles NaN by not drawing
    rsi14_plot = [val if val is not None else np.nan for val in rsi14]
    ax2.plot(timestamps, rsi14_plot, label='RSI(14)', color='green')
    ax2.axhline(70, color='red', linestyle='--', alpha=0.5, label='Overbought (70)')
    ax2.axhline(50, color='gray', linestyle='--', alpha=0.5, label='Mid-Level (50)')
    ax2.axhline(30, color='blue', linestyle='--', alpha=0.5, label='Oversold (30)')

    ax2.set_title('RSI(14)')
    ax2.set_ylabel('RSI Value')
    ax2.set_xlabel('Period (Timestamp/Index)')
    ax2.set_ylim([0, 100]) # RSI is bounded by 0 and 100
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout() # Adjusts subplot params for a tight layout
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plotting tool for trading strategies.")
    parser.add_argument(
        "--strategy",
        type=str,
        choices=['SP500', 'NASDAQ', 'DJI'],
        default='SP500',
        help="Strategy to plot: SP500, NASDAQ, or DJI (default: SP500)"
    )
    args = parser.parse_args()

    print("Loading market data for plotting...")
    ohlcv_data = load_market_data()

    if not ohlcv_data:
        print("No data loaded, cannot proceed with plotting.")
    else:
        closes = [d['close'] for d in ohlcv_data]
        rsi14_values = calculate_rsi(closes, 14) # Common for both strategies

        signal_dicts = []
        strategy_name_for_plot = ""

        sma_s, sma_m, sma_l = [], [], []
        s_period, m_period, l_period = 0, 0, 0

        print(f"Selected strategy for plotting: {args.strategy}")

        if args.strategy == 'SP500':
            strategy_name_for_plot = "S&P 500 Enhanced SMA Strategy"
            s_period, m_period, l_period = 10, 50, 200
            sma_s = calculate_sma(closes, s_period)
            sma_m = calculate_sma(closes, m_period)
            sma_l = calculate_sma(closes, l_period)
            print(f"Calculating indicators for {strategy_name_for_plot}...")
            signal_dicts = sp500_sma_enhanced_strategy(ohlcv_data)
        elif args.strategy == 'NASDAQ':
            strategy_name_for_plot = "NASDAQ Enhanced SMA Strategy"
            s_period, m_period, l_period = 20, 100, 250
            sma_s = calculate_sma(closes, s_period)
            sma_m = calculate_sma(closes, m_period)
            sma_l = calculate_sma(closes, l_period)
            print(f"Calculating indicators for {strategy_name_for_plot}...")
            signal_dicts = nasdaq_sma_enhanced_strategy(ohlcv_data)
        elif args.strategy == 'DJI':
            strategy_name_for_plot = "DJI Enhanced SMA Strategy"
            # For DJI, SMAs are 30, 60, 90, 300, 600, 900.
            # Plotting: 90 (short), 300 (medium), 600 (long)
            s_period, m_period, l_period = 90, 300, 600
            sma_s = calculate_sma(closes, s_period)
            sma_m = calculate_sma(closes, m_period)
            sma_l = calculate_sma(closes, l_period)
            print(f"Calculating indicators for {strategy_name_for_plot}...")
            signal_dicts = dji_sma_enhanced_strategy(ohlcv_data)
        else: # Should not be reached
            print(f"Error: Unknown strategy '{args.strategy}' for plotting. Exiting.")
            exit()

        active_signal_count = sum(1 for s in signal_dicts if s['signal'] != 'HOLD')
        if not active_signal_count:
            print("No active BUY/SELL signals generated by the strategy for the current dataset.")
            print("Plot will show price and indicators, but no signal markers or SL/TP lines.")
        else:
            buy_count = sum(1 for s in signal_dicts if s['signal'] == 'BUY')
            sell_count = sum(1 for s in signal_dicts if s['signal'] == 'SELL')
            print(f"Found {buy_count} BUY signals and {sell_count} SELL signals for {args.strategy} strategy.")

        print("Generating plot...")
        plot_signals(
            ohlcv_data, signal_dicts,
            strategy_name=strategy_name_for_plot,
            sma_short=sma_s, sma_medium=sma_m, sma_long=sma_l,
            sma_short_period=s_period, sma_medium_period=m_period, sma_long_period=l_period,
            rsi14=rsi14_values
        )
        print("Plot display initiated. Close the plot window to continue.")
