import numpy as np
from .indicators import (
    calculate_sma, calculate_rsi, calculate_obv,
    did_cross_above, did_cross_below, calculate_atr
)

# Utility functions (calculate_sma, calculate_rsi, calculate_obv, did_cross_above, did_cross_below)
# were previously copied here but are now imported from .indicators

def nasdaq_sma_enhanced_strategy(
    ohlcv_data: list[dict],
    sl_atr_multiplier: float = 2.0,
    tp_atr_multiplier: float = 4.0,
    trailing_sl_atr_multiplier: float = 1.5
) -> list[dict]:
    """
    Implements an enhanced NASDAQ strategy with SMAs, RSI, OBV, ATR-based initial SL/TP, and ATR-based trailing SL.

    Args:
        ohlcv_data: List of OHLCV data dictionaries.
        sl_atr_multiplier: Multiplier for ATR for initial stop-loss.
        tp_atr_multiplier: Multiplier for ATR for initial take-profit.
        trailing_sl_atr_multiplier: Multiplier for ATR for trailing stop-loss.

    Returns:
        A list of signal dictionaries. Each dictionary contains:
        - 'signal': 'BUY', 'SELL', or 'HOLD'
        - 'timestamp': Timestamp from ohlcv_data
        - For BUY/SELL: 'entry_price', 'stop_loss', 'take_profit'
    """
    if not ohlcv_data:
        return []

    closes = [d['close'] for d in ohlcv_data]
    volumes = [d['volume'] for d in ohlcv_data] # Needed for OBV calculation
    n = len(ohlcv_data)

    # Calculate indicators
    sma20 = calculate_sma(closes, 20)
    sma100 = calculate_sma(closes, 100)
    sma250 = calculate_sma(closes, 250) # Longest SMA for trend direction
    rsi14 = calculate_rsi(closes, 14)

    obv_series = calculate_obv(ohlcv_data)
    obv_sma20 = calculate_sma(obv_series, 20) # SMA of OBV
    atr_series = calculate_atr(ohlcv_data, period=14) # Default ATR period

    # Initialize signals with HOLD dictionaries ensuring all elements are dicts
    signals = [{'signal': 'HOLD', 'timestamp': ohlcv_data[i].get('timestamp', f"T{i}")} for i in range(n)]

    # Trade state variables
    active_trade_type = None  # Can be 'LONG', 'SHORT', or None
    current_trailing_stop_price = 0.0
    entry_price_of_active_trade = 0.0
    initial_take_profit_price = 0.0

    # Determine the starting point for applying strategy logic
    # Longest period is sma250. OBV_SMA20 needs 20 for OBV + 19 for SMA part.
    # RSI needs 14+1.
    start_index = 0
    for i in range(n):
        if sma20[i] is not None and \
           sma100[i] is not None and \
           sma250[i] is not None and \
           rsi14[i] is not None and \
           obv_sma20[i] is not None and \
           obv_series[i] is not None:
            # ATR check will be done inside the loop.
            start_index = i
            break
    else:
        return signals # Not enough data for any indicator

    for i in range(n): # Iterate through all data points
        current_timestamp = ohlcv_data[i].get('timestamp', f"T{i}")
        # signals[i] is already initialized to a HOLD dict.

        if i < start_index:
            # Ensure signals[i] remains the default HOLD dict if no other action is taken
            signals[i] = {'signal': 'HOLD', 'timestamp': current_timestamp}
            continue

        current_price_low = ohlcv_data[i]['low']
        current_price_high = ohlcv_data[i]['high']
        current_price_close = ohlcv_data[i]['close']

        if current_price_low is None or current_price_high is None or current_price_close is None or \
           sma20[i] is None or sma100[i] is None or sma250[i] is None or \
           rsi14[i] is None or obv_series[i] is None or obv_sma20[i] is None or \
           atr_series[i] is None:
            continue

        atr_val = atr_series[i]
        if atr_val is None or atr_val <= 0:
            atr_val = current_price_close * 0.01

        # --- Exit Logic for Active Trades (Opposing Signal, TSL, TP) ---
        if active_trade_type == 'LONG':
            # Check for Opposing SELL Signal (NASDAQ specific conditions)
            sell_condition_met = did_cross_below(sma20, sma100, i) and \
                                 current_price_close < sma250[i] and \
                                 rsi14[i] < 50 and \
                                 obv_series[i] < obv_sma20[i]
            if sell_condition_met:
                signals[i] = {'signal': 'EXIT_LONG_OPPOSING_SIGNAL', 'exit_price': current_price_close, 'timestamp': current_timestamp}
                active_trade_type = None
                current_trailing_stop_price = 0.0
                entry_price_of_active_trade = 0.0
                initial_take_profit_price = 0.0
                continue

            # Check for TSL Hit
            if current_price_low <= current_trailing_stop_price:
                signals[i] = {'signal': 'EXIT_LONG_TSL', 'exit_price': current_trailing_stop_price, 'timestamp': current_timestamp}
                active_trade_type = None
                continue
            # Check for TP Hit
            elif current_price_close >= initial_take_profit_price:
                signals[i] = {'signal': 'EXIT_LONG_TP', 'exit_price': initial_take_profit_price, 'timestamp': current_timestamp}
                active_trade_type = None
                continue
            else: # Update trailing stop
                potential_new_stop = current_price_high - (atr_val * trailing_sl_atr_multiplier)
                current_trailing_stop_price = max(current_trailing_stop_price, potential_new_stop)
                signals[i]['current_trailing_stop'] = current_trailing_stop_price

        elif active_trade_type == 'SHORT':
            # Check for Opposing BUY Signal (NASDAQ specific conditions)
            buy_condition_met = did_cross_above(sma20, sma100, i) and \
                                current_price_close > sma250[i] and \
                                rsi14[i] > 50 and \
                                obv_series[i] > obv_sma20[i]
            if buy_condition_met:
                signals[i] = {'signal': 'EXIT_SHORT_OPPOSING_SIGNAL', 'exit_price': current_price_close, 'timestamp': current_timestamp}
                active_trade_type = None
                current_trailing_stop_price = 0.0
                entry_price_of_active_trade = 0.0
                initial_take_profit_price = 0.0
                continue

            # Check for TSL Hit
            if current_price_high >= current_trailing_stop_price:
                signals[i] = {'signal': 'EXIT_SHORT_TSL', 'exit_price': current_trailing_stop_price, 'timestamp': current_timestamp}
                active_trade_type = None
                continue
            # Check for TP Hit
            elif current_price_close <= initial_take_profit_price:
                signals[i] = {'signal': 'EXIT_SHORT_TP', 'exit_price': initial_take_profit_price, 'timestamp': current_timestamp}
                active_trade_type = None
                continue
            else: # Update trailing stop
                potential_new_stop = current_price_low + (atr_val * trailing_sl_atr_multiplier)
                current_trailing_stop_price = min(current_trailing_stop_price, potential_new_stop)
                signals[i]['current_trailing_stop'] = current_trailing_stop_price

        # --- New Entry Signal Logic (only if no active trade after exit checks) ---
        if active_trade_type is None: # Re-check as an exit might have occurred
            # Buy Signal (NASDAQ Golden Cross Enhanced)
            # These are the same conditions as checked for opposing signals above
            if did_cross_above(sma20, sma100, i) and \
               current_price_close > sma250[i] and \
               rsi14[i] > 50 and \
               obv_series[i] > obv_sma20[i]:
                active_trade_type = 'LONG'
                entry_price_of_active_trade = current_price_close
                current_trailing_stop_price = round(entry_price_of_active_trade - (atr_val * sl_atr_multiplier), 2)
                initial_take_profit_price = round(entry_price_of_active_trade + (atr_val * tp_atr_multiplier), 2)
                signals[i] = {
                    'signal': 'BUY', 'entry_price': entry_price_of_active_trade,
                    'stop_loss': current_trailing_stop_price, 'take_profit': initial_take_profit_price,
                    'timestamp': current_timestamp
                }
            # Sell Signal (NASDAQ Death Cross Enhanced)
            elif did_cross_below(sma20, sma100, i) and \
                 current_price_close < sma250[i] and \
                 rsi14[i] < 50 and \
                 obv_series[i] < obv_sma20[i]:
                active_trade_type = 'SHORT'
                entry_price_of_active_trade = current_price_close
                current_trailing_stop_price = round(entry_price_of_active_trade + (atr_val * sl_atr_multiplier), 2)
                initial_take_profit_price = round(entry_price_of_active_trade - (atr_val * tp_atr_multiplier), 2)
                signals[i] = {
                    'signal': 'SELL', 'entry_price': entry_price_of_active_trade,
                    'stop_loss': current_trailing_stop_price, 'take_profit': initial_take_profit_price,
                    'timestamp': current_timestamp
                }
    return signals

if __name__ == '__main__':
    # Generate sample OHLCV data (similar to sp500_sma_enhanced.py for consistency)
    sample_ohlcv_data = []
    base_price = 200 # Adjusted base price for "NASDAQ" feel
    base_volume = 5000
    num_periods = 300 # Ensure enough data for SMA250

    for i in range(num_periods):
        price_movement = np.random.randn() * 2.5 # Slightly more volatile
        volume_movement = np.random.randint(-500, 500)

        open_p = base_price + np.random.randn()
        close_p = open_p + price_movement
        high_p = max(open_p, close_p) + np.random.rand() * 2
        low_p = min(open_p, close_p) - np.random.rand() * 2
        close_p = max(low_p, min(high_p, close_p)) # Ensure close is within H/L

        sample_ohlcv_data.append({
            'timestamp': f'NDQ_T{i+1}',
            'open': round(open_p, 2),
            'high': round(high_p, 2),
            'low': round(low_p, 2),
            'close': round(close_p, 2),
            'volume': max(1000, base_volume + volume_movement + i * 10)
        })
        base_price = close_p

    print(f"Generated {len(sample_ohlcv_data)} sample OHLCV data points for NASDAQ strategy.")

    # Test utility functions (briefly, assuming they are robust from sp500 version)
    print("\n--- Basic Test of Copied Utility Functions ---")
    sample_closes = [d['close'] for d in sample_ohlcv_data[:10]]
    sma_5_test = calculate_sma(sample_closes, 5)
    print(f"Test SMA(5) on first 10 closes: {sma_5_test}")
    assert len(sma_5_test) == 10

    # Corrected RSI test data extraction
    rsi_test_closes = [d['close'] for d in sample_ohlcv_data[:20]] if len(sample_ohlcv_data) >= 20 else [d['close'] for d in sample_ohlcv_data]
    rsi_14_test = calculate_rsi(rsi_test_closes, 14)
    print(f"Test RSI(14) on first 20 closes (length): {len(rsi_14_test)}")
    assert len(rsi_14_test) == len(rsi_test_closes)

    obv_test = calculate_obv(sample_ohlcv_data[:10])
    print(f"Test OBV on first 10 data points (length): {len(obv_test)}")
    assert len(obv_test) == 10

    print("\n--- Testing nasdaq_sma_enhanced_strategy ---")
    signal_dicts = nasdaq_sma_enhanced_strategy(sample_ohlcv_data)
    print(f"Generated signal dictionaries ({len(signal_dicts)} total):")

    buy_signals_details = []
    sell_signals_details = []
    for i, sig_dict in enumerate(signal_dicts):
        if sig_dict['signal'] == 'BUY':
            buy_signals_details.append({'index': i, **sig_dict})
        elif sig_dict['signal'] == 'SELL':
            sell_signals_details.append({'index': i, **sig_dict})

    print(f"Number of BUY signals: {len(buy_signals_details)}")
    if buy_signals_details:
        print("First few BUY signals details:")
        for detail in buy_signals_details[:3]:
            print(f"  Index: {detail['index']}, TS: {detail['timestamp']}, Entry: {detail['entry_price']:.2f}, SL: {detail['stop_loss']:.2f}, TP: {detail['take_profit']:.2f}")

    print(f"Number of SELL signals: {len(sell_signals_details)}")
    if sell_signals_details:
        print("First few SELL signals details:")
        for detail in sell_signals_details[:3]:
            print(f"  Index: {detail['index']}, TS: {detail['timestamp']}, Entry: {detail['entry_price']:.2f}, SL: {detail['stop_loss']:.2f}, TP: {detail['take_profit']:.2f}")

    # Assertions for warm-up period
    min_len_for_signals = 249 # Index for SMA250 (period 250 means index 249 is first point)
    if len(signal_dicts) > min_len_for_signals:
        initial_signal_types = [s['signal'] for s in signal_dicts[:min_len_for_signals]]
        assert all(s_type == 'HOLD' for s_type in initial_signal_types), \
            f"Initial {min_len_for_signals} signals should be HOLD. Found: {initial_signal_types[:10]}..." # Print first few if fails
        print(f"Initial {min_len_for_signals} signal types are HOLD as expected.")
    elif len(signal_dicts) > 0 : # If data is shorter than warm-up, all should be HOLD
         initial_signal_types = [s['signal'] for s in signal_dicts]
         assert all(s_type == 'HOLD' for s_type in initial_signal_types), "All signals should be HOLD for very short data."


    print("\n--- Strategy Test with Edge Case (Short Data) ---")
    short_ohlcv_data = sample_ohlcv_data[:50] # Not enough for SMA250
    short_signal_dicts = nasdaq_sma_enhanced_strategy(short_ohlcv_data)
    assert all(s['signal'] == 'HOLD' for s in short_signal_dicts), "Signals for data shorter than SMA250 should be all HOLDs"
    print("Signals for short data are all HOLDs as expected.")

    print("\n--- Strategy Test with Empty Data ---")
    empty_signal_dicts = nasdaq_sma_enhanced_strategy([])
    assert not empty_signal_dicts, "Signal dicts for empty data should be an empty list"
    print("Signal dicts for empty data are empty as expected.")

    print("\nAll basic tests in __main__ passed for NASDAQ strategy.")
