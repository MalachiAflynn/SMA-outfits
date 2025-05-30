import numpy as np
from .indicators import (
    calculate_sma, calculate_rsi, calculate_obv, calculate_atr,
    did_cross_above, did_cross_below
)

def dji_sma_enhanced_strategy(
    ohlcv_data: list[dict],
    sl_atr_multiplier: float = 1.5,
    tp_atr_multiplier: float = 3.0,
    trailing_sl_atr_multiplier: float = 1.25 # Potentially tighter TSL for DJI
) -> list[dict]:
    """
    Implements an enhanced DJI strategy with multiple SMAs, RSI, OBV, ATR-based initial SL/TP, and ATR-based trailing SL.

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

    n = len(ohlcv_data)
    closes = [d['close'] for d in ohlcv_data]

    # Calculate Indicators
    sma30 = calculate_sma(closes, 30)
    sma60 = calculate_sma(closes, 60)
    sma90 = calculate_sma(closes, 90)
    sma300 = calculate_sma(closes, 300)
    sma600 = calculate_sma(closes, 600)
    sma900 = calculate_sma(closes, 900) # Longest SMA for major trend

    rsi14 = calculate_rsi(closes, 14)
    obv_series = calculate_obv(ohlcv_data)
    obv_sma20 = calculate_sma(obv_series, 20)
    atr_series = calculate_atr(ohlcv_data, period=14) # ATR with default 14 period

    # Initialize signals with HOLD dictionaries ensuring all elements are dicts
    signals = [{'signal': 'HOLD', 'timestamp': ohlcv_data[i].get('timestamp', f"T{i}")} for i in range(n)]

    # Trade state variables
    active_trade_type = None  # Can be 'LONG', 'SHORT', or None
    current_trailing_stop_price = 0.0
    entry_price_of_active_trade = 0.0
    initial_take_profit_price = 0.0

    # Determine the starting point for applying strategy logic.
    # Must be at least the longest SMA period (900).
    # Other indicators like RSI(14), OBV_SMA(20), ATR(14) have shorter warm-ups.
    # The loop itself will check for None in these shorter period indicators.
    start_index = 0
    for i in range(n):
        if sma900[i] is not None: # Wait for the longest SMA to be available
            # Further ensure other critical SMAs for crossover are present
            if sma90[i] is not None and sma300[i] is not None and \
               sma30[i] is not None and sma60[i] is not None and \
               sma600[i] is not None and \
               rsi14[i] is not None and \
               obv_series[i] is not None and obv_sma20[i] is not None and \
               atr_series[i] is not None:
                start_index = i
                break
    else: # Not enough data for the longest SMA
        return signals

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
           sma30[i] is None or sma60[i] is None or sma90[i] is None or \
           sma300[i] is None or sma600[i] is None or sma900[i] is None or \
           rsi14[i] is None or \
           obv_series[i] is None or obv_sma20[i] is None or \
           atr_series[i] is None:
            continue

        atr_val = atr_series[i]
        if atr_val is None or atr_val <= 0:
            atr_val = current_price_close * 0.005 # Fallback for DJI

        # --- Exit Logic for Active Trades (Opposing Signal, TSL, TP) ---
        if active_trade_type == 'LONG':
            # Check for Opposing SELL Signal (DJI specific conditions)
            sell_condition1 = did_cross_below(sma90, sma300, i)
            sell_condition2 = sma30[i] < sma60[i] and sma60[i] < sma90[i]
            sell_condition3 = current_price_close < sma600[i]
            sell_condition4 = rsi14[i] < 50
            sell_condition5 = obv_series[i] < obv_sma20[i]
            if sell_condition1 and sell_condition2 and sell_condition3 and sell_condition4 and sell_condition5:
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
            # Check for Opposing BUY Signal (DJI specific conditions)
            buy_condition1 = did_cross_above(sma90, sma300, i)
            buy_condition2 = sma30[i] > sma60[i] and sma60[i] > sma90[i]
            buy_condition3 = current_price_close > sma600[i]
            buy_condition4 = rsi14[i] > 50
            buy_condition5 = obv_series[i] > obv_sma20[i]
            if buy_condition1 and buy_condition2 and buy_condition3 and buy_condition4 and buy_condition5:
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
            buy_condition1 = did_cross_above(sma90, sma300, i)
            buy_condition2 = sma30[i] > sma60[i] and sma60[i] > sma90[i]
            buy_condition3 = current_price_close > sma600[i]
            buy_condition4 = rsi14[i] > 50
            buy_condition5 = obv_series[i] > obv_sma20[i]

            if buy_condition1 and buy_condition2 and buy_condition3 and buy_condition4 and buy_condition5:
                active_trade_type = 'LONG'
                entry_price_of_active_trade = current_price_close
                current_trailing_stop_price = round(entry_price_of_active_trade - (atr_val * sl_atr_multiplier), 2)
                initial_take_profit_price = round(entry_price_of_active_trade + (atr_val * tp_atr_multiplier), 2)
                signals[i] = {
                    'signal': 'BUY', 'entry_price': entry_price_of_active_trade,
                    'stop_loss': current_trailing_stop_price, 'take_profit': initial_take_profit_price,
                    'timestamp': current_timestamp
                }
                # No 'continue' here, as the signal for this bar is now 'BUY'

            else: # Only check sell if not a buy
                sell_condition1 = did_cross_below(sma90, sma300, i)
                sell_condition2 = sma30[i] < sma60[i] and sma60[i] < sma90[i]
                sell_condition3 = current_price_close < sma600[i]
                sell_condition4 = rsi14[i] < 50
                sell_condition5 = obv_series[i] < obv_sma20[i]

                if sell_condition1 and sell_condition2 and sell_condition3 and sell_condition4 and sell_condition5:
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
    num_periods = 1200 # Needs to be significantly more than 900 for all indicators to warm up
    sample_ohlcv_data = []
    base_price = 30000 # DJI base price
    base_volume = 1000000

    for i in range(num_periods):
        price_movement = np.random.randn() * 50 # DJI points movement
        volume_movement = np.random.randint(-100000, 100000)

        open_p = base_price + np.random.randn() * 10
        close_p = open_p + price_movement
        # Ensure high/low are reasonable relative to open/close
        high_p = max(open_p, close_p) + np.random.rand() * 30
        low_p = min(open_p, close_p) - np.random.rand() * 30
        close_p = max(low_p, min(high_p, close_p)) # Ensure close is within H/L bounds

        sample_ohlcv_data.append({
            'timestamp': f'DJI_T{i+1}',
            'open': round(open_p, 2),
            'high': round(high_p, 2),
            'low': round(low_p, 2),
            'close': round(close_p, 2),
            'volume': max(500000, base_volume + volume_movement + i * 100)
        })
        base_price = close_p # Next day's price is based on current close

    print(f"Generated {len(sample_ohlcv_data)} sample OHLCV data points for DJI strategy.")

    # Test the strategy
    print("\n--- Testing dji_sma_enhanced_strategy ---")
    # Using default ATR multipliers
    signal_dicts = dji_sma_enhanced_strategy(sample_ohlcv_data)
    print(f"Generated signal dictionaries ({len(signal_dicts)} total).")

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
        for detail in buy_signals_details[:min(3, len(buy_signals_details))]: # Print details of first 3
            print(f"  Index: {detail['index']}, TS: {detail['timestamp']}, Entry: {detail['entry_price']:.2f}, "
                  f"SL: {detail['stop_loss']:.2f}, TP: {detail['take_profit']:.2f}, ATR used for SL: approx {(detail['entry_price'] - detail['stop_loss'])/1.5 :.2f}")

    print(f"Number of SELL signals: {len(sell_signals_details)}")
    if sell_signals_details:
        print("First few SELL signals details:")
        for detail in sell_signals_details[:min(3, len(sell_signals_details))]:
            print(f"  Index: {detail['index']}, TS: {detail['timestamp']}, Entry: {detail['entry_price']:.2f}, "
                  f"SL: {detail['stop_loss']:.2f}, TP: {detail['take_profit']:.2f}, ATR used for SL: approx {(detail['stop_loss'] - detail['entry_price'])/1.5 :.2f}")

    # Assertions for warm-up period
    # Longest SMA is 900. ATR is 14. OBV_SMA is 20.
    # So, first possible signal is at index 899 (0-indexed for 900th data point).
    # Start_index determined by the loop should be at least 899.
    # All signals before this start_index (where all indicators are valid) should be HOLD.

    # Find the actual start_index used by the strategy by looking for the first non-HOLD or first point where all data is valid
    # This is tricky to assert perfectly without replicating the start_index logic here.
    # A simpler assertion: check that a significant number of initial signals are HOLD.
    # The sma900 will have 899 None values. ATR14 has 14 Nones. OBVSMA20 has 19+1=20 Nones.
    # Thus, the first point where sma900 is not None is index 899.
    # The atr_series will have None up to index 13. obv_sma20 up to index 19.
    # The loop's check for `atr_series[i] is not None` is key.

    min_len_for_signals = 899 # Index for SMA900
    # The actual start_index might be slightly later if ATR or OBV_SMA haven't warmed up by index 899.
    # ATR(14) has 14 Nones. So atr_series[13] is first non-None.
    # OBV_SMA(20) has 19 Nones on OBV series, which has 1 None. So obv_sma20[19] is first.
    # These are shorter than sma900. So sma900 dominates.

    initial_hold_count = 0
    for k_idx in range(min_len_for_signals):
        if k_idx < len(signal_dicts) and signal_dicts[k_idx]['signal'] == 'HOLD':
            initial_hold_count += 1

    if len(signal_dicts) > min_len_for_signals:
        assert initial_hold_count >= min_len_for_signals -1 , \
            f"Expected at least {min_len_for_signals-1} initial HOLD signals. Got {initial_hold_count} relevant HOLDs."
        print(f"Initial {initial_hold_count} (up to {min_len_for_signals-1}) signal types are HOLD as expected.")
    elif len(signal_dicts) > 0: # Data shorter than longest SMA warm-up
        assert all(s['signal'] == 'HOLD' for s in signal_dicts), "All signals should be HOLD for very short data."
        print("All signals are HOLD for very short data, as expected.")


    print("\n--- Strategy Test with Edge Case (Short Data) ---")
    short_ohlcv_data = sample_ohlcv_data[:800] # Not enough for SMA900
    short_signal_dicts = dji_sma_enhanced_strategy(short_ohlcv_data)
    assert all(s['signal'] == 'HOLD' for s in short_signal_dicts), \
        "Signals for data shorter than SMA900 period should be all HOLDs"
    print("Signals for short data (not enough for SMA900) are all HOLDs as expected.")

    print("\n--- Strategy Test with Empty Data ---")
    empty_signal_dicts = dji_sma_enhanced_strategy([])
    assert not empty_signal_dicts, "Signal dicts for empty data should be an empty list"
    print("Signal dicts for empty data are empty as expected.")

    print("\nAll basic tests in __main__ passed for DJI strategy.")
