import numpy as np
from .indicators import (
    calculate_sma, calculate_rsi, calculate_obv,
    did_cross_above, did_cross_below, calculate_atr
)

def sp500_sma_enhanced_strategy(
    ohlcv_data: list[dict],
    sl_atr_multiplier: float = 2.0,
    tp_atr_multiplier: float = 4.0,
    trailing_sl_atr_multiplier: float = 1.5
) -> list[dict]:
    """
    Implements an S&P 500 strategy with SMAs, RSI, OBV, ATR-based initial SL/TP, and ATR-based trailing SL.

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
    volumes = [d['volume'] for d in ohlcv_data]
    n = len(ohlcv_data)

    # Calculate indicators
    sma10 = calculate_sma(closes, 10)
    sma50 = calculate_sma(closes, 50)
    sma200 = calculate_sma(closes, 200)
    rsi14 = calculate_rsi(closes, 14)

    obv_series = calculate_obv(ohlcv_data)
    obv_sma20 = calculate_sma(obv_series, 20) # Calculate SMA of OBV
    atr_series = calculate_atr(ohlcv_data, period=14) # Default ATR period of 14

    # Initialize signals with HOLD dictionaries ensuring all elements are dicts
    signals = [{'signal': 'HOLD', 'timestamp': ohlcv_data[i].get('timestamp', f"T{i}")} for i in range(n)]

    # Trade state variables
    active_trade_type = None  # Can be 'LONG', 'SHORT', or None
    current_trailing_stop_price = 0.0
    entry_price_of_active_trade = 0.0
    initial_take_profit_price = 0.0

    # Determine the starting point for applying strategy logic
    # This is when all indicators have valid (non-None) values.
    # Longest periods: SMA200 for price, OBV_SMA(20) for OBV.
    # RSI needs period + 1. OBV_SMA needs its period.
    start_index = 0
    for i in range(n):
        if sma10[i] is not None and \
           sma50[i] is not None and \
           sma200[i] is not None and \
           rsi14[i] is not None and \
           obv_sma20[i] is not None and \
           obv_series[i] is not None:
            # ATR check will be done inside the loop, as ATR might have different warmup
            # or we might use a fallback if ATR is None at start_index.
            # For start_index, we primarily care about SMAs, RSI, OBV.
            start_index = i
            break
    else: # If no such index found (e.g., data too short for all indicators)
        return signals # Return all HOLDs

    # Adjust start_index to ensure we can look back one period for crossover detection
    if start_index == 0 and n > 0 : # If all indicators are valid from the first data point (unlikely with long periods)
        # We still need a previous point for crossover, so if start_index is 0, we can't really check crossover at index 0.
        # The loop below starts from start_index, so index i-1 is safe.
        # However, the first point where all indicators are valid is where we can *start* making decisions.
        # Crossover checks inherently require one previous data point.
        pass


    for i in range(n): # Iterate through all data points
        current_timestamp = ohlcv_data[i].get('timestamp', f"T{i}")
        # signals[i] is already initialized to a HOLD dict.
        # It will be updated if a trade action occurs or if an active trade's TSL is updated.

        if i < start_index: # Not enough data for primary indicators yet
            # Ensure signals[i] remains the default HOLD dict if no other action is taken
            signals[i] = {'signal': 'HOLD', 'timestamp': current_timestamp}
            continue

        current_price_low = ohlcv_data[i]['low']
        current_price_high = ohlcv_data[i]['high']
        current_price_close = ohlcv_data[i]['close']

        # Ensure all necessary data for this iteration is valid
        if current_price_low is None or current_price_high is None or current_price_close is None or \
           sma10[i] is None or sma50[i] is None or sma200[i] is None or \
           rsi14[i] is None or obv_series[i] is None or obv_sma20[i] is None or \
           atr_series[i] is None: # Check ATR value here
            continue # Skip if any crucial data is missing

        atr_val = atr_series[i]
        if atr_val is None or atr_val <= 0:
            atr_val = current_price_close * 0.01 # Fallback: 1% of close price

        # --- Exit Logic for Active Trades (Opposing Signal, TSL, TP) ---
        if active_trade_type == 'LONG':
            # Check for Opposing SELL Signal
            sell_condition_met = did_cross_below(sma10, sma50, i) and \
                                 current_price_close < sma200[i] and \
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
            # Check for Opposing BUY Signal
            buy_condition_met = did_cross_above(sma10, sma50, i) and \
                                current_price_close > sma200[i] and \
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

        # --- New Entry Signal Logic (only if no active trade) ---
        if active_trade_type is None:
            # Buy Signal Conditions
            if did_cross_above(sma10, sma50, i) and \
               current_price_close > sma200[i] and \
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
            # Sell Signal Conditions
            elif did_cross_below(sma10, sma50, i) and \
                 current_price_close < sma200[i] and \
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
    # Generate sample OHLCV data (more than 200 entries for SMA200)
    sample_ohlcv_data = []
    base_price = 100
    base_volume = 1000
    for i in range(250): # Increased data points
        price_movement = np.random.randn() * 2
        volume_movement = np.random.randint(-100, 100)
        open_p = base_price + np.random.randn()
        close_p = open_p + price_movement
        high_p = max(open_p, close_p) + np.random.rand()
        low_p = min(open_p, close_p) - np.random.rand()

        sample_ohlcv_data.append({
            'timestamp': f'2023-01-01T00:{i:02d}:00Z', # Simplified timestamp
            'open': round(open_p, 2),
            'high': round(high_p, 2),
            'low': round(low_p, 2),
            'close': round(close_p, 2),
            'volume': base_volume + volume_movement + i * 5 # Add some trend to volume
        })
        base_price = close_p # Next day's price is based on current close

    print(f"Generated {len(sample_ohlcv_data)} sample OHLCV data points.")

    # Test calculate_sma
    print("\n--- Testing calculate_sma ---")
    sample_closes_for_sma = [d['close'] for d in sample_ohlcv_data[:10]]
    sma_5 = calculate_sma(sample_closes_for_sma, 5)
    print(f"Sample closes (first 10): {sample_closes_for_sma}")
    print(f"SMA(5) for first 10: {sma_5}")
    assert len(sma_5) == len(sample_closes_for_sma)

    # Test calculate_rsi
    print("\n--- Testing calculate_rsi ---")
    sample_closes_for_rsi = [d['close'] for d in sample_ohlcv_data[:20]] # Need enough for period 14
    rsi_14_test = calculate_rsi(sample_closes_for_rsi, 14)
    print(f"Sample closes (first 20 for RSI): {sample_closes_for_rsi}")
    print(f"RSI(14) for first 20: {rsi_14_test}")
    assert len(rsi_14_test) == len(sample_closes_for_rsi)

    # calculate_volume_sma was a local helper, its test is removed as calculate_sma (imported) is tested in indicators.py
    # print("\n--- Testing calculate_volume_sma ---")
    # sample_volumes_for_sma = [d['volume'] for d in sample_ohlcv_data[:10]]
    # vol_sma_5 = calculate_volume_sma(sample_volumes_for_sma, 5) # This would cause NameError
    # print(f"Sample volumes (first 10): {sample_volumes_for_sma}")
    # print(f"Volume SMA(5) for first 10: {vol_sma_5}")
    # assert len(vol_sma_5) == len(sample_volumes_for_sma)

    print("\n--- Testing calculate_obv ---")
    sample_obv_data = sample_ohlcv_data[:10]
    obv_test = calculate_obv(sample_obv_data)
    print(f"Sample OHLCV (first 10 closes): {[d['close'] for d in sample_obv_data]}")
    print(f"Sample OHLCV (first 10 volumes): {[d['volume'] for d in sample_obv_data]}")
    print(f"OBV for first 10: {obv_test}")
    assert len(obv_test) == len(sample_obv_data)
    if len(obv_test) > 1:
        # Basic check: if price went up, OBV should increase by volume, if down, decrease by volume
        # This is a simplified check for one step.
        idx_check = 1
        expected_obv_at_idx = obv_test[idx_check-1]
        if sample_obv_data[idx_check]['close'] > sample_obv_data[idx_check-1]['close']:
            expected_obv_at_idx += sample_obv_data[idx_check]['volume']
        elif sample_obv_data[idx_check]['close'] < sample_obv_data[idx_check-1]['close']:
            expected_obv_at_idx -= sample_obv_data[idx_check]['volume']
        assert obv_test[idx_check] == expected_obv_at_idx, \
            f"OBV at index {idx_check} was {obv_test[idx_check]}, expected {expected_obv_at_idx}"


    print("\n--- Testing sp500_sma_enhanced_strategy (with OBV) ---")
    # Using all 250 data points for the strategy
    signal_dicts = sp500_sma_enhanced_strategy(sample_ohlcv_data)
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
        for detail in buy_signals_details[:3]: # Print details of first 3 buy signals
            print(f"  Index: {detail['index']}, Timestamp: {detail['timestamp']}, "
                  f"Entry: {detail['entry_price']:.2f}, SL: {detail['stop_loss']:.2f}, TP: {detail['take_profit']:.2f}")

    print(f"Number of SELL signals: {len(sell_signals_details)}")
    if sell_signals_details:
        print("First few SELL signals details:")
        for detail in sell_signals_details[:3]: # Print details of first 3 sell signals
            print(f"  Index: {detail['index']}, Timestamp: {detail['timestamp']}, "
                  f"Entry: {detail['entry_price']:.2f}, SL: {detail['stop_loss']:.2f}, TP: {detail['take_profit']:.2f}")

    # A quick check to see if signals are mostly HOLDs at the beginning
    # due to indicator warm-up periods. Longest is SMA200 or OBV_SMA20.
    # Max warm-up is 200 for SMA200.
    min_len_for_signals = 199 # Index for SMA200 (period 200 means index 199 is first point)

    if len(signal_dicts) > min_len_for_signals :
        initial_signal_types = [s['signal'] for s in signal_dicts[:min_len_for_signals]]
        assert all(s_type == 'HOLD' for s_type in initial_signal_types), \
            f"Initial {min_len_for_signals} signals should be HOLD due to indicator warm-up. Found: {initial_signal_types}"
        print(f"Initial {min_len_for_signals} signal types are HOLD as expected (due to indicator warm-up).")
    else:
        print(f"Data length ({len(signal_dicts)}) is less than min_len_for_signals ({min_len_for_signals}), all should be HOLD.")
        initial_signal_types = [s['signal'] for s in signal_dicts]
        assert all(s_type == 'HOLD' for s_type in initial_signal_types), "All signals should be HOLD for very short data."


    print("\n--- Strategy Test with Edge Case (Short Data) ---")
    short_ohlcv_data = sample_ohlcv_data[:50] # Not enough for SMA200
    short_signal_dicts = sp500_sma_enhanced_strategy(short_ohlcv_data)
    print(f"Signal dicts for short data ({len(short_signal_dicts)} total):")
    # for sd in short_signal_dicts: print(sd) # Can be verbose
    assert all(s['signal'] == 'HOLD' for s in short_signal_dicts), \
        "Signals for data shorter than longest SMA period should be all HOLDs"
    print("Signals for short data are all HOLDs as expected.")

    print("\n--- Strategy Test with Empty Data ---")
    empty_ohlcv_data = []
    empty_signal_dicts = sp500_sma_enhanced_strategy(empty_ohlcv_data)
    print(f"Signal dicts for empty data: {empty_signal_dicts}")
    assert not empty_signal_dicts, "Signal dicts for empty data should be an empty list"
    print("Signal dicts for empty data are empty as expected.")

    print("\nAll basic tests in __main__ passed.")
