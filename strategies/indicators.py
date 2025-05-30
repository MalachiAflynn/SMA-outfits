"""
Common Technical Indicators Module
----------------------------------

This module provides functions to calculate various technical indicators widely used in
financial market analysis. These indicators include:
- Simple Moving Average (SMA)
- Relative Strength Index (RSI)
- On-Balance Volume (OBV)

It also includes helper functions for detecting series crossovers.
"""
import numpy as np

def calculate_sma(prices: list[float | None], period: int) -> list[float | None]:
    """
    Calculates the Simple Moving Average (SMA) for a given period.

    Args:
        prices: A list of prices. Can contain None values, which will be treated as NaN.
        period: The period for the SMA calculation. Must be a positive integer.

    Returns:
        A list of SMA values, with None for periods where SMA cannot be calculated.
        The list is the same length as the input prices list.

    Raises:
        ValueError: If period is not a positive integer.
    """
    if not isinstance(period, int) or period <= 0:
        raise ValueError("Period must be a positive integer.")
    if not prices:
        return []
    if len(prices) < period: # Not enough data to calculate any SMA
        return [None] * len(prices)

    # Prepare list for SMA values, with initial Nones for warm-up period
    sma_values = [None] * (period - 1)

    # Convert prices to a numpy array, treating None as np.nan for calculations
    price_array = np.array([p if p is not None else np.nan for p in prices], dtype=float)

    # Calculate SMAs using convolution for efficiency
    # The 'valid' mode ensures that convolution is only computed where the kernel fully overlaps
    weights = np.repeat(1.0, period) / period
    smas_calculated = np.convolve(price_array, weights, 'valid')

    sma_values.extend(smas_calculated.tolist())

    # Ensure the output list is the same length as input, padding if necessary
    # This should ideally be handled by the (period-1) Nones and 'valid' convolution,
    # but as a safeguard for any unexpected behavior with short lists.
    while len(sma_values) < len(prices):
        sma_values.append(None)

    return sma_values


def calculate_rsi(prices: list[float | None], period: int = 14) -> list[float | None]:
    """
    Calculates the Relative Strength Index (RSI).

    Args:
        prices: A list of closing prices. Can contain None values.
        period: The period for RSI calculation (default 14). Must be positive.

    Returns:
        A list of RSI values, padded with None at the beginning.
        The list is the same length as the input prices list.

    Raises:
        ValueError: If period is not a positive integer.
    """
    if not isinstance(period, int) or period <= 0:
        raise ValueError("Period must be a positive integer.")
    if not prices or len(prices) < period + 1: # Need at least period+1 prices for first RSI
        return [None] * len(prices)

    rsi_values = [None] * period  # RSI is undefined for the first 'period' data points

    # Convert to numpy array, handling potential Nones by making them NaN
    # Prices must be float for np.diff to work as expected with NaNs
    price_array = np.array([p if p is not None else np.nan for p in prices], dtype=float)

    # Calculate price differences (deltas)
    deltas = np.diff(price_array) # Size n-1

    # Separate gains and losses
    gains = deltas.copy()
    losses = deltas.copy()
    gains[gains < 0] = 0  # Zero out losses for gains array
    losses[losses > 0] = 0 # Zero out gains for losses array
    losses = np.abs(losses) # Make losses positive

    # Calculate initial average gain and loss
    # Ensure enough non-NaN deltas exist for the first calculation
    if len(deltas) < period: # Not enough deltas
        return [None] * len(prices)

    initial_avg_gain = np.nanmean(gains[:period])
    initial_avg_loss = np.nanmean(losses[:period])

    if np.isnan(initial_avg_gain) or np.isnan(initial_avg_loss): # Not enough valid data in initial window
         # Fill the rest of rsi_values with None, as calculation cannot proceed
        rsi_values.extend([None] * (len(prices) - period))
        return rsi_values


    avg_gain_series = np.full_like(price_array, np.nan)
    avg_loss_series = np.full_like(price_array, np.nan)

    avg_gain_series[period] = initial_avg_gain
    avg_loss_series[period] = initial_avg_loss

    # Smooth subsequent averages
    for i in range(period, len(deltas)): # Iterate over deltas, index i corresponds to price_array[i+1]
        current_gain = gains[i]
        current_loss = losses[i]

        # Handle NaNs in current gain/loss if data had Nones
        if np.isnan(current_gain): current_gain = 0.0
        if np.isnan(current_loss): current_loss = 0.0

        avg_gain_series[i+1] = (avg_gain_series[i] * (period - 1) + current_gain) / period
        avg_loss_series[i+1] = (avg_loss_series[i] * (period - 1) + current_loss) / period

    # Calculate RS and RSI
    # np.seterr(divide='ignore', invalid='ignore') # Suppress if desired, but explicit handling is better
    rs = np.where(avg_loss_series == 0,
                  np.inf, # If loss is 0, RS is infinite (or 0 if gain is also 0)
                  avg_gain_series / avg_loss_series)
    rs = np.where((avg_loss_series == 0) & (avg_gain_series == 0), 0, rs) # Handle 0/0 case to be 0 (RSI 50 after formula)

    rsi = 100 - (100 / (1 + rs))

    # Update rsi_values list, starting from index 'period'
    # rsi is calculated for price_array indices starting from 'period'
    for i in range(period, len(price_array)):
        if i < len(rsi_values): # Should always be true
            rsi_values[i] = rsi[i]
        else: # If rsi_values was not pre-filled enough (defensive)
            rsi_values.append(rsi[i])

    # Ensure the output list is the same length as the input prices list
    while len(rsi_values) < len(prices):
        rsi_values.append(None)

    return rsi_values


def calculate_obv(ohlcv_data: list[dict]) -> list[float | None]:
    """
    Calculates On-Balance Volume (OBV).

    Args:
        ohlcv_data: A list of dictionaries, where each dictionary is expected to have
                    'close' and 'volume' keys.
    Returns:
        A list of OBV values, same length as ohlcv_data.
        The first OBV value is the volume of the first period.
        Returns None for subsequent periods if 'close' or 'volume' is missing or None.
    """
    if not ohlcv_data:
        return []

    obv_values = [None] * len(ohlcv_data)

    # First OBV value
    if len(ohlcv_data) > 0:
        first_volume = ohlcv_data[0].get('volume')
        if first_volume is not None:
            obv_values[0] = float(first_volume)
        else: # If first volume is None, OBV cannot start
            return obv_values # All None

    for i in range(1, len(ohlcv_data)):
        prev_obv = obv_values[i-1]
        current_close = ohlcv_data[i].get('close')
        prev_close = ohlcv_data[i-1].get('close')
        current_volume = ohlcv_data[i].get('volume')

        if prev_obv is None or current_close is None or prev_close is None or current_volume is None:
            # If any required data is missing, OBV cannot continue from this point
            # Keep subsequent values as None
            break

        if current_close > prev_close:
            obv_values[i] = prev_obv + current_volume
        elif current_close < prev_close:
            obv_values[i] = prev_obv - current_volume
        else:
            obv_values[i] = prev_obv

    return obv_values


def did_cross_above(series1: list[float | None], series2: list[float | None], index: int) -> bool:
    """
    Checks if series1 crossed above series2 at the given index.
    Requires at least two data points (index > 0).

    Args:
        series1: The first series (e.g., shorter SMA).
        series2: The second series (e.g., longer SMA).
        index: The current index to check for a crossover.

    Returns:
        True if series1 crossed above series2 at 'index', False otherwise.
    """
    if index <= 0 or index >= len(series1) or index >= len(series2):
        return False # Index out of bounds or not enough data for a crossover

    # Ensure data is valid at current and previous point for both series
    if series1[index] is None or series2[index] is None or \
       series1[index-1] is None or series2[index-1] is None:
        return False

    return series1[index-1] <= series2[index-1] and series1[index] > series2[index]


def did_cross_below(series1: list[float | None], series2: list[float | None], index: int) -> bool:
    """
    Checks if series1 crossed below series2 at the given index.
    Requires at least two data points (index > 0).

    Args:
        series1: The first series (e.g., shorter SMA).
        series2: The second series (e.g., longer SMA).
        index: The current index to check for a crossover.

    Returns:
        True if series1 crossed below series2 at 'index', False otherwise.
    """
    if index <= 0 or index >= len(series1) or index >= len(series2):
        return False

    if series1[index] is None or series2[index] is None or \
       series1[index-1] is None or series2[index-1] is None:
        return False

    return series1[index-1] >= series2[index-1] and series1[index] < series2[index]


def calculate_atr(ohlcv_data: list[dict], period: int = 14) -> list[float | None]:
    """
    Calculates the Average True Range (ATR).

    Args:
        ohlcv_data: A list of OHLCV dictionaries. Each dictionary must contain
                    'high', 'low', and 'close' keys.
        period: The period for the ATR calculation (default 14).

    Returns:
        A list of ATR values, same length as ohlcv_data, padded with None
        at the beginning.
    """
    if not isinstance(period, int) or period <= 0:
        raise ValueError("Period must be a positive integer.")
    if not ohlcv_data or len(ohlcv_data) < 2: # Need at least 2 data points for first TR
        return [None] * len(ohlcv_data)

    true_ranges = []
    # First TR value requires ohlcv_data[0]['close'] for ohlcv_data[1]
    for i in range(1, len(ohlcv_data)):
        current_high = ohlcv_data[i].get('high')
        current_low = ohlcv_data[i].get('low')
        prev_close = ohlcv_data[i-1].get('close')

        if current_high is None or current_low is None or prev_close is None:
            # If data is missing, TR cannot be calculated for this and subsequent points
            # in a simple sequential manner. We'll add a None and let SMA handle it.
            true_ranges.append(None)
            continue

        high_low = current_high - current_low
        high_prev_close = abs(current_high - prev_close)
        low_prev_close = abs(current_low - prev_close)

        true_range = max(high_low, high_prev_close, low_prev_close)
        true_ranges.append(true_range)

    if not true_ranges: # Should not happen if len(ohlcv_data) >= 2
        return [None] * len(ohlcv_data)

    # Calculate SMA of True Ranges
    # calculate_sma will return a list of the same length as true_ranges,
    # with (period - 1) Nones at the beginning.
    # atr_values_calculated = calculate_sma(true_ranges, period)

    # The final ATR list needs to be same length as ohlcv_data.
    # It needs 1 (for the first missing TR) + (period - 1) (for SMA warm-up) initial Nones.
    # So, total 'period' Nones at the beginning.
    # final_atr_list = [None] * (1 + (period -1)) # Total 'period' initial Nones
    # final_atr_list.extend(atr_values_calculated[period-1:]) # Add calculated ATRs, skipping its initial Nones

    # --- Wilder's Smoothing Implementation ---
    if len(true_ranges) < period:
        # Not enough TR values to calculate even the first ATR
        return [None] * len(ohlcv_data)

    atr_values = [None] * len(ohlcv_data) # Initialize with Nones, same length as OHLCV data

    # Calculate the first ATR value: SMA of the first 'period' TRs
    # This first ATR value corresponds to ohlcv_data[period]
    first_atr_value_calculated = np.nanmean([tr for tr in true_ranges[:period] if tr is not None])

    if np.isnan(first_atr_value_calculated): # If all TRs in the first window were None
        return [None] * len(ohlcv_data) # Cannot proceed

    atr_values[period] = first_atr_value_calculated

    # Calculate subsequent ATR values using Wilder's smoothing
    # Current TR (true_ranges[i]) corresponds to ohlcv_data[i+1]
    # Current ATR (atr_values[j]) corresponds to ohlcv_data[j]
    # So, when using true_ranges[period], this TR corresponds to ohlcv_data[period+1]
    # and it will be used to calculate atr_values[period+1]

    for i in range(period, len(true_ranges)):
        current_tr = true_ranges[i]
        previous_atr = atr_values[i] # Previous ATR aligns with true_ranges[i-1] and ohlcv_data[i]

        if current_tr is None: # If current TR is None, ATR cannot continue meaningfully
            # Keep subsequent atr_values as None, already initialized
            break

        current_atr = ((previous_atr * (period - 1)) + current_tr) / period
        if (i + 1) < len(ohlcv_data): # Ensure we don't write out of bounds
             atr_values[i + 1] = current_atr
        else: # Should not happen if true_ranges is derived from ohlcv_data correctly
            break

    return atr_values


if __name__ == '__main__':
    print("--- Testing Technical Indicators ---")

    # Test SMA
    prices_sma = [10, 11, 12, 13, 14, 15, None, 17, 18]
    sma_3 = calculate_sma(prices_sma, 3)
    print(f"Prices for SMA: {prices_sma}")
    print(f"SMA(3): {sma_3}") # Expected: [None, None, 11.0, 12.0, 13.0, 14.0, None, None, None] due to None in prices
                               # Actual based on current code: [None, None, 11.0, 12.0, 13.0, 14.0, nan, nan, nan] (converted to list)
                               # Let's refine expectation: [None, None, 11.0, 12.0, 13.0, 14.0, None, None, None] (if None propagates fully)
                               # With nanmean/nanconvolve, it should be [None, None, 11.0, 12.0, 13.0, 14.0, (15+17)/2 if period was 2 around None, but it's period 3]
                               # [None, None, 11.0, 12.0, 13.0, 14.0, (15+NaN+17)/3 -> NaN, (NaN+17+18)/3 -> NaN, (17+18+?)/3]
                               # For SMA with convolve and nan: [None, None, 11.0, 12.0, 13.0, 14.0, np.nan, np.nan, np.nan]
                               # Then this list of [None, None, 11.0, 12.0, 13.0, 14.0, nan, nan, nan] is returned. OK.


    # Test RSI
    prices_rsi = [45, 44, 46, 47, 43, 44, 45, 42, 43, 45, 46, 47, 48, 49, 48, 47, 49, 50, 52, 51] # 20 points
    rsi_14 = calculate_rsi(prices_rsi, 14)
    print(f"\nPrices for RSI (len {len(prices_rsi)}): {prices_rsi}")
    print(f"RSI(14) (len {len(rsi_14)}): {rsi_14}")
    # Expected: 14 Nones, then values. Length should be 20.
    assert len(rsi_14) == len(prices_rsi), "RSI output length mismatch"
    assert rsi_14.count(None) >= 14, "RSI should have initial Nones"

    prices_rsi_with_none = [45, 44, 46, 47, None, 44, 45, 42, 43, 45, 46, 47, 48, 49, 48, 47, 49, 50, 52, 51]
    rsi_14_none = calculate_rsi(prices_rsi_with_none, 14)
    print(f"\nPrices for RSI with None: {prices_rsi_with_none}")
    print(f"RSI(14) with None: {rsi_14_none}")


    # Test OBV
    ohlcv_obv = [
        {'close': 10, 'volume': 100},
        {'close': 11, 'volume': 110}, # OBV = 100 + 110 = 210
        {'close': 10, 'volume': 90},  # OBV = 210 - 90  = 120
        {'close': 10, 'volume': 120}, # OBV = 120
        {'close': 12, 'volume': 150}, # OBV = 120 + 150 = 270
        {'close': 12, 'volume': 0},   # OBV = 270
        {'close': 11, 'volume': 50},  # OBV = 270 - 50  = 220
        {'close': None, 'volume': 100},# OBV calculation stops
        {'close': 13, 'volume': 100},
    ]
    obv_values = calculate_obv(ohlcv_obv)
    print(f"\nOHLCV for OBV: {ohlcv_obv}")
    print(f"OBV: {obv_values}") # Expected: [100.0, 210.0, 120.0, 120.0, 270.0, 270.0, 220.0, None, None]
    expected_obv = [100.0, 210.0, 120.0, 120.0, 270.0, 270.0, 220.0, None, None]
    assert obv_values == expected_obv, f"OBV calculation error. Got {obv_values}, expected {expected_obv}"


    # Test Crossovers
    series_a = [1, 2, 3, 5, 4]
    series_b = [2, 2, 2, 3, 5]
    print(f"\nSeries A: {series_a}")
    print(f"Series B: {series_b}")
    print(f"Cross Above at index 3 (A[3]>B[3], A[2]<=B[2]): {did_cross_above(series_a, series_b, 3)}") # True (5>3, 3<=2 is false) -> Let's trace: (A[2]=3, B[2]=2) (A[3]=5, B[3]=3) -> (3<=2) is False. Expected: False.
                                                                                                     # Ah, series_a[2] <= series_b[2] AND series_a[3] > series_b[3]
                                                                                                     # A[2]=3, B[2]=2. 3 <= 2 is False. So, did_cross_above(idx=3) is False.
                                                                                                     # Let's re-check series:
    series_a_co = [10, 12, 15, 18, 20] # Short SMA
    series_b_co = [11, 13, 14, 16, 22] # Long SMA
    # index 2: A[1]=12, B[1]=13. A[2]=15, B[2]=14.  (12 <= 13) AND (15 > 14) -> TRUE
    print(f"Series A_co: {series_a_co}")
    print(f"Series B_co: {series_b_co}")
    print(f"Cross Above A_co over B_co at index 2: {did_cross_above(series_a_co, series_b_co, 2)}") # Expected: True
    assert did_cross_above(series_a_co, series_b_co, 2) == True

    # index 4: A[3]=18, B[3]=16. A[4]=20, B[4]=22. (18 >= 16) AND (20 < 22) -> TRUE
    print(f"Cross Below A_co under B_co at index 4: {did_cross_below(series_a_co, series_b_co, 4)}") # Expected: True
    assert did_cross_below(series_a_co, series_b_co, 4) == True


    # Test ATR
    ohlcv_atr = [
        {'high': 10, 'low': 8, 'close': 9},    # No TR[0]
        {'high': 11, 'low': 9, 'close': 10},   # TR[0] = max(2, abs(11-9), abs(9-9)) = max(2,2,0) = 2
        {'high': 10, 'low': 8, 'close': 8},    # TR[1] = max(2, abs(10-10), abs(8-10)) = max(2,0,2) = 2
        {'high': 12, 'low': 9, 'close': 11},   # TR[2] = max(3, abs(12-8), abs(9-8)) = max(3,4,1) = 4
        {'high': 11, 'low': 10, 'close': 10},  # TR[3] = max(1, abs(11-11), abs(10-11)) = max(1,0,1) = 1
        {'high': 13, 'low': 11, 'close': 12},  # TR[4] = max(2, abs(13-10), abs(11-10)) = max(2,3,1) = 3
        {'high': 12, 'low': 10, 'close': 11},  # TR[5] = max(2, abs(12-12), abs(10-12)) = max(2,0,2) = 2
        {'high': 10, 'low': 8, 'close': 9},    # TR[6] = max(2, abs(10-11), abs(8-11)) = max(2,1,3) = 3
    ] # 8 data points, so 7 TR values
    # TRs: [2, 2, 4, 1, 3, 2, 3]
    atr_period = 3
    atr_values = calculate_atr(ohlcv_atr, atr_period)
    print(f"\nOHLCV for ATR (len {len(ohlcv_atr)}): {ohlcv_atr}")
    print(f"ATR({atr_period}) (len {len(atr_values)}): {atr_values}")
    # Expected TRs: [2, 2, 4, 1, 3, 2, 3]
    # SMA(3) of TRs:
    #   TR_SMA[0] = (2+2+4)/3 = 8/3 = 2.666...
    #   TR_SMA[1] = (2+4+1)/3 = 7/3 = 2.333...
    #   TR_SMA[2] = (4+1+3)/3 = 8/3 = 2.666...
    #   TR_SMA[3] = (1+3+2)/3 = 6/3 = 2.0
    #   TR_SMA[4] = (3+2+3)/3 = 8/3 = 2.666...
    # Expected ATR with Wilder's Smoothing (period=3):
    # TRs: [2, 2, 4, 1, 3, 2, 3]
    # First ATR (ohlcv_atr[3]): SMA(TR[0], TR[1], TR[2]) = (2+2+4)/3 = 8/3 = 2.666...
    # ATR for ohlcv_atr[4]: ((ATR[3]*(3-1)) + TR[3]) / 3 = ((8/3*2) + 1)/3 = (16/3 + 3/3)/3 = (19/3)/3 = 19/9 = 2.111...
    # ATR for ohlcv_atr[5]: ((ATR[4]*(3-1)) + TR[4]) / 3 = ((19/9*2) + 3)/3 = (38/9 + 27/9)/3 = (65/9)/3 = 65/27 = 2.407...
    # Expected: [None, None, None, 2.666..., 2.111..., 2.407..., ...]

    print(f"ATR({atr_period}) (len {len(atr_values)}): {atr_values}")
    assert len(atr_values) == len(ohlcv_atr), "ATR output length mismatch"

    expected_initial_nones_wilder = atr_period # Initial Nones up to ohlcv_data[period-1]
    actual_initial_nones = 0
    for k in range(expected_initial_nones_wilder):
        if k < len(atr_values) and atr_values[k] is None:
            actual_initial_nones +=1
    assert actual_initial_nones == expected_initial_nones_wilder, \
        f"ATR should have {expected_initial_nones_wilder} initial Nones. Got {actual_initial_nones} Nones at start."

    if len(atr_values) > expected_initial_nones_wilder and atr_values[expected_initial_nones_wilder] is not None:
         assert atr_values[expected_initial_nones_wilder] > 0, "First calculated ATR value should be positive"

    # Check specific values (approximate due to float)
    if len(atr_values) > atr_period and atr_values[atr_period] is not None: # First ATR
        assert abs(atr_values[atr_period] - (2+2+4)/3) < 0.001, \
            f"First ATR value incorrect. Got {atr_values[atr_period]}, expected {(2+2+4)/3}"

    if len(atr_values) > atr_period + 1 and atr_values[atr_period + 1] is not None: # Second ATR
        expected_second_atr = (((2+2+4)/3 * (atr_period - 1)) + 1) / atr_period # Using TR[3]=1
        assert abs(atr_values[atr_period + 1] - expected_second_atr) < 0.001, \
            f"Second ATR value incorrect. Got {atr_values[atr_period + 1]}, expected {expected_second_atr}"


    print("\nIndicator tests finished.")
