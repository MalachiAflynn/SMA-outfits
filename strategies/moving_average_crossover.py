import numpy as np

def calculate_sma(prices: list[float], period: int) -> list[float | None]:
    """
    Calculates the Simple Moving Average (SMA) for a given period.

    Args:
        prices: A list of prices.
        period: The period for the SMA calculation.

    Returns:
        A list of SMA values, with None for periods where SMA cannot be calculated.
    """
    if period <= 0:
        raise ValueError("Period must be a positive integer.")
    if len(prices) < period:
        return [None] * len(prices)  # Not enough data to calculate SMA for any point

    sma_values = [None] * (period - 1)  # Fill initial entries with None
    for i in range(period - 1, len(prices)):
        sma_values.append(np.mean(prices[i - period + 1 : i + 1]))
    return sma_values

def moving_average_crossover_strategy(
    prices: list[float], short_window: int, long_window: int
) -> list[str]:
    """
    Implements a moving average crossover strategy.

    Args:
        prices: A list of prices.
        short_window: The period for the short SMA.
        long_window: The period for the long SMA.

    Returns:
        A list of signals ('BUY', 'SELL', or 'HOLD').
    """
    if not prices:
        return []
    if short_window <= 0 or long_window <= 0:
        raise ValueError("Window periods must be positive integers.")
    if short_window >= long_window:
        raise ValueError("Short window period must be less than long window period.")
    if len(prices) < long_window:
        # Not enough data for the long window, so cannot generate meaningful signals
        return ["HOLD"] * len(prices)

    short_sma = calculate_sma(prices, short_window)
    long_sma = calculate_sma(prices, long_window)

    signals = ["HOLD"] * len(prices)  # Initialize signals with 'HOLD'

    for i in range(1, len(prices)):
        # Ensure SMAs are calculable for current and previous period
        if (
            short_sma[i] is not None
            and long_sma[i] is not None
            and short_sma[i - 1] is not None
            and long_sma[i - 1] is not None
        ):
            # Check for BUY signal
            if short_sma[i - 1] <= long_sma[i - 1] and short_sma[i] > long_sma[i]:
                signals[i] = "BUY"
            # Check for SELL signal
            elif short_sma[i - 1] >= long_sma[i - 1] and short_sma[i] < long_sma[i]:
                signals[i] = "SELL"
    return signals

if __name__ == "__main__":
    # Example Usage
    sample_prices = [
        100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
        110, 109, 108, 107, 106, 105, 104, 103, 102, 101,
        100, 99, 98, 97, 96
    ]
    short_period = 5
    long_period = 10

    print(f"Sample Prices: {sample_prices}")

    # Test calculate_sma
    sma5 = calculate_sma(sample_prices, short_period)
    print(f"SMA({short_period}): {sma5}")
    sma10 = calculate_sma(sample_prices, long_period)
    print(f"SMA({long_period}): {sma10}")

    # Test moving_average_crossover_strategy
    generated_signals = moving_average_crossover_strategy(
        sample_prices, short_period, long_period
    )
    print(f"Signals (Short: {short_period}, Long: {long_period}): {generated_signals}")

    # Example with insufficient data for long window
    short_prices = [10, 11, 12, 13, 14]
    print(f"\nShort Prices: {short_prices}")
    signals_short_data = moving_average_crossover_strategy(short_prices, 5, 10)
    print(f"Signals for short data (5, 10): {signals_short_data}")

    # Example with short window >= long window
    try:
        moving_average_crossover_strategy(sample_prices, 10, 5)
    except ValueError as e:
        print(f"\nError for invalid windows: {e}")

    # Example with zero period
    try:
        calculate_sma(sample_prices, 0)
    except ValueError as e:
        print(f"\nError for zero period SMA: {e}")

    # Example with insufficient data for SMA calculation itself
    very_short_prices = [10, 11]
    sma_very_short = calculate_sma(very_short_prices, 5)
    print(f"\nSMA for very short prices (period 5): {sma_very_short}")
    signals_very_short_data = moving_average_crossover_strategy(very_short_prices, 2, 4) # Adjusted windows to be valid
    print(f"Signals for very short data (2, 4): {signals_very_short_data}")

    # Edge case: Prices list is empty
    empty_prices = []
    signals_empty_prices = moving_average_crossover_strategy(empty_prices, 5, 10)
    print(f"\nSignals for empty prices: {signals_empty_prices}")

    # Edge case: All prices are the same
    flat_prices = [100] * 20
    signals_flat_prices = moving_average_crossover_strategy(flat_prices, 5, 10)
    print(f"\nSignals for flat prices: {signals_flat_prices}")
