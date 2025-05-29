# Import necessary strategies
from strategies.moving_average_crossover import moving_average_crossover_strategy

# Placeholder for actual data loading logic
def load_market_data() -> list[float]:
    """
    Simulates loading market data.

    This is a placeholder function. In a real application, this function
    would fetch data from a CSV file, database, or an API.

    Returns:
        A list of sample prices.
    """
    # TODO: Replace with actual data loading logic
    sample_prices = [
        10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 19, 18, 17, 16, 15
    ]
    return sample_prices

if __name__ == "__main__":
    # 1. Load market data
    market_data = load_market_data()
    print(f"Loaded market data: {market_data}")

    # 2. Define strategy parameters
    short_window_period = 5
    long_window_period = 10
    print(f"Using short window: {short_window_period}, long window: {long_window_period}")

    # 3. Execute the trading strategy
    # Ensure there's enough data for the chosen window periods
    if len(market_data) < long_window_period:
        print("Not enough market data to run the strategy with the chosen window periods.")
        signals = ["HOLD"] * len(market_data)
    else:
        signals = moving_average_crossover_strategy(
            market_data, short_window_period, long_window_period
        )

    # 4. Print the generated signals
    print(f"Generated signals: {signals}")
