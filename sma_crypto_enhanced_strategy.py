# --- Do not remove these libs ---
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, CategoricalParameter
from pandas import DataFrame
# --------------------------------

# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class SMACryptoEnhanced(IStrategy):
    '''
    Freqtrade strategy based on SMA crossovers, RSI, OBV, and ATR.
    This version is adapted for crypto trading.
    Using 10, 50, 200 period SMAs as the base for signals.
    '''
    INTERFACE_VERSION = 3

    # Timeframe: 1-hour candles. Can be changed to '5m', '15m', etc.
    timeframe = '1h'

    # Optimal stoploss calculation options. For this initial version, a static stoploss is used.
    # See Freqtrade documentation for more advanced stoploss options.
    stoploss = -0.10  # Static stoploss of 10%

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "60": 0.01,  # 1% profit after 60 minutes
        "30": 0.02,  # 2% profit after 30 minutes
        "0": 0.04    # 4% profit immediately
    }

    # Optimal timeframe for the strategy.
    startup_candle_count: int = 200  # Based on the longest SMA (200 periods)

    # Trailing stop: Freqtrade offers built-in trailing stop features.
    # This is a placeholder if we want to enable it via class attribute.
    # trailing_stop = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.02
    # trailing_only_offset_is_reached = False

    # --- Plotting configuration ---
    # Define plotconfig if you want to plot indicators with Freqtrade.
    # This is optional and can be expanded later.
    plot_config = {
        'main_plot': {
            'sma_short': {'color': 'blue'},
            'sma_medium': {'color': 'orange'},
            'sma_long': {'color': 'purple'},
        },
        'subplots': {
            "RSI": {
                'rsi': {'color': 'red'},
            },
            "OBV": {
                'obv': {'color': 'green'},
                'obv_sma': {'color': 'darkgreen'},
            }
        }
    }

    # --- Strategy parameters for Hyperopt ---
    # These are examples if we want to enable Hyperopt later
    # buy_rsi = IntParameter(low=30, high=50, default=35, space="buy")
    # sell_rsi = IntParameter(low=50, high=70, default=65, space="sell")


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        '''
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        '''
        # --- SMAs ---
        dataframe['sma_short'] = ta.SMA(dataframe, timeperiod=10)
        dataframe['sma_medium'] = ta.SMA(dataframe, timeperiod=50)
        dataframe['sma_long'] = ta.SMA(dataframe, timeperiod=200)

        # --- RSI ---
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # --- OBV (On-Balance Volume) ---
        # TALIB's OBV uses close and volume. Ensure 'volume' column is present and float.
        # Freqtrade DataFrames typically have 'volume' as float.
        dataframe['obv'] = ta.OBV(dataframe['close'], dataframe['volume'])
        dataframe['obv_sma'] = ta.SMA(dataframe['obv'], timeperiod=20)

        # --- ATR (Average True Range - Wilder's Smoothing by default in TA-Lib) ---
        # TALIB's ATR needs high, low, close.
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the 'enter_long' and 'enter_tag' columns for buy signals.

        Conditions for Long Entry (Golden Cross Enhanced):
        1. SMA Short (10) crosses above SMA Medium (50).
        2. Current close price is above SMA Long (200).
        3. RSI (14) is above 50 (indicating bullish momentum).
        4. OBV is above its 20-period SMA (confirming volume accumulation).
        5. Current volume is greater than 0.

        Args:
            dataframe: Pandas DataFrame with OHLCV data and populated indicators.
            metadata: Dictionary containing pair metadata.

        Returns:
            Pandas DataFrame with 'enter_long' and 'enter_tag' columns populated.
        """
        # Initialize 'enter_long' and 'enter_short' columns
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0  # Strategy does not short, but good practice

        # Define conditions for long entry
        conditions_long = (
            (qtpylib.crossed_above(dataframe['sma_short'], dataframe['sma_medium'])) &
            (dataframe['close'] > dataframe['sma_long']) &
            (dataframe['rsi'] > 50) &
            (dataframe['obv'] > dataframe['obv_sma']) &
            (dataframe['volume'] > 0)  # Ensure there is trading volume
        )

        # Apply conditions to set 'enter_long' and 'enter_tag'
        dataframe.loc[conditions_long, ['enter_long', 'enter_tag']] = (1, 'golden_cross_enhanced')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the 'exit_long' and 'exit_tag' columns for sell signals.
        This method defines condition-based exits. Freqtrade also handles exits
        based on ROI, stoploss, and time TSL if configured at class/config level.

        Conditions for Long Exit (Opposing Signal - Death Cross Enhanced):
        1. SMA Short (10) crosses below SMA Medium (50).
        2. RSI (14) is below 50 (indicating bearish momentum).
        3. OBV is below its 20-period SMA (confirming volume distribution).
        4. Current volume is greater than 0.

        Args:
            dataframe: Pandas DataFrame with OHLCV data and populated indicators.
            metadata: Dictionary containing pair metadata.

        Returns:
            Pandas DataFrame with 'exit_long' and 'exit_tag' columns populated.
        """
        # Initialize 'exit_long' and 'exit_short' columns
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0 # Strategy does not short

        # Define conditions for long exit
        conditions_exit_long = (
            (qtpylib.crossed_below(dataframe['sma_short'], dataframe['sma_medium'])) &
            (dataframe['rsi'] < 50) &
            (dataframe['obv'] < dataframe['obv_sma']) &
            (dataframe['volume'] > 0)  # Ensure there is trading volume
        )

        # Apply conditions to set 'exit_long' and 'exit_tag'
        dataframe.loc[conditions_exit_long, ['exit_long', 'exit_tag']] = (1, 'death_cross_exit')

        return dataframe

# --- Optional: Main block for testing populate_indicators ---
if __name__ == '__main__':
    import pandas as pd
    import numpy as np

    # Create a sample DataFrame (mimicking Freqtrade's structure)
    data = {
        'date': pd.to_datetime([f'2023-01-{(i%30)+1:02d} 00:00:00' for i in range(250)]), # Sample dates
        'open': np.random.rand(250) * 100 + 1000,
        'high': np.random.rand(250) * 10 + 1100, # Ensure high is higher
        'low': np.random.rand(250) * 10 + 990,   # Ensure low is lower
        'close': np.random.rand(250) * 100 + 1000,
        'volume': np.random.rand(250) * 1000 + 100
    }
    sample_df = DataFrame(data)
    # Ensure 'high' is always >= 'open' and 'close', and 'low' is always <= 'open' and 'close'
    sample_df['high'] = sample_df[['high', 'open', 'close']].max(axis=1)
    sample_df['low'] = sample_df[['low', 'open', 'close']].min(axis=1)

    # Instantiate the strategy (config can be empty for this test)
    strategy = SMACryptoEnhanced(config={})

    # Populate indicators
    df_with_indicators = strategy.populate_indicators(sample_df.copy(), {'pair': 'BTC/USDT'})

    print("--- DataFrame with Indicators (last 5 rows) ---")
    print(df_with_indicators.tail())

    print("\n--- Checking for NaNs in last few rows of added indicators ---")
    last_rows = df_with_indicators.tail()
    indicator_cols = ['sma_short', 'sma_medium', 'sma_long', 'rsi', 'obv', 'obv_sma', 'atr']
    for col in indicator_cols:
        if col in last_rows:
            print(f"'{col}' last 5 values: {last_rows[col].values}")
            if last_rows[col].isnull().any():
                print(f"WARNING: NaNs found in recent values of '{col}'")
        else:
            print(f"ERROR: Column '{col}' not found in DataFrame.")

    # A simple check to see if the longest SMA has values (not NaN) in the last row
    # (assuming 250 candles is enough for SMA200 to have a value)
    if 'sma_long' in df_with_indicators and not df_with_indicators['sma_long'].iloc[-1:].isnull().all():
        print("\nLongest SMA (sma_long) has non-NaN values in the last row, good.")
    else:
        print("\nWARNING: Longest SMA (sma_long) might still be NaN in the last row, check data length or startup_candle_count.")
