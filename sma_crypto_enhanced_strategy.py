# --- Do not remove these libs ---
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, CategoricalParameter
from pandas import DataFrame
from datetime import datetime
from freqtrade.persistence import Trade # For type hinting Trade object
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
    # minimal_roi = {
    #     "60": 0.01,  # 1% profit after 60 minutes
    #     "30": 0.02,  # 2% profit after 30 minutes
    #     "0": 0.04    # 4% profit immediately
    # }

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
    # SMA Periods
    sma_short_period = IntParameter(low=5, high=25, default=10, space="buy", optimize=True, load=True)
    sma_medium_period = IntParameter(low=30, high=70, default=50, space="buy", optimize=True, load=True)
    # sma_long_period is fixed at 200

    # RSI Parameters
    rsi_period = IntParameter(low=10, high=25, default=14, space="buy", optimize=True, load=True)
    rsi_buy_level = IntParameter(low=40, high=60, default=50, space="buy", optimize=True, load=True)
    rsi_sell_level = IntParameter(low=40, high=60, default=50, space="sell", optimize=True, load=True) # For exit condition

    # OBV SMA Period
    obv_sma_period = IntParameter(low=10, high=30, default=20, space="buy", optimize=True, load=True)

    # Stoploss Percentage (Freqtrade will use this class attribute 'stoploss' if it's a Parameter)
    stoploss = DecimalParameter(-0.20, -0.01, default=-0.10, decimals=2, space="sell", optimize=True, load=True)

    # Minimal ROI - optimizing the target for 0 minutes
    # Other ROI points can be kept static or also made optimizable
    roi_time0_profit = DecimalParameter(low=0.01, high=0.10, default=0.04, decimals=3, space="roi", optimize=True, load=True)

    # Second ROI point (optimizable time and profit)
    roi_time1_minutes = IntParameter(low=15, high=120, default=30, step=15, space="roi", optimize=True, load=True)
    roi_time1_profit = DecimalParameter(low=0.005, high=0.05, default=0.02, decimals=3, space="roi", optimize=True, load=True)

    # ATR Trailing Stop Multiplier
    tsl_atr_multiplier = DecimalParameter(low=1.0, high=5.0, default=2.0, decimals=1, space="protection", optimize=True, load=True)
    # ATR Period for Trailing Stop
    tsl_atr_period = IntParameter(low=7, high=28, default=14, space="protection", optimize=True, load=True)

    @property
    def minimal_roi(self):
        # Freqtrade expects keys to be integers (minutes).
        # The dictionary items are processed by Freqtrade to determine the ROI behavior.
        # It's important that time_ H.W. Bush, George H.W. Bush, George H. W. Bush, G. H. W. Bush, Bush Sr., Bush 41, Bush the Elder, George Herbert Walker Bushvalues are unique and sorted, which dict literals do by insertion order for Python 3.7+
        # Freqtrade typically handles sorting of these times internally if needed.

        # Ensure the time for the third static point is distinct and later than the optimized second point
        # A simple way is to add a fixed duration to the optimized time of the second point.
        time_for_third_point = self.roi_time1_minutes.value + 120

        return {
            0: self.roi_time0_profit.value,
            self.roi_time1_minutes.value: self.roi_time1_profit.value,
            time_for_third_point: 0.001 # Static minimal profit target long after the second point
        }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame using hyperoptable parameters.
        """
        # --- SMAs ---
        dataframe['sma_short'] = ta.SMA(dataframe, timeperiod=self.sma_short_period.value)
        dataframe['sma_medium'] = ta.SMA(dataframe, timeperiod=self.sma_medium_period.value)
        dataframe['sma_long'] = ta.SMA(dataframe, timeperiod=200) # Fixed long SMA

        # --- RSI ---
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.rsi_period.value)

        # --- OBV (On-Balance Volume) ---
        dataframe['obv'] = ta.OBV(dataframe['close'], dataframe['volume'])
        dataframe['obv_sma'] = ta.SMA(dataframe['obv'], timeperiod=self.obv_sma_period.value)

        # --- ATR (Average True Range - Wilder's Smoothing by default in TA-Lib) ---
        # TALIB's ATR needs high, low, close.
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=self.tsl_atr_period.value)

        return dataframe

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: 'datetime',
                        current_rate: float, current_profit: float, **kwargs) -> float:
        '''
        Custom stoploss logic, implementing an ATR-based trailing stop.
        This method is called by Freqtrade for open trades each tick.
        :param pair: Pair that's currently open (e.g. 'BTC/USDT')
        :param trade: Trade object. Attributes like `trade.open_rate`, `trade.is_short`,
                      `trade.stop_loss_pct` (initial stoploss percentage),
                      `trade.stop_loss` (current absolute stop price).
        :param current_time: datetime object, current candle datetime
        :param current_rate: Current rate for pair
        :param current_profit: Current profit (as ratio, e.g. 0.01 for 1%)
        :return: New absolute stop-loss price. Returning -1 leaves stoploss untouched.
                 A positive value sets the new absolute stoploss.
        '''
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return -1 # Data not found, leave stoploss untouched

        # Ensure 'atr' column exists
        if 'atr' not in dataframe.columns:
            # This might happen if populate_indicators hasn't run or 'atr' was dropped
            return -1

        last_candle_atr = dataframe['atr'].iloc[-1]

        # Ensure last_candle_atr is a valid number, otherwise, don't update stop
        if not isinstance(last_candle_atr, (float, int)) or last_candle_atr <= 0 or np.isnan(last_candle_atr):
            return -1 # Keep current stop_loss price

        atr_val = last_candle_atr * self.tsl_atr_multiplier.value

        # trade.stop_loss should hold the current absolute stop price set by Freqtrade
        # (either from initial stoploss or previous custom_stoploss calls)
        current_stop_price = trade.stop_loss

        if trade.is_short:
            new_potential_stop = current_rate + atr_val
            # For short, stop moves down. We want the lower of current stop and new potential stop.
            # (min ensures it doesn't move against us, only trails or stays)
            if current_stop_price is not None:
                return min(current_stop_price, new_potential_stop)
            else: # Should not happen if initial stoploss is set
                return new_potential_stop
        else: # Long trade
            new_potential_stop = current_rate - atr_val
            # For long, stop moves up. We want the higher of current stop and new potential stop.
            # (max ensures it doesn't move against us, only trails or stays)
            if current_stop_price is not None:
                return max(current_stop_price, new_potential_stop)
            else: # Should not happen if initial stoploss is set
                return new_potential_stop

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
            (dataframe['rsi'] > self.rsi_buy_level.value) &
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
            (dataframe['rsi'] < self.rsi_sell_level.value) &
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
