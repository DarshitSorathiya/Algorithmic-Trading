"""
Technical indicator calculations (ATR & RSI use Wilder's smoothing)
"""
import pandas as pd
import numpy as np


class TechnicalIndicators:
    """Calculate various technical indicators"""
    
    @staticmethod
    def calculate_ema(data: pd.Series, period: int) -> pd.Series:
        """
        Calculate Exponential Moving Average
        
        Args:
            data: Price series
            period: EMA period
            
        Returns:
            EMA series
        """
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_sma(data: pd.Series, period: int) -> pd.Series:
        """
        Calculate Simple Moving Average
        
        Args:
            data: Price series
            period: SMA period
            
        Returns:
            SMA series
        """
        return data.rolling(window=period).mean()
    
    @staticmethod
    def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, 
                       signal: int = 9) -> tuple:
        """
        Calculate MACD indicator
        
        Args:
            df: DataFrame with Close prices
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            
        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        ema_fast = TechnicalIndicators.calculate_ema(df['Close'], fast)
        ema_slow = TechnicalIndicators.calculate_ema(df['Close'], slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def calculate_bollinger_bands(df: pd.DataFrame, period: int = 20, 
                                   std_dev: int = 2) -> tuple:
        """
        Calculate Bollinger Bands
        
        Args:
            df: DataFrame with Close prices
            period: Moving average period
            std_dev: Standard deviation multiplier
            
        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        sma = df['Close'].rolling(window=period).mean()
        std = df['Close'].rolling(window=period).std()
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        return upper_band, sma, lower_band
    
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range using Wilder's smoothing:
        - TR computed per standard definition
        - initial ATR (at index period-1) = simple average of first 'period' TR values
        - subsequent ATRs: ATR_t = (ATR_{t-1} * (period-1) + TR_t) / period
        """
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        tr_values = tr.values
        atr = np.full(len(tr_values), np.nan, dtype=float)
        
        if len(tr_values) >= period:
            # first ATR value: simple average of first `period` TRs (at index period-1)
            first_atr = np.nanmean(tr_values[:period])
            atr[period - 1] = first_atr
            # Wilder smoothing for subsequent values
            for i in range(period, len(tr_values)):
                atr[i] = (atr[i - 1] * (period - 1) + tr_values[i]) / period
        
        return pd.Series(atr, index=df.index)
    
    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index using Wilder's smoothing:
        - initial average gain/loss = simple average of first 'period' gains/losses
          (value placed at index period)
        - subsequent averages use Wilder's smoothing:
          avg_gain_t = (avg_gain_{t-1} * (period-1) + gain_t) / period
        - RSI = 100 - (100 / (1 + RS)), with edge handling:
          * if avg_loss == 0 and avg_gain == 0 -> RSI = 50 (no movement)
          * if avg_loss == 0 -> RSI = 100
        """
        close = df['Close']
        delta = close.diff().values
        
        gains = np.where(delta > 0, delta, 0.0)
        losses = np.where(delta < 0, -delta, 0.0)
        
        avg_gain = np.full(len(delta), np.nan, dtype=float)
        avg_loss = np.full(len(delta), np.nan, dtype=float)
        
        if len(delta) > period:
            # first average (placed at index `period`)
            avg_gain[period] = np.nanmean(gains[1:period+1])  # exclude delta[0] which is NaN
            avg_loss[period] = np.nanmean(losses[1:period+1])
            
            # Wilder smoothing for subsequent values
            for i in range(period + 1, len(delta)):
                avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i]) / period
                avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i]) / period
        
        # compute RSI with proper edge handling
        rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss != 0)
        rsi = np.full(len(delta), np.nan, dtype=float)
        
        for i in range(len(delta)):
            ag = avg_gain[i]
            al = avg_loss[i]
            if np.isnan(ag) or np.isnan(al):
                rsi[i] = np.nan
            else:
                if al == 0.0:
                    if ag == 0.0:
                        rsi[i] = 50.0  # no movement
                    else:
                        rsi[i] = 100.0  # gains but no losses
                else:
                    rs_i = ag / al
                    rsi[i] = 100.0 - (100.0 / (1.0 + rs_i))
        
        return pd.Series(rsi, index=df.index)
    
    @staticmethod
    def add_all_indicators(df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """
        Add all technical indicators to DataFrame
        
        Args:
            df: DataFrame with OHLCV data
            params: Dictionary of indicator parameters
            
        Returns:
            DataFrame with added indicators
        """
        df = df.copy()
        
        # MACD
        macd_params = params.get('macd', {})
        df['MACD'], df['Signal_Line'], df['MACD_Hist'] = TechnicalIndicators.calculate_macd(
            df, 
            fast=macd_params.get('fast', 12),
            slow=macd_params.get('slow', 26),
            signal=macd_params.get('signal', 9)
        )
        
        # Additional Moving Averages - Dynamic periods
        ma_params = params.get('moving_averages', {})
        sma_short = ma_params.get('sma_short', 20)
        sma_long = ma_params.get('sma_long', 50)
        ema_short = ma_params.get('ema_short', 20)
        ema_long = ma_params.get('ema_long', 50)
        
        df[f'SMA_{sma_short}'] = TechnicalIndicators.calculate_sma(df['Close'], sma_short)
        df[f'SMA_{sma_long}'] = TechnicalIndicators.calculate_sma(df['Close'], sma_long)
        df[f'EMA_{ema_short}'] = TechnicalIndicators.calculate_ema(df['Close'], ema_short)
        df[f'EMA_{ema_long}'] = TechnicalIndicators.calculate_ema(df['Close'], ema_long)
        
        # Bollinger Bands
        bb_params = params.get('bollinger_bands', {})
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = TechnicalIndicators.calculate_bollinger_bands(
            df,
            period=bb_params.get('period', 20),
            std_dev=bb_params.get('std_dev', 2)
        )
        
        # ATR (Wilder)
        atr_params = params.get('atr', {})
        df['ATR'] = TechnicalIndicators.calculate_atr(
            df,
            period=atr_params.get('period', 14)
        )
        
        # RSI (Wilder)
        rsi_params = params.get('rsi', {})
        df['RSI'] = TechnicalIndicators.calculate_rsi(
            df,
            period=rsi_params.get('period', 14)
        )
        
        # MACD crosses and trends
        df['MACD_Cross_Up'] = (df['MACD'] > df['Signal_Line']) & (df['MACD'].shift(1) <= df['Signal_Line'].shift(1))
        df['MACD_Cross_Down'] = (df['MACD'] < df['Signal_Line']) & (df['MACD'].shift(1) >= df['Signal_Line'].shift(1))
        df['MACD_Rising'] = (df['MACD_Hist'] > df['MACD_Hist'].shift(1)) & (df['MACD_Hist'].shift(1) > df['MACD_Hist'].shift(2))
        df['MACD_Falling'] = (df['MACD_Hist'] < df['MACD_Hist'].shift(1)) & (df['MACD_Hist'].shift(1) < df['MACD_Hist'].shift(2))
        
        return df