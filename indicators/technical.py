"""
Technical indicator calculations
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
    def calculate_kama(df: pd.DataFrame, period: int = 200, 
                       fast_ema: int = 2, slow_ema: int = 30) -> pd.Series:
        """
        Calculate Kaufman Adaptive Moving Average
        
        Args:
            df: DataFrame with Close prices
            period: Lookback period
            fast_ema: Fast EMA period
            slow_ema: Slow EMA period
            
        Returns:
            KAMA series
        """
        close = df['Close'].values
        kama = np.zeros(len(close))
        kama[0] = float(close[0])
        
        fastest = 2.0 / (fast_ema + 1)
        slowest = 2.0 / (slow_ema + 1)
        
        for i in range(1, len(close)):
            if i < period:
                kama[i] = float(close[i])
            else:
                change = abs(close[i] - close[i - period])
                volatility = np.sum(np.abs(np.diff(close[i - period:i + 1])))
                
                if volatility != 0:
                    er = change / volatility
                else:
                    er = 0
                    
                sc = (er * (fastest - slowest) + slowest) ** 2
                kama[i] = float(kama[i - 1] + sc * (close[i] - kama[i - 1]))
        
        return pd.Series(kama, index=df.index)
    
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
        Calculate Average True Range
        
        Args:
            df: DataFrame with High, Low, Close prices
            period: ATR period
            
        Returns:
            ATR series
        """
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index
        
        Args:
            df: DataFrame with Close prices
            period: RSI period
            
        Returns:
            RSI series
        """
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
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
        
        # KAMA
        kama_params = params.get('kama', {})
        kama_period = kama_params.get('period', 100)
        df[f'KAMA_{kama_period}'] = TechnicalIndicators.calculate_kama(
            df,
            period=kama_period,
            fast_ema=kama_params.get('fast_ema', 2),
            slow_ema=kama_params.get('slow_ema', 30)
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
        
        # ATR
        atr_params = params.get('atr', {})
        df['ATR'] = TechnicalIndicators.calculate_atr(
            df,
            period=atr_params.get('period', 14)
        )
        
        # RSI
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