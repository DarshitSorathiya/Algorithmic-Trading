"""
Candlestick pattern detection and swing point identification
"""
import pandas as pd
import numpy as np


class SwingDetector:
    """Detect swing highs and lows"""
    
    @staticmethod
    def find_swing_points(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
        """
        Find swing highs and lows
        
        Args:
            df: DataFrame with High and Low prices
            window: Lookback/lookforward window size
            
        Returns:
            DataFrame with swing points identified
        """
        df = df.copy()
        df['Is_Swing_High'] = 0
        df['Is_Swing_Low'] = 0
        
        for i in range(window, len(df)):
            lookback_high = df['High'].iloc[max(0, i-window):i+1].max()
            lookback_low = df['Low'].iloc[max(0, i-window):i+1].min()
            
            if i + window < len(df):
                lookforward_high = df['High'].iloc[i:min(len(df), i+window+1)].max()
                lookforward_low = df['Low'].iloc[i:min(len(df), i+window+1)].min()
                
                if df['High'].iloc[i] == lookback_high and df['High'].iloc[i] == lookforward_high:
                    df.iloc[i, df.columns.get_loc('Is_Swing_High')] = 1
                
                if df['Low'].iloc[i] == lookback_low and df['Low'].iloc[i] == lookforward_low:
                    df.iloc[i, df.columns.get_loc('Is_Swing_Low')] = 1
            else:
                if df['High'].iloc[i] == lookback_high:
                    df.iloc[i, df.columns.get_loc('Is_Swing_High')] = 1
                
                if df['Low'].iloc[i] == lookback_low:
                    df.iloc[i, df.columns.get_loc('Is_Swing_Low')] = 1
        
        return df


class PatternDetector:
    """Detect candlestick patterns and swing points"""
    
    @staticmethod
    def detect_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect common candlestick patterns
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with pattern indicators added
        """
        df = df.copy()
        
        # Calculate body and shadow sizes
        df['Body'] = abs(df['Close'] - df['Open'])
        df['Upper_Shadow'] = df['High'] - df[['Close', 'Open']].max(axis=1)
        df['Lower_Shadow'] = df[['Close', 'Open']].min(axis=1) - df['Low']
        df['Range'] = df['High'] - df['Low']
        
        # Doji
        df['Doji'] = (df['Body'] <= df['Range'] * 0.1) & (df['Range'] > 0)
        
        # Hammer (bullish)
        df['Hammer'] = (
            (df['Lower_Shadow'] > 2 * df['Body']) &
            (df['Upper_Shadow'] < df['Body'] * 0.3) &
            (df['Body'] > 0)
        )
        
        # Shooting Star (bearish)
        df['Shooting_Star'] = (
            (df['Upper_Shadow'] > 2 * df['Body']) &
            (df['Lower_Shadow'] < df['Body'] * 0.3) &
            (df['Body'] > 0)
        )
        
        # Engulfing patterns
        df['Bullish_Engulfing'] = (
            (df['Close'] > df['Open']) &
            (df['Close'].shift(1) < df['Open'].shift(1)) &
            (df['Open'] < df['Close'].shift(1)) &
            (df['Close'] > df['Open'].shift(1))
        )
        
        df['Bearish_Engulfing'] = (
            (df['Close'] < df['Open']) &
            (df['Close'].shift(1) > df['Open'].shift(1)) &
            (df['Open'] > df['Close'].shift(1)) &
            (df['Close'] < df['Open'].shift(1))
        )
        
        return df
    
    @staticmethod
    def detect_all_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect all candlestick patterns
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with all pattern indicators added
        """
        return PatternDetector.detect_candlestick_patterns(df)