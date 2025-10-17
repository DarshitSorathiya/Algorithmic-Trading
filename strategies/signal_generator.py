"""
Trading signal generation strategy
"""
import pandas as pd
import numpy as np
from indicators.technical import TechnicalIndicators
from indicators.patterns import PatternDetector, SwingDetector


class SignalGenerator:
    """Generate trading signals based on technical analysis"""
    
    def __init__(self, params: dict = None):
        """
        Initialize SignalGenerator
        
        Args:
            params: Dictionary with 'swing_window' and 'min_distance' keys
        """
        if params is None:
            params = {}
        self.swing_window = params.get('swing_window', 12)
        self.min_distance = params.get('min_distance', 10)
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals
        
        Args:
            df: DataFrame with OHLCV data (with indicators already added)
            
        Returns:
            DataFrame with signals added
        """
        # Handle multi-level columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df = df.copy()
        
        # Initialize signal columns
        df['Signal'] = 0
        df['Signal_Type'] = ''
        df['Entry_Price'] = np.nan
        df['Stop_Loss'] = np.nan
        
        last_signal_bar = -self.min_distance
        
        # Generate signals
        for i in range(50, len(df)):
            bars_since_signal = i - last_signal_bar
            
            if bars_since_signal < self.min_distance:
                continue
            
            # Buy signal at swing low
            if df['Is_Swing_Low'].iloc[i] == 1:
                macd_confirm = (df['MACD'].iloc[i] < 0.5 and 
                               (df['MACD_Cross_Up'].iloc[i] or df['MACD_Rising'].iloc[i] or 
                                df['MACD'].iloc[i] > df['MACD'].iloc[i-3]))
                
                rsi_confirm = df['RSI'].iloc[i] < 55
                bb_confirm = df['Close'].iloc[i] < df['BB_Middle'].iloc[i]
                price_confirm = df['Close'].iloc[i] < df['Close'].iloc[i-5:i].mean()
                
                if macd_confirm or rsi_confirm or bb_confirm or price_confirm:
                    df.loc[df.index[i], 'Signal'] = 1
                    df.loc[df.index[i], 'Signal_Type'] = 'Swing Low'
                    df.loc[df.index[i], 'Entry_Price'] = df['Close'].iloc[i]
                    df.loc[df.index[i], 'Stop_Loss'] = df['Close'].iloc[i] - 2 * df['ATR'].iloc[i]
                    last_signal_bar = i
            
            # Sell signal at swing high
            elif df['Is_Swing_High'].iloc[i] == 1:
                macd_confirm = (df['MACD'].iloc[i] > -0.5 and 
                               (df['MACD_Cross_Down'].iloc[i] or df['MACD_Falling'].iloc[i] or
                                df['MACD'].iloc[i] < df['MACD'].iloc[i-3]))
                
                rsi_confirm = df['RSI'].iloc[i] > 45
                bb_confirm = df['Close'].iloc[i] > df['BB_Middle'].iloc[i]
                price_confirm = df['Close'].iloc[i] > df['Close'].iloc[i-5:i].mean()
                
                if macd_confirm or rsi_confirm or bb_confirm or price_confirm:
                    df.loc[df.index[i], 'Signal'] = -1
                    df.loc[df.index[i], 'Signal_Type'] = 'Swing High'
                    df.loc[df.index[i], 'Entry_Price'] = df['Close'].iloc[i]
                    df.loc[df.index[i], 'Stop_Loss'] = df['Close'].iloc[i] + 2 * df['ATR'].iloc[i]
                    last_signal_bar = i
        
        return df