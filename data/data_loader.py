"""
Data loading and preprocessing module
"""
import yfinance as yf
import pandas as pd
from typing import Optional


class DataLoader:
    """Handles data fetching and preprocessing for stock market data"""
    
    def __init__(self, ticker: str, start_date: str, end_date: str, 
                 interval: str = '1d', auto_adjust: bool = True):
        """
        Initialize DataLoader
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for data (YYYY-MM-DD)
            end_date: End date for data (YYYY-MM-DD)
            interval: Data interval (1d, 1h, etc.)
            auto_adjust: Whether to auto-adjust prices
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.auto_adjust = auto_adjust
        self.data = None
    
    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch data from Yahoo Finance
        
        Returns:
            DataFrame with OHLCV data
        """
        print(f"Downloading {self.ticker} data from {self.start_date} to {self.end_date}...")
        
        self.data = yf.download(
            self.ticker,
            start=self.start_date,
            end=self.end_date,
            interval=self.interval,
            progress=False,
            auto_adjust=self.auto_adjust
        )
        
        # Handle multi-level columns if present
        if isinstance(self.data.columns, pd.MultiIndex):
            self.data.columns = self.data.columns.get_level_values(0)
        
        print(f"Downloaded {len(self.data)} rows of data")
        return self.data
    
    def get_feature_columns(self, exclude_cols: list) -> list:
        """
        Get feature columns excluding specified columns
        
        Args:
            exclude_cols: List of column names to exclude
            
        Returns:
            List of feature column names
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call fetch_data() first.")
        
        return [col for col in self.data.columns if col not in exclude_cols]
    
    def prepare_ml_data(self, feature_cols: list, target_col: str = 'Signal'):
        """
        Prepare data for machine learning
        
        Args:
            feature_cols: List of feature column names
            target_col: Target column name
            
        Returns:
            Tuple of (X, y) where X is features and y is target
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call fetch_data() first.")
        
        # Drop NaN values
        df_clean = self.data.dropna().copy()
        
        X = df_clean[feature_cols]
        y = df_clean[target_col]
        
        print(f"\nDataset shape: {X.shape}")
        print(f"Signal distribution:\n{y.value_counts()}")
        
        return X, y
    
    def get_data_summary(self) -> dict:
        """
        Get summary statistics of the loaded data
        
        Returns:
            Dictionary with summary statistics
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call fetch_data() first.")
        
        return {
            'ticker': self.ticker,
            'start_date': self.data.index.min(),
            'end_date': self.data.index.max(),
            'total_rows': len(self.data),
            'columns': list(self.data.columns),
            'missing_values': self.data.isnull().sum().to_dict()
        }