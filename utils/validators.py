"""
Validation utilities for trading strategy system
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional


class DataValidator:
    """Validate data quality and integrity"""
    
    @staticmethod
    def validate_ohlcv(df: pd.DataFrame) -> List[str]:
        """
        Validate OHLCV data integrity
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Check required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
            return errors
        
        # Check for negative values
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if (df[col] < 0).any():
                errors.append(f"{col} contains negative values")
        
        # Check High >= Low
        invalid_hl = df[df['High'] < df['Low']]
        if len(invalid_hl) > 0:
            errors.append(f"Found {len(invalid_hl)} bars where High < Low")
        
        # Check High >= Open/Close
        invalid_ho = df[df['High'] < df['Open']]
        if len(invalid_ho) > 0:
            errors.append(f"Found {len(invalid_ho)} bars where High < Open")
        
        invalid_hc = df[df['High'] < df['Close']]
        if len(invalid_hc) > 0:
            errors.append(f"Found {len(invalid_hc)} bars where High < Close")
        
        # Check Low <= Open/Close
        invalid_lo = df[df['Low'] > df['Open']]
        if len(invalid_lo) > 0:
            errors.append(f"Found {len(invalid_lo)} bars where Low > Open")
        
        invalid_lc = df[df['Low'] > df['Close']]
        if len(invalid_lc) > 0:
            errors.append(f"Found {len(invalid_lc)} bars where Low > Close")
        
        return errors
    
    @staticmethod
    def check_data_gaps(df: pd.DataFrame, max_gap_days: int = 7) -> List[Dict]:
        """
        Check for gaps in data
        
        Args:
            df: DataFrame with date index
            max_gap_days: Maximum acceptable gap in days
            
        Returns:
            List of gap information dicts
        """
        gaps = []
        
        if len(df) < 2:
            return gaps
        
        dates = pd.to_datetime(df.index)
        date_diffs = dates.to_series().diff()
        
        large_gaps = date_diffs[date_diffs > timedelta(days=max_gap_days)]
        
        for idx, gap_size in large_gaps.items():
            gaps.append({
                'date': idx,
                'gap_days': gap_size.days,
                'previous_date': dates[dates.get_loc(idx) - 1]
            })
        
        return gaps
    
    @staticmethod
    def detect_outliers(df: pd.DataFrame, column: str, 
                       threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect outliers using z-score method
        
        Args:
            df: DataFrame
            column: Column to check
            threshold: Z-score threshold (default 3.0)
            
        Returns:
            DataFrame of outliers
        """
        if column not in df.columns:
            return pd.DataFrame()
        
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        outliers = df[z_scores > threshold].copy()
        outliers['z_score'] = z_scores[z_scores > threshold]
        
        return outliers
    
    @staticmethod
    def check_data_completeness(df: pd.DataFrame, 
                                min_rows: int = 100) -> Tuple[bool, str]:
        """
        Check if data is complete enough for analysis
        
        Args:
            df: DataFrame to check
            min_rows: Minimum required rows
            
        Returns:
            Tuple of (is_valid, message)
        """
        if len(df) < min_rows:
            return False, f"Insufficient data: {len(df)} rows (minimum {min_rows})"
        
        # Check for too many NaN values
        nan_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if nan_ratio > 0.3:
            return False, f"Too many missing values: {nan_ratio*100:.1f}%"
        
        return True, "Data is complete"


class ConfigValidator:
    """Validate configuration parameters"""
    
    @staticmethod
    def validate_date_range(start_date: str, end_date: str) -> Tuple[bool, str]:
        """
        Validate date range
        
        Args:
            start_date: Start date string
            end_date: End date string
            
        Returns:
            Tuple of (is_valid, message)
        """
        try:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
        except Exception as e:
            return False, f"Invalid date format: {e}"
        
        if start >= end:
            return False, "Start date must be before end date"
        
        if end > datetime.now():
            return False, "End date cannot be in the future"
        
        days_diff = (end - start).days
        if days_diff < 30:
            return False, f"Date range too short: {days_diff} days (minimum 30)"
        
        return True, "Date range is valid"
    
    @staticmethod
    def validate_numeric_param(value: float, min_val: float, max_val: float,
                               param_name: str) -> Tuple[bool, str]:
        """
        Validate numeric parameter
        
        Args:
            value: Parameter value
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            param_name: Parameter name for error message
            
        Returns:
            Tuple of (is_valid, message)
        """
        if not isinstance(value, (int, float)):
            return False, f"{param_name} must be a number"
        
        if value < min_val or value > max_val:
            return False, f"{param_name} must be between {min_val} and {max_val}"
        
        return True, f"{param_name} is valid"
    
    @staticmethod
    def validate_config(config: Dict) -> List[str]:
        """
        Validate entire configuration
        
        Args:
            config: Configuration dictionary
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Validate date range
        if 'start_date' in config and 'end_date' in config:
            is_valid, msg = ConfigValidator.validate_date_range(
                config['start_date'], config['end_date']
            )
            if not is_valid:
                errors.append(msg)
        
        # Validate position size
        if 'position_size' in config:
            is_valid, msg = ConfigValidator.validate_numeric_param(
                config['position_size'], 0.01, 1.0, "position_size"
            )
            if not is_valid:
                errors.append(msg)
        
        # Validate initial capital
        if 'initial_capital' in config:
            is_valid, msg = ConfigValidator.validate_numeric_param(
                config['initial_capital'], 100, 10000000, "initial_capital"
            )
            if not is_valid:
                errors.append(msg)
        
        return errors


class TickerValidator:
    """Validate stock ticker symbols"""
    
    @staticmethod
    def validate_ticker(ticker: str) -> Tuple[bool, str]:
        """
        Validate ticker symbol format
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            Tuple of (is_valid, message)
        """
        if not ticker:
            return False, "Ticker cannot be empty"
        
        if not isinstance(ticker, str):
            return False, "Ticker must be a string"
        
        ticker = ticker.strip().upper()
        
        if len(ticker) < 1 or len(ticker) > 10:
            return False, "Ticker must be 1-10 characters"
        
        if not ticker.isalnum() and not all(c.isalnum() or c in ['.', '-'] for c in ticker):
            return False, "Ticker contains invalid characters"
        
        return True, f"Ticker '{ticker}' is valid"
    
    @staticmethod
    def normalize_ticker(ticker: str) -> str:
        """
        Normalize ticker symbol
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            Normalized ticker
        """
        return ticker.strip().upper()


class InputValidator:
    """Validate user inputs"""
    
    @staticmethod
    def validate_streamlit_inputs(train_ticker: str, test_ticker: str,
                                  start_date: datetime, end_date: datetime,
                                  initial_capital: float, 
                                  position_size: float) -> List[str]:
        """
        Validate all Streamlit inputs
        
        Args:
            train_ticker: Training ticker
            test_ticker: Test ticker
            start_date: Start date
            end_date: End date
            initial_capital: Initial capital
            position_size: Position size
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Validate tickers
        is_valid, msg = TickerValidator.validate_ticker(train_ticker)
        if not is_valid:
            errors.append(f"Training ticker: {msg}")
        
        is_valid, msg = TickerValidator.validate_ticker(test_ticker)
        if not is_valid:
            errors.append(f"Test ticker: {msg}")
        
        # Validate date range
        is_valid, msg = ConfigValidator.validate_date_range(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        if not is_valid:
            errors.append(msg)
        
        # Validate capital
        is_valid, msg = ConfigValidator.validate_numeric_param(
            initial_capital, 100, 10000000, "Initial capital"
        )
        if not is_valid:
            errors.append(msg)
        
        # Validate position size
        is_valid, msg = ConfigValidator.validate_numeric_param(
            position_size, 0.01, 1.0, "Position size"
        )
        if not is_valid:
            errors.append(msg)
        
        return errors
