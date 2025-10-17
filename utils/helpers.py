"""
Utility functions and helpers
"""
import pandas as pd
import numpy as np


class MetricsCalculator:
    """Calculate trading and portfolio metrics"""
    
    @staticmethod
    def calculate_drawdown(capital_series: pd.Series) -> pd.Series:
        """
        Calculate drawdown series
        
        Args:
            capital_series: Series of capital values
            
        Returns:
            Drawdown series (percentage)
        """
        running_max = capital_series.cummax()
        drawdown = (capital_series - running_max) / running_max * 100
        return drawdown
    
    @staticmethod
    def calculate_max_drawdown(capital_series: pd.Series) -> float:
        """
        Calculate maximum drawdown
        
        Args:
            capital_series: Series of capital values
            
        Returns:
            Maximum drawdown percentage
        """
        drawdown = MetricsCalculator.calculate_drawdown(capital_series)
        return drawdown.min()
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio
        
        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Sharpe ratio
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    
    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sortino ratio
        
        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Sortino ratio
        """
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / 252)
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        
        return np.sqrt(252) * excess_returns.mean() / downside_returns.std()
    
    @staticmethod
    def calculate_win_rate(trades_df: pd.DataFrame) -> float:
        """
        Calculate win rate
        
        Args:
            trades_df: DataFrame with trades
            
        Returns:
            Win rate percentage
        """
        if len(trades_df) == 0:
            return 0.0
        
        winning_trades = trades_df[trades_df['PnL'] > 0]
        return (len(winning_trades) / len(trades_df)) * 100
    
    @staticmethod
    def calculate_profit_factor(trades_df: pd.DataFrame) -> float:
        """
        Calculate profit factor
        
        Args:
            trades_df: DataFrame with trades
            
        Returns:
            Profit factor
        """
        if len(trades_df) == 0:
            return 0.0
        
        winning_trades = trades_df[trades_df['PnL'] > 0]
        losing_trades = trades_df[trades_df['PnL'] < 0]
        
        if len(losing_trades) == 0 or losing_trades['PnL'].sum() == 0:
            return float('inf')
        
        return abs(winning_trades['PnL'].sum() / losing_trades['PnL'].sum())
    
    @staticmethod
    def calculate_expectancy(trades_df: pd.DataFrame) -> float:
        """
        Calculate expectancy per trade
        
        Args:
            trades_df: DataFrame with trades
            
        Returns:
            Average expectancy per trade
        """
        if len(trades_df) == 0:
            return 0.0
        
        return trades_df['PnL'].mean()


class DataValidator:
    """Validate and clean data"""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: list) -> bool:
        """
        Validate if DataFrame has required columns
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            True if valid, False otherwise
        """
        missing_cols = [col for col in required_columns if col not in df.columns]
        
        if missing_cols:
            print(f"Missing columns: {missing_cols}")
            return False
        
        return True
    
    @staticmethod
    def remove_outliers(df: pd.DataFrame, column: str, 
                       n_std: float = 3.0) -> pd.DataFrame:
        """
        Remove outliers from DataFrame based on standard deviation
        
        Args:
            df: DataFrame
            column: Column name to check for outliers
            n_std: Number of standard deviations
            
        Returns:
            DataFrame with outliers removed
        """
        mean = df[column].mean()
        std = df[column].std()
        
        lower_bound = mean - n_std * std
        upper_bound = mean + n_std * std
        
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    @staticmethod
    def fill_missing_values(df: pd.DataFrame, method: str = 'ffill') -> pd.DataFrame:
        """
        Fill missing values in DataFrame
        
        Args:
            df: DataFrame
            method: Method to fill ('ffill', 'bfill', 'mean', 'median')
            
        Returns:
            DataFrame with filled values
        """
        df_filled = df.copy()
        
        if method in ['ffill', 'bfill']:
            df_filled = df_filled.fillna(method=method)
        elif method == 'mean':
            df_filled = df_filled.fillna(df_filled.mean())
        elif method == 'median':
            df_filled = df_filled.fillna(df_filled.median())
        else:
            raise ValueError(f"Invalid method: {method}")
        
        return df_filled


class ReportGenerator:
    """Generate trading reports"""
    
    @staticmethod
    def generate_summary_report(metrics: dict, ticker: str) -> str:
        """
        Generate text summary report
        
        Args:
            metrics: Dictionary of metrics
            ticker: Stock ticker symbol
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 70)
        report.append(f"{ticker} - BACKTEST SUMMARY REPORT")
        report.append("=" * 70)
        report.append("")
        
        report.append("PERFORMANCE METRICS:")
        report.append("-" * 70)
        
        for key, value in metrics.items():
            if isinstance(value, float):
                if abs(value) > 1000:
                    report.append(f"  {key:<30}: ${value:,.2f}")
                else:
                    report.append(f"  {key:<30}: {value:.2f}")
            else:
                report.append(f"  {key:<30}: {value}")
        
        report.append("")
        report.append("=" * 70)
        
        return "\n".join(report)
    
    @staticmethod
    def generate_trade_report(trades_df: pd.DataFrame, ticker: str) -> str:
        """
        Generate trade details report
        
        Args:
            trades_df: DataFrame with trades
            ticker: Stock ticker symbol
            
        Returns:
            Formatted report string
        """
        if len(trades_df) == 0:
            return f"No trades executed for {ticker}"
        
        report = []
        report.append("=" * 70)
        report.append(f"{ticker} - TRADE DETAILS REPORT")
        report.append("=" * 70)
        report.append("")
        
        report.append(f"Total Trades: {len(trades_df)}")
        report.append(f"Winning Trades: {len(trades_df[trades_df['PnL'] > 0])}")
        report.append(f"Losing Trades: {len(trades_df[trades_df['PnL'] < 0])}")
        report.append("")
        
        report.append("TRADE SUMMARY:")
        report.append("-" * 70)
        
        for idx, trade in trades_df.head(10).iterrows():
            report.append(f"\nTrade #{idx + 1}:")
            report.append(f"  Type: {trade['Type']}")
            report.append(f"  Entry: {trade['Entry_Date']} @ ${trade['Entry_Price']:.2f}")
            report.append(f"  Exit:  {trade['Exit_Date']} @ ${trade['Exit_Price']:.2f}")
            report.append(f"  P&L:   ${trade['PnL']:.2f} ({trade['Return_%']:.2f}%)")
            report.append(f"  Reason: {trade['Exit_Reason']}")
        
        if len(trades_df) > 10:
            report.append(f"\n... and {len(trades_df) - 10} more trades")
        
        report.append("")
        report.append("=" * 70)
        
        return "\n".join(report)
    
    @staticmethod
    def save_report_to_file(report: str, filename: str):
        """
        Save report to text file
        
        Args:
            report: Report string
            filename: Output filename
        """
        with open(filename, 'w') as f:
            f.write(report)
        print(f"Report saved to {filename}")


class ProgressPrinter:
    """Print progress messages"""
    
    @staticmethod
    def print_header(title: str, width: int = 70):
        """Print header"""
        print("\n" + "=" * width)
        print(title.center(width))
        print("=" * width)
    
    @staticmethod
    def print_subheader(title: str, width: int = 70):
        """Print subheader"""
        print("\n" + "-" * width)
        print(title)
        print("-" * width)
    
    @staticmethod
    def print_step(step: str):
        """Print step message"""
        print(f"\n➤ {step}...")
    
    @staticmethod
    def print_success(message: str):
        """Print success message"""
        print(f"✓ {message}")
    
    @staticmethod
    def print_warning(message: str):
        """Print warning message"""
        print(f"⚠ WARNING: {message}")
    
    @staticmethod
    def print_error(message: str):
        """Print error message"""
        print(f"✗ ERROR: {message}")
