"""
Backtesting engine for trading strategies
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict


class BacktestEngine:
    """Backtest trading strategies"""
    
    def __init__(self, initial_capital: float = 10000, 
                 position_size: float = 0.1, max_loss: float = 0.02):
        """
        Initialize BacktestEngine
        
        Args:
            initial_capital: Starting capital
            position_size: Fraction of capital per trade
            max_loss: Maximum loss per trade (not used currently)
        """
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.max_loss = max_loss
    
    def run_backtest(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Run backtest on DataFrame with signals
        
        Args:
            df: DataFrame with signals and prices
            
        Returns:
            Tuple of (df_with_capital, trades_df, metrics_dict)
        """
        df = df.copy()
        df['Return'] = df['Close'].pct_change()
        
        capital = self.initial_capital
        position = 0
        entry_price = 0
        stop_loss = 0
        shares = 0
        entry_date = None
        
        capital_history = []
        position_history = []
        trade_log = []
        
        for i in range(len(df)):
            current_price = df['Close'].iloc[i]
            signal = df['Signal'].iloc[i]
            
            # Check stop loss for long position
            if position == 1:
                if current_price <= stop_loss:
                    pnl = (current_price - entry_price) * shares
                    capital += pnl
                    
                    trade_log.append({
                        'Entry_Date': entry_date,
                        'Exit_Date': df.index[i],
                        'Type': 'Long',
                        'Entry_Price': entry_price,
                        'Exit_Price': current_price,
                        'Shares': shares,
                        'PnL': pnl,
                        'Return_%': (pnl / (entry_price * shares)) * 100,
                        'Exit_Reason': 'Stop Loss'
                    })
                    
                    position = 0
                    shares = 0
            
            # Check stop loss for short position
            elif position == -1:
                if current_price >= stop_loss:
                    pnl = (entry_price - current_price) * shares
                    capital += pnl
                    
                    trade_log.append({
                        'Entry_Date': entry_date,
                        'Exit_Date': df.index[i],
                        'Type': 'Short',
                        'Entry_Price': entry_price,
                        'Exit_Price': current_price,
                        'Shares': shares,
                        'PnL': pnl,
                        'Return_%': (pnl / (entry_price * shares)) * 100,
                        'Exit_Reason': 'Stop Loss'
                    })
                    
                    position = 0
                    shares = 0
            
            # Handle buy signal
            if signal == 1:
                if position == -1:
                    # Close short position
                    pnl = (entry_price - current_price) * shares
                    capital += pnl
                    
                    trade_log.append({
                        'Entry_Date': entry_date,
                        'Exit_Date': df.index[i],
                        'Type': 'Short',
                        'Entry_Price': entry_price,
                        'Exit_Price': current_price,
                        'Shares': shares,
                        'PnL': pnl,
                        'Return_%': (pnl / (entry_price * shares)) * 100,
                        'Exit_Reason': 'Signal Reversal'
                    })
                
                # Open long position
                trade_amount = capital * self.position_size
                shares = trade_amount / current_price
                entry_price = current_price
                entry_date = df.index[i]
                stop_loss = df['Stop_Loss'].iloc[i]
                position = 1
            
            # Handle sell signal
            elif signal == -1:
                if position == 1:
                    # Close long position
                    pnl = (current_price - entry_price) * shares
                    capital += pnl
                    
                    trade_log.append({
                        'Entry_Date': entry_date,
                        'Exit_Date': df.index[i],
                        'Type': 'Long',
                        'Entry_Price': entry_price,
                        'Exit_Price': current_price,
                        'Shares': shares,
                        'PnL': pnl,
                        'Return_%': (pnl / (entry_price * shares)) * 100,
                        'Exit_Reason': 'Signal Reversal'
                    })
                
                # Open short position
                trade_amount = capital * self.position_size
                shares = trade_amount / current_price
                entry_price = current_price
                entry_date = df.index[i]
                stop_loss = df['Stop_Loss'].iloc[i]
                position = -1
            
            # Calculate current capital
            if position == 1:
                unrealized_pnl = (current_price - entry_price) * shares
                current_capital = capital + unrealized_pnl
            elif position == -1:
                unrealized_pnl = (entry_price - current_price) * shares
                current_capital = capital + unrealized_pnl
            else:
                current_capital = capital
            
            capital_history.append(current_capital)
            position_history.append(position)
        
        # Close any remaining position
        if position != 0:
            current_price = df['Close'].iloc[-1]
            if position == 1:
                pnl = (current_price - entry_price) * shares
            else:
                pnl = (entry_price - current_price) * shares
            
            capital += pnl
            
            trade_log.append({
                'Entry_Date': entry_date,
                'Exit_Date': df.index[-1],
                'Type': 'Long' if position == 1 else 'Short',
                'Entry_Price': entry_price,
                'Exit_Price': current_price,
                'Shares': shares,
                'PnL': pnl,
                'Return_%': (pnl / (entry_price * shares)) * 100,
                'Exit_Reason': 'End of Period'
            })
        
        df['Capital'] = capital_history
        df['Position'] = position_history
        
        trades_df = pd.DataFrame(trade_log)
        metrics = self._calculate_metrics(df, trades_df, capital)
        
        return df, trades_df, metrics
    
    def _calculate_metrics(self, df: pd.DataFrame, trades_df: pd.DataFrame, 
                          final_capital: float) -> Dict:
        """
        Calculate performance metrics
        
        Args:
            df: DataFrame with capital history
            trades_df: DataFrame with trade log
            final_capital: Final capital value
            
        Returns:
            Dictionary of metrics
        """
        total_return = ((final_capital - self.initial_capital) / self.initial_capital) * 100
        
        if len(trades_df) > 0:
            winning_trades = trades_df[trades_df['PnL'] > 0]
            losing_trades = trades_df[trades_df['PnL'] < 0]
            
            win_rate = (len(winning_trades) / len(trades_df)) * 100
            avg_win = winning_trades['PnL'].mean() if len(winning_trades) > 0 else 0
            avg_loss = losing_trades['PnL'].mean() if len(losing_trades) > 0 else 0
            
            if len(losing_trades) > 0 and losing_trades['PnL'].sum() != 0:
                profit_factor = abs(winning_trades['PnL'].sum() / losing_trades['PnL'].sum())
            else:
                profit_factor = float('inf')
            
            cumulative = df['Capital']
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max * 100
            max_drawdown = drawdown.min()
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            max_drawdown = 0
            winning_trades = pd.DataFrame()
            losing_trades = pd.DataFrame()
        
        return {
            'Initial Capital': self.initial_capital,
            'Final Capital': final_capital,
            'Total Return %': total_return,
            'Total Trades': len(trades_df),
            'Winning Trades': len(winning_trades),
            'Losing Trades': len(losing_trades),
            'Win Rate %': win_rate,
            'Average Win': avg_win,
            'Average Loss': avg_loss,
            'Profit Factor': profit_factor,
            'Max Drawdown %': max_drawdown
        }