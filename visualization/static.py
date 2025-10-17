"""
Static visualization module using Matplotlib
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


class StaticVisualizer:
    """Create static Matplotlib visualizations"""
    
    def __init__(self, style: str = 'default'):
        """
        Initialize visualizer
        
        Args:
            style: Matplotlib style to use
        """
        self.style = style
        if style != 'default':
            plt.style.use(style)
    
    def plot_comprehensive_analysis(self, df_train: pd.DataFrame, 
                                   df_test: pd.DataFrame,
                                   results_df: pd.DataFrame,
                                   ticker_train: str,
                                   ticker_test: str,
                                   feature_cols: list,
                                   best_model,
                                   best_model_name: str,
                                   from_date: str = '2024-01-01') -> plt.Figure:
        """
        Create comprehensive static analysis with 6 subplots
        
        Args:
            df_train: Training dataset with backtest results
            df_test: Test dataset with backtest results
            results_df: ML model results DataFrame
            ticker_train: Training ticker symbol
            ticker_test: Test ticker symbol
            feature_cols: List of feature column names
            best_model: Trained best model
            best_model_name: Name of the best model
            from_date: Start date for plotting
            
        Returns:
            Matplotlib Figure object
        """
        fig, axes = plt.subplots(3, 2, figsize=(20, 15))
        fig.suptitle('Comprehensive Trading Strategy Analysis', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Train Price with Signals
        self._plot_price_signals(axes[0, 0], df_train, ticker_train, from_date)
        
        # Plot 2: Test Price with Signals
        self._plot_price_signals(axes[0, 1], df_test, ticker_test, from_date)
        
        # Plot 3: Train Capital Curve
        self._plot_capital_curve(axes[1, 0], df_train, ticker_train, from_date)
        
        # Plot 4: Test Capital Curve
        self._plot_capital_curve(axes[1, 1], df_test, ticker_test, from_date)
        
        # Plot 5: Model Comparison
        self._plot_model_comparison(axes[2, 0], results_df)
        
        # Plot 6: Feature Importance
        self._plot_feature_importance(axes[2, 1], best_model, best_model_name, 
                                     feature_cols)
        
        plt.tight_layout()
        return fig
    
    def _plot_price_signals(self, ax, df: pd.DataFrame, ticker: str, 
                           from_date: str):
        """Plot price with trading signals"""
        df_plot = df[df.index >= from_date].copy()
        
        ax.plot(df_plot.index, df_plot['Close'], label='Close', 
               linewidth=2, color='blue')
        ax.plot(df_plot.index, df_plot['BB_Upper'], '--', alpha=0.5, 
               color='red', label='BB Upper')
        ax.plot(df_plot.index, df_plot['BB_Lower'], '--', alpha=0.5, 
               color='green', label='BB Lower')
        
        buy = df_plot[df_plot['Signal'] == 1]
        sell = df_plot[df_plot['Signal'] == -1]
        
        ax.scatter(buy.index, buy['Close'], marker='^', color='lime', 
                  s=200, label='Buy', zorder=5, edgecolors='darkgreen', 
                  linewidths=2)
        ax.scatter(sell.index, sell['Close'], marker='v', color='red', 
                  s=200, label='Sell', zorder=5, edgecolors='darkred', 
                  linewidths=2)
        
        ax.set_title(f'{ticker} - Trading Signals', fontsize=14, fontweight='bold')
        ax.set_ylabel('Price (USD)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_capital_curve(self, ax, df: pd.DataFrame, ticker: str, 
                           from_date: str):
        """Plot capital curve"""
        df_plot = df[df.index >= from_date].copy()
        
        ax.plot(df_plot.index, df_plot['Capital'], linewidth=2, 
               color='green', label='Portfolio Value')
        ax.axhline(y=10000, color='gray', linestyle='--', 
                  label='Initial Capital', linewidth=2)
        
        ax.fill_between(df_plot.index, 10000, df_plot['Capital'], 
                       where=(df_plot['Capital'] >= 10000), 
                       alpha=0.3, color='green', label='Profit')
        ax.fill_between(df_plot.index, 10000, df_plot['Capital'], 
                       where=(df_plot['Capital'] < 10000), 
                       alpha=0.3, color='red', label='Loss')
        
        ax.set_title(f'{ticker} - Portfolio Value', fontsize=14, fontweight='bold')
        ax.set_ylabel('Capital (USD)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_model_comparison(self, ax, results_df: pd.DataFrame):
        """Plot ML model comparison"""
        x_pos = np.arange(len(results_df))
        bars = ax.barh(x_pos, results_df['F1 Score'], 
                      color=plt.cm.RdYlGn(results_df['F1 Score']))
        
        ax.set_yticks(x_pos)
        ax.set_yticklabels(results_df['Model'])
        ax.set_xlabel('F1 Score', fontsize=12)
        ax.set_title('ML Model Performance Comparison', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        for i, v in enumerate(results_df['F1 Score']):
            ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=10)
    
    def _plot_feature_importance(self, ax, best_model, best_model_name: str, 
                                 feature_cols: list):
        """Plot feature importance for tree-based models"""
        if any(name in best_model_name for name in ['Random Forest', 'XGBoost', 'LightGBM', 'Gradient Boosting']):
            model_step = best_model.named_steps[list(best_model.named_steps.keys())[1]]
            
            if hasattr(model_step, 'feature_importances_'):
                importances = model_step.feature_importances_
                indices = np.argsort(importances)[-15:]
                
                ax.barh(range(len(indices)), importances[indices], color='teal')
                ax.set_yticks(range(len(indices)))
                ax.set_yticklabels([feature_cols[i] for i in indices], fontsize=9)
                ax.set_xlabel('Importance', fontsize=12)
                ax.set_title(f'Top 15 Feature Importances - {best_model_name}', 
                           fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='x')
            else:
                self._plot_no_feature_importance(ax, best_model_name)
        else:
            self._plot_no_feature_importance(ax, best_model_name)
    
    def _plot_no_feature_importance(self, ax, model_name: str):
        """Show message when feature importance is not available"""
        ax.text(0.5, 0.5, f'{model_name}\nFeature Importance Not Available', 
               ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_title('Feature Importance', fontsize=14, fontweight='bold')
        ax.axis('off')
    
    def plot_trade_details(self, trades_df: pd.DataFrame, 
                          ticker: str) -> Optional[plt.Figure]:
        """
        Create detailed trade analysis plots
        
        Args:
            trades_df: DataFrame with trade details
            ticker: Stock ticker symbol
            
        Returns:
            Matplotlib Figure object or None
        """
        if len(trades_df) == 0:
            print("No trades to visualize")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f'{ticker} - Trade Analysis', fontsize=16, fontweight='bold')
        
        # P&L Distribution
        axes[0, 0].hist(trades_df['PnL'], bins=30, 
                       color=['green' if x > 0 else 'red' for x in trades_df['PnL']],
                       edgecolor='black', alpha=0.7)
        axes[0, 0].set_title('Trade P&L Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('P&L ($)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(x=0, color='black', linestyle='--', linewidth=2)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Win/Loss by Type
        trade_summary = trades_df.groupby('Type').apply(
            lambda x: pd.Series({
                'Wins': len(x[x['PnL'] > 0]),
                'Losses': len(x[x['PnL'] < 0])
            })
        )
        
        x = np.arange(len(trade_summary))
        width = 0.35
        axes[0, 1].bar(x - width/2, trade_summary['Wins'], width, 
                      label='Wins', color='green')
        axes[0, 1].bar(x + width/2, trade_summary['Losses'], width, 
                      label='Losses', color='red')
        axes[0, 1].set_title('Win/Loss by Trade Type', fontsize=12, fontweight='bold')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(trade_summary.index)
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Cumulative P&L
        trades_sorted = trades_df.sort_values('Exit_Date')
        cumulative_pnl = trades_sorted['PnL'].cumsum()
        axes[1, 0].plot(range(len(cumulative_pnl)), cumulative_pnl, 
                       linewidth=2, color='blue')
        axes[1, 0].fill_between(range(len(cumulative_pnl)), 0, cumulative_pnl, 
                               alpha=0.3, color='blue')
        axes[1, 0].set_title('Cumulative P&L', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Trade Number')
        axes[1, 0].set_ylabel('Cumulative P&L ($)')
        axes[1, 0].axhline(y=0, color='black', linestyle='--', linewidth=1)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Trade Duration
        trades_df['Duration'] = (pd.to_datetime(trades_df['Exit_Date']) - 
                                pd.to_datetime(trades_df['Entry_Date'])).dt.days
        
        trade_types = trades_df['Type'].unique()
        durations = [trades_df[trades_df['Type'] == t]['Duration'].values 
                    for t in trade_types]
        
        axes[1, 1].boxplot(durations, labels=trade_types, patch_artist=True)
        axes[1, 1].set_title('Trade Duration Analysis', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Duration (days)')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def show(self):
        """Display all plots"""
        plt.show()
    
    def save_figure(self, fig: plt.Figure, filename: str, dpi: int = 300):
        """
        Save figure to file
        
        Args:
            fig: Matplotlib Figure object
            filename: Output filename
            dpi: Resolution in dots per inch
        """
        fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to {filename}")
