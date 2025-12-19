"""
Interactive visualization module using Plotly
"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional


class InteractiveVisualizer:
    """Create interactive Plotly visualizations"""
    
    def __init__(self, theme: str = 'plotly_dark'):
        """
        Initialize visualizer
        
        Args:
            theme: Plotly theme to use
        """
        self.theme = theme
    
    def plot_candlestick(self, df: pd.DataFrame, ticker: str, 
                         from_date: str = '2024-01-01',
                         trades_df: Optional[pd.DataFrame] = None,
                         height: int = 1400) -> go.Figure:
        """
        Create interactive candlestick chart with all indicators
        
        Args:
            df: DataFrame with price data and indicators
            ticker: Stock ticker symbol
            from_date: Start date for plotting
            trades_df: Optional DataFrame with trades
            height: Chart height
            
        Returns:
            Plotly Figure object
        """
        df_plot = df[df.index >= from_date].copy()
        
        fig = make_subplots(
            rows=5, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(
                f'{ticker} - Candlestick with Signals', 
                'MACD', 
                'RSI', 
                'Volume', 
                'Bollinger Bands %B'
            ),
            row_heights=[0.4, 0.15, 0.15, 0.15, 0.15]
        )
        
        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df_plot.index,
            open=df_plot['Open'],
            high=df_plot['High'],
            low=df_plot['Low'],
            close=df_plot['Close'],
            name='OHLC',
            increasing_line_color='green',
            decreasing_line_color='red'
        ), row=1, col=1)
        
        # Bollinger Bands
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot['BB_Upper'],
            name='BB Upper', line=dict(color='rgba(250, 128, 114, 0.5)', width=1)
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot['BB_Middle'],
            name='BB Middle', line=dict(color='rgba(135, 206, 250, 0.8)', width=1, dash='dash')
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot['BB_Lower'],
            name='BB Lower', line=dict(color='rgba(250, 128, 114, 0.5)', width=1),
            fill='tonexty', fillcolor='rgba(250, 128, 114, 0.1)'
        ), row=1, col=1)
        
        # Buy Signals
        buy_signals = df_plot[df_plot['Signal'] == 1]
        fig.add_trace(go.Scatter(
            x=buy_signals.index, y=buy_signals['Low'] * 0.995,
            mode='markers', name='Buy Signal',
            marker=dict(symbol='triangle-up', size=15, color='lime', 
                       line=dict(color='darkgreen', width=2))
        ), row=1, col=1)
        
        # Sell Signals
        sell_signals = df_plot[df_plot['Signal'] == -1]
        fig.add_trace(go.Scatter(
            x=sell_signals.index, y=sell_signals['High'] * 1.005,
            mode='markers', name='Sell Signal',
            marker=dict(symbol='triangle-down', size=15, color='red', 
                       line=dict(color='darkred', width=2))
        ), row=1, col=1)
        
        # Candlestick Patterns
        hammers = df_plot[df_plot['Hammer'] == True]
        if len(hammers) > 0:
            fig.add_trace(go.Scatter(
                x=hammers.index, y=hammers['Low'] * 0.99,
                mode='markers', name='Hammer',
                marker=dict(symbol='star', size=10, color='yellow')
            ), row=1, col=1)
        
        shooting_stars = df_plot[df_plot['Shooting_Star'] == True]
        if len(shooting_stars) > 0:
            fig.add_trace(go.Scatter(
                x=shooting_stars.index, y=shooting_stars['High'] * 1.01,
                mode='markers', name='Shooting Star',
                marker=dict(symbol='star', size=10, color='orange')
            ), row=1, col=1)
        
        # MACD
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot['MACD'],
            name='MACD', line=dict(color='blue', width=1.5)
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot['Signal_Line'],
            name='Signal', line=dict(color='orange', width=1.5)
        ), row=2, col=1)
        
        colors = ['green' if val >= 0 else 'red' for val in df_plot['MACD_Hist']]
        fig.add_trace(go.Bar(
            x=df_plot.index, y=df_plot['MACD_Hist'],
            name='Histogram', marker_color=colors, opacity=0.5
        ), row=2, col=1)
        
        # RSI
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot['RSI'],
            name='RSI', line=dict(color='purple', width=2)
        ), row=3, col=1)
        
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1, opacity=0.5)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1, opacity=0.5)
        
        # Volume
        if 'Volume' in df_plot.columns:
            colors = ['green' if df_plot['Close'].iloc[i] >= df_plot['Open'].iloc[i] else 'red' 
                      for i in range(len(df_plot))]
            fig.add_trace(go.Bar(
                x=df_plot.index, y=df_plot['Volume'],
                name='Volume', marker_color=colors, opacity=0.5
            ), row=4, col=1)
        
        # %B (Bollinger Band percentage)
        df_plot['BB_Percent'] = (df_plot['Close'] - df_plot['BB_Lower']) / (df_plot['BB_Upper'] - df_plot['BB_Lower'])
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot['BB_Percent'],
            name='%B', line=dict(color='teal', width=2), 
            fill='tozeroy', fillcolor='rgba(0, 128, 128, 0.2)'
        ), row=5, col=1)
        
        fig.add_hline(y=1, line_dash="dash", line_color="red", row=5, col=1, opacity=0.5)
        fig.add_hline(y=0, line_dash="dash", line_color="green", row=5, col=1, opacity=0.5)
        
        fig.update_layout(
            title=f'{ticker} - Complete Technical Analysis Dashboard',
            xaxis_rangeslider_visible=False,
            height=height,
            showlegend=True,
            hovermode='x unified',
            template=self.theme
        )
        
        fig.update_xaxes(title_text="Date", row=5, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="MACD", row=2, col=1)
        fig.update_yaxes(title_text="RSI", row=3, col=1)
        fig.update_yaxes(title_text="Volume", row=4, col=1)
        fig.update_yaxes(title_text="%B", row=5, col=1)
        
        return fig
    
    def plot_portfolio_analysis(self, df: pd.DataFrame, metrics: dict, 
                                ticker: str, from_date: str = '2024-01-01',
                                height: int = 1000) -> go.Figure:
        """
        Create interactive portfolio performance dashboard
        
        Args:
            df: DataFrame with backtest results
            metrics: Dictionary of performance metrics
            ticker: Stock ticker symbol
            from_date: Start date for plotting
            height: Chart height
            
        Returns:
            Plotly Figure object
        """
        df_plot = df[df.index >= from_date].copy()
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Portfolio Value', 'Drawdown', 'Daily Returns Distribution', 
                'Cumulative Returns', 'Position History', 'Return by Month'
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ]
        )
        
        # Portfolio Value
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot['Capital'],
            name='Portfolio Value', line=dict(color='blue', width=2),
            fill='tozeroy', fillcolor='rgba(0, 0, 255, 0.1)'
        ), row=1, col=1)
        
        fig.add_hline(y=10000, line_dash="dash", line_color="gray", row=1, col=1)
        
        # Drawdown
        cumulative = df_plot['Capital']
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max * 100
        
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=drawdown,
            name='Drawdown', line=dict(color='red', width=2),
            fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.2)'
        ), row=1, col=2)
        
        # Daily Returns Distribution
        daily_returns = df_plot['Capital'].pct_change() * 100
        fig.add_trace(go.Histogram(
            x=daily_returns.dropna(),
            name='Returns', marker_color='purple', opacity=0.7, nbinsx=50
        ), row=2, col=1)
        
        # Cumulative Returns
        cumulative_returns = ((df_plot['Capital'] / df_plot['Capital'].iloc[0]) - 1) * 100
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=cumulative_returns,
            name='Cumulative Return %', line=dict(color='green', width=2)
        ), row=2, col=2)
        
        # Position History
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot['Position'],
            name='Position', line=dict(color='orange', width=2),
            fill='tozeroy'
        ), row=3, col=1)
        
        # Return by Month
        df_plot['Month'] = df_plot.index.to_period('M')
        monthly_returns = df_plot.groupby('Month')['Capital'].last().pct_change() * 100
        
        fig.add_trace(go.Bar(
            x=[str(m) for m in monthly_returns.index],
            y=monthly_returns.values,
            name='Monthly Return %',
            marker_color=['green' if v > 0 else 'red' for v in monthly_returns.values]
        ), row=3, col=2)
        
        fig.update_layout(
            title=f'{ticker} - Portfolio Performance Dashboard',
            height=height,
            showlegend=True,
            template=self.theme
        )
        
        return fig
    
    def plot_ml_comparison(self, results_df: pd.DataFrame, 
                          confusion_matrices: dict, ticker_test: str,
                          height: int = 900) -> go.Figure:
        """
        Create ML model comparison visualizations
        
        Args:
            results_df: DataFrame with model results
            confusion_matrices: Dictionary of confusion matrices
            ticker_test: Test ticker symbol
            height: Chart height
            
        Returns:
            Plotly Figure object
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Model Performance Comparison', 'F1 Score by Model', 
                'Precision vs Recall', 'Model Metrics Heatmap'
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "heatmap"}]
            ]
        )
        
        # Model Performance Comparison
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        for metric in metrics:
            fig.add_trace(go.Bar(
                name=metric,
                x=results_df['Model'],
                y=results_df[metric],
                text=results_df[metric].round(3),
                textposition='auto'
            ), row=1, col=1)
        
        # F1 Score Ranking
        sorted_df = results_df.sort_values('F1 Score', ascending=True)
        colors = ['green' if x > 0.5 else 'orange' if x > 0.3 else 'red' 
                  for x in sorted_df['F1 Score']]
        
        fig.add_trace(go.Bar(
            x=sorted_df['F1 Score'],
            y=sorted_df['Model'],
            orientation='h',
            marker_color=colors,
            text=sorted_df['F1 Score'].round(3),
            textposition='auto'
        ), row=1, col=2)
        
        # Precision vs Recall Scatter
        fig.add_trace(go.Scatter(
            x=results_df['Precision'],
            y=results_df['Recall'],
            mode='markers+text',
            text=results_df['Model'],
            textposition='top center',
            marker=dict(
                size=results_df['F1 Score']*100, 
                color=results_df['F1 Score'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="F1 Score")
            ),
            name='Models'
        ), row=2, col=1)
        
        # Add diagonal line for reference
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            line=dict(dash='dash', color='gray'),
            showlegend=False
        ), row=2, col=1)
        
        # Metrics Heatmap
        heatmap_data = results_df[['Accuracy', 'Precision', 'Recall', 'F1 Score']].values
        fig.add_trace(go.Heatmap(
            z=heatmap_data.T,
            x=results_df['Model'],
            y=['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            colorscale='RdYlGn',
            text=heatmap_data.T.round(3),
            texttemplate='%{text}',
            textfont={"size": 10}
        ), row=2, col=2)
        
        fig.update_layout(
            title=f'Machine Learning Model Comparison - Tested on {ticker_test}',
            height=height,
            showlegend=True,
            template=self.theme
        )
        
        fig.update_xaxes(title_text="Model", row=1, col=1)
        fig.update_yaxes(title_text="Score", row=1, col=1)
        fig.update_xaxes(title_text="F1 Score", row=1, col=2)
        fig.update_xaxes(title_text="Precision", row=2, col=1)
        fig.update_yaxes(title_text="Recall", row=2, col=1)
        
        return fig
    
    def plot_confusion_matrix(self, cm, class_names: list, 
                             model_name: str) -> go.Figure:
        """
        Create interactive confusion matrix
        
        Args:
            cm: Confusion matrix
            class_names: List of class names
            model_name: Name of the model
            
        Returns:
            Plotly Figure object
        """
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=class_names,
            y=class_names,
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 16},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=f'Confusion Matrix - {model_name}',
            xaxis_title='Predicted',
            yaxis_title='Actual',
            height=500,
            template=self.theme
        )
        
        return fig
    
    def plot_trade_analysis(self, trades_df: pd.DataFrame, 
                           ticker: str) -> Optional[go.Figure]:
        """
        Create detailed trade analysis visualizations
        
        Args:
            trades_df: DataFrame with trade details
            ticker: Stock ticker symbol
            
        Returns:
            Plotly Figure object or None
        """
        if len(trades_df) == 0:
            print("No trades to visualize")
            return None
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Trade P&L Distribution', 'Win/Loss by Trade Type', 
                'Cumulative P&L', 'Trade Duration Analysis'
            ),
            specs=[
                [{"type": "histogram"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "box"}]
            ]
        )
        
        # P&L Distribution
        fig.add_trace(go.Histogram(
            x=trades_df['PnL'],
            name='P&L',
            marker_color=['green' if x > 0 else 'red' for x in trades_df['PnL']],
            nbinsx=30
        ), row=1, col=1)
        
        # Win/Loss by Type
        trade_summary = trades_df.groupby('Type').apply(
            lambda x: pd.Series({
                'Wins': len(x[x['PnL'] > 0]),
                'Losses': len(x[x['PnL'] < 0])
            })
        ).reset_index()
        
        fig.add_trace(go.Bar(
            name='Wins',
            x=trade_summary['Type'],
            y=trade_summary['Wins'],
            marker_color='green'
        ), row=1, col=2)
        
        fig.add_trace(go.Bar(
            name='Losses',
            x=trade_summary['Type'],
            y=trade_summary['Losses'],
            marker_color='red'
        ), row=1, col=2)
        
        # Cumulative P&L
        trades_df_sorted = trades_df.sort_values('Exit_Date')
        cumulative_pnl = trades_df_sorted['PnL'].cumsum()
        
        fig.add_trace(go.Scatter(
            x=trades_df_sorted['Exit_Date'],
            y=cumulative_pnl,
            name='Cumulative P&L',
            line=dict(color='blue', width=2),
            fill='tozeroy'
        ), row=2, col=1)
        
        # Trade Duration
        trades_df['Duration'] = (pd.to_datetime(trades_df['Exit_Date']) - 
                                pd.to_datetime(trades_df['Entry_Date'])).dt.days
        
        fig.add_trace(go.Box(
            y=trades_df['Duration'],
            x=trades_df['Type'],
            name='Duration',
            marker_color='purple'
        ), row=2, col=2)
        
        fig.update_layout(
            title=f'{ticker} - Trade Analysis Dashboard',
            height=800,
            showlegend=True,
            template=self.theme
        )
        
        return fig
    
    def plot_performance_metrics(self, metrics: dict) -> go.Figure:
        """
        Create visual representation of performance metrics
        
        Args:
            metrics: Dictionary of performance metrics
            
        Returns:
            Plotly Figure object
        """
        gauges_fig = make_subplots(
            rows=2, cols=3,
            specs=[
                [{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}],
                [{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]
            ],
            subplot_titles=(
                'Total Return %', 'Win Rate %', 'Profit Factor', 
                'Max Drawdown %', 'Total Trades', 'Avg Win/Loss Ratio'
            )
        )
        
        # Total Return
        gauges_fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=metrics['Total Return %'],
            delta={'reference': 0},
            gauge={
                'axis': {'range': [-50, 100]},
                'bar': {'color': "darkgreen" if metrics['Total Return %'] > 0 else "darkred"},
                'steps': [
                    {'range': [-50, 0], 'color': "lightcoral"},
                    {'range': [0, 100], 'color': "lightgreen"}
                ],
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 0}
            },
            title={'text': "Total Return %"}
        ), row=1, col=1)
        
        # Win Rate
        gauges_fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=metrics['Win Rate %'],
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "green"},
                'steps': [
                    {'range': [0, 40], 'color': "lightcoral"},
                    {'range': [40, 60], 'color': "lightyellow"},
                    {'range': [60, 100], 'color': "lightgreen"}
                ]
            },
            title={'text': "Win Rate %"}
        ), row=1, col=2)
        
        # Profit Factor
        pf_value = min(metrics['Profit Factor'], 5) if metrics['Profit Factor'] != float('inf') else 5
        gauges_fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=pf_value,
            gauge={
                'axis': {'range': [0, 5]},
                'bar': {'color': "blue"},
                'steps': [
                    {'range': [0, 1], 'color': "lightcoral"},
                    {'range': [1, 2], 'color': "lightyellow"},
                    {'range': [2, 5], 'color': "lightgreen"}
                ]
            },
            title={'text': "Profit Factor"}
        ), row=1, col=3)
        
        # Max Drawdown
        gauges_fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=abs(metrics['Max Drawdown %']),
            gauge={
                'axis': {'range': [0, 50]},
                'bar': {'color': "red"},
                'steps': [
                    {'range': [0, 10], 'color': "lightgreen"},
                    {'range': [10, 25], 'color': "lightyellow"},
                    {'range': [25, 50], 'color': "lightcoral"}
                ]
            },
            title={'text': "Max Drawdown %"}
        ), row=2, col=1)
        
        # Total Trades
        gauges_fig.add_trace(go.Indicator(
            mode="number",
            value=metrics['Total Trades'],
            title={'text': "Total Trades"},
            number={'font': {'size': 50}}
        ), row=2, col=2)
        
        # Avg Win/Loss Ratio
        ratio = abs(metrics['Average Win'] / metrics['Average Loss']) if metrics['Average Loss'] != 0 else 0
        gauges_fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=ratio,
            gauge={
                'axis': {'range': [0, 5]},
                'bar': {'color': "purple"},
                'steps': [
                    {'range': [0, 1], 'color': "lightcoral"},
                    {'range': [1, 2], 'color': "lightyellow"},
                    {'range': [2, 5], 'color': "lightgreen"}
                ]
            },
            title={'text': "Win/Loss Ratio"}
        ), row=2, col=3)
        
        gauges_fig.update_layout(
            title='Performance Metrics Dashboard',
            height=600,
            template=self.theme
        )
        
        return gauges_fig
