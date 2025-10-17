"""
Advanced TradingView-style Interactive Charts
Provides rich, interactive charting with toggleable indicators
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class TradingViewChart:
    """Create TradingView-style interactive charts"""
    
    def __init__(self, df: pd.DataFrame, title: str = "Trading Chart", ma_config: dict = None):
        """
        Initialize TradingView-style chart
        
        Args:
            df: DataFrame with OHLCV and indicator data
            title: Chart title
            ma_config: Dict with MA periods {'sma_short': 20, 'sma_long': 50, 'ema_short': 20, 'ema_long': 50, 'kama': 100}
        """
        self.df = df.copy()
        self.title = title
        self.fig = None
        self.show_patterns = True  # Default to showing patterns
        self.show_bollinger = True
        self.show_moving_avg = True
        self.show_swing_points = True
        self.show_trade_signals = True
        
        # MA configuration with defaults
        self.ma_config = ma_config or {
            'sma_short': 20,
            'sma_long': 50,
            'ema_short': 20,
            'ema_long': 50,
            'kama': 100
        }
    
    def create_chart(self, 
                    show_volume: bool = True,
                    show_macd: bool = True,
                    show_rsi: bool = True,
                    indicators: Dict = None) -> go.Figure:
        """
        Create comprehensive interactive chart
        
        Args:
            show_volume: Show volume subplot
            show_macd: Show MACD subplot
            show_rsi: Show RSI subplot
            indicators: Dict of indicator settings
            
        Returns:
            Plotly figure object
        """
        # Determine subplot configuration
        rows = 1
        row_heights = [0.6]  # Main price chart
        subplot_titles = [self.title]
        
        if show_macd:
            rows += 1
            row_heights.append(0.15)
            subplot_titles.append("MACD")
        
        if show_rsi:
            rows += 1
            row_heights.append(0.15)
            subplot_titles.append("RSI")
        
        if show_volume:
            rows += 1
            row_heights.append(0.10)
            subplot_titles.append("Volume")
        
        # Create subplots
        self.fig = make_subplots(
            rows=rows, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=row_heights,
            subplot_titles=subplot_titles
        )
        
        current_row = 1
        
        # 1. Main Price Chart
        self._add_candlestick(current_row)
        self._add_bollinger_bands(current_row)
        self._add_moving_averages(current_row)
        self._add_swing_points(current_row)
        self._add_candlestick_patterns(current_row)  # Add pattern markers
        self._add_trade_signals(current_row)
        current_row += 1
        
        # 2. MACD
        if show_macd:
            self._add_macd(current_row)
            current_row += 1
        
        # 3. RSI
        if show_rsi:
            self._add_rsi(current_row)
            current_row += 1
        
        # 4. Volume
        if show_volume:
            self._add_volume(current_row)
            current_row += 1
        
        # Update layout for TradingView-style appearance
        self._update_layout()
        
        return self.fig
    
    def _add_candlestick(self, row: int):
        """Add candlestick chart"""
        self.fig.add_trace(
            go.Candlestick(
                x=self.df.index,
                open=self.df['Open'],
                high=self.df['High'],
                low=self.df['Low'],
                close=self.df['Close'],
                name='Price',
                increasing=dict(line=dict(color='#26a69a'), fillcolor='#26a69a'),
                decreasing=dict(line=dict(color='#ef5350'), fillcolor='#ef5350'),
                showlegend=True
            ),
            row=row, col=1
        )
    
    def _add_bollinger_bands(self, row: int):
        """Add Bollinger Bands"""
        if not self.show_bollinger:
            return
            
        if all(col in self.df.columns for col in ['BB_Upper', 'BB_Middle', 'BB_Lower']):
            # Upper band
            self.fig.add_trace(
                go.Scatter(
                    x=self.df.index,
                    y=self.df['BB_Upper'],
                    name='BB Upper',
                    line=dict(color='rgba(173, 204, 255, 0.5)', width=1, dash='dash'),
                    showlegend=True,
                    legendgroup='bb'
                ),
                row=row, col=1
            )
            
            # Middle band
            self.fig.add_trace(
                go.Scatter(
                    x=self.df.index,
                    y=self.df['BB_Middle'],
                    name='BB Middle',
                    line=dict(color='rgba(173, 204, 255, 0.8)', width=1),
                    showlegend=True,
                    legendgroup='bb'
                ),
                row=row, col=1
            )
            
            # Lower band
            self.fig.add_trace(
                go.Scatter(
                    x=self.df.index,
                    y=self.df['BB_Lower'],
                    name='BB Lower',
                    line=dict(color='rgba(173, 204, 255, 0.5)', width=1, dash='dash'),
                    fill='tonexty',
                    fillcolor='rgba(173, 204, 255, 0.1)',
                    showlegend=True,
                    legendgroup='bb'
                ),
                row=row, col=1
            )
    
    def _add_moving_averages(self, row: int):
        """Add moving averages if available"""
        if not self.show_moving_avg:
            return
        
        # Dynamic MA configuration based on settings
        ma_columns = {}
        
        # KAMA
        kama_col = f'KAMA_{self.ma_config["kama"]}'
        if kama_col in self.df.columns:
            ma_columns[kama_col] = dict(
                color='rgba(255, 152, 0, 0.8)', 
                width=2, 
                name=f'KAMA {self.ma_config["kama"]}',
                legendgroup='ma'
            )
        
        # SMA Long
        sma_long_col = f'SMA_{self.ma_config["sma_long"]}'
        if sma_long_col in self.df.columns:
            ma_columns[sma_long_col] = dict(
                color='rgba(33, 150, 243, 0.8)', 
                width=2, 
                name=f'SMA {self.ma_config["sma_long"]}',
                legendgroup='ma'
            )
        
        # SMA Short
        sma_short_col = f'SMA_{self.ma_config["sma_short"]}'
        if sma_short_col in self.df.columns:
            ma_columns[sma_short_col] = dict(
                color='rgba(76, 175, 80, 0.8)', 
                width=1.5, 
                name=f'SMA {self.ma_config["sma_short"]}',
                legendgroup='ma'
            )
        
        # EMA Long
        ema_long_col = f'EMA_{self.ma_config["ema_long"]}'
        if ema_long_col in self.df.columns:
            ma_columns[ema_long_col] = dict(
                color='rgba(233, 30, 99, 0.8)', 
                width=2, 
                name=f'EMA {self.ma_config["ema_long"]}',
                legendgroup='ma'
            )
        
        # EMA Short
        ema_short_col = f'EMA_{self.ma_config["ema_short"]}'
        if ema_short_col in self.df.columns:
            ma_columns[ema_short_col] = dict(
                color='rgba(156, 39, 176, 0.8)', 
                width=1.5, 
                name=f'EMA {self.ma_config["ema_short"]}',
                legendgroup='ma'
            )
        
        # Add all configured MAs to chart
        for col, style in ma_columns.items():
            self.fig.add_trace(
                go.Scatter(
                    x=self.df.index,
                    y=self.df[col],
                    name=style['name'],
                    line=dict(color=style['color'], width=style['width']),
                    showlegend=True,
                    legendgroup=style.get('legendgroup', '')
                ),
                row=row, col=1
            )
    
    def _add_swing_points(self, row: int):
        """Add swing high/low markers"""
        if not self.show_swing_points:
            return
            
        if 'Is_Swing_High' in self.df.columns:
            swing_highs = self.df[self.df['Is_Swing_High'] == 1]
            if len(swing_highs) > 0:
                self.fig.add_trace(
                    go.Scatter(
                        x=swing_highs.index,
                        y=swing_highs['High'],
                        mode='markers',
                        name='Swing High',
                        marker=dict(
                            symbol='triangle-down',
                            size=12,
                            color='rgba(239, 83, 80, 0.8)',
                            line=dict(width=1, color='white')
                        ),
                        showlegend=True
                    ),
                    row=row, col=1
                )
        
        if 'Is_Swing_Low' in self.df.columns:
            swing_lows = self.df[self.df['Is_Swing_Low'] == 1]
            if len(swing_lows) > 0:
                self.fig.add_trace(
                    go.Scatter(
                        x=swing_lows.index,
                        y=swing_lows['Low'],
                        mode='markers',
                        name='Swing Low',
                        marker=dict(
                            symbol='triangle-up',
                            size=12,
                            color='rgba(38, 166, 154, 0.8)',
                            line=dict(width=1, color='white')
                        ),
                        showlegend=True
                    ),
                    row=row, col=1
                )
    
    def _add_trade_signals(self, row: int):
        """Add buy/sell signal markers"""
        if not self.show_trade_signals:
            return
            
        if 'Signal' not in self.df.columns:
            return
        
        # Buy signals
        buy_signals = self.df[self.df['Signal'] == 1]
        if len(buy_signals) > 0:
            self.fig.add_trace(
                go.Scatter(
                    x=buy_signals.index,
                    y=buy_signals['Low'] * 0.995,  # Slightly below low
                    mode='markers+text',
                    name='BUY Signal',
                    marker=dict(
                        symbol='triangle-up',
                        size=15,
                        color='#00ff00',
                        line=dict(width=2, color='darkgreen')
                    ),
                    text='BUY',
                    textposition='bottom center',
                    textfont=dict(size=10, color='green', family='Arial Black'),
                    showlegend=True
                ),
                row=row, col=1
            )
        
        # Sell signals
        sell_signals = self.df[self.df['Signal'] == -1]
        if len(sell_signals) > 0:
            self.fig.add_trace(
                go.Scatter(
                    x=sell_signals.index,
                    y=sell_signals['High'] * 1.005,  # Slightly above high
                    mode='markers+text',
                    name='SELL Signal',
                    marker=dict(
                        symbol='triangle-down',
                        size=15,
                        color='#ff0000',
                        line=dict(width=2, color='darkred')
                    ),
                    text='SELL',
                    textposition='top center',
                    textfont=dict(size=10, color='red', family='Arial Black'),
                    showlegend=True
                ),
                row=row, col=1
            )
    
    def _add_candlestick_patterns(self, row: int):
        """Add candlestick pattern markers"""
        # Skip if patterns are disabled
        if not self.show_patterns:
            return
        
        pattern_configs = {
            'Doji': {
                'symbol': 'star',
                'color': '#FFC107',
                'name': 'Doji',
                'text': 'D',
                'size': 10
            },
            'Hammer': {
                'symbol': 'diamond',
                'color': '#4CAF50',
                'name': 'Hammer',
                'text': 'H',
                'size': 10
            },
            'Shooting_Star': {
                'symbol': 'diamond',
                'color': '#F44336',
                'name': 'Shooting Star',
                'text': 'S',
                'size': 10
            },
            'Bullish_Engulfing': {
                'symbol': 'circle',
                'color': '#00E676',
                'name': 'Bullish Engulfing',
                'text': 'BE',
                'size': 12
            },
            'Bearish_Engulfing': {
                'symbol': 'circle',
                'color': '#FF1744',
                'name': 'Bearish Engulfing',
                'text': 'SE',
                'size': 12
            }
        }
        
        for pattern_name, config in pattern_configs.items():
            if pattern_name not in self.df.columns:
                continue
            
            # Find where pattern occurs
            pattern_data = self.df[self.df[pattern_name] == True]
            
            if len(pattern_data) == 0:
                continue
            
            # Position markers above the high for bearish patterns, below low for bullish
            if 'Bearish' in pattern_name or pattern_name == 'Shooting_Star':
                y_position = pattern_data['High'] * 1.003
                text_position = 'top center'
            else:
                y_position = pattern_data['Low'] * 0.997
                text_position = 'bottom center'
            
            self.fig.add_trace(
                go.Scatter(
                    x=pattern_data.index,
                    y=y_position,
                    mode='markers+text',
                    name=config['name'],
                    marker=dict(
                        symbol=config['symbol'],
                        size=config['size'],
                        color=config['color'],
                        line=dict(width=1, color='white')
                    ),
                    text=config['text'],
                    textposition=text_position,
                    textfont=dict(size=8, color=config['color'], family='Arial Bold'),
                    showlegend=True,
                    hovertemplate=f"<b>{config['name']}</b><br>" +
                                 "Date: %{x}<br>" +
                                 "Price: %{y:.2f}<br>" +
                                 "<extra></extra>"
                ),
                row=row, col=1
            )
    
    def _add_macd(self, row: int):
        """Add MACD indicator"""
        if not all(col in self.df.columns for col in ['MACD', 'Signal_Line', 'MACD_Hist']):
            return
        
        # MACD line
        self.fig.add_trace(
            go.Scatter(
                x=self.df.index,
                y=self.df['MACD'],
                name='MACD',
                line=dict(color='#2196F3', width=2),
                showlegend=True
            ),
            row=row, col=1
        )
        
        # Signal line
        self.fig.add_trace(
            go.Scatter(
                x=self.df.index,
                y=self.df['Signal_Line'],
                name='Signal',
                line=dict(color='#FF6D00', width=2),
                showlegend=True
            ),
            row=row, col=1
        )
        
        # Histogram
        colors = ['#26a69a' if val >= 0 else '#ef5350' 
                 for val in self.df['MACD_Hist']]
        
        self.fig.add_trace(
            go.Bar(
                x=self.df.index,
                y=self.df['MACD_Hist'],
                name='Histogram',
                marker=dict(color=colors),
                showlegend=True
            ),
            row=row, col=1
        )
        
        # Zero line
        self.fig.add_hline(y=0, line=dict(color='gray', width=1, dash='dash'), 
                          row=row, col=1)
    
    def _add_rsi(self, row: int):
        """Add RSI indicator"""
        if 'RSI' not in self.df.columns:
            return
        
        # RSI line
        self.fig.add_trace(
            go.Scatter(
                x=self.df.index,
                y=self.df['RSI'],
                name='RSI',
                line=dict(color='#9C27B0', width=2),
                fill='tozeroy',
                fillcolor='rgba(156, 39, 176, 0.1)',
                showlegend=True
            ),
            row=row, col=1
        )
        
        # Overbought/Oversold lines
        self.fig.add_hline(y=70, line=dict(color='red', width=1, dash='dash'), 
                          row=row, col=1, annotation_text="Overbought")
        self.fig.add_hline(y=30, line=dict(color='green', width=1, dash='dash'), 
                          row=row, col=1, annotation_text="Oversold")
        self.fig.add_hline(y=50, line=dict(color='gray', width=1, dash='dot'), 
                          row=row, col=1)
    
    def _add_volume(self, row: int):
        """Add volume bars"""
        if 'Volume' not in self.df.columns:
            return
        
        # Color volume bars based on price movement
        colors = ['#26a69a' if self.df['Close'].iloc[i] >= self.df['Open'].iloc[i] 
                 else '#ef5350' 
                 for i in range(len(self.df))]
        
        self.fig.add_trace(
            go.Bar(
                x=self.df.index,
                y=self.df['Volume'],
                name='Volume',
                marker=dict(color=colors),
                showlegend=True
            ),
            row=row, col=1
        )
    
    def _update_layout(self):
        """Update layout for TradingView-style appearance"""
        self.fig.update_layout(
            # Dark theme (TradingView style)
            template='plotly_dark',
            
            # Size
            height=900,
            
            # Remove range slider (we'll use zoom/pan instead)
            xaxis=dict(
                rangeslider=dict(visible=False),
                type='date',
                showgrid=True,
                gridcolor='rgba(128, 128, 128, 0.2)'
            ),
            
            # Hover mode
            hovermode='x unified',
            
            # Improved Legend - Prevent collapsing with vertical layout
            legend=dict(
                orientation='v',  # Vertical to prevent collapsing
                yanchor='top',
                y=0.99,
                xanchor='left',
                x=1.01,  # Position outside plot area
                bgcolor='rgba(17, 17, 17, 0.9)',
                bordercolor='rgba(128, 128, 128, 0.3)',
                borderwidth=1,
                font=dict(size=10),
                itemsizing='constant',
                traceorder='grouped',  # Group by legendgroup
                tracegroupgap=5
            ),
            
            # Margins - Add space on right for legend
            margin=dict(l=50, r=150, t=50, b=50),
            
            # Interaction
            dragmode='pan',  # Pan by default (zoom with scroll)
            
            # Grid
            plot_bgcolor='rgba(17, 17, 17, 1)',
            paper_bgcolor='rgba(17, 17, 17, 1)'
        )
        
        # Update all y-axes
        self.fig.update_yaxes(
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)',
            showline=True,
            linecolor='rgba(128, 128, 128, 0.5)'
        )
        
        # Update all x-axes
        self.fig.update_xaxes(
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)',
            showline=True,
            linecolor='rgba(128, 128, 128, 0.5)',
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label='1M', step='month', stepmode='backward'),
                    dict(count=3, label='3M', step='month', stepmode='backward'),
                    dict(count=6, label='6M', step='month', stepmode='backward'),
                    dict(count=1, label='YTD', step='year', stepmode='todate'),
                    dict(count=1, label='1Y', step='year', stepmode='backward'),
                    dict(step='all', label='ALL')
                ]),
                bgcolor='rgba(128, 128, 128, 0.2)',
                activecolor='rgba(38, 166, 154, 0.7)'
            ),
            rangeslider=dict(visible=False)
        )
    
    def add_pattern_annotations(self, patterns: pd.DataFrame):
        """
        Add pattern annotations to chart
        
        Args:
            patterns: DataFrame with pattern detections
        """
        # Add annotations for detected patterns
        for pattern in ['Doji', 'Hammer', 'Shooting_Star', 
                       'Bullish_Engulfing', 'Bearish_Engulfing']:
            if pattern in patterns.columns:
                pattern_dates = patterns[patterns[pattern] == 1].index
                for date in pattern_dates:
                    self.fig.add_annotation(
                        x=date,
                        y=patterns.loc[date, 'High'] * 1.01,
                        text=pattern.replace('_', ' '),
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor='yellow',
                        bgcolor='rgba(255, 255, 0, 0.2)',
                        bordercolor='yellow',
                        font=dict(size=10, color='yellow'),
                        row=1, col=1
                    )


def create_advanced_chart(df: pd.DataFrame, 
                         title: str = "Trading Analysis",
                         show_volume: bool = True,
                         show_macd: bool = True,
                         show_rsi: bool = True,
                         show_bollinger: bool = True,
                         show_moving_avg: bool = True,
                         show_swing_points: bool = True,
                         show_patterns: bool = True,
                         show_trade_signals: bool = True,
                         ma_config: dict = None) -> go.Figure:
    """
    Convenience function to create advanced trading chart
    
    Args:
        df: DataFrame with OHLCV and indicator data
        title: Chart title
        show_volume: Show volume subplot
        show_macd: Show MACD subplot
        show_rsi: Show RSI subplot
        show_bollinger: Show Bollinger Bands on main chart
        show_moving_avg: Show moving averages on main chart
        show_swing_points: Show swing high/low markers
        show_patterns: Show candlestick pattern markers
        show_trade_signals: Show buy/sell signal arrows
        ma_config: Dict with MA periods {'sma_short': 20, 'sma_long': 50, 'ema_short': 20, 'ema_long': 50, 'kama': 100}
        
    Returns:
        Plotly figure object
    """
    chart = TradingViewChart(df, title, ma_config)
    # Store all preferences
    chart.show_patterns = show_patterns
    chart.show_bollinger = show_bollinger
    chart.show_moving_avg = show_moving_avg
    chart.show_swing_points = show_swing_points
    chart.show_trade_signals = show_trade_signals
    
    return chart.create_chart(
        show_volume=show_volume,
        show_macd=show_macd,
        show_rsi=show_rsi
    )
