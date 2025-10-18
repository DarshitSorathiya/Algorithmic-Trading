"""
Streamlit Web Application for Trading Strategy Analysis System
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import modules
from config import (
    DATA_CONFIG, INDICATOR_PARAMS, SIGNAL_PARAMS, 
    BACKTEST_PARAMS, ML_CONFIG, VIZ_CONFIG, 
    EXCLUDED_FEATURES, MODEL_PARAMS
)
from data.data_loader import DataLoader
from indicators.technical import TechnicalIndicators
from indicators.patterns import PatternDetector, SwingDetector
from strategies.signal_generator import SignalGenerator
from backtesting.backtest_engine import BacktestEngine
from ml.models import ModelFactory
from ml.trainer import MLTrainer
from utils.helpers import MetricsCalculator, DataValidator, ReportGenerator
from visualization.tradingview_charts import create_advanced_chart

# Page configuration
st.set_page_config(
    page_title="Trading Strategy Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean, minimal CSS
st.markdown("""
    <style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Clean spacing */
    .block-container {
        padding-top: 1rem;
        max-width: 1400px;
    }
    
    /* Better button styling */
    .stButton>button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'results' not in st.session_state:
    st.session_state.results = {}

def main():
    """Main application function"""
    
    # Simple header
    st.title("📊 Trading Strategy Analyzer")
    st.caption("Advanced technical analysis with machine learning")
    st.divider()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Ticker selection
        st.subheader("Stocks")
        train_ticker = st.text_input(
            "Training Ticker", 
            value="GOOGL",
            help="Stock for training"
        )
        test_ticker = st.text_input(
            "Testing Ticker", 
            value="AMZN",
            help="Stock for testing"
        )
        
        # Date range
        st.subheader("Date Range")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start",
                value=datetime(2023, 1, 1),
                help="Analysis start date"
            )
        with col2:
            end_date = st.date_input(
                "End",
                value=datetime(2025, 12, 31)
            )
        
        # Trading parameters
        st.subheader("Trading")
        initial_capital = st.number_input(
            "Initial Capital ($)",
            min_value=1000,
            max_value=1000000,
            value=10000,
            step=1000
        )
        position_size = st.slider(
            "Position Size (%)",
            min_value=5,
            max_value=100,
            value=10,
            step=5
        )
        
        # Advanced options
        with st.expander("Advanced Settings"):
            st.write("**Indicators**")
            macd_fast = st.number_input("MACD Fast", value=12, min_value=5, max_value=50)
            macd_slow = st.number_input("MACD Slow", value=26, min_value=10, max_value=100)
            rsi_period = st.number_input("RSI Period", value=14, min_value=5, max_value=50)
            bb_period = st.number_input("BB Period", value=20, min_value=10, max_value=50)
            
            st.write("**Moving Averages**")
            col_ma1, col_ma2 = st.columns(2)
            with col_ma1:
                sma_short = st.number_input("SMA Short", value=20, min_value=5, max_value=100)
                sma_long = st.number_input("SMA Long", value=50, min_value=20, max_value=200)
            with col_ma2:
                ema_short = st.number_input("EMA Short", value=20, min_value=5, max_value=100)
                ema_long = st.number_input("EMA Long", value=50, min_value=20, max_value=200)
            kama_period = st.number_input("KAMA", value=100, min_value=50, max_value=200)
            
            st.write("**Signals**")
            swing_order = st.slider("Swing Order", 1, 10, 5)
            min_swing_pct = st.slider("Min Swing %", 0.5, 5.0, 2.0, 0.5)
        
        st.divider()
        
        # Run analysis button
        run_button = st.button("Run Analysis", type="primary", use_container_width=True)
    
    # Main content area
    if run_button:
        with st.spinner("Running analysis..."):
            try:
                # Store MA configuration in session state
                st.session_state.ma_config = {
                    'sma_short': sma_short,
                    'sma_long': sma_long,
                    'ema_short': ema_short,
                    'ema_long': ema_long,
                    'kama': kama_period
                }
                
                results = run_analysis(
                    train_ticker, test_ticker,
                    start_date, end_date,
                    initial_capital, position_size
                )
                st.session_state.results = results
                st.session_state.analysis_complete = True
                st.success("✓ Analysis complete! View results below.")
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                st.exception(e)
    
    # Display results if available
    if st.session_state.analysis_complete and st.session_state.results:
        display_results(st.session_state.results)
    else:
        # Minimal welcome hint
        st.caption("Set tickers and dates in the sidebar, then click ‘Run Analysis’.")

def run_analysis(train_ticker, test_ticker, start_date, end_date, initial_capital, position_size):
    """Run the complete trading strategy analysis"""
    
    results = {
        'train_ticker': train_ticker,
        'test_ticker': test_ticker,
        'start_date': start_date,
        'end_date': end_date,
        'initial_capital': initial_capital
    }
    
    # Update config with user inputs
    BACKTEST_PARAMS['initial_capital'] = initial_capital
    BACKTEST_PARAMS['position_size'] = position_size / 100
    
    # Update MA configuration if available in session state
    if 'ma_config' in st.session_state:
        ma_config = st.session_state.ma_config
        INDICATOR_PARAMS['moving_averages'] = {
            'sma_short': ma_config['sma_short'],
            'sma_long': ma_config['sma_long'],
            'ema_short': ma_config['ema_short'],
            'ema_long': ma_config['ema_long']
        }
        INDICATOR_PARAMS['kama']['period'] = ma_config['kama']
    
    # Process training ticker
    st.write(f"### Processing {train_ticker}...")
    progress_bar = st.progress(0)
    
    df_train, trades_df_train, backtest_train = process_ticker(
        train_ticker, str(start_date), str(end_date), progress_bar, 0
    )
    
    results['df_train'] = df_train
    results['trades_train'] = trades_df_train
    results['backtest_train'] = backtest_train
    
    # Process test ticker
    st.write(f"### Processing {test_ticker}...")
    df_test, trades_df_test, backtest_test = process_ticker(
        test_ticker, str(start_date), str(end_date), progress_bar, 0.3
    )
    
    results['df_test'] = df_test
    results['trades_test'] = trades_df_test
    results['backtest_test'] = backtest_test
    
    # ML Training
    st.write("### Training Machine Learning Models...")
    ml_results = train_ml_models(df_train, df_test, progress_bar)
    results.update(ml_results)
    
    progress_bar.progress(100)
    
    return results

def process_ticker(ticker, start_date, end_date, progress_bar, base_progress):
    """Process a single ticker through the analysis pipeline"""
    
    # Load data
    loader = DataLoader(ticker, start_date, end_date)
    loader.fetch_data()
    progress_bar.progress(int(base_progress * 100) + 5)
    
    # Add indicators
    tech = TechnicalIndicators()
    df = tech.add_all_indicators(loader.data, INDICATOR_PARAMS)
    progress_bar.progress(int(base_progress * 100) + 10)
    
    # Detect patterns
    pattern_detector = PatternDetector()
    df = pattern_detector.detect_all_patterns(df)
    progress_bar.progress(int(base_progress * 100) + 15)
    
    # Find swings
    swing_detector = SwingDetector()
    df = swing_detector.find_swing_points(df, SIGNAL_PARAMS['swing_window'])
    progress_bar.progress(int(base_progress * 100) + 20)
    
    # Generate signals
    signal_gen = SignalGenerator(SIGNAL_PARAMS)
    df = signal_gen.generate_signals(df)
    progress_bar.progress(int(base_progress * 100) + 25)
    
    # Run backtest
    backtest = BacktestEngine(
        initial_capital=BACKTEST_PARAMS['initial_capital'],
        position_size=BACKTEST_PARAMS['position_size'],
        max_loss=BACKTEST_PARAMS['max_loss']
    )
    df, trades_df, metrics = backtest.run_backtest(df)
    
    # Add cumulative capital column to trades
    if len(trades_df) > 0:
        trades_df['Cumulative Capital'] = BACKTEST_PARAMS['initial_capital'] + trades_df['PnL'].cumsum()
    
    progress_bar.progress(int(base_progress * 100) + 30)
    
    return df, trades_df, metrics

def train_ml_models(df_train, df_test, progress_bar):
    """Train and evaluate ML models"""
    
    # Prepare data
    signal_rows_train = df_train[df_train['Signal'] != 0].copy()
    signal_rows_test = df_test[df_test['Signal'] != 0].copy()
    
    feature_cols = [col for col in df_train.columns if col not in EXCLUDED_FEATURES]
    
    # Clean data
    df_train_clean = signal_rows_train.dropna(subset=feature_cols + ['Signal'])
    df_test_clean = signal_rows_test.dropna(subset=feature_cols + ['Signal'])
    
    X_train = df_train_clean[feature_cols]
    y_train = df_train_clean['Signal']
    X_test = df_test_clean[feature_cols]
    y_test = df_test_clean['Signal']
    
    progress_bar.progress(70)
    
    # Train models
    trainer = MLTrainer()
    X_train_scaled, y_train_encoded, X_test_scaled, y_test_encoded = \
        trainer.prepare_data(X_train, y_train, X_test, y_test)
    
    progress_bar.progress(75)
    
    results_df, best_model_name = trainer.train_models(
        X_train_scaled, y_train_encoded,
        X_test_scaled, y_test_encoded,
        MODEL_PARAMS
    )
    
    progress_bar.progress(90)
    
    best_name, best_model, best_y_pred = trainer.get_best_model()
    
    return {
        'ml_results': results_df,
        'best_model_name': best_model_name,
        'best_model': best_model,
        'trainer': trainer,
        'y_test_encoded': y_test_encoded,
        'feature_cols': feature_cols
    }

def display_results(results):
    """Display analysis results in organized tabs"""
    
    st.markdown("## 📊 Analysis Results")
    st.markdown("---")
    
    # Create tabs
    tabs = st.tabs([
        "📈 Overview",
        "💼 Backtesting",
        "🤖 Machine Learning",
        "📉 Charts",
        "📋 Trade Details",
        "📊 Technical Analysis"
    ])
    
    # Tab 1: Overview
    with tabs[0]:
        display_overview(results)
    
    # Tab 2: Backtesting
    with tabs[1]:
        display_backtesting(results)
    
    # Tab 3: Machine Learning
    with tabs[2]:
        display_ml_results(results)
    
    # Tab 4: Charts
    with tabs[3]:
        display_charts(results)
    
    # Tab 5: Trade Details
    with tabs[4]:
        display_trade_details(results)
    
    # Tab 6: Technical Analysis
    with tabs[5]:
        display_technical_analysis(results)

def display_overview(results):
    """Display overview metrics"""
    
    st.subheader("📊 Performance Summary")
    
    # Key metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### 🔵 {results['train_ticker']} (Training)")
        metrics_train = results['backtest_train']
        
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        with metrics_col1:
            st.metric(
                "Final Capital",
                f"${metrics_train['Final Capital']:,.2f}",
                f"{metrics_train['Total Return %']:.2f}%"
            )
        with metrics_col2:
            st.metric("Win Rate", f"{metrics_train['Win Rate %']:.2f}%")
        with metrics_col3:
            st.metric("Total Trades", metrics_train['Total Trades'])
        
        # Additional metrics
        st.markdown("#### Detailed Metrics")
        col_a, col_b = st.columns(2)
        with col_a:
            st.write(f"**Winning Trades:** {metrics_train['Winning Trades']}")
            st.write(f"**Losing Trades:** {metrics_train['Losing Trades']}")
            st.write(f"**Average Win:** ${metrics_train['Average Win']:,.2f}")
        with col_b:
            st.write(f"**Average Loss:** ${metrics_train['Average Loss']:,.2f}")
            st.write(f"**Profit Factor:** {metrics_train['Profit Factor']:.2f}")
            st.write(f"**Max Drawdown:** {metrics_train['Max Drawdown %']:.2f}%")
    
    with col2:
        st.markdown(f"### 🟢 {results['test_ticker']} (Testing)")
        metrics_test = results['backtest_test']
        
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        with metrics_col1:
            st.metric(
                "Final Capital",
                f"${metrics_test['Final Capital']:,.2f}",
                f"{metrics_test['Total Return %']:.2f}%"
            )
        with metrics_col2:
            st.metric("Win Rate", f"{metrics_test['Win Rate %']:.2f}%")
        with metrics_col3:
            st.metric("Total Trades", metrics_test['Total Trades'])
        
        # Additional metrics
        st.markdown("#### Detailed Metrics")
        col_a, col_b = st.columns(2)
        with col_a:
            st.write(f"**Winning Trades:** {metrics_test['Winning Trades']}")
            st.write(f"**Losing Trades:** {metrics_test['Losing Trades']}")
            st.write(f"**Average Win:** ${metrics_test['Average Win']:,.2f}")
        with col_b:
            st.write(f"**Average Loss:** ${metrics_test['Average Loss']:,.2f}")
            st.write(f"**Profit Factor:** {metrics_test['Profit Factor']:.2f}")
            st.write(f"**Max Drawdown:** {metrics_test['Max Drawdown %']:.2f}%")
    
    st.markdown("---")
    
    # ML Performance
    st.subheader("🤖 Machine Learning Performance")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Best Model", results['best_model_name'])
    with col2:
        best_f1 = results['ml_results'].iloc[0]['F1 Score']
        st.metric("F1 Score", f"{best_f1:.4f}")
    with col3:
        best_acc = results['ml_results'].iloc[0]['Accuracy']
        st.metric("Accuracy", f"{best_acc:.4f}")
    with col4:
        st.metric("Models Trained", len(results['ml_results']))

def display_backtesting(results):
    """Display detailed backtesting results"""
    
    st.subheader("💼 Backtesting Analysis")
    
    # Training results
    with st.expander(f"🔵 {results['train_ticker']} - Training Results", expanded=True):
        metrics = results['backtest_train']
        trades_df = results['trades_train']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Initial Capital", f"${metrics['Initial Capital']:,.2f}")
        with col2:
            st.metric("Final Capital", f"${metrics['Final Capital']:,.2f}")
        with col3:
            st.metric("Total Return", f"{metrics['Total Return %']:.2f}%")
        with col4:
            st.metric("Total Trades", metrics['Total Trades'])
        
        # Equity curve
        st.markdown("#### 📈 Equity Curve")
        if len(trades_df) > 0:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=trades_df['Exit_Date'],
                y=trades_df['Cumulative Capital'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='blue', width=2)
            ))
            fig.update_layout(
                title=f"{results['train_ticker']} Portfolio Value Over Time",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                height=400,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Trade statistics
        if len(trades_df) > 0:
            st.markdown("#### 📊 Trade Statistics")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Winning Trades**")
                winning_trades = trades_df[trades_df['PnL'] > 0]
                if len(winning_trades) > 0:
                    st.write(f"Count: {len(winning_trades)}")
                    st.write(f"Avg Profit: ${winning_trades['PnL'].mean():,.2f}")
                    st.write(f"Max Profit: ${winning_trades['PnL'].max():,.2f}")
                    st.write(f"Total Profit: ${winning_trades['PnL'].sum():,.2f}")
            
            with col2:
                st.write("**Losing Trades**")
                losing_trades = trades_df[trades_df['PnL'] < 0]
                if len(losing_trades) > 0:
                    st.write(f"Count: {len(losing_trades)}")
                    st.write(f"Avg Loss: ${losing_trades['PnL'].mean():,.2f}")
                    st.write(f"Max Loss: ${losing_trades['PnL'].min():,.2f}")
                    st.write(f"Total Loss: ${losing_trades['PnL'].sum():,.2f}")
    
    # Test results
    with st.expander(f"🟢 {results['test_ticker']} - Test Results", expanded=True):
        metrics = results['backtest_test']
        trades_df = results['trades_test']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Initial Capital", f"${metrics['Initial Capital']:,.2f}")
        with col2:
            st.metric("Final Capital", f"${metrics['Final Capital']:,.2f}")
        with col3:
            st.metric("Total Return", f"{metrics['Total Return %']:.2f}%")
        with col4:
            st.metric("Total Trades", metrics['Total Trades'])
        
        # Equity curve
        st.markdown("#### 📈 Equity Curve")
        if len(trades_df) > 0:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=trades_df['Exit_Date'],
                y=trades_df['Cumulative Capital'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='green', width=2)
            ))
            fig.update_layout(
                title=f"{results['test_ticker']} Portfolio Value Over Time",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                height=400,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Trade statistics
        if len(trades_df) > 0:
            st.markdown("#### 📊 Trade Statistics")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Winning Trades**")
                winning_trades = trades_df[trades_df['PnL'] > 0]
                if len(winning_trades) > 0:
                    st.write(f"Count: {len(winning_trades)}")
                    st.write(f"Avg Profit: ${winning_trades['PnL'].mean():,.2f}")
                    st.write(f"Max Profit: ${winning_trades['PnL'].max():,.2f}")
                    st.write(f"Total Profit: ${winning_trades['PnL'].sum():,.2f}")
            
            with col2:
                st.write("**Losing Trades**")
                losing_trades = trades_df[trades_df['PnL'] < 0]
                if len(losing_trades) > 0:
                    st.write(f"Count: {len(losing_trades)}")
                    st.write(f"Avg Loss: ${losing_trades['PnL'].mean():,.2f}")
                    st.write(f"Max Loss: ${losing_trades['PnL'].min():,.2f}")
                    st.write(f"Total Loss: ${losing_trades['PnL'].sum():,.2f}")

def display_ml_results(results):
    """Display machine learning results"""
    
    st.subheader("🤖 Machine Learning Model Comparison")
    
    # Model comparison table
    st.markdown("#### 📊 Model Performance")
    st.dataframe(
        results['ml_results'].style.format({
            'Accuracy': '{:.4f}',
            'Precision': '{:.4f}',
            'Recall': '{:.4f}',
            'F1 Score': '{:.4f}'
        }).background_gradient(subset=['F1 Score'], cmap='RdYlGn'),
        width='stretch'
    )
    
    # Best model details
    st.markdown(f"#### 🏆 Best Model: {results['best_model_name']}")
    
    col1, col2, col3, col4 = st.columns(4)
    best_results = results['ml_results'].iloc[0]
    
    with col1:
        st.metric("Accuracy", f"{best_results['Accuracy']:.4f}")
    with col2:
        st.metric("Precision", f"{best_results['Precision']:.4f}")
    with col3:
        st.metric("Recall", f"{best_results['Recall']:.4f}")
    with col4:
        st.metric("F1 Score", f"{best_results['F1 Score']:.4f}")
    
    # Confusion Matrix
    st.markdown("#### 📊 Confusion Matrix")
    trainer = results['trainer']
    cm = trainer.confusion_matrices[results['best_model_name']]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Sell', 'Buy'],
        y=['Sell', 'Buy'],
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 20},
    ))
    fig.update_layout(
        title=f"Confusion Matrix - {results['best_model_name']}",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Classification Report
    st.markdown("#### 📋 Classification Report")
    report = trainer.get_classification_report(
        results['y_test_encoded'],
        results['best_model_name']
    )
    st.text(report)
    
    # Feature Importance
    st.markdown("#### 🎯 Feature Importance (Top 15)")
    importance_df = trainer.get_feature_importance(
        results['feature_cols'],
        results['best_model_name'],
        top_n=15
    )
    
    if importance_df is not None:
        fig = go.Figure(go.Bar(
            x=importance_df['Importance'],
            y=importance_df['Feature'],
            orientation='h',
            marker=dict(color=importance_df['Importance'], colorscale='Viridis')
        ))
        fig.update_layout(
            title="Top 15 Most Important Features",
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=500,
            yaxis={'categoryorder': 'total ascending'}
        )
        st.plotly_chart(fig, use_container_width=True)
    # Removed info message for models without feature importance

def display_charts(results):
    """Display interactive charts"""
    
    st.markdown("### � Interactive Charts")
    
    # Professional chart controls with better grouping
    with st.container():
        col_left, col_right = st.columns([3, 1])
        
        with col_left:
            st.markdown("**Indicators & Overlays**")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                show_bb = st.checkbox("Bollinger Bands", value=True)
            with col2:
                show_ma = st.checkbox("Moving Averages", value=True)
            with col3:
                show_swings = st.checkbox("Swing Points", value=True)
            with col4:
                show_patterns = st.checkbox("Patterns", value=True)
            with col5:
                show_signals = st.checkbox("Trade Signals", value=True)
            
            col6, col7, col8, _, _ = st.columns(5)
            with col6:
                show_volume = st.checkbox("Volume", value=True)
            with col7:
                show_macd = st.checkbox("MACD", value=True)
            with col8:
                show_rsi = st.checkbox("RSI", value=True)
        
        with col_right:
            st.markdown("**Display Options**")
            chart_period = st.selectbox("Timeframe", ["Last 100", "Last 200", "Last 500", "All"], index=1, label_visibility="collapsed")
            chart_style = st.selectbox("Theme", ["Dark", "Light"], index=0, label_visibility="collapsed")
    
    st.markdown("---")
    
    # Price chart with signals
    tab1, tab2 = st.tabs([f"Training: {results['train_ticker']}", f"Testing: {results['test_ticker']}"])
    
    # Get MA configuration from session state or use defaults
    ma_config = st.session_state.get('ma_config', {
        'sma_short': 20,
        'sma_long': 50,
        'ema_short': 20,
        'ema_long': 50,
        'kama': 100
    })
    
    with tab1:
        create_advanced_tradingview_chart(
            results['df_train'], 
            results['train_ticker'],
            show_volume, show_macd, show_rsi, 
            show_bb, show_ma, show_swings, show_patterns, show_signals,
            chart_period, chart_style, ma_config
        )
    
    with tab2:
        create_advanced_tradingview_chart(
            results['df_test'], 
            results['test_ticker'],
            show_volume, show_macd, show_rsi,
            show_bb, show_ma, show_swings, show_patterns, show_signals,
            chart_period, chart_style, ma_config
        )

def create_advanced_tradingview_chart(df, ticker, 
                                     show_volume, show_macd, show_rsi,
                                     show_bb, show_ma, show_swings, show_patterns, show_signals,
                                     period, style, ma_config=None):
    """Create advanced TradingView-style chart"""
    
    # Filter data based on period
    period_map = {
        "Last 100": 100,
        "Last 200": 200,
        "Last 500": 500,
        "All": len(df)
    }
    df_filtered = df.tail(period_map[period]).copy()
    
    # Create chart using the new TradingViewChart class
    fig = create_advanced_chart(
        df_filtered,
        title=f"{ticker} Trading Analysis",
        show_volume=show_volume,
        show_macd=show_macd,
        show_rsi=show_rsi,
        show_bollinger=show_bb,
        show_moving_avg=show_ma,
        show_swing_points=show_swings,
        show_patterns=show_patterns,
        show_trade_signals=show_signals,
        ma_config=ma_config
    )
    
    # Apply light theme if selected
    if style == "Light":
        fig.update_layout(
            template='plotly_white',
            plot_bgcolor='rgba(255, 255, 255, 1)',
            paper_bgcolor='rgba(255, 255, 255, 1)'
        )
    
    # Display chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Chart statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Data Points", len(df_filtered))
    with col2:
        buy_signals = len(df_filtered[df_filtered['Signal'] == 1])
        st.metric("Buy Signals", buy_signals)
    with col3:
        sell_signals = len(df_filtered[df_filtered['Signal'] == -1])
        st.metric("Sell Signals", sell_signals)
    with col4:
        price_change = ((df_filtered['Close'].iloc[-1] / df_filtered['Close'].iloc[0]) - 1) * 100
        st.metric("Price Change", f"{price_change:.2f}%")
    
    # Pattern statistics (if patterns are shown)
    if show_patterns:
        st.markdown("#### 🕯️ Candlestick Pattern Occurrences")
        pattern_cols = st.columns(5)
        patterns_to_check = ['Doji', 'Hammer', 'Shooting_Star', 'Bullish_Engulfing', 'Bearish_Engulfing']
        pattern_display = ['Doji', 'Hammer', 'Shooting Star', 'Bullish Engulfing', 'Bearish Engulfing']
        
        for idx, (pattern_col, pattern_display_name) in enumerate(zip(patterns_to_check, pattern_display)):
            if pattern_col in df_filtered.columns:
                count = df_filtered[pattern_col].sum()
                pattern_cols[idx].metric(pattern_display_name, int(count))
            else:
                pattern_cols[idx].metric(pattern_display_name, "N/A")

def create_candlestick_chart(df, ticker):
    """Create candlestick chart with indicators and signals (Legacy function)"""
    
    # Filter recent data for better visualization
    df_recent = df.tail(200).copy()
    
    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            f'{ticker} Price & Signals',
            'MACD',
            'RSI',
            'Volume'
        ),
        row_heights=[0.5, 0.15, 0.15, 0.2]
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df_recent.index,
            open=df_recent['Open'],
            high=df_recent['High'],
            low=df_recent['Low'],
            close=df_recent['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Buy/Sell signals
    buy_signals = df_recent[df_recent['Signal'] == 1]
    sell_signals = df_recent[df_recent['Signal'] == -1]
    
    if len(buy_signals) > 0:
        fig.add_trace(
            go.Scatter(
                x=buy_signals.index,
                y=buy_signals['Low'] * 0.98,
                mode='markers',
                name='Buy Signal',
                marker=dict(symbol='triangle-up', size=15, color='green')
            ),
            row=1, col=1
        )
    
    if len(sell_signals) > 0:
        fig.add_trace(
            go.Scatter(
                x=sell_signals.index,
                y=sell_signals['High'] * 1.02,
                mode='markers',
                name='Sell Signal',
                marker=dict(symbol='triangle-down', size=15, color='red')
            ),
            row=1, col=1
        )
    
    # Bollinger Bands
    if 'BB_Upper' in df_recent.columns:
        fig.add_trace(
            go.Scatter(
                x=df_recent.index, y=df_recent['BB_Upper'],
                mode='lines', name='BB Upper', line=dict(dash='dash', color='gray')
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df_recent.index, y=df_recent['BB_Lower'],
                mode='lines', name='BB Lower', line=dict(dash='dash', color='gray'),
                fill='tonexty', fillcolor='rgba(128,128,128,0.1)'
            ),
            row=1, col=1
        )
    
    # MACD
    if 'MACD' in df_recent.columns:
        fig.add_trace(
            go.Scatter(x=df_recent.index, y=df_recent['MACD'], name='MACD',
                      line=dict(color='blue')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=df_recent.index, y=df_recent['Signal_Line'], name='Signal',
                      line=dict(color='red')),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(x=df_recent.index, y=df_recent['MACD_Hist'], name='Histogram',
                  marker=dict(color='gray')),
            row=2, col=1
        )
    
    # RSI
    if 'RSI' in df_recent.columns:
        fig.add_trace(
            go.Scatter(x=df_recent.index, y=df_recent['RSI'], name='RSI',
                      line=dict(color='purple')),
            row=3, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    # Volume
    fig.add_trace(
        go.Bar(x=df_recent.index, y=df_recent['Volume'], name='Volume',
              marker=dict(color='lightblue')),
        row=4, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=1000,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    fig.update_yaxes(title_text="Volume", row=4, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

def display_trade_details(results):
    """Display detailed trade information"""
    
    st.subheader("📋 Trade Details")
    
    tab1, tab2 = st.tabs([
        f"🔵 {results['train_ticker']} Trades",
        f"🟢 {results['test_ticker']} Trades"
    ])
    
    with tab1:
        trades_df = results['trades_train']
        if len(trades_df) > 0:
            st.dataframe(
                trades_df.style.format({
                    'Entry_Price': '${:.2f}',
                    'Exit_Price': '${:.2f}',
                    'PnL': '${:.2f}',
                    'Return_%': '{:.2f}%',
                    'Cumulative Capital': '${:.2f}'
                }).applymap(
                    lambda x: 'color: green' if isinstance(x, (int, float)) and x > 0 else 'color: red',
                    subset=['PnL', 'Return_%']
                ),
                width='stretch'
            )
            
            # Download button
            csv = trades_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Trades CSV",
                data=csv,
                file_name=f"{results['train_ticker']}_trades.csv",
                mime="text/csv"
            )
        else:
            st.info("No trades executed.")
    
    with tab2:
        trades_df = results['trades_test']
        if len(trades_df) > 0:
            st.dataframe(
                trades_df.style.format({
                    'Entry_Price': '${:.2f}',
                    'Exit_Price': '${:.2f}',
                    'PnL': '${:.2f}',
                    'Return_%': '{:.2f}%',
                    'Cumulative Capital': '${:.2f}'
                }).applymap(
                    lambda x: 'color: green' if isinstance(x, (int, float)) and x > 0 else 'color: red',
                    subset=['PnL', 'Return_%']
                ),
                width='stretch'
            )
            
            # Download button
            csv = trades_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Trades CSV",
                data=csv,
                file_name=f"{results['test_ticker']}_trades.csv",
                mime="text/csv"
            )
        else:
            st.info("No trades executed.")

def display_technical_analysis(results):
    """Display technical analysis details"""
    
    st.subheader("📊 Technical Analysis")
    
    tab1, tab2 = st.tabs([
        f"🔵 {results['train_ticker']}",
        f"🟢 {results['test_ticker']}"
    ])
    
    with tab1:
        display_tech_details(results['df_train'], results['train_ticker'])
    
    with tab2:
        display_tech_details(results['df_test'], results['test_ticker'])

def display_tech_details(df, ticker):
    """Display technical indicator details"""
    
    # Latest values
    st.markdown("#### 📈 Latest Indicator Values")
    latest = df.iloc[-1]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Price", f"${latest['Close']:.2f}")
        if 'RSI' in df.columns:
            st.metric("RSI", f"{latest['RSI']:.2f}")
    
    with col2:
        if 'MACD' in df.columns:
            st.metric("MACD", f"{latest['MACD']:.4f}")
        if 'ATR' in df.columns:
            st.metric("ATR", f"{latest['ATR']:.2f}")
    
    with col3:
        if 'BB_Upper' in df.columns:
            st.metric("BB Upper", f"${latest['BB_Upper']:.2f}")
            st.metric("BB Lower", f"${latest['BB_Lower']:.2f}")
    
    with col4:
        if 'KAMA' in df.columns:
            st.metric("KAMA", f"${latest['KAMA']:.2f}")
        st.metric("Volume", f"{latest['Volume']:,.0f}")
    
    # Signal summary
    st.markdown("#### 🎯 Signal Summary")
    buy_signals = len(df[df['Signal'] == 1])
    sell_signals = len(df[df['Signal'] == -1])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Buy Signals", buy_signals)
    with col2:
        st.metric("Sell Signals", sell_signals)
    with col3:
        st.metric("Total Signals", buy_signals + sell_signals)
    
    # Pattern detection
    st.markdown("#### 🕯️ Candlestick Patterns Detected")
    
    # List of actual pattern column names from PatternDetector
    pattern_columns = ['Doji', 'Hammer', 'Shooting_Star', 'Bullish_Engulfing', 'Bearish_Engulfing']
    pattern_cols = [col for col in pattern_columns if col in df.columns]
    
    if pattern_cols:
        pattern_summary = {}
        for col in pattern_cols:
            count = df[col].sum()
            if count > 0:
                # Convert snake_case to Title Case
                display_name = col.replace('_', ' ')
                pattern_summary[display_name] = int(count)
        
        if pattern_summary:
            pattern_df = pd.DataFrame(
                list(pattern_summary.items()),
                columns=['Pattern', 'Count']
            ).sort_values('Count', ascending=False)
            
            fig = go.Figure(go.Bar(
                x=pattern_df['Pattern'],
                y=pattern_df['Count'],
                marker=dict(color=pattern_df['Count'], colorscale='Viridis')
            ))
            fig.update_layout(
                title="Candlestick Pattern Frequency",
                xaxis_title="Pattern",
                yaxis_title="Count",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No candlestick patterns detected in this dataset.")
    else:
        st.info("Pattern detection columns not found.")

if __name__ == "__main__":
    main()
