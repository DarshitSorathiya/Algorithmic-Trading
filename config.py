"""
Configuration module for trading strategy parameters
"""
import warnings
warnings.filterwarnings('ignore')

# Data Configuration
DATA_CONFIG = {
    'start_date': '2023-01-01',
    'end_date': '2025-12-31',
    'interval': '1d',
    'auto_adjust': True
}

# Technical Indicator Parameters
INDICATOR_PARAMS = {
    'macd': {
        'fast': 12,
        'slow': 26,
        'signal': 9
    },
    'moving_averages': {
        'sma_short': 20,
        'sma_long': 50,
        'ema_short': 20,
        'ema_long': 50
    },
    'bollinger_bands': {
        'period': 20,
        'std_dev': 2
    },
    'atr': {
        'period': 14
    },
    'rsi': {
        'period': 14
    }
}

# Signal Generation Parameters
SIGNAL_PARAMS = {
    'swing_window': 5,       # Reduced from 12 for more swing detection (was too restrictive)
    'min_distance': 3,       # Reduced from 10 to allow more frequent trades
    'lookback_period': 50,
}

# Backtesting Parameters
BACKTEST_PARAMS = {
    'initial_capital': 10000,
    'position_size': 0.1,
    'max_loss': 0.02
}

# Machine Learning Configuration
ML_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5
}

# Visualization Configuration
VIZ_CONFIG = {
    'plot_from_date': '2024-01-01',
    'theme': 'plotly_dark',
    'chart_height': 1400,
    'portfolio_height': 1000,
    'ml_height': 900
}

# Feature columns to exclude
EXCLUDED_FEATURES = [
    # Target and signal-related
    'Signal', 'Signal_Type', 'Entry_Price', 'Stop_Loss', 
    'Capital', 'Position',
    # Candlestick pattern features (optional - can cause overfitting)
    'Body', 'Upper_Shadow', 'Lower_Shadow', 'Range', 
    'Doji', 'Hammer', 'Shooting_Star', 
    'Bullish_Engulfing', 'Bearish_Engulfing',
    # **CRITICAL** - Swing detection features (lookahead bias!)
    # These use FUTURE data (bars i+1 to i+n) to identify swings at bar i
    'Is_Swing_High', 'Is_Swing_Low', 'Swing_High', 'Swing_Low',
    # Crossover and trend features (may contain lookahead bias)
    # Note: These might be okay if properly implemented with shift()
    # but excluding them makes predictions more realistic
    'MACD_Cross_Up', 'MACD_Cross_Down', 'MACD_Rising', 'MACD_Falling',
    'RSI_Cross_Up', 'RSI_Cross_Down',
    # **NEW** - Features used to CREATE signals (target leakage!)
    # These are directly used in signal_generator.py logic:
    # - RSI < 55 → Buy, RSI > 45 → Sell
    # - MACD conditions for confirmation
    # Using them for ML creates circular logic: predict what we used to create labels
    'RSI', 'MACD', 'MACD_Hist', 'Signal_Line'
]

# Model configurations
MODEL_PARAMS = {
    'logistic_regression': {
        'max_iter': 1000,
        'class_weight': 'balanced',
        'random_state': 42
    },
    'random_forest': {
        'n_estimators': 100,
        'class_weight': 'balanced',
        'random_state': 42
    },
    'gradient_boosting': {
        'n_estimators': 100,
        'random_state': 42
    },
    'svm': {
        'class_weight': 'balanced',
        'random_state': 42
    },
    'knn': {
        'n_neighbors': 5
    },
    'decision_tree': {
        'class_weight': 'balanced',
        'random_state': 42
    },
    'xgboost': {
        'use_label_encoder': False,
        'eval_metric': 'mlogloss',
        'random_state': 42
    },
    'lightgbm': {
        'class_weight': 'balanced',
        'random_state': 42,
        'verbose': -1
    }
}