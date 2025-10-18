# Trading Strategy Analysis System

A comprehensive trading strategy analysis platform with technical indicators, machine learning models, backtesting engine, and interactive Streamlit web interface.

## 🚀 Quick Start

### Option 1: Web Interface (Recommended for Beginners)

```powershell
# Double-click run_streamlit.bat or run in terminal:
.\run_streamlit.ps1
```

Then open your browser to `http://localhost:8501` and enjoy the interactive UI! 🎨

### Option 2: Command Line Interface

```powershell
.\.venv\Scripts\python.exe main.py --train-ticker GOOGL --test-ticker AMZN
```


## 🎯 Features

### 🌐 Interactive Web Interface

- **Streamlit UI**: Clean, professional web interface with real-time controls
- **Configurable Moving Averages**: Customize KAMA, SMA, and EMA periods dynamically
- **TradingView-Style Charts**: Professional candlestick charts with multiple indicators
- **Chart Controls**: Toggle 8+ indicators/overlays (Volume, RSI, MACD, Bollinger Bands, Support/Resistance, etc.)
- **Live Updates**: Real-time chart updates based on configuration changes

### 📊 Technical Analysis

- **Moving Averages**: KAMA, SMA (short/long), EMA (short/long) - all user-configurable
- **Momentum Indicators**: MACD, RSI, ATR
- **Volatility Indicators**: Bollinger Bands
- **Pattern Detection**: Doji, Hammer, Shooting Star, Engulfing patterns
- **Support/Resistance**: Automatic swing high/low detection

### 🤖 Machine Learning

- **8 ML Models**: Logistic Regression, Random Forest, Gradient Boosting, SVM, KNN, Decision Tree, XGBoost, LightGBM
- **Model Comparison**: Side-by-side performance metrics
- **Feature Importance**: Visual analysis of key trading features
- **Confusion Matrix**: Detailed prediction analysis

### 💼 Backtesting & Trading

- **Full Backtest Engine**: Position management with stop-loss
- **Portfolio Tracking**: Real-time capital curve visualization
- **Signal Generation**: Multi-factor signal system
- **Trade Analysis**: Detailed buy/sell signal tracking
- **Performance Metrics**: Win rate, profit factor, max drawdown, returns

### 📈 Visualization

- **Interactive Charts**: Plotly-based responsive dashboards
- **Multi-Panel Layout**: Price, MACD, RSI, Volume in synchronized views
- **Performance Gauges**: Visual metrics display
- **Pattern Frequency**: Candlestick pattern distribution analysis

## 📁 Project Structure

```
trading_strategy/
│
├── __init__.py                    # Package initialization
├── config.py                      # Configuration and constants
├── main.py                        # CLI execution script
├── app.py                         # Streamlit web application ⭐
├── setup.py                       # Package setup configuration
├── requirements.txt               # Python dependencies
├── run_streamlit.bat              # Windows batch launcher
├── run_streamlit.ps1              # PowerShell launcher
├── .gitignore                     # Git ignore rules
├── README.md                      # This file
│
├── .venv/                         # Virtual environment (generated)
├── __pycache__/                   # Python cache (generated)
│
├── backtesting/
│   ├── __init__.py
│   ├── backtest_engine.py         # Backtesting engine with position management
│   └── __pycache__/
│
├── data/
│   ├── __init__.py
│   ├── data_loader.py             # Yahoo Finance data fetching
│   └── __pycache__/
│
├── indicators/
│   ├── __init__.py
│   ├── technical.py               # Technical indicators (MACD, RSI, BB, KAMA, etc.)
│   ├── patterns.py                # Candlestick pattern detection
│   └── __pycache__/
│
├── ml/
│   ├── __init__.py
│   ├── models.py                  # ML model definitions (8 models)
│   ├── trainer.py                 # Model training and evaluation
│   └── __pycache__/
│
├── strategies/
│   ├── __init__.py
│   ├── signal_generator.py        # Multi-factor signal generation
│   └── __pycache__/
│
├── utils/
│   ├── __init__.py
│   ├── helpers.py                 # Utility functions
│   ├── logger.py                  # Logging configuration
│   ├── validators.py              # Input validation functions
│   └── __pycache__/
│
└── visualization/
    ├── __init__.py
    ├── interactive.py             # Plotly interactive dashboards
    ├── static.py                  # Matplotlib static plots
    ├── tradingview_charts.py      # TradingView-style chart generator
    └── __pycache__/
```

### Core Files Description

| File        | Purpose                 | Key Features                                                       |
| ----------- | ----------------------- | ------------------------------------------------------------------ |
| `app.py`    | Streamlit web interface | Clean UI, configurable MAs, TradingView charts, real-time controls |
| `main.py`   | Command-line interface  | Batch processing, automation-friendly                              |
| `config.py` | Central configuration   | All parameters, indicator settings, model configs                  |
| `setup.py`  | Package installer       | Makes project pip-installable                                      |

### Module Breakdown

| Module            | Components                                             | Responsibility                                        |
| ----------------- | ------------------------------------------------------ | ----------------------------------------------------- |
| **backtesting**   | `backtest_engine.py`                                   | Execute trades, track positions, calculate metrics    |
| **data**          | `data_loader.py`                                       | Fetch from Yahoo Finance, preprocess, validate        |
| **indicators**    | `technical.py`, `patterns.py`                          | Calculate 15+ indicators, detect candlestick patterns |
| **ml**            | `models.py`, `trainer.py`                              | 8 ML models, training pipeline, evaluation            |
| **strategies**    | `signal_generator.py`                                  | Generate buy/sell signals from indicators             |
| **utils**         | `helpers.py`, `logger.py`, `validators.py`             | Shared utilities, logging, validation                 |
| **visualization** | `interactive.py`, `static.py`, `tradingview_charts.py` | Plotly/Matplotlib charts, TradingView styling         |

## 🛠️ Installation

1. **Clone or download the repository**

2. **Create virtual environment** (recommended):

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux/Mac
python -m venv .venv
source .venv/bin/activate
```

3. **Install dependencies**:

```powershell
pip install -r requirements.txt
```

## 📦 Dependencies

- **Data**: yfinance, pandas, numpy
- **Machine Learning**: scikit-learn, xgboost, lightgbm
- **Visualization**: plotly, matplotlib, streamlit
- **Technical Analysis**: ta-lib (optional), pandas-ta

## 🎯 Usage

### Basic Usage

Run the analysis with default parameters (AAPL for training, TSLA for testing):

```bash
python main.py
```

### Custom Parameters

```bash
python main.py --train-ticker AAPL --test-ticker MSFT --start-date 2023-01-01 --end-date 2025-12-31
```

### All Available Options

```bash
python main.py --help
```

**Arguments**:

- `--train-ticker`: Training ticker symbol (default: AAPL)
- `--test-ticker`: Test ticker symbol (default: TSLA)
- `--start-date`: Start date YYYY-MM-DD (default: 2023-01-01)
- `--end-date`: End date YYYY-MM-DD (default: 2025-12-31)
- `--initial-capital`: Initial capital in USD (default: 10000)
- `--position-size`: Position size as fraction (default: 0.1)
- `--no-visualization`: Skip visualization generation
- `--save-reports`: Save reports to text files

### Streamlit Web Interface Features

**Configuration Panel:**

- Select training and testing tickers
- Configure date ranges
- Set initial capital and position size
- Customize 5 moving average periods (KAMA, SMA short/long, EMA short/long)

**Chart Controls:**

- Toggle Volume display
- Toggle RSI indicator
- Toggle MACD indicator
- Toggle Bollinger Bands
- Toggle Support/Resistance levels
- Toggle Buy/Sell signals
- Toggle Moving Averages overlay
- Toggle Candlestick Patterns

**Analysis Sections:**

- Portfolio performance visualization
- Model performance comparison
- Feature importance analysis
- Trade details and signals
- Pattern frequency distribution

### Example Commands

```powershell
# Web Interface (Recommended)
.\run_streamlit.ps1

# CLI with custom tickers
python main.py --train-ticker GOOGL --test-ticker AMZN

# Custom capital and position size
python main.py --initial-capital 50000 --position-size 0.2

# Fast mode (no visualizations)
python main.py --no-visualization

# Save reports to files
python main.py --save-reports
```

## 📊 Output

The system generates:

1. **Console Output**:

   - Data loading progress
   - Signal generation statistics
   - Backtest performance metrics
   - ML model comparison results

2. **Interactive Visualizations** (Plotly):

   - Candlestick charts with signals and indicators
   - Portfolio performance dashboard
   - Performance metrics gauges
   - Trade analysis
   - ML model comparison charts
   - Confusion matrices

3. **Static Visualizations** (Matplotlib):

   - Comprehensive 6-panel analysis
   - Price charts with signals
   - Capital curves
   - Model performance comparison
   - Feature importance

4. **Reports** (if `--save-reports` is used):
   - Performance summary reports
   - Trade details reports
   - Timestamped filenames

## ⚙️ Configuration

Edit `config.py` to customize:

- Data parameters (date ranges, intervals)
- Technical indicator parameters (MACD, RSI, etc.)
- Signal generation parameters
- Backtesting parameters
- ML model configurations
- Visualization settings

## 📈 Performance Metrics

The system calculates:

- Total Return %
- Win Rate %
- Profit Factor
- Maximum Drawdown
- Average Win/Loss
- Sharpe Ratio (coming soon)
- Sortino Ratio (coming soon)

## 🤖 Machine Learning Models

Supported models:

1. Logistic Regression
2. Random Forest
3. Gradient Boosting
4. Support Vector Machine (SVM)
5. K-Nearest Neighbors (KNN)
6. Decision Tree
7. XGBoost
8. LightGBM

Models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## 📝 Code Structure

### Modular Design

Each module has a single responsibility:

- **data**: Data loading and preprocessing
- **indicators**: Technical indicators and patterns
- **strategies**: Signal generation logic
- **backtesting**: Backtesting engine
- **ml**: Machine learning models and training
- **visualization**: Charts and plots
- **utils**: Helper functions and utilities

### Key Classes

- `DataLoader`: Fetch and prepare data
- `TechnicalIndicators`: Calculate indicators
- `PatternDetector`: Detect candlestick patterns
- `SignalGenerator`: Generate trading signals
- `BacktestEngine`: Run backtests
- `ModelFactory`: Create ML models
- `MLTrainer`: Train and evaluate models
- `InteractiveVisualizer`: Create interactive charts
- `StaticVisualizer`: Create static plots

## 🎨 UI Features

### Clean, Minimal Design

- Simple sidebar configuration
- No unnecessary text or clutter
- Professional appearance
- Responsive layout

### Dynamic Configuration

- Session state persistence for MA settings
- Real-time chart updates
- Configurable indicator periods
- Toggle controls for all chart elements

### TradingView-Style Charts

- Vertical legend layout (prevents collapsing with multiple MAs)
- Multi-panel synchronized charts
- Professional color scheme
- Interactive tooltips and zoom

## 🔧 Development

### Adding New Indicators

Add to `indicators/technical.py`:

```python
@staticmethod
def calculate_new_indicator(df: pd.DataFrame, period: int) -> pd.Series:
    # Your calculation
    return result
```

### Adding New ML Models

Add to `ml/models.py` in `ModelFactory.create_all_models()`:

```python
models["New Model"] = YourModel(params)
```

### Customizing the Streamlit UI

Edit `app.py`:

- Modify CSS in the `st.markdown()` section (lines 36-50)
- Add new chart controls in the sidebar
- Update chart display functions for new visualizations

## 🐛 Troubleshooting

**Import Errors**:

```powershell
pip install -r requirements.txt --upgrade
```

**Streamlit Port Already in Use**:

```powershell
# Kill the process using port 8501
netstat -ano | findstr :8501
taskkill /PID <PID> /F
```

**Data Download Issues**:

- Check internet connection
- Verify ticker symbols are valid (use Yahoo Finance format)
- Try different date ranges (some older data may not be available)
- Ensure yfinance is up to date: `pip install yfinance --upgrade`

**Memory Issues**:

- Use `--no-visualization` flag in CLI mode
- Reduce date range to process less data
- Close other applications to free up RAM

**Chart Display Issues**:

- Clear browser cache and refresh
- Try a different browser (Chrome/Edge recommended)
- Check console for JavaScript errors
- Ensure Plotly is installed: `pip install plotly --upgrade`

## 🎯 Tips & Best Practices

### For Best Results:

1. **Start with liquid stocks**: Use high-volume tickers (AAPL, GOOGL, MSFT, etc.)
2. **Test period length**: Use at least 1 year of data for reliable backtesting
3. **Position sizing**: Keep position size between 0.05-0.2 (5%-20% of capital)
4. **Multiple timeframes**: Test on different date ranges to validate strategy
5. **Model selection**: Compare all 8 models to find the best performer for your data

### Performance Optimization:

- Reduce MA periods for faster calculation
- Disable unused chart indicators
- Use smaller date ranges for initial testing
- Enable caching in production deployments

## � Acknowledgments

This project uses the following open-source libraries:

- **yfinance** - Financial data from Yahoo Finance
- **scikit-learn** - Machine learning models
- **XGBoost & LightGBM** - Gradient boosting models
- **Plotly** - Interactive visualizations
- **Streamlit** - Web application framework
- **pandas & numpy** - Data processing

---

**Happy Trading! 📈💰**
