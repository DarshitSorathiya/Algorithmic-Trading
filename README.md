# Professional Trading Strategy Analysis System

A comprehensive, modular Python-based trading strategy analysis system featuring technical analysis, machine learning models, backtesting, and interactive visualizations.

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

📖 **For detailed Streamlit UI guide, see [STREAMLIT_GUIDE.md](STREAMLIT_GUIDE.md)**

## 🎯 Features

- **🌐 Streamlit Web UI**: Beautiful, interactive web interface (NEW!)
- **Data Loading**: Automated data fetching from Yahoo Finance
- **Technical Analysis**:
  - MACD, RSI, Bollinger Bands, ATR, KAMA
  - Candlestick pattern detection (Doji, Hammer, Shooting Star, Engulfing patterns)
  - Swing high/low detection
- **Signal Generation**: Multi-factor signal generation with configurable parameters
- **Backtesting Engine**: Full-featured backtesting with position management and stop-loss
- **Machine Learning**: 8 different ML models for signal prediction
  - Logistic Regression, Random Forest, Gradient Boosting
  - SVM, KNN, Decision Tree, XGBoost, LightGBM
- **Visualization**:
  - Interactive Plotly dashboards
  - Static Matplotlib comprehensive plots
  - Performance metrics gauges
  - Trade analysis charts
- **Professional Reports**: Automated report generation

## 📁 Project Structure

```
trading_strategy/
│
├── __init__.py
├── config.py                 # Configuration and constants
├── main.py                   # Main execution script (CLI)
├── app.py                    # Streamlit web application (NEW!)
├── requirements.txt          # Dependencies
├── run_streamlit.bat         # Quick launch script (Windows)
├── run_streamlit.ps1         # Quick launch script (PowerShell)
│
├── data/
│   ├── __init__.py
│   └── data_loader.py       # Data fetching and preprocessing
│
├── indicators/
│   ├── __init__.py
│   ├── technical.py         # Technical indicators
│   └── patterns.py          # Candlestick patterns
│
├── strategies/
│   ├── __init__.py
│   └── signal_generator.py  # Signal generation logic
│
├── backtesting/
│   ├── __init__.py
│   └── backtest_engine.py   # Backtesting engine
│
├── ml/
│   ├── __init__.py
│   ├── models.py            # ML model definitions
│   └── trainer.py           # Training and evaluation
│
├── visualization/
│   ├── __init__.py
│   ├── interactive.py       # Plotly interactive charts
│   └── static.py            # Matplotlib static charts
│
└── utils/
    ├── __init__.py
    └── helpers.py           # Utility functions
```

## 🛠️ Installation

1. **Clone the repository**:

```bash
git clone <your-repo-url>
cd trading_strategy
```

2. **Create virtual environment** (recommended):

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

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

### Example Commands

```bash
# Run with custom tickers
python main.py --train-ticker GOOGL --test-ticker AMZN

# Run with custom capital and position size
python main.py --initial-capital 50000 --position-size 0.2

# Run without visualizations (faster)
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

### Adding New Visualizations

Add methods to `visualization/interactive.py` or `visualization/static.py`

## 🐛 Troubleshooting

**Import Errors**:

```bash
pip install -r requirements.txt --upgrade
```

**Data Download Issues**:

- Check internet connection
- Verify ticker symbols
- Try different date ranges

**Memory Issues**:

- Use `--no-visualization` flag
- Reduce date range
- Process fewer tickers

## 📄 License

MIT License - Feel free to use and modify!

## 👤 Author

Darshit Sorathiya

## 🙏 Acknowledgments

- yfinance for data
- scikit-learn for ML
- Plotly for interactive charts
- Matplotlib for static plots

## 📮 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Happy Trading! 📈💰**
