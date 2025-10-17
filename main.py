"""
Main execution script for trading strategy analysis
"""
import argparse
import sys
import numpy as np
from datetime import datetime

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
from visualization.interactive import InteractiveVisualizer
from visualization.static import StaticVisualizer
from utils.helpers import ProgressPrinter, ReportGenerator


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Professional Trading Strategy Analysis System'
    )
    
    parser.add_argument('--train-ticker', type=str, default='AAPL',
                       help='Training ticker symbol (default: AAPL)')
    parser.add_argument('--test-ticker', type=str, default='TSLA',
                       help='Test ticker symbol (default: TSLA)')
    parser.add_argument('--start-date', type=str, default=DATA_CONFIG['start_date'],
                       help=f'Start date (default: {DATA_CONFIG["start_date"]})')
    parser.add_argument('--end-date', type=str, default=DATA_CONFIG['end_date'],
                       help=f'End date (default: {DATA_CONFIG["end_date"]})')
    parser.add_argument('--initial-capital', type=float, 
                       default=BACKTEST_PARAMS['initial_capital'],
                       help=f'Initial capital (default: {BACKTEST_PARAMS["initial_capital"]})')
    parser.add_argument('--position-size', type=float, 
                       default=BACKTEST_PARAMS['position_size'],
                       help=f'Position size (default: {BACKTEST_PARAMS["position_size"]})')
    parser.add_argument('--no-visualization', action='store_true',
                       help='Skip visualization generation')
    parser.add_argument('--save-reports', action='store_true',
                       help='Save reports to files')
    
    return parser.parse_args()


def process_ticker(ticker: str, start_date: str, end_date: str,
                  initial_capital: float, position_size: float,
                  visualize: bool = True) -> tuple:
    """
    Process a single ticker: load data, generate signals, backtest
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date
        end_date: End date
        initial_capital: Initial capital
        position_size: Position size
        visualize: Whether to generate visualizations
        
    Returns:
        Tuple of (df, trades_df, metrics)
    """
    ProgressPrinter.print_header(f"PROCESSING {ticker}")
    
    # Load data
    ProgressPrinter.print_step(f"Loading {ticker} data")
    loader = DataLoader(ticker, start_date, end_date)
    df = loader.fetch_data()
    ProgressPrinter.print_success(f"Loaded {len(df)} rows")
    
    # Add technical indicators
    ProgressPrinter.print_step("Calculating technical indicators")
    df = TechnicalIndicators.add_all_indicators(df, INDICATOR_PARAMS)
    ProgressPrinter.print_success("Technical indicators added")
    
    # Detect patterns
    ProgressPrinter.print_step("Detecting candlestick patterns")
    df = PatternDetector.detect_all_patterns(df)
    ProgressPrinter.print_success("Patterns detected")
    
    # Find swing points
    ProgressPrinter.print_step("Finding swing highs and lows")
    df = SwingDetector.find_swing_points(df, SIGNAL_PARAMS['swing_window'])
    ProgressPrinter.print_success("Swing points identified")
    
    # Generate signals
    ProgressPrinter.print_step("Generating trading signals")
    signal_gen = SignalGenerator(SIGNAL_PARAMS)
    df = signal_gen.generate_signals(df)
    signal_count = len(df[df['Signal'] != 0])
    ProgressPrinter.print_success(f"Generated {signal_count} signals")
    
    # Backtest
    ProgressPrinter.print_step("Running backtest")
    engine = BacktestEngine(initial_capital, position_size, 
                           BACKTEST_PARAMS['max_loss'])
    df, trades_df, metrics = engine.run_backtest(df)
    ProgressPrinter.print_success(f"Backtest complete - {len(trades_df)} trades executed")
    
    # Print metrics
    print("\n" + ReportGenerator.generate_summary_report(metrics, ticker))
    
    return df, trades_df, metrics, loader


def main():
    """Main execution function"""
    args = parse_arguments()
    
    ProgressPrinter.print_header("PROFESSIONAL TRADING STRATEGY ANALYSIS SYSTEM")
    print(f"\nStart Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Training Ticker: {args.train_ticker}")
    print(f"Test Ticker: {args.test_ticker}")
    print(f"Date Range: {args.start_date} to {args.end_date}")
    print(f"Initial Capital: ${args.initial_capital:,.2f}")
    print(f"Position Size: {args.position_size * 100}%")
    
    try:
        # Process training ticker
        df_train, trades_df_train, metrics_train, loader_train = process_ticker(
            args.train_ticker, args.start_date, args.end_date,
            args.initial_capital, args.position_size, 
            not args.no_visualization
        )
        
        # Process test ticker
        df_test, trades_df_test, metrics_test, loader_test = process_ticker(
            args.test_ticker, args.start_date, args.end_date,
            args.initial_capital, args.position_size,
            not args.no_visualization
        )
        
        # Prepare ML data
        ProgressPrinter.print_header("MACHINE LEARNING TRAINING & EVALUATION")
        
        ProgressPrinter.print_step("Preparing training data")
        # Get feature columns (exclude signal-related and temporary columns)
        feature_cols = [col for col in df_train.columns if col not in EXCLUDED_FEATURES]
        
        # Clean and prepare training data
        df_train_clean = df_train.dropna().copy()
        X_train = df_train_clean[feature_cols]
        y_train = df_train_clean['Signal']
        
        print(f"\nTraining Dataset shape: {X_train.shape}")
        print(f"Signal distribution:\n{y_train.value_counts()}")
        
        ProgressPrinter.print_step("Preparing test data")
        # Clean and prepare test data
        df_test_clean = df_test.dropna().copy()
        X_test = df_test_clean[feature_cols]
        y_test = df_test_clean['Signal']
        
        print(f"\nTest Dataset shape: {X_test.shape}")
        print(f"Signal distribution:\n{y_test.value_counts()}")
        
        # Initialize ML trainer
        ProgressPrinter.print_step("Initializing ML models")
        trainer = MLTrainer()
        
        # Prepare data
        X_train_scaled, y_train_encoded, X_test_scaled, y_test_encoded = \
            trainer.prepare_data(X_train, y_train, X_test, y_test)
        
        # Train and evaluate
        ProgressPrinter.print_step("Training and evaluating models")
        results_df, best_model_name = trainer.train_models(
            X_train_scaled, y_train_encoded, 
            X_test_scaled, y_test_encoded,
            MODEL_PARAMS
        )
        
        ProgressPrinter.print_success(f"Best Model: {best_model_name}")
        
        # Get best model and predictions
        _, best_model, best_y_pred = trainer.get_best_model()
        
        # Generate classification report
        print("\n" + "="*70)
        print("CLASSIFICATION REPORT")
        print("="*70)
        print(trainer.get_classification_report(y_test_encoded, best_model_name))
        
        # Generate visualizations
        if not args.no_visualization:
            ProgressPrinter.print_header("GENERATING VISUALIZATIONS")
            
            # Interactive visualizations
            ProgressPrinter.print_step("Creating interactive charts")
            interactive_viz = InteractiveVisualizer(VIZ_CONFIG['theme'])
            
            # Training ticker charts
            fig1 = interactive_viz.plot_candlestick(
                df_train, args.train_ticker, VIZ_CONFIG['plot_from_date'],
                trades_df_train, VIZ_CONFIG['chart_height']
            )
            fig1.show()
            
            fig2 = interactive_viz.plot_portfolio_analysis(
                df_train, metrics_train, args.train_ticker,
                VIZ_CONFIG['plot_from_date'], VIZ_CONFIG['portfolio_height']
            )
            fig2.show()
            
            fig3 = interactive_viz.plot_performance_metrics(metrics_train)
            fig3.show()
            
            if len(trades_df_train) > 0:
                fig4 = interactive_viz.plot_trade_analysis(trades_df_train, args.train_ticker)
                if fig4:
                    fig4.show()
            
            # Test ticker charts
            fig5 = interactive_viz.plot_candlestick(
                df_test, args.test_ticker, VIZ_CONFIG['plot_from_date'],
                trades_df_test, VIZ_CONFIG['chart_height']
            )
            fig5.show()
            
            fig6 = interactive_viz.plot_portfolio_analysis(
                df_test, metrics_test, args.test_ticker,
                VIZ_CONFIG['plot_from_date'], VIZ_CONFIG['portfolio_height']
            )
            fig6.show()
            
            fig7 = interactive_viz.plot_performance_metrics(metrics_test)
            fig7.show()
            
            if len(trades_df_test) > 0:
                fig8 = interactive_viz.plot_trade_analysis(trades_df_test, args.test_ticker)
                if fig8:
                    fig8.show()
            
            # ML comparison charts
            fig9 = interactive_viz.plot_ml_comparison(
                results_df, trainer.confusion_matrices, args.test_ticker,
                VIZ_CONFIG['ml_height']
            )
            fig9.show()
            
            # Confusion matrix for best model
            class_names_map = {0: 'Sell', 1: 'Hold', 2: 'Buy'}
            unique_classes_test = np.unique(y_test_encoded)
            present_class_names = [class_names_map[c] for c in unique_classes_test]
            
            fig10 = interactive_viz.plot_confusion_matrix(
                trainer.confusion_matrices[best_model_name],
                present_class_names,
                best_model_name
            )
            fig10.show()
            
            ProgressPrinter.print_success("Interactive charts displayed")
            
            # Static visualizations
            ProgressPrinter.print_step("Creating static comprehensive plot")
            static_viz = StaticVisualizer()
            
            fig_static = static_viz.plot_comprehensive_analysis(
                df_train, df_test, results_df,
                args.train_ticker, args.test_ticker,
                feature_cols, best_model, best_model_name,
                VIZ_CONFIG['plot_from_date']
            )
            static_viz.show()
            
            ProgressPrinter.print_success("Static plot displayed")
        
        # Save reports
        if args.save_reports:
            ProgressPrinter.print_header("SAVING REPORTS")
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Training report
            train_report = ReportGenerator.generate_summary_report(
                metrics_train, args.train_ticker
            )
            train_filename = f"report_{args.train_ticker}_{timestamp}.txt"
            ReportGenerator.save_report_to_file(train_report, train_filename)
            
            # Test report
            test_report = ReportGenerator.generate_summary_report(
                metrics_test, args.test_ticker
            )
            test_filename = f"report_{args.test_ticker}_{timestamp}.txt"
            ReportGenerator.save_report_to_file(test_report, test_filename)
            
            # Trade reports
            if len(trades_df_train) > 0:
                train_trade_report = ReportGenerator.generate_trade_report(
                    trades_df_train, args.train_ticker
                )
                train_trade_filename = f"trades_{args.train_ticker}_{timestamp}.txt"
                ReportGenerator.save_report_to_file(train_trade_report, train_trade_filename)
            
            if len(trades_df_test) > 0:
                test_trade_report = ReportGenerator.generate_trade_report(
                    trades_df_test, args.test_ticker
                )
                test_trade_filename = f"trades_{args.test_ticker}_{timestamp}.txt"
                ReportGenerator.save_report_to_file(test_trade_report, test_trade_filename)
            
            ProgressPrinter.print_success("Reports saved")
        
        # Final summary
        ProgressPrinter.print_header("ANALYSIS COMPLETE")
        print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\n{args.train_ticker} Final Capital: ${metrics_train['Final Capital']:,.2f} "
              f"({metrics_train['Total Return %']:+.2f}%)")
        print(f"{args.test_ticker} Final Capital: ${metrics_test['Final Capital']:,.2f} "
              f"({metrics_test['Total Return %']:+.2f}%)")
        print(f"\nBest ML Model: {best_model_name} "
              f"(F1 Score: {results_df[results_df['Model'] == best_model_name]['F1 Score'].values[0]:.3f})")
        print("\n✓ All visualizations and analysis completed successfully!")
        
    except Exception as e:
        ProgressPrinter.print_error(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
