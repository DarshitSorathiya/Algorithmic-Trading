"""
Quick setup and test script
"""
import sys
import subprocess

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing packages: {e}")
        return False

def test_imports():
    """Test if all modules can be imported"""
    print("\nTesting module imports...")
    
    modules = [
        "config",
        "data.data_loader",
        "indicators.technical",
        "indicators.patterns",
        "strategies.signal_generator",
        "backtesting.backtest_engine",
        "ml.models",
        "ml.trainer",
        "visualization.interactive",
        "visualization.static",
        "utils.helpers"
    ]
    
    all_ok = True
    for module in modules:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError as e:
            print(f"  ✗ {module}: {e}")
            all_ok = False
    
    return all_ok

def main():
    """Main setup function"""
    print("=" * 70)
    print("TRADING STRATEGY ANALYSIS SYSTEM - SETUP")
    print("=" * 70)
    
    # Install requirements
    if not install_requirements():
        print("\n✗ Setup failed!")
        sys.exit(1)
    
    # Test imports
    if not test_imports():
        print("\n⚠ Some modules failed to import. Please check the errors above.")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("✓ SETUP COMPLETE!")
    print("=" * 70)
    print("\nYou can now run the analysis:")
    print("  python main.py")
    print("\nFor help:")
    print("  python main.py --help")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
