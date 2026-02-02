"""
Example: Running the Quantitative Trading System
================================================

This example demonstrates how to use the trading system
for paper trading and backtesting.
"""

import logging
from quant_trading import (
    TradingSystem,
    SystemConfig,
    TradingMode,
    DataManager,
    FeatureEngine,
    AlphaEngine
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def example_paper_trading():
    """
    Example: Paper Trading Mode
    
    Runs the system in simulation mode with mock broker.
    """
    print("\n" + "="*60)
    print("PAPER TRADING EXAMPLE")
    print("="*60 + "\n")
    
    # Create configuration
    config = SystemConfig()
    config.mode = TradingMode.PAPER
    config.initial_capital = 1000000  # 10 Lakhs
    
    # Customize symbols
    config.data.symbols = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK']
    
    # Customize risk parameters
    config.risk.max_position_size_pct = 0.15  # Max 15% per position
    config.risk.max_drawdown_pct = 0.10       # Kill switch at 10% drawdown
    
    # Create trading system
    system = TradingSystem(config)
    
    # Initialize
    system.initialize()
    
    # Run for a few iterations (in real usage, this would be continuous)
    print("Running 5 iterations...")
    for i in range(5):
        system._run_iteration()
        status = system.get_status()
        print(f"\nIteration {i+1}:")
        print(f"  Equity: ₹{status['current_equity']:,.2f}")
        print(f"  Positions: {status['positions_count']}")
        print(f"  Exposure: {status['exposure_pct']:.1%}")
        print(f"  Drawdown: {status['drawdown']:.2%}")
    
    # Print final report
    print("\n" + system.monitoring.generate_report())
    
    # Shutdown
    system.shutdown()


def example_backtest():
    """
    Example: Backtesting Mode
    
    Runs the system over historical data.
    """
    print("\n" + "="*60)
    print("BACKTEST EXAMPLE")
    print("="*60 + "\n")
    
    # Create configuration
    config = SystemConfig()
    config.initial_capital = 1000000
    
    # Create trading system
    system = TradingSystem(config)
    
    # Run backtest
    results = system.run_backtest(
        start_date='2024-01-01',
        end_date='2024-12-31'
    )
    
    # Print results
    print("\nBacktest Results:")
    print("-" * 40)
    print(f"Period: {results['start_date']} to {results['end_date']}")
    print(f"Initial Capital: ₹{results['initial_capital']:,.2f}")
    print(f"Final Equity: ₹{results['final_equity']:,.2f}")
    print(f"Total Return: {results['total_return_pct']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"Win Rate: {results['win_rate']:.1%}")
    print(f"Total Trades: {results['total_trades']}")


def example_components():
    """
    Example: Using Individual Components
    
    Shows how to use data, features, and alpha independently.
    """
    print("\n" + "="*60)
    print("COMPONENTS EXAMPLE")
    print("="*60 + "\n")
    
    # 1. Data Manager
    print("1. Loading Data...")
    data_manager = DataManager(use_mock=True)
    data_manager.initialize()
    
    symbol = 'RELIANCE'
    df = data_manager.get_combined_data(symbol)
    print(f"   Loaded {len(df)} rows for {symbol}")
    print(f"   Columns: {list(df.columns)}")
    
    # 2. Feature Engineering
    print("\n2. Computing Features...")
    feature_engine = FeatureEngine()
    features = feature_engine.compute_features(df, symbol)
    print(f"   Computed {len(features.feature_names)} features")
    print(f"   Sample features: {features.feature_names[:10]}")
    
    # 3. Alpha Generation
    print("\n3. Generating Signals...")
    alpha_engine = AlphaEngine()
    
    # Get market regime
    regime = alpha_engine.get_regime(features.features)
    print(f"   Market Regime: {regime}")
    
    # Generate signals
    alpha_output = alpha_engine.generate_signals(features.features, symbol)
    print(f"   Combined Signal: {alpha_output.combined_signal:.3f}")
    print(f"   Confidence: {alpha_output.combined_confidence:.3f}")
    print(f"   Position Suggestion: {alpha_output.position_suggestion}")
    
    # Individual strategy signals
    print("\n   Individual Strategy Signals:")
    for signal in alpha_output.signals:
        print(f"     {signal.strategy_name}: {signal.signal_type.name} "
              f"(confidence: {signal.confidence:.2f})")


def example_custom_strategy():
    """
    Example: Adding a Custom Strategy
    
    Shows how to create and add a custom alpha model.
    """
    print("\n" + "="*60)
    print("CUSTOM STRATEGY EXAMPLE")
    print("="*60 + "\n")
    
    from quant_trading.alpha import AlphaModel, Signal, SignalType
    import pandas as pd
    
    # Define custom alpha model
    class VWAPReversion(AlphaModel):
        """
        Custom strategy: VWAP Mean Reversion
        
        Buy when price is significantly below VWAP
        Sell when price is significantly above VWAP
        """
        
        def __init__(self, threshold: float = 0.02):
            super().__init__("VWAPReversion")
            self.threshold = threshold
        
        def generate_signal(self, features: pd.DataFrame, symbol: str) -> Signal:
            latest = features.iloc[-1]
            
            # Get price relative to VWAP
            if 'close_to_vwap' in features.columns:
                deviation = latest['close_to_vwap']
            else:
                deviation = 0
            
            # Generate signal based on deviation
            if deviation < -self.threshold:
                signal_value = min(abs(deviation) / self.threshold, 1)
                confidence = 0.6 + 0.3 * signal_value
            elif deviation > self.threshold:
                signal_value = -min(abs(deviation) / self.threshold, 1)
                confidence = 0.6 + 0.3 * abs(signal_value)
            else:
                signal_value = 0
                confidence = 0.3
            
            return self._create_signal(
                symbol=symbol,
                signal_value=signal_value,
                confidence=confidence,
                vwap_deviation=deviation
            )
    
    # Use the custom strategy
    data_manager = DataManager(use_mock=True)
    data_manager.initialize()
    
    feature_engine = FeatureEngine()
    df = data_manager.get_combined_data('RELIANCE')
    features = feature_engine.compute_features(df, 'RELIANCE')
    
    # Test custom strategy
    custom_alpha = VWAPReversion(threshold=0.015)
    signal = custom_alpha.generate_signal(features.features, 'RELIANCE')
    
    print(f"Custom Strategy: {custom_alpha.name}")
    print(f"Signal Type: {signal.signal_type.name}")
    print(f"Confidence: {signal.confidence:.2f}")
    print(f"VWAP Deviation: {signal.metadata.get('vwap_deviation', 0):.4f}")


if __name__ == "__main__":
    # Run all examples
    example_components()
    example_custom_strategy()
    example_backtest()
    # example_paper_trading()  # Uncomment to run paper trading