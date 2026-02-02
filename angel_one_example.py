"""
Angel One Trading Example
=========================

Example showing how to use the trading system with Angel One broker.
"""

import logging
from quant_trading import TradingSystem, SystemConfig, TradingMode
from quant_trading.execution import AngelOneAPI
from quant_trading.angel_one_config import ANGEL_ONE_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_angel_one_connection():
    """Test Angel One API connection."""
    print("\n" + "="*60)
    print("TESTING ANGEL ONE CONNECTION")
    print("="*60 + "\n")
    
    # Check if credentials are configured
    if not ANGEL_ONE_CONFIG['client_id']:
        print("⚠️  Please configure your Angel One credentials in:")
        print("   quant_trading/angel_one_config.py")
        print("\n   Required fields:")
        print("   - client_id: Your Angel One User ID")
        print("   - password: Your Angel One Password")
        print("   - totp_secret: Your TOTP secret or 6-digit code")
        return False
    
    # Create broker instance
    broker = AngelOneAPI(
        api_key=ANGEL_ONE_CONFIG['api_key'],
        secret_key=ANGEL_ONE_CONFIG['secret_key'],
        client_id=ANGEL_ONE_CONFIG['client_id'],
        password=ANGEL_ONE_CONFIG['password'],
        totp=ANGEL_ONE_CONFIG['totp_secret']
    )
    
    # Connect
    if broker.connect():
        print("✅ Connected to Angel One!")
        
        # Get account info
        print("\n? Account Info:")
        account = broker.get_account_info()
        for key, value in account.items():
            print(f"   {key}: ₹{value:,.2f}" if isinstance(value, (int, float)) else f"   {key}: {value}")
        
        # Get positions
        print("\n? Current Positions:")
        positions = broker.get_positions()
        if positions:
            for symbol, pos in positions.items():
                print(f"   {symbol}: {pos['quantity']} @ ₹{pos['avg_price']:.2f} "
                      f"({pos['side']}) P&L: ₹{pos['pnl']:.2f}")
        else:
            print("   No open positions")
        
        # Get LTP for some stocks
        print("\n? Live Prices:")
        for symbol in ['RELIANCE', 'TCS', 'INFY']:
            ltp = broker.get_ltp(symbol)
            print(f"   {symbol}: ₹{ltp:.2f}")
        
        # Disconnect
        broker.disconnect()
        print("\n✅ Disconnected from Angel One")
        return True
    else:
        print("❌ Failed to connect to Angel One")
        return False


def run_paper_trading_with_angel_one_data():
    """
    Run paper trading but use Angel One for live price data.
    Orders are simulated (not sent to broker).
    """
    print("\n" + "="*60)
    print("PAPER TRADING WITH ANGEL ONE DATA")
    print("="*60 + "\n")
    
    # Create configuration
    config = SystemConfig()
    config.mode = TradingMode.PAPER
    config.initial_capital = 500000  # 5 Lakhs
    
    # Customize symbols
    config.data.symbols = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK']
    
    # Create trading system
    system = TradingSystem(config)
    
    # Initialize with mock broker (paper trading)
    system.initialize()
    
    print("Running paper trading simulation...")
    print("(Press Ctrl+C to stop)\n")
    
    try:
        # Run 5 iterations for demo
        for i in range(5):
            system._run_iteration()
            status = system.get_status()
            print(f"Iteration {i+1}: Equity=₹{status['current_equity']:,.2f}, "
                  f"Positions={status['positions_count']}, "
                  f"Drawdown={status['drawdown']:.2%}")
    except KeyboardInterrupt:
        print("\nStopping...")
    
    # Print final report
    print("\n" + system.monitoring.generate_report())
    system.shutdown()


def run_live_trading_with_angel_one():
    """
    Run live trading with Angel One.
    ⚠️ WARNING: This will place REAL orders!
    """
    print("\n" + "="*60)
    print("⚠️  LIVE TRADING WITH ANGEL ONE")
    print("="*60 + "\n")
    
    # Safety check
    confirm = input("This will place REAL orders. Type 'CONFIRM' to proceed: ")
    if confirm != 'CONFIRM':
        print("Cancelled.")
        return
    
    # Check credentials
    if not ANGEL_ONE_CONFIG['client_id']:
        print("Please configure Angel One credentials first.")
        return
    
    # Create Angel One broker
    broker = AngelOneAPI(
        api_key=ANGEL_ONE_CONFIG['api_key'],
        secret_key=ANGEL_ONE_CONFIG['secret_key'],
        client_id=ANGEL_ONE_CONFIG['client_id'],
        password=ANGEL_ONE_CONFIG['password'],
        totp=ANGEL_ONE_CONFIG['totp_secret']
    )
    
    # Create configuration
    config = SystemConfig()
    config.mode = TradingMode.LIVE
    config.initial_capital = 500000
    
    # Conservative risk settings for live trading
    config.risk.max_position_size_pct = 0.05  # Max 5% per position
    config.risk.max_drawdown_pct = 0.05       # Kill switch at 5% drawdown
    config.risk.daily_loss_limit_pct = 0.02   # Stop at 2% daily loss
    
    # Only trade liquid stocks
    config.data.symbols = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK']
    
    # Create trading system with Angel One broker
    from quant_trading.orchestrator import TradingSystem
    system = TradingSystem(config)
    system.execution_engine.broker = broker
    
    # Initialize
    if not broker.connect():
        print("Failed to connect to Angel One")
        return
    
    system.data_manager.initialize()
    system.monitoring.start()
    
    print("Starting live trading...")
    print("(Press Ctrl+C to stop)\n")
    
    try:
        system.run()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        system.shutdown()


if __name__ == "__main__":
    import sys
    
    print("\nAngel One Trading System")
    print("========================\n")
    print("Options:")
    print("1. Test Angel One connection")
    print("2. Paper trading (simulation)")
    print("3. Live trading (REAL orders)")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == '1':
        test_angel_one_connection()
    elif choice == '2':
        run_paper_trading_with_angel_one_data()
    elif choice == '3':
        run_live_trading_with_angel_one()
    else:
        print("Invalid option")
