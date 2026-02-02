"""
Professional NIFTY 50 F&O Trading Dashboard with Paper Trading
===============================================================

Live intraday options trading signals with:
- Specific strike prices and premium levels
- Paper trading with virtual portfolio
- Detailed entry/exit prices
- Hold time recommendations

Core Principles:
- Capital protection > profits
- NO TRADE over weak setups
- Max 2 signals per day
- Risk per trade: ≤ 1% of capital
"""

import sys
import os

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
import threading
import time as time_module
import traceback
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our enhanced modules
from quant_trading.fno_signal_generator import (
    NiftyFnOSignalGenerator, 
    TradingSignal, 
    MarketTrend, 
    SignalType,
    TechnicalIndicators
)
from quant_trading.paper_trading import (
    PaperTradingEngine,
    TradeRecommendation
)
from quant_trading.data_manager import (
    MarketDataManager,
    get_data_manager
)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'nifty_fno_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize engines
signal_generator = NiftyFnOSignalGenerator()
paper_trading = PaperTradingEngine(initial_capital=100000)  # 1 Lakh virtual capital
data_manager = get_data_manager()  # Cached data manager

# Global state
current_signal = None
current_recommendation = None
last_update = None
historical_signals = []
update_lock = threading.Lock()


def get_nifty_data(period='5d', interval='5m'):
    """Fetch NIFTY 50 data with caching."""
    try:
        df = data_manager.get_nifty_data(period=period, interval=interval)
        return df
    except Exception as e:
        logger.error(f"Error fetching NIFTY data: {e}")
        return None


def get_india_vix():
    """Fetch India VIX with caching."""
    try:
        return data_manager.get_india_vix()
    except Exception as e:
        logger.error(f"Error fetching VIX: {e}")
        return None


def generate_market_analysis():
    """Generate comprehensive market analysis with trade recommendation."""
    global current_signal, current_recommendation, last_update
    
    with update_lock:
        try:
            df = get_nifty_data()
            if df is None or len(df) < 50:
                logger.warning("Insufficient data for analysis")
                return _create_error_response("Unable to fetch market data")
            
            # Generate signal using enhanced analyzer
            signal = signal_generator.generate_signal(df)
            signal_dict = signal.to_dict()
    
            # Generate trade recommendation if signal is not NO TRADE
            recommendation = None
            if signal.signal != SignalType.NO_TRADE:
                try:
                    recommendation = paper_trading.generate_trade_recommendation(signal_dict)
                except Exception as e:
                    logger.error(f"Error generating recommendation: {e}")
            
            # Get additional data
            india_vix = get_india_vix()
            
            # Update global state
            current_signal = signal
            current_recommendation = recommendation
            last_update = datetime.now()
            
            # Check open positions
            alerts = []
            try:
                if signal.spot_price > 0:
                    alerts = paper_trading.check_and_update_positions(signal.spot_price)
            except Exception as e:
                logger.error(f"Error checking positions: {e}")
            
            # Store in history
            if signal.signal != SignalType.NO_TRADE:
                historical_signals.append(signal_dict)
                if len(historical_signals) > 10:
                    historical_signals.pop(0)
            
            # Prepare chart data
            chart_data = prepare_chart_data(df)
            
            # Get data manager metrics
            data_metrics = data_manager.get_metrics()
            market_status_info = data_manager.get_market_status()
            
            return {
                'signal': signal_dict,
                'recommendation': recommendation.to_dict() if recommendation else None,
                'chart_data': chart_data,
                'india_vix': india_vix,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'market_status': market_status_info.get('status', get_market_status()),
                'market_message': market_status_info.get('message', ''),
                'signals_today': signal_generator.signals_today,
                'max_signals': signal_generator.max_signals_per_day,
                'portfolio': paper_trading.get_portfolio_summary(),
                'open_positions': paper_trading.get_open_positions(),
                'alerts': alerts,
                'data_metrics': data_metrics
            }
            
        except Exception as e:
            logger.error(f"Error in market analysis: {e}")
            traceback.print_exc()
            return _create_error_response(str(e))


def _create_error_response(error_msg: str) -> dict:
    """Create error response with default values."""
    return {
        'signal': {
            'signal': 'NO TRADE',
            'market_trend': 'SIDEWAYS',
            'reasoning': [f'Error: {error_msg}'],
            'confidence': 'Low',
            'spot_price': 0,
            'rsi': 50,
            'ema_20': 0,
            'ema_50': 0,
            'vwap': 0,
            'volume_status': 'Unknown',
            'score': 0
        },
        'recommendation': None,
        'chart_data': None,
        'india_vix': None,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'market_status': get_market_status(),
        'signals_today': 0,
        'max_signals': 2,
        'portfolio': paper_trading.get_portfolio_summary(),
        'open_positions': [],
        'alerts': [],
        'error': error_msg
    }


def prepare_chart_data(df: pd.DataFrame) -> dict:
    """Prepare chart data for frontend."""
    if df is None or len(df) < 50:
        return None
    
    df_chart = df.tail(50).copy()
    
    # Calculate EMAs
    close_prices = df['close'].values
    ema_20_values = []
    ema_50_values = []
    
    ema_20 = close_prices[0]
    ema_50 = close_prices[0]
    mult_20 = 2 / 21
    mult_50 = 2 / 51
    
    for price in close_prices:
        ema_20 = (price - ema_20) * mult_20 + ema_20
        ema_50 = (price - ema_50) * mult_50 + ema_50
        ema_20_values.append(ema_20)
        ema_50_values.append(ema_50)
    
    ema_20_chart = ema_20_values[-50:]
    ema_50_chart = ema_50_values[-50:]
    
    # Calculate VWAP
    typical_price = (df_chart['high'] + df_chart['low'] + df_chart['close']) / 3
    vwap_values = (typical_price * df_chart['volume']).cumsum() / df_chart['volume'].cumsum()
    
    # Helper function to sanitize values for JSON (NaN/Inf not allowed in JSON)
    def sanitize_value(x):
        if pd.isna(x) or np.isinf(x):
            return None
        return round(float(x), 2)
    
    def sanitize_list(lst):
        return [sanitize_value(x) for x in lst]
    
    return {
        'labels': [d.strftime('%H:%M') if hasattr(d, 'strftime') else str(d) 
                   for d in df_chart['Datetime' if 'Datetime' in df_chart.columns else df_chart.index]],
        'ohlc': [
            {
                'open': sanitize_value(row['open']),
                'high': sanitize_value(row['high']),
                'low': sanitize_value(row['low']),
                'close': sanitize_value(row['close'])
            }
            for _, row in df_chart.iterrows()
        ],
        'close': sanitize_list(df_chart['close'].tolist()),
        'ema_20': sanitize_list(ema_20_chart),
        'ema_50': sanitize_list(ema_50_chart),
        'vwap': sanitize_list(vwap_values.tolist()),
        'volume': sanitize_list(df_chart['volume'].tolist()) if 'volume' in df_chart.columns else []
    }


def get_market_status():
    """Get current market status."""
    now = datetime.now().time()
    
    if time(9, 15) <= now <= time(15, 30):
        return "MARKET OPEN"
    elif time(9, 0) <= now < time(9, 15):
        return "PRE-OPEN"
    elif time(15, 30) < now <= time(16, 0):
        return "POST-CLOSE"
    else:
        return "MARKET CLOSED"


def background_updater():
    """Background thread for real-time updates."""
    while True:
        try:
            data = generate_market_analysis()
            if data:
                socketio.emit('market_update', data)
        except Exception as e:
            print(f"Update error: {e}")
        
        time_module.sleep(30)


# ═══════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════

@app.route('/')
def dashboard():
    """Main dashboard page."""
    return render_template('trading_dashboard.html')


@app.route('/api/signal')
def get_signal():
    """Get current trading signal with recommendation."""
    try:
        data = generate_market_analysis()
        if data:
            return jsonify(data)
        return jsonify({'error': 'Unable to fetch data'}), 500
    except Exception as e:
        print(f"API Error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/portfolio')
def get_portfolio():
    """Get paper trading portfolio."""
    return jsonify({
        'portfolio': paper_trading.get_portfolio_summary(),
        'open_positions': paper_trading.get_open_positions(),
        'trade_history': paper_trading.get_trade_history(20)
    })


@app.route('/api/execute-trade', methods=['POST'])
def execute_trade():
    """Execute a paper trade."""
    try:
        if current_recommendation is None:
            return jsonify({'error': 'No active recommendation'}), 400
        
        trade = paper_trading.execute_paper_trade(current_recommendation)
        return jsonify({
            'success': True,
            'trade': trade.to_dict(),
            'portfolio': paper_trading.get_portfolio_summary()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/close-trade', methods=['POST'])
def close_trade():
    """Close a paper trade."""
    try:
        data = request.json
        trade_id = data.get('trade_id')
        exit_premium = float(data.get('exit_premium', 0))
        exit_reason = data.get('exit_reason', 'Manual Exit')
        
        trade = paper_trading.close_paper_trade(trade_id, exit_premium, exit_reason)
        if trade:
            return jsonify({
                'success': True,
                'trade': trade.to_dict(),
                'portfolio': paper_trading.get_portfolio_summary()
            })
        return jsonify({'error': 'Trade not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/reset-portfolio', methods=['POST'])
def reset_portfolio():
    """Reset paper trading portfolio."""
    paper_trading.reset_paper_trading()
    return jsonify({
        'success': True,
        'portfolio': paper_trading.get_portfolio_summary()
    })


@app.route('/api/history')
def get_history():
    """Get signal and trade history."""
    return jsonify({
        'signals': historical_signals,
        'trades': paper_trading.get_trade_history(20)
    })


@app.route('/api/metrics')
def get_metrics():
    """Get system metrics including data fetch stats."""
    return jsonify({
        'data_metrics': data_manager.get_metrics(),
        'market_status': data_manager.get_market_status(),
        'signal_generator': {
            'signals_today': signal_generator.signals_today,
            'max_signals': signal_generator.max_signals_per_day
        },
        'portfolio_summary': paper_trading.get_portfolio_summary()
    })


@app.route('/api/clear-cache', methods=['POST'])
def clear_cache():
    """Clear data cache to force fresh data fetch."""
    data_manager.clear_cache()
    return jsonify({'success': True, 'message': 'Cache cleared'})


@app.route('/api/health')
def health_check():
    """Health check endpoint."""
    try:
        df = get_nifty_data()
        data_ok = df is not None and len(df) > 0
        return jsonify({
            'status': 'healthy' if data_ok else 'degraded',
            'data_available': data_ok,
            'timestamp': datetime.now().isoformat(),
            'market_status': data_manager.get_market_status()
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


# ═══════════════════════════════════════════════════════════════
# SOCKET.IO EVENTS
# ═══════════════════════════════════════════════════════════════

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print(f"Client connected at {datetime.now()}")
    data = generate_market_analysis()
    if data:
        socketio.emit('market_update', data)


@socketio.on('request_update')
def handle_update_request():
    """Handle manual update request."""
    data = generate_market_analysis()
    if data:
        socketio.emit('market_update', data)


@socketio.on('execute_paper_trade')
def handle_execute_trade():
    """Handle paper trade execution via socket."""
    if current_recommendation:
        trade = paper_trading.execute_paper_trade(current_recommendation)
        socketio.emit('trade_executed', {
            'trade': trade.to_dict(),
            'portfolio': paper_trading.get_portfolio_summary()
        })


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║     PROFESSIONAL NIFTY 50 F&O TRADING DASHBOARD                   ║
    ║                    WITH PAPER TRADING                             ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║  Features:                                                        ║
    ║  • Specific Strike & Premium prices                               ║
    ║  • Entry, Stop Loss, Target levels                                ║
    ║  • Hold time recommendations                                      ║
    ║  • Virtual Paper Trading (₹1,00,000 capital)                      ║
    ║  • P&L tracking and analytics                                     ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║  Core Principles:                                                 ║
    ║  • Capital protection > profits                                   ║
    ║  • NO TRADE over weak setups                                      ║
    ║  • Max 2 signals per day | Risk ≤ 1% per trade                    ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║  Trading Window: 9:20 AM - 2:45 PM IST                            ║
    ║  Starting server at http://localhost:5000                         ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    # Start background updater
    updater_thread = threading.Thread(target=background_updater, daemon=True)
    updater_thread.start()
    
    # Run server
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)