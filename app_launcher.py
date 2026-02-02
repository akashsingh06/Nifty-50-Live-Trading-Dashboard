"""
Unified Trading Dashboard Launcher
===================================
Single entry point for all trading services.

Services:
1. NIFTY 50 F&O Trading - Live signals with paper trading
2. Stock Analyzer - Analyze any Indian stock  
3. ML Dashboard - Machine learning based predictions
4. Paper Trading Portfolio - Track virtual trades

Run: python app_launcher.py
Open: http://localhost:5000
"""

import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from flask import Flask, render_template, jsonify, request, redirect, url_for
from flask_socketio import SocketIO
import pandas as pd
import numpy as np
import yfinance as yf
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

# Import our modules
from quant_trading.fno_signal_generator import (
    NiftyFnOSignalGenerator, 
    TradingSignal, 
    SignalType
)
from quant_trading.paper_trading import PaperTradingEngine
from quant_trading.data_manager import get_data_manager

app = Flask(__name__)
app.config['SECRET_KEY'] = 'unified_trading_dashboard'
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize engines
signal_generator = NiftyFnOSignalGenerator()
paper_trading = PaperTradingEngine(initial_capital=100000)
data_manager = get_data_manager()

# Global state
current_signal = None
current_recommendation = None
last_update = None
historical_signals = []
update_lock = threading.Lock()


def get_chart_data(df: pd.DataFrame, limit: int = 50) -> dict:
    """Get chart data for frontend."""
    if df is None or len(df) == 0:
        return None
    
    df_limited = df.tail(limit)
    
    # Calculate EMAs for chart
    close = df['close'].values
    ema_20 = pd.Series(close).ewm(span=20).mean().values[-limit:]
    ema_50 = pd.Series(close).ewm(span=50).mean().values[-limit:]
    
    # Calculate VWAP
    if 'volume' in df.columns and df['volume'].sum() > 0:
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        vwap_values = vwap.tail(limit).values
    else:
        vwap_values = close[-limit:]
    
    # Handle index - could be DatetimeIndex or RangeIndex
    if hasattr(df_limited.index, 'strftime'):
        labels = df_limited.index.strftime('%H:%M').tolist()
    else:
        labels = [str(i) for i in range(len(df_limited))]
    
    # Helper function to clean NaN values
    def clean_values(arr):
        result = []
        for x in arr:
            if pd.isna(x) or np.isnan(x):
                result.append(None)
            else:
                result.append(round(float(x), 2))
        return result
    
    return {
        'labels': labels,
        'close': clean_values(df_limited['close'].values),
        'ema_20': clean_values(ema_20),
        'ema_50': clean_values(ema_50),
        'vwap': clean_values(vwap_values)
    }


def get_market_status():
    """Get current market status."""
    now = datetime.now()
    current_time = now.time()
    
    market_open = time(9, 15)
    market_close = time(15, 30)
    
    if now.weekday() >= 5:
        return "CLOSED (Weekend)"
    elif current_time < market_open:
        return "PRE-MARKET"
    elif current_time > market_close:
        return "CLOSED"
    else:
        return "MARKET OPEN"


def analyze_stock(symbol: str):
    """Analyze a single stock."""
    try:
        # Add .NS suffix if not present
        ticker_symbol = f"{symbol}.NS" if not symbol.endswith('.NS') else symbol
        
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period="6mo")
        
        if df.empty:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="6mo")
        
        if df.empty:
            return {'error': f'Could not fetch data for {symbol}'}
        
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        
        # Calculate indicators
        close = df['close'].values
        
        # EMAs
        ema_20 = pd.Series(close).ewm(span=20).mean().iloc[-1]
        ema_50 = pd.Series(close).ewm(span=50).mean().iloc[-1]
        
        # RSI
        delta = pd.Series(close).diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = (100 - (100 / (1 + rs))).iloc[-1]
        
        # MACD
        ema_12 = pd.Series(close).ewm(span=12).mean()
        ema_26 = pd.Series(close).ewm(span=26).mean()
        macd = (ema_12 - ema_26).iloc[-1]
        signal_line = (ema_12 - ema_26).ewm(span=9).mean().iloc[-1]
        
        # Current price and changes
        current_price = close[-1]
        prev_close = close[-2] if len(close) > 1 else close[-1]
        day_change = ((current_price - prev_close) / prev_close) * 100
        
        week_ago = close[-5] if len(close) > 5 else close[0]
        week_change = ((current_price - week_ago) / week_ago) * 100
        
        month_ago = close[-22] if len(close) > 22 else close[0]
        month_change = ((current_price - month_ago) / month_ago) * 100
        
        # Generate signal
        bullish_points = 0
        bearish_points = 0
        reasons = []
        
        if current_price > ema_20:
            bullish_points += 1
            reasons.append("Price above EMA 20")
        else:
            bearish_points += 1
            reasons.append("Price below EMA 20")
            
        if ema_20 > ema_50:
            bullish_points += 1
            reasons.append("EMA 20 > EMA 50 (Bullish crossover)")
        else:
            bearish_points += 1
            reasons.append("EMA 20 < EMA 50 (Bearish crossover)")
            
        if rsi < 30:
            bullish_points += 2
            reasons.append(f"RSI oversold ({rsi:.1f})")
        elif rsi > 70:
            bearish_points += 2
            reasons.append(f"RSI overbought ({rsi:.1f})")
        elif rsi > 50:
            bullish_points += 1
            reasons.append(f"RSI bullish ({rsi:.1f})")
        else:
            bearish_points += 1
            reasons.append(f"RSI bearish ({rsi:.1f})")
            
        if macd > signal_line:
            bullish_points += 1
            reasons.append("MACD bullish")
        else:
            bearish_points += 1
            reasons.append("MACD bearish")
        
        # Determine recommendation
        if bullish_points >= 4:
            recommendation = "STRONG BUY"
            rec_color = "#10b981"
        elif bullish_points >= 3:
            recommendation = "BUY"
            rec_color = "#34d399"
        elif bearish_points >= 4:
            recommendation = "STRONG SELL"
            rec_color = "#ef4444"
        elif bearish_points >= 3:
            recommendation = "SELL"
            rec_color = "#f87171"
        else:
            recommendation = "HOLD"
            rec_color = "#f59e0b"
        
        return {
            'symbol': symbol.upper().replace('.NS', ''),
            'current_price': round(current_price, 2),
            'day_change': round(day_change, 2),
            'week_change': round(week_change, 2),
            'month_change': round(month_change, 2),
            'ema_20': round(ema_20, 2),
            'ema_50': round(ema_50, 2),
            'rsi': round(rsi, 2),
            'macd': round(macd, 2),
            'signal_line': round(signal_line, 2),
            'recommendation': recommendation,
            'rec_color': rec_color,
            'bullish_points': bullish_points,
            'bearish_points': bearish_points,
            'reasons': reasons,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        return {'error': str(e)}


def generate_nifty_analysis():
    """Generate NIFTY 50 analysis."""
    global current_signal, current_recommendation, last_update
    
    with update_lock:
        try:
            df = data_manager.get_nifty_data()
            if df is None or len(df) < 50:
                return {'error': 'Unable to fetch market data'}
            
            signal = signal_generator.generate_signal(df)
            signal_dict = signal.to_dict()
            
            recommendation = None
            if signal.signal != SignalType.NO_TRADE:
                try:
                    recommendation = paper_trading.generate_trade_recommendation(signal_dict)
                except Exception as e:
                    logger.error(f"Error generating recommendation: {e}")
            
            india_vix = data_manager.get_india_vix()
            
            current_signal = signal
            current_recommendation = recommendation
            last_update = datetime.now()
            
            alerts = []
            if signal.spot_price > 0:
                alerts = paper_trading.check_and_update_positions(signal.spot_price)
            
            return {
                'signal': signal_dict,
                'recommendation': recommendation.to_dict() if recommendation else None,
                'india_vix': india_vix,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'market_status': get_market_status(),
                'signals_today': signal_generator.signals_today,
                'max_signals': signal_generator.max_signals_per_day,
                'portfolio': paper_trading.get_portfolio_summary(),
                'open_positions': paper_trading.get_open_positions(),
                'alerts': alerts,
                'chart_data': get_chart_data(df) if df is not None else None
            }
            
        except Exception as e:
            logger.error(f"Error in NIFTY analysis: {e}")
            traceback.print_exc()
            return {'error': str(e)}


# ==================== ROUTES ====================

@app.route('/')
def home():
    """Main launcher page."""
    return render_template('launcher.html')


@app.route('/nifty')
def nifty_dashboard():
    """NIFTY 50 F&O Trading Dashboard."""
    return render_template('trading_dashboard.html')


@app.route('/stock-analyzer')
def stock_analyzer():
    """Stock Analyzer page."""
    return render_template('stock_analyzer.html')


@app.route('/portfolio')
def portfolio():
    """Paper Trading Portfolio page."""
    return render_template('portfolio.html')


# ==================== API ENDPOINTS ====================

@app.route('/api/nifty')
def api_nifty():
    """Get NIFTY analysis."""
    result = generate_nifty_analysis()
    return jsonify(result)


# Alias for trading_dashboard.html compatibility
@app.route('/api/signal')
def api_signal():
    """Get trading signal - alias for /api/nifty."""
    result = generate_nifty_analysis()
    return jsonify(result)


@app.route('/api/stock/<symbol>')
def api_stock(symbol):
    """Analyze a stock."""
    result = analyze_stock(symbol)
    return jsonify(result)


@app.route('/api/portfolio')
def api_portfolio():
    """Get portfolio summary."""
    return jsonify({
        'portfolio': paper_trading.get_portfolio_summary(),
        'open_positions': paper_trading.get_open_positions(),
        'trade_history': paper_trading.get_trade_history()
    })


@app.route('/api/execute-trade', methods=['POST'])
def execute_trade():
    """Execute a paper trade."""
    try:
        if current_recommendation:
            trade = paper_trading.execute_paper_trade(current_recommendation)
            if trade:
                return jsonify({'success': True, 'trade': trade.to_dict()})
        return jsonify({'success': False, 'error': 'No recommendation available'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/close-trade', methods=['POST'])
def close_trade_post():
    """Close a paper trade via POST body."""
    try:
        data = request.json or {}
        trade_id = data.get('trade_id')
        exit_premium = data.get('exit_premium')
        result = paper_trading.close_paper_trade(trade_id, exit_premium)
        if result:
            return jsonify({'success': True, 'trade': result.to_dict() if hasattr(result, 'to_dict') else result})
        return jsonify({'success': False, 'error': 'Trade not found'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/close-trade/<trade_id>', methods=['POST'])
def close_trade(trade_id):
    """Close a paper trade."""
    try:
        data = request.json or {}
        exit_premium = data.get('exit_premium')
        result = paper_trading.close_paper_trade(trade_id, exit_premium)
        if result:
            return jsonify({'success': True, 'trade': result})
        return jsonify({'success': False, 'error': 'Trade not found'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/reset-portfolio', methods=['POST'])
def reset_portfolio():
    """Reset paper trading portfolio."""
    try:
        paper_trading.reset()
        return jsonify({'success': True, 'message': 'Portfolio reset'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/market-status')
def api_market_status():
    """Get market status."""
    return jsonify({
        'status': get_market_status(),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })


@app.route('/api/watchlist', methods=['GET', 'POST'])
def api_watchlist():
    """Get or update watchlist."""
    if request.method == 'POST':
        # Add stock to watchlist (stored in session/memory for now)
        data = request.json
        symbol = data.get('symbol', '').upper()
        if symbol:
            result = analyze_stock(symbol)
            return jsonify(result)
    
    # Default watchlist
    default_stocks = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK']
    results = []
    for symbol in default_stocks:
        result = analyze_stock(symbol)
        if 'error' not in result:
            results.append(result)
    
    return jsonify(results)


# ==================== SOCKET EVENTS ====================

@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected at {datetime.now()}")


@socketio.on('request_update')
def handle_update_request():
    """Handle update request from client."""
    result = generate_nifty_analysis()
    socketio.emit('market_update', result)


def background_updates():
    """Background thread for real-time updates."""
    while True:
        try:
            result = generate_nifty_analysis()
            socketio.emit('market_update', result)
        except Exception as e:
            logger.error(f"Background update error: {e}")
        time_module.sleep(30)  # Update every 30 seconds


# ==================== MAIN ====================

if __name__ == '__main__':
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║         UNIFIED TRADING DASHBOARD LAUNCHER                        ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║  Services Available:                                              ║
    ║  • NIFTY 50 F&O Trading Dashboard                                 ║
    ║  • Stock Analyzer (Any Indian Stock)                              ║
    ║  • Paper Trading Portfolio                                        ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║  Open http://localhost:5000 to get started                        ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    # Start background update thread
    update_thread = threading.Thread(target=background_updates, daemon=True)
    update_thread.start()
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)