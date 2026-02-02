"""
Professional NIFTY 50 F&O Trading Dashboard
============================================

Live intraday options trading signals with strict risk management.
Based on expert Indian equity derivatives trading principles.

Core Principles:
- Capital protection > profits
- NO TRADE over weak setups
- Trade only clear trends or strong breakouts
- Max 2 signals per day
- Risk per trade: ≤ 1% of capital
"""

from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
import threading
import time as time_module

# Import our professional signal generator
from quant_trading.fno_signal_generator import (
    NiftyFnOSignalGenerator, 
    TradingSignal, 
    MarketTrend, 
    SignalType
)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'nifty_fno_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize professional signal generator
signal_generator = NiftyFnOSignalGenerator()

# Global state
current_signal = None
last_update = None
historical_signals = []


def get_nifty_data(period='5d', interval='5m'):
    """Fetch NIFTY 50 data from Yahoo Finance."""
    try:
        nifty = yf.Ticker("^NSEI")
        df = nifty.history(period=period, interval=interval)
        
        if df.empty:
            return None
        
        df.columns = df.columns.str.lower()
        df = df.reset_index()
        
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None


def get_india_vix():
    """Fetch India VIX."""
    try:
        vix = yf.Ticker("^INDIAVIX")
        data = vix.history(period='1d')
        if not data.empty:
            return round(data['Close'].iloc[-1], 2)
    except:
        pass
    return None


def generate_market_analysis():
    """Generate comprehensive market analysis."""
    global current_signal, last_update
    
    df = get_nifty_data()
    if df is None:
        return None
    
    # Generate signal using professional analyzer
    signal = signal_generator.generate_signal(df)
    
    # Get additional data
    india_vix = get_india_vix()
    
    # Prepare response
    current_signal = signal
    last_update = datetime.now()
    
    # Store in history (keep last 10)
    if signal.signal != SignalType.NO_TRADE:
        historical_signals.append(signal.to_dict())
        if len(historical_signals) > 10:
            historical_signals.pop(0)
    
    # Prepare chart data
    chart_data = prepare_chart_data(df)
    
    return {
        'signal': signal.to_dict(),
        'chart_data': chart_data,
        'india_vix': india_vix,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'market_status': get_market_status(),
        'signals_today': signal_generator.signals_today,
        'max_signals': signal_generator.max_signals_per_day
    }


def prepare_chart_data(df: pd.DataFrame) -> dict:
    """Prepare chart data for frontend."""
    if df is None or len(df) < 50:
        return None
    
    # Get last 50 candles for chart
    df_chart = df.tail(50).copy()
    
    # Calculate EMAs for chart
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
    
    # Take last 50 values
    ema_20_chart = ema_20_values[-50:]
    ema_50_chart = ema_50_values[-50:]
    
    # Calculate VWAP
    typical_price = (df_chart['high'] + df_chart['low'] + df_chart['close']) / 3
    vwap_values = (typical_price * df_chart['volume']).cumsum() / df_chart['volume'].cumsum()
    
    return {
        'labels': [d.strftime('%H:%M') if hasattr(d, 'strftime') else str(d) 
                   for d in df_chart['Datetime' if 'Datetime' in df_chart.columns else df_chart.index]],
        'ohlc': [
            {
                'open': round(row['open'], 2),
                'high': round(row['high'], 2),
                'low': round(row['low'], 2),
                'close': round(row['close'], 2)
            }
            for _, row in df_chart.iterrows()
        ],
        'close': [round(x, 2) for x in df_chart['close'].tolist()],
        'ema_20': [round(x, 2) for x in ema_20_chart],
        'ema_50': [round(x, 2) for x in ema_50_chart],
        'vwap': [round(x, 2) for x in vwap_values.tolist()],
        'volume': df_chart['volume'].tolist() if 'volume' in df_chart.columns else []
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
        
        time_module.sleep(30)  # Update every 30 seconds


# Routes
@app.route('/')
def dashboard():
    """Main dashboard page."""
    return render_template('professional_dashboard.html')


@app.route('/api/signal')
def get_signal():
    """Get current trading signal."""
    data = generate_market_analysis()
    if data:
        return jsonify(data)
    return jsonify({'error': 'Unable to fetch data'}), 500


@app.route('/api/history')
def get_history():
    """Get historical signals."""
    return jsonify({'signals': historical_signals})


@app.route('/api/status')
def get_status():
    """Get system status."""
    return jsonify({
        'market_status': get_market_status(),
        'signals_today': signal_generator.signals_today,
        'max_signals': signal_generator.max_signals_per_day,
        'last_update': last_update.strftime('%Y-%m-%d %H:%M:%S') if last_update else None,
        'trading_window': '9:20 AM - 2:45 PM IST'
    })


# SocketIO events
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


if __name__ == '__main__':
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║     PROFESSIONAL NIFTY 50 F&O TRADING DASHBOARD              ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Core Principles:                                            ║
    ║  • Capital protection > profits                              ║
    ║  • NO TRADE over weak setups                                 ║
    ║  • Max 2 signals per day                                     ║
    ║  • Risk per trade: ≤ 1% of capital                           ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Trading Window: 9:20 AM - 2:45 PM IST                       ║
    ║  Timeframe: 5-minute candles                                 ║
    ║  Instrument: NIFTY 50 Index Options (CE/PE)                  ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Starting server at http://localhost:5000                    ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Start background updater
    updater_thread = threading.Thread(target=background_updater, daemon=True)
    updater_thread.start()
    
    # Run server
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)