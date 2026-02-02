"""
F&O Nifty 50 Live Trading Dashboard
===================================
Flask-based web application for live trading signals.

Features:
- Real-time Nifty 50 data
- ML-based buy/sell signals
- Stop Loss & Target prices
- Live charts with TradingView style
- F&O specific analysis

Run: python app.py
Open: http://localhost:5000
"""

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import threading
import time
import json
import sys

sys.path.insert(0, '.')

from quant_trading.ml import MLPredictor
from quant_trading.features import FeatureEngine
from quant_trading.alpha import AlphaEngine

app = Flask(__name__)
app.config['SECRET_KEY'] = 'quant_trading_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global state
market_data = {
    'nifty50': None,
    'banknifty': None,
    'last_update': None,
    'signals': {},
    'is_running': False
}

# Initialize ML components
ml_predictor = MLPredictor()
feature_engine = FeatureEngine()
alpha_engine = AlphaEngine()


class FnOAnalyzer:
    """F&O specific analysis for Nifty 50."""
    
    def __init__(self):
        self.lot_size = {
            'NIFTY': 50,
            'BANKNIFTY': 15
        }
        
    def get_live_data(self, symbol: str = '^NSEI', period: str = '5d', interval: str = '5m'):
        """Fetch live/recent data for Nifty."""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                return None
            
            df = df.reset_index()
            df.columns = [c.lower().replace(' ', '_') for c in df.columns]
            
            # Rename datetime column
            if 'datetime' in df.columns:
                df['date'] = df['datetime']
            
            df['symbol'] = 'NIFTY50' if symbol == '^NSEI' else 'BANKNIFTY'
            
            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
    
    def calculate_signals(self, df: pd.DataFrame) -> dict:
        """Calculate trading signals with SL and Target."""
        if df is None or len(df) < 50:
            return None
        
        current_price = df['close'].iloc[-1]
        
        # Calculate ATR for SL/Target
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        tr = np.maximum(high[1:] - low[1:], 
                       np.maximum(abs(high[1:] - close[:-1]), 
                                 abs(low[1:] - close[:-1])))
        atr = np.mean(tr[-14:])
        
        # Get ML prediction
        try:
            # Prepare data for ML
            df_ml = df.copy()
            if 'volume' not in df_ml.columns:
                df_ml['volume'] = 1000000
            
            prediction = ml_predictor.predict(df_ml, 'NIFTY50', '1d')
            
            predicted_return = prediction.predicted_return
            prob_up = prediction.prob_up
            prob_down = prediction.prob_down
            regime = prediction.regime.value
            confidence = prediction.overall_confidence
            
        except Exception as e:
            print(f"ML prediction error: {e}")
            predicted_return = 0
            prob_up = 0.5
            prob_down = 0.5
            regime = 'unknown'
            confidence = 0.3
        
        # Calculate technical indicators
        sma_20 = np.mean(close[-20:])
        sma_50 = np.mean(close[-50:]) if len(close) >= 50 else sma_20
        
        # RSI
        delta = np.diff(close)
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses[-14:])
        rs = avg_gain / avg_loss if avg_loss > 0 else 100
        rsi = 100 - (100 / (1 + rs))
        
        # VWAP (if volume available)
        if 'volume' in df.columns and df['volume'].sum() > 0:
            vwap = (df['close'] * df['volume']).sum() / df['volume'].sum()
        else:
            vwap = current_price
        
        # Determine signal
        signal = self._determine_signal(
            current_price, sma_20, sma_50, rsi, 
            predicted_return, prob_up, prob_down, confidence
        )
        
        # Calculate Stop Loss and Target
        if signal['action'] == 'BUY':
            sl = current_price - (atr * 1.5)
            target1 = current_price + (atr * 2)
            target2 = current_price + (atr * 3)
            risk_reward = (target1 - current_price) / (current_price - sl)
        elif signal['action'] == 'SELL':
            sl = current_price + (atr * 1.5)
            target1 = current_price - (atr * 2)
            target2 = current_price - (atr * 3)
            risk_reward = (current_price - target1) / (sl - current_price)
        else:
            sl = current_price - (atr * 1.5)
            target1 = current_price + (atr * 2)
            target2 = current_price + (atr * 3)
            risk_reward = 1.33
        
        return {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'symbol': 'NIFTY 50',
            'current_price': round(current_price, 2),
            'signal': signal,
            'stop_loss': round(sl, 2),
            'target1': round(target1, 2),
            'target2': round(target2, 2),
            'risk_reward': round(risk_reward, 2),
            'atr': round(atr, 2),
            'indicators': {
                'rsi': round(rsi, 2),
                'sma_20': round(sma_20, 2),
                'sma_50': round(sma_50, 2),
                'vwap': round(vwap, 2),
                'atr': round(atr, 2)
            },
            'ml_prediction': {
                'predicted_return': round(predicted_return * 100, 2),
                'prob_up': round(prob_up * 100, 1),
                'prob_down': round(prob_down * 100, 1),
                'regime': regime,
                'confidence': round(confidence * 100, 1)
            },
            'lot_size': 50,
            'points_risk': round(current_price - sl, 2),
            'points_target': round(target1 - current_price, 2)
        }
    
    def _determine_signal(self, price, sma_20, sma_50, rsi, 
                          pred_return, prob_up, prob_down, confidence):
        """Determine trading signal based on multiple factors."""
        
        # Score-based signal generation
        score = 0
        reasons = []
        
        # Trend analysis
        if price > sma_20 > sma_50:
            score += 2
            reasons.append("Strong uptrend (Price > SMA20 > SMA50)")
        elif price > sma_20:
            score += 1
            reasons.append("Above SMA20")
        elif price < sma_20 < sma_50:
            score -= 2
            reasons.append("Strong downtrend (Price < SMA20 < SMA50)")
        elif price < sma_20:
            score -= 1
            reasons.append("Below SMA20")
        
        # RSI analysis
        if rsi < 30:
            score += 1.5
            reasons.append(f"RSI oversold ({rsi:.1f})")
        elif rsi < 40:
            score += 0.5
            reasons.append(f"RSI approaching oversold ({rsi:.1f})")
        elif rsi > 70:
            score -= 1.5
            reasons.append(f"RSI overbought ({rsi:.1f})")
        elif rsi > 60:
            score -= 0.5
            reasons.append(f"RSI approaching overbought ({rsi:.1f})")
        
        # ML prediction
        if pred_return > 0.01 and prob_up > 0.55:
            score += 1.5
            reasons.append(f"ML bullish ({prob_up*100:.0f}% up probability)")
        elif pred_return < -0.01 and prob_down > 0.55:
            score -= 1.5
            reasons.append(f"ML bearish ({prob_down*100:.0f}% down probability)")
        
        # Determine action
        if score >= 2:
            action = 'BUY'
            strength = 'STRONG' if score >= 3 else 'MODERATE'
            color = '#00ff00' if strength == 'STRONG' else '#90EE90'
        elif score <= -2:
            action = 'SELL'
            strength = 'STRONG' if score <= -3 else 'MODERATE'
            color = '#ff0000' if strength == 'STRONG' else '#FFB6C1'
        else:
            action = 'WAIT'
            strength = 'NEUTRAL'
            color = '#ffff00'
        
        return {
            'action': action,
            'strength': strength,
            'score': round(score, 2),
            'color': color,
            'reasons': reasons,
            'confidence': round(min(abs(score) / 5 * 100, 100), 1)
        }

    def get_option_chain_data(self, spot_price: float) -> dict:
        """Generate synthetic option chain data for display."""
        # Calculate strike prices around spot
        atm_strike = round(spot_price / 50) * 50
        strikes = [atm_strike + (i * 50) for i in range(-5, 6)]
        
        chain = []
        for strike in strikes:
            # Simplified Black-Scholes approximation
            moneyness = (spot_price - strike) / spot_price
            
            # CE (Call) pricing approximation
            ce_price = max(spot_price - strike, 0) + abs(moneyness) * 100 + np.random.uniform(10, 50)
            ce_oi = int(np.random.uniform(50000, 500000))
            ce_change = np.random.uniform(-10, 10)
            
            # PE (Put) pricing approximation  
            pe_price = max(strike - spot_price, 0) + abs(moneyness) * 100 + np.random.uniform(10, 50)
            pe_oi = int(np.random.uniform(50000, 500000))
            pe_change = np.random.uniform(-10, 10)
            
            chain.append({
                'strike': strike,
                'ce_ltp': round(ce_price, 2),
                'ce_oi': ce_oi,
                'ce_change': round(ce_change, 2),
                'pe_ltp': round(pe_price, 2),
                'pe_oi': pe_oi,
                'pe_change': round(pe_change, 2),
                'is_atm': strike == atm_strike
            })
        
        return {
            'spot': round(spot_price, 2),
            'atm_strike': atm_strike,
            'chain': chain
        }


# Initialize analyzer
fno_analyzer = FnOAnalyzer()


def background_data_updater():
    """Background thread to update market data."""
    global market_data
    
    while market_data['is_running']:
        try:
            # Fetch Nifty 50 data
            df_nifty = fno_analyzer.get_live_data('^NSEI', '5d', '5m')
            
            if df_nifty is not None:
                market_data['nifty50'] = df_nifty
                
                # Calculate signals
                signals = fno_analyzer.calculate_signals(df_nifty)
                if signals:
                    market_data['signals'] = signals
                    market_data['last_update'] = datetime.now().strftime('%H:%M:%S')
                    
                    # Emit to connected clients
                    socketio.emit('market_update', {
                        'signals': signals,
                        'chart_data': get_chart_data(df_nifty),
                        'last_update': market_data['last_update']
                    })
            
            # Update every 30 seconds
            time.sleep(30)
            
        except Exception as e:
            print(f"Background updater error: {e}")
            time.sleep(10)


def get_chart_data(df: pd.DataFrame, limit: int = 100) -> dict:
    """Prepare chart data for frontend."""
    if df is None or df.empty:
        return {}
    
    df_recent = df.tail(limit)
    
    # OHLC data
    ohlc = []
    for _, row in df_recent.iterrows():
        timestamp = row.get('date', row.get('datetime', datetime.now()))
        if hasattr(timestamp, 'timestamp'):
            ts = int(timestamp.timestamp() * 1000)
        else:
            ts = int(datetime.now().timestamp() * 1000)
        
        ohlc.append({
            'time': ts,
            'open': round(row['open'], 2),
            'high': round(row['high'], 2),
            'low': round(row['low'], 2),
            'close': round(row['close'], 2)
        })
    
    # Calculate SMA for chart
    close = df_recent['close'].values
    sma_20 = pd.Series(close).rolling(20).mean().fillna(close[0]).tolist()
    
    return {
        'ohlc': ohlc,
        'sma_20': [{'time': ohlc[i]['time'], 'value': round(sma_20[i], 2)} 
                   for i in range(len(sma_20))],
        'labels': [datetime.fromtimestamp(o['time']/1000).strftime('%H:%M') 
                   for o in ohlc]
    }


@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('dashboard.html')


@app.route('/api/signals')
def get_signals():
    """API endpoint for current signals."""
    if market_data['signals']:
        return jsonify(market_data['signals'])
    
    # Fetch fresh data if not available
    df = fno_analyzer.get_live_data('^NSEI', '5d', '5m')
    if df is not None:
        signals = fno_analyzer.calculate_signals(df)
        return jsonify(signals)
    
    return jsonify({'error': 'Data not available'})


@app.route('/api/chart-data')
def get_chart_data_api():
    """API endpoint for chart data."""
    df = market_data.get('nifty50')
    
    if df is None:
        df = fno_analyzer.get_live_data('^NSEI', '5d', '5m')
    
    if df is not None:
        return jsonify(get_chart_data(df))
    
    return jsonify({'error': 'Data not available'})


@app.route('/api/option-chain')
def get_option_chain():
    """API endpoint for option chain."""
    signals = market_data.get('signals', {})
    spot = signals.get('current_price', 23500)
    
    chain_data = fno_analyzer.get_option_chain_data(spot)
    return jsonify(chain_data)


@app.route('/api/historical/<symbol>')
def get_historical(symbol):
    """API endpoint for historical data."""
    period = request.args.get('period', '1mo')
    interval = request.args.get('interval', '1d')
    
    ticker_map = {
        'nifty': '^NSEI',
        'banknifty': '^NSEBANK'
    }
    
    ticker_symbol = ticker_map.get(symbol.lower(), '^NSEI')
    df = fno_analyzer.get_live_data(ticker_symbol, period, interval)
    
    if df is not None:
        return jsonify(get_chart_data(df, limit=200))
    
    return jsonify({'error': 'Data not available'})


@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print('Client connected')
    
    # Send initial data
    if market_data['signals']:
        emit('market_update', {
            'signals': market_data['signals'],
            'chart_data': get_chart_data(market_data.get('nifty50')),
            'last_update': market_data['last_update']
        })


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print('Client disconnected')


@socketio.on('request_update')
def handle_request_update():
    """Handle manual update request."""
    df = fno_analyzer.get_live_data('^NSEI', '5d', '5m')
    if df is not None:
        signals = fno_analyzer.calculate_signals(df)
        market_data['signals'] = signals
        market_data['nifty50'] = df
        market_data['last_update'] = datetime.now().strftime('%H:%M:%S')
        
        emit('market_update', {
            'signals': signals,
            'chart_data': get_chart_data(df),
            'last_update': market_data['last_update']
        })


def start_background_updater():
    """Start the background data updater thread."""
    global market_data
    market_data['is_running'] = True
    
    thread = threading.Thread(target=background_data_updater, daemon=True)
    thread.start()
    print("Background updater started")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("  ? F&O NIFTY 50 LIVE TRADING DASHBOARD")
    print("  Starting server at http://localhost:5000")
    print("="*60 + "\n")
    
    # Start background updater
    start_background_updater()
    
    # Run Flask with SocketIO
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)