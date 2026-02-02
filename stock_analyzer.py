"""
Stock Performance Analyzer
==========================
Get stock performance and BUY/SELL recommendations for Indian stocks.
Now with ML-based predictions inspired by Jim Simons/Renaissance Technologies!

Usage:
    python stock_analyzer.py RELIANCE
    python stock_analyzer.py TCS INFY WIPRO
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent to path
sys.path.insert(0, '.')

from quant_trading.data import DataManager
from quant_trading.features import FeatureEngine
from quant_trading.alpha import AlphaEngine
from quant_trading.ml import MLPredictor, MarketRegime


def analyze_stock(symbol: str, use_real_data: bool = True):
    """Analyze a single stock and provide recommendation."""
    
    print(f"\n{'='*60}")
    print(f"  STOCK ANALYSIS: {symbol}")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}")
    
    # Initialize components
    data_manager = DataManager(use_mock=not use_real_data)
    feature_engine = FeatureEngine()
    alpha_engine = AlphaEngine()
    
    # Fetch data
    print(f"\n? Fetching data for {symbol}...")
    
    try:
        if use_real_data:
            import yfinance as yf
            ticker = yf.Ticker(f"{symbol}.NS")
            df = ticker.history(period="6mo")
            
            if df.empty:
                # Try without .NS
                ticker = yf.Ticker(symbol)
                df = ticker.history(period="6mo")
            
            if df.empty:
                print(f"❌ Could not fetch data for {symbol}")
                return None
            
            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]
            df['symbol'] = symbol
            
            # Get current info
            info = ticker.info
            current_price = info.get('currentPrice', df['close'].iloc[-1])
            prev_close = info.get('previousClose', df['close'].iloc[-2])
            day_high = info.get('dayHigh', df['high'].iloc[-1])
            day_low = info.get('dayLow', df['low'].iloc[-1])
            volume = info.get('volume', df['volume'].iloc[-1])
            market_cap = info.get('marketCap', 0)
            pe_ratio = info.get('trailingPE', 0)
            fifty_two_week_high = info.get('fiftyTwoWeekHigh', df['high'].max())
            fifty_two_week_low = info.get('fiftyTwoWeekLow', df['low'].min())
        else:
            # Use mock data
            df = data_manager.get_data(symbol)
            current_price = df['close'].iloc[-1]
            prev_close = df['close'].iloc[-2]
            day_high = df['high'].iloc[-1]
            day_low = df['low'].iloc[-1]
            volume = df['volume'].iloc[-1]
            market_cap = 0
            pe_ratio = 0
            fifty_two_week_high = df['high'].max()
            fifty_two_week_low = df['low'].min()
            
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        print("   Using mock data instead...")
        df = data_manager.get_data(symbol)
        current_price = df['close'].iloc[-1]
        prev_close = df['close'].iloc[-2]
        day_high = df['high'].iloc[-1]
        day_low = df['low'].iloc[-1]
        volume = df['volume'].iloc[-1]
        market_cap = 0
        pe_ratio = 0
        fifty_two_week_high = df['high'].max()
        fifty_two_week_low = df['low'].min()
    
    # Calculate performance metrics
    daily_change = current_price - prev_close
    daily_change_pct = (daily_change / prev_close) * 100
    
    # Weekly, Monthly returns
    if len(df) >= 5:
        weekly_return = ((current_price - df['close'].iloc[-5]) / df['close'].iloc[-5]) * 100
    else:
        weekly_return = 0
        
    if len(df) >= 22:
        monthly_return = ((current_price - df['close'].iloc[-22]) / df['close'].iloc[-22]) * 100
    else:
        monthly_return = 0
        
    if len(df) >= 66:
        quarterly_return = ((current_price - df['close'].iloc[-66]) / df['close'].iloc[-66]) * 100
    else:
        quarterly_return = 0
    
    # 52-week position
    fifty_two_week_range = fifty_two_week_high - fifty_two_week_low
    position_in_range = ((current_price - fifty_two_week_low) / fifty_two_week_range) * 100 if fifty_two_week_range > 0 else 50
    
    # ============================================
    # PRICE & PERFORMANCE
    # ============================================
    print(f"\n┌{'─'*58}┐")
    print(f"│{'PRICE & PERFORMANCE':^58}│")
    print(f"├{'─'*58}┤")
    print(f"│  Current Price:        ₹{current_price:>15,.2f}               │")
    print(f"│  Day Change:           ₹{daily_change:>+15,.2f} ({daily_change_pct:+.2f}%)     │")
    print(f"│  Day Range:            ₹{day_low:,.2f} - ₹{day_high:,.2f}           │")
    print(f"├{'─'*58}┤")
    print(f"│  Weekly Return:        {weekly_return:>+15.2f}%               │")
    print(f"│  Monthly Return:       {monthly_return:>+15.2f}%               │")
    print(f"│  Quarterly Return:     {quarterly_return:>+15.2f}%               │")
    print(f"├{'─'*58}┤")
    print(f"│  52-Week High:         ₹{fifty_two_week_high:>15,.2f}               │")
    print(f"│  52-Week Low:          ₹{fifty_two_week_low:>15,.2f}               │")
    print(f"│  Position in Range:    {position_in_range:>15.1f}%               │")
    if market_cap > 0:
        print(f"│  Market Cap:           ₹{market_cap/1e9:>12,.0f} Bn               │")
    if pe_ratio > 0:
        print(f"│  P/E Ratio:            {pe_ratio:>18.2f}               │")
    print(f"└{'─'*58}┘")
    
    # ============================================
    # TECHNICAL INDICATORS
    # ============================================
    print(f"\n┌{'─'*58}┐")
    print(f"│{'TECHNICAL INDICATORS':^58}│")
    print(f"├{'─'*58}┤")
    
    # Compute features
    feature_set = feature_engine.compute_features(df)
    features_df = feature_set.features
    latest = features_df.iloc[-1]
    
    # RSI
    rsi = latest.get('rsi_14', 50)
    rsi_signal = "OVERSOLD ?" if rsi < 30 else "OVERBOUGHT ?" if rsi > 70 else "NEUTRAL ⚪"
    print(f"│  RSI (14):             {rsi:>15.2f}  {rsi_signal:>10}│")
    
    # MACD
    macd = latest.get('macd', 0)
    macd_signal_line = latest.get('macd_signal', 0)
    macd_status = "BULLISH ?" if macd > macd_signal_line else "BEARISH ?"
    print(f"│  MACD:                 {macd:>+15.2f}  {macd_status:>10}│")
    
    # SMA comparison
    sma_20 = latest.get('sma_20', current_price)
    sma_50 = latest.get('sma_50', current_price)
    sma_status = "ABOVE SMA ?" if current_price > sma_20 else "BELOW SMA ?"
    print(f"│  SMA 20:               ₹{sma_20:>14,.2f}  {sma_status:>10}│")
    print(f"│  SMA 50:               ₹{sma_50:>14,.2f}               │")
    
    # Bollinger Bands
    bb_upper = latest.get('bb_upper', current_price * 1.02)
    bb_lower = latest.get('bb_lower', current_price * 0.98)
    bb_width = latest.get('bb_width', 0.04)
    if current_price > bb_upper:
        bb_status = "ABOVE BAND ?"
    elif current_price < bb_lower:
        bb_status = "BELOW BAND ?"
    else:
        bb_status = "IN BAND ⚪"
    print(f"│  Bollinger Upper:      ₹{bb_upper:>14,.2f}               │")
    print(f"│  Bollinger Lower:      ₹{bb_lower:>14,.2f}  {bb_status:>10}│")
    
    # Volatility
    volatility = latest.get('volatility_20', 0) * 100
    vol_status = "HIGH ⚠️" if volatility > 3 else "LOW ✅" if volatility < 1.5 else "MEDIUM"
    print(f"│  Volatility (20d):     {volatility:>15.2f}%  {vol_status:>10}│")
    
    print(f"└{'─'*58}┘")
    
    # ============================================
    # TRADING SIGNALS
    # ============================================
    print(f"\n┌{'─'*58}┐")
    print(f"│{'TRADING SIGNALS':^58}│")
    print(f"├{'─'*58}┤")
    
    # Generate signals from alpha models
    alpha_output = alpha_engine.generate_signals(features_df, symbol)
    
    # Individual strategies
    strategy_signals = {}
    for signal in alpha_output.signals:
        strategy_name = signal.strategy_name
        strategy_signals[strategy_name] = signal
        
        if signal.signal_type.value == 'buy':
            sig_icon = "? BUY "
        elif signal.signal_type.value == 'sell':
            sig_icon = "? SELL"
        else:
            sig_icon = "⚪ HOLD"
        
        conf_bar = "█" * int(signal.confidence * 10) + "░" * (10 - int(signal.confidence * 10))
        print(f"│  {strategy_name:20s} {sig_icon}  [{conf_bar}] {signal.confidence*100:>3.0f}%│")
    
    print(f"├{'─'*58}┤")
    
    # Combined recommendation
    buy_count = sum(1 for s in strategy_signals.values() if s.signal_type.value == 'buy')
    sell_count = sum(1 for s in strategy_signals.values() if s.signal_type.value == 'sell')
    hold_count = sum(1 for s in strategy_signals.values() if s.signal_type.value == 'neutral')
    
    avg_confidence = np.mean([s.confidence for s in strategy_signals.values()])
    
    # Determine final recommendation
    if buy_count > sell_count and buy_count > hold_count and avg_confidence > 0.5:
        recommendation = "? BUY"
        rec_detail = "Multiple strategies suggest buying"
    elif sell_count > buy_count and sell_count > hold_count and avg_confidence > 0.5:
        recommendation = "? SELL"
        rec_detail = "Multiple strategies suggest selling"
    elif buy_count > sell_count:
        recommendation = "? WEAK BUY"
        rec_detail = "Slight bullish bias, proceed with caution"
    elif sell_count > buy_count:
        recommendation = "? WEAK SELL"
        rec_detail = "Slight bearish bias, consider reducing"
    else:
        recommendation = "⚪ HOLD"
        rec_detail = "No clear direction, wait for clarity"
    
    print(f"│{'':58}│")
    print(f"│  {'RECOMMENDATION:':20s} {recommendation:>34}│")
    print(f"│  {rec_detail:^56}│")
    print(f"│{'':58}│")
    print(f"│  Strategies: {buy_count} BUY | {sell_count} SELL | {hold_count} HOLD                 │")
    print(f"│  Avg Confidence: {avg_confidence*100:.1f}%                                    │")
    print(f"└{'─'*58}┘")
    
    # ============================================
    # RISK ASSESSMENT
    # ============================================
    print(f"\n┌{'─'*58}┐")
    print(f"│{'RISK ASSESSMENT':^58}│")
    print(f"├{'─'*58}┤")
    
    # Support/Resistance levels
    support = bb_lower
    resistance = bb_upper
    
    # Risk/Reward
    risk_per_share = current_price - support
    reward_per_share = resistance - current_price
    risk_reward = reward_per_share / risk_per_share if risk_per_share > 0 else 0
    
    rr_status = "FAVORABLE ✅" if risk_reward > 2 else "ACCEPTABLE" if risk_reward > 1 else "POOR ⚠️"
    
    print(f"│  Support Level:        ₹{support:>14,.2f}               │")
    print(f"│  Resistance Level:     ₹{resistance:>14,.2f}               │")
    print(f"│  Risk/Reward Ratio:    {risk_reward:>15.2f}  {rr_status:>10}│")
    print(f"│  Suggested Stop-Loss:  ₹{support * 0.98:>14,.2f}               │")
    print(f"│  Suggested Target:     ₹{resistance * 1.02:>14,.2f}               │")
    print(f"└{'─'*58}┘")
    
    # ============================================
    # ML PREDICTIONS (Jim Simons Style)
    # ============================================
    print(f"\n┌{'─'*58}┐")
    print(f"│{'? ML PREDICTIONS (Renaissance Style)':^58}│")
    print(f"├{'─'*58}┤")
    
    try:
        ml_predictor = MLPredictor()
        
        # 5-day prediction
        pred_5d = ml_predictor.predict(df, symbol, horizon='5d')
        
        # Direction indicator
        if pred_5d.predicted_return > 0.01:
            ml_direction = "? BULLISH"
        elif pred_5d.predicted_return < -0.01:
            ml_direction = "? BEARISH"
        else:
            ml_direction = "➡️ NEUTRAL"
        
        print(f"│  {'5-Day Forecast:':25} {ml_direction:>30}│")
        print(f"│  Predicted Price:       ₹{pred_5d.predicted_price:>14,.2f}               │")
        print(f"│  Expected Return:       {pred_5d.predicted_return:>+14.2%}               │")
        print(f"│  95% Confidence Range:  ₹{pred_5d.confidence_interval[0]:>6,.0f} - ₹{pred_5d.confidence_interval[1]:>6,.0f}        │")
        print(f"├{'─'*58}┤")
        print(f"│  {'PROBABILITY ANALYSIS':^56}│")
        print(f"│  Prob Stock Goes UP:    {pred_5d.prob_up:>14.1%}               │")
        print(f"│  Prob Stock Goes DOWN:  {pred_5d.prob_down:>14.1%}               │")
        print(f"│  Prob Stays SIDEWAYS:   {pred_5d.prob_sideways:>14.1%}               │")
        print(f"├{'─'*58}┤")
        print(f"│  {'MARKET REGIME':^56}│")
        
        regime_emoji = {
            'bull_quiet': '? Bull (Calm)',
            'bull_volatile': '? Bull (Volatile)',
            'bear_quiet': '? Bear (Calm)',
            'bear_volatile': '? Bear (Volatile)',
            'sideways': '⚪ Sideways',
            'crisis': '? Crisis Mode'
        }
        regime_str = regime_emoji.get(pred_5d.regime.value, pred_5d.regime.value)
        print(f"│  Current Regime:        {regime_str:>30}│")
        print(f"│  Pattern Detected:      {pred_5d.pattern_detected[:30]:>30}│")
        print(f"│  Anomaly Score:         {pred_5d.anomaly_score:>14.2f}               │")
        print(f"├{'─'*58}┤")
        print(f"│  {'MODEL ENSEMBLE VOTES':^56}│")
        
        for model_name, pred_val in pred_5d.model_predictions.items():
            vote = "?" if pred_val > 0.005 else "?" if pred_val < -0.005 else "➡️"
            print(f"│    {model_name:20s}: {vote} {pred_val:>+10.2%}                │")
        
        print(f"├{'─'*58}┤")
        print(f"│  {'RISK METRICS':^56}│")
        print(f"│  Predicted Volatility:  {pred_5d.predicted_volatility:>14.1%}               │")
        print(f"│  Value at Risk (95%):   ₹{pred_5d.var_95:>14,.2f}               │")
        print(f"│  Expected Sharpe:       {pred_5d.expected_sharpe:>14.2f}               │")
        print(f"├{'─'*58}┤")
        
        # ML-based recommendation
        if pred_5d.prob_up > 0.6 and pred_5d.signal_strength == 'strong':
            ml_rec = "? STRONG BUY"
            ml_detail = "ML models agree: High probability of upside"
        elif pred_5d.prob_up > 0.55 and pred_5d.predicted_return > 0.01:
            ml_rec = "? MODERATE BUY"
            ml_detail = "Positive outlook with moderate confidence"
        elif pred_5d.prob_down > 0.6 and pred_5d.signal_strength == 'strong':
            ml_rec = "? STRONG SELL"
            ml_detail = "ML models agree: High probability of downside"
        elif pred_5d.prob_down > 0.55 and pred_5d.predicted_return < -0.01:
            ml_rec = "? MODERATE SELL"
            ml_detail = "Negative outlook with moderate confidence"
        else:
            ml_rec = "⚪ WAIT"
            ml_detail = "No clear edge - patience recommended"
        
        print(f"│  {'ML RECOMMENDATION:':20s} {ml_rec:>34}│")
        print(f"│  {ml_detail:^56}│")
        print(f"│  Signal Strength:       {pred_5d.signal_strength.upper():>30}│")
        print(f"│  Overall Confidence:    {pred_5d.overall_confidence:>14.1%}               │")
        print(f"└{'─'*58}┘")
        
        # Store ML prediction for return
        ml_prediction = {
            'predicted_price': pred_5d.predicted_price,
            'predicted_return': pred_5d.predicted_return,
            'prob_up': pred_5d.prob_up,
            'prob_down': pred_5d.prob_down,
            'regime': pred_5d.regime.value,
            'ml_recommendation': ml_rec
        }
        
    except Exception as e:
        print(f"│  ⚠️  ML Prediction unavailable: {str(e)[:25]}      │")
        print(f"└{'─'*58}┘")
        ml_prediction = None
    
    # Summary
    print(f"\n{'='*60}")
    if "BUY" in recommendation:
        print(f"  ? SUMMARY: Consider BUYING {symbol}")
        print(f"     Entry: ₹{current_price:,.2f} | Stop: ₹{support*0.98:,.2f} | Target: ₹{resistance*1.02:,.2f}")
    elif "SELL" in recommendation:
        print(f"  ? SUMMARY: Consider SELLING {symbol}")
        print(f"     Exit around ₹{current_price:,.2f}")
    else:
        print(f"  ? SUMMARY: HOLD {symbol} - Wait for clear signal")
    print(f"{'='*60}\n")
    
    result = {
        'symbol': symbol,
        'price': current_price,
        'recommendation': recommendation,
        'confidence': avg_confidence,
        'rsi': rsi,
        'support': support,
        'resistance': resistance
    }
    
    if ml_prediction:
        result['ml_prediction'] = ml_prediction
    
    return result


def main():
    """Main function to analyze stocks."""
    
    print("\n" + "="*60)
    print("  ? STOCK PERFORMANCE ANALYZER")
    print("  Quantitative Analysis for Indian Stocks")
    print("="*60)
    
    # Default stocks to analyze
    if len(sys.argv) > 1:
        symbols = [s.upper() for s in sys.argv[1:]]
    else:
        # Interactive mode
        print("\nEnter stock symbols (comma-separated) or press Enter for defaults:")
        print("Examples: RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK")
        user_input = input("\n> ").strip()
        
        if user_input:
            symbols = [s.strip().upper() for s in user_input.split(',')]
        else:
            symbols = ['RELIANCE', 'TCS', 'INFY']
    
    print(f"\n? Analyzing: {', '.join(symbols)}")
    
    results = []
    for symbol in symbols:
        result = analyze_stock(symbol, use_real_data=True)
        if result:
            results.append(result)
    
    # Summary table
    if len(results) > 1:
        print("\n" + "="*60)
        print("  ? SUMMARY TABLE")
        print("="*60)
        print(f"\n{'Symbol':<12} {'Price':>12} {'Recommendation':<15} {'RSI':>8}")
        print("-"*50)
        for r in results:
            print(f"{r['symbol']:<12} ₹{r['price']:>10,.2f} {r['recommendation']:<15} {r['rsi']:>7.1f}")
        print("-"*50)


if __name__ == "__main__":
    main()
