"""
Professional NIFTY 50 F&O Signal Generator
==========================================

Expert Indian equity derivatives trader logic for
NIFTY 50 intraday options trading.

Core Principles:
- Capital protection > profits
- NO TRADE over weak setups
- Trade only clear trends or strong breakouts
- Avoid sideways, low-volume, choppy markets
- Never overtrade

Trading Constraints:
- Market: NSE India
- Instrument: NIFTY 50 Index
- Style: Intraday options buying (CE / PE)
- Timeframe: 5-minute candles
- Trading window: 9:20 AM – 2:45 PM IST
- Max signals per day: 2
- Risk per trade: ≤ 1% of capital
"""

import numpy as np
import pandas as pd
from datetime import datetime, time
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from enum import Enum


class MarketTrend(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    SIDEWAYS = "SIDEWAYS"


class SignalType(Enum):
    BUY_CALL = "BUY CALL (CE)"
    BUY_PUT = "BUY PUT (PE)"
    NO_TRADE = "NO TRADE"


class Confidence(Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class StrikeType(Enum):
    ATM = "ATM"
    ITM = "ITM"
    OTM = "OTM"


@dataclass
class TradingSignal:
    """Professional trading signal output."""
    market_trend: MarketTrend
    signal: SignalType
    option_type: Optional[str]
    strike_price: Optional[int]
    strike_type: Optional[StrikeType]
    entry: Optional[str]
    stop_loss: Optional[float]
    target: Optional[float]
    risk_reward_ratio: Optional[str]
    confidence: Confidence
    reasoning: List[str]
    timestamp: datetime
    
    # Additional data
    spot_price: float
    ema_20: float
    ema_50: float
    rsi: float
    vwap: float
    volume_status: str
    
    def to_dict(self) -> dict:
        return {
            'market_trend': self.market_trend.value,
            'signal': self.signal.value,
            'option_type': self.option_type,
            'strike_price': self.strike_price,
            'strike_type': self.strike_type.value if self.strike_type else None,
            'entry': self.entry,
            'stop_loss': self.stop_loss,
            'target': self.target,
            'risk_reward_ratio': self.risk_reward_ratio,
            'confidence': self.confidence.value,
            'reasoning': self.reasoning,
            'timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'spot_price': self.spot_price,
            'ema_20': self.ema_20,
            'ema_50': self.ema_50,
            'rsi': self.rsi,
            'vwap': self.vwap,
            'volume_status': self.volume_status
        }
    
    def format_output(self) -> str:
        """Format signal in strict output format."""
        output = f"""
╔══════════════════════════════════════════════════════════════╗
║           NIFTY 50 F&O SIGNAL - {self.timestamp.strftime('%d %b %Y %H:%M')}          ║
╠══════════════════════════════════════════════════════════════╣
║  Market Trend:        {self.market_trend.value:<35} ║
║  Signal:              {self.signal.value:<35} ║"""
        
        if self.signal != SignalType.NO_TRADE:
            output += f"""
║  Option Type:         {self.option_type:<35} ║
║  Strike Price:        {str(self.strike_price) + ' ' + self.strike_type.value:<35} ║
║  Entry:               {self.entry:<35} ║
║  Stop Loss:           ₹{self.stop_loss:<34} ║
║  Target:              ₹{self.target:<34} ║
║  Risk-Reward Ratio:   {self.risk_reward_ratio:<35} ║"""
        
        output += f"""
║  Confidence:          {self.confidence.value:<35} ║
╠══════════════════════════════════════════════════════════════╣
║  REASONING:                                                  ║"""
        
        for i, reason in enumerate(self.reasoning[:3], 1):
            output += f"\n║  {i}. {reason:<55} ║"
        
        output += """
╚══════════════════════════════════════════════════════════════╝"""
        
        return output


class NiftyFnOSignalGenerator:
    """
    Professional NIFTY 50 F&O Signal Generator.
    
    Generates high-probability BUY signals with strict risk management.
    """
    
    def __init__(self):
        # Trading constraints
        self.market_open = time(9, 20)
        self.market_close = time(14, 45)
        self.max_signals_per_day = 2
        self.risk_per_trade = 0.01  # 1% of capital
        
        # Signal tracking
        self.signals_today = 0
        self.last_signal_date = None
        
        # Thresholds
        self.rsi_oversold = 35
        self.rsi_overbought = 65
        self.rsi_extreme_oversold = 25
        self.rsi_extreme_overbought = 75
        
    def is_trading_hours(self) -> bool:
        """Check if current time is within trading window."""
        now = datetime.now().time()
        return self.market_open <= now <= self.market_close
    
    def reset_daily_counters(self):
        """Reset daily signal counter."""
        today = datetime.now().date()
        if self.last_signal_date != today:
            self.signals_today = 0
            self.last_signal_date = today
    
    def calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return prices[-1]
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price - ema) * multiplier + ema
        
        return ema
    
    def calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate Relative Strength Index."""
        if len(prices) < period + 1:
            return 50
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return round(rsi, 2)
    
    def calculate_vwap(self, df: pd.DataFrame) -> float:
        """Calculate Volume Weighted Average Price."""
        if 'volume' not in df.columns or df['volume'].sum() == 0:
            return df['close'].iloc[-1]
        
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).sum() / df['volume'].sum()
        
        return round(vwap, 2)
    
    def analyze_volume(self, df: pd.DataFrame) -> str:
        """Analyze volume vs average."""
        if 'volume' not in df.columns:
            return "Normal"
        
        recent_vol = df['volume'].iloc[-5:].mean()
        avg_vol = df['volume'].mean()
        
        ratio = recent_vol / avg_vol if avg_vol > 0 else 1
        
        if ratio > 1.5:
            return "High"
        elif ratio < 0.7:
            return "Low"
        else:
            return "Normal"
    
    def analyze_candle_pattern(self, df: pd.DataFrame) -> Tuple[str, str]:
        """Analyze last 2 candle patterns."""
        if len(df) < 2:
            return "Insufficient data", "Insufficient data"
        
        def describe_candle(row):
            body = row['close'] - row['open']
            range_size = row['high'] - row['low']
            body_pct = abs(body) / range_size * 100 if range_size > 0 else 0
            
            if body > 0:
                direction = "Bullish"
            elif body < 0:
                direction = "Bearish"
            else:
                direction = "Doji"
            
            if body_pct > 70:
                size = "Strong"
            elif body_pct > 40:
                size = "Normal"
            else:
                size = "Weak/Indecisive"
            
            # Check for specific patterns
            upper_wick = row['high'] - max(row['open'], row['close'])
            lower_wick = min(row['open'], row['close']) - row['low']
            
            if lower_wick > abs(body) * 2 and upper_wick < abs(body) * 0.5:
                return f"{direction} Hammer"
            elif upper_wick > abs(body) * 2 and lower_wick < abs(body) * 0.5:
                return f"{direction} Shooting Star"
            else:
                return f"{size} {direction}"
        
        current = describe_candle(df.iloc[-1])
        previous = describe_candle(df.iloc[-2])
        
        return current, previous
    
    def determine_market_trend(self, spot: float, ema_20: float, ema_50: float,
                                rsi: float, vwap: float) -> MarketTrend:
        """Determine market regime based on multiple indicators."""
        bullish_signals = 0
        bearish_signals = 0
        
        # EMA alignment
        if ema_20 > ema_50:
            bullish_signals += 1
        elif ema_20 < ema_50:
            bearish_signals += 1
        
        # Price vs EMA 20
        if spot > ema_20:
            bullish_signals += 1
        elif spot < ema_20:
            bearish_signals += 1
        
        # Price vs VWAP
        if spot > vwap:
            bullish_signals += 1
        elif spot < vwap:
            bearish_signals += 1
        
        # RSI trend
        if rsi > 55:
            bullish_signals += 1
        elif rsi < 45:
            bearish_signals += 1
        
        # Determine trend
        if bullish_signals >= 3 and bearish_signals <= 1:
            return MarketTrend.BULLISH
        elif bearish_signals >= 3 and bullish_signals <= 1:
            return MarketTrend.BEARISH
        else:
            return MarketTrend.SIDEWAYS
    
    def select_strike_price(self, spot: float, signal_type: SignalType) -> Tuple[int, StrikeType]:
        """Select optimal strike price."""
        # Round to nearest 50
        atm_strike = round(spot / 50) * 50
        
        if signal_type == SignalType.BUY_CALL:
            # For momentum, use ATM or slightly OTM
            strike = atm_strike
            strike_type = StrikeType.ATM
        elif signal_type == SignalType.BUY_PUT:
            strike = atm_strike
            strike_type = StrikeType.ATM
        else:
            strike = atm_strike
            strike_type = StrikeType.ATM
        
        return strike, strike_type
    
    def calculate_targets(self, spot: float, signal_type: SignalType,
                          atr: float) -> Tuple[float, float, str]:
        """Calculate stop loss and target based on ATR."""
        if signal_type == SignalType.BUY_CALL:
            # For CE: SL below support, Target above
            stop_loss = round(spot - atr * 1.2, 2)
            target = round(spot + atr * 2.5, 2)
        elif signal_type == SignalType.BUY_PUT:
            # For PE: SL above resistance, Target below
            stop_loss = round(spot + atr * 1.2, 2)
            target = round(spot - atr * 2.5, 2)
        else:
            return None, None, None
        
        risk = abs(spot - stop_loss)
        reward = abs(target - spot)
        rr_ratio = round(reward / risk, 2) if risk > 0 else 0
        
        return stop_loss, target, f"1:{rr_ratio}"
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        if len(df) < period:
            return (df['high'].iloc[-1] - df['low'].iloc[-1])
        
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        tr = np.maximum(high[1:] - low[1:],
                       np.maximum(abs(high[1:] - close[:-1]),
                                 abs(low[1:] - close[:-1])))
        
        atr = np.mean(tr[-period:])
        return round(atr, 2)
    
    def generate_signal(self, df: pd.DataFrame) -> TradingSignal:
        """
        Generate trading signal based on market data.
        
        Returns TradingSignal with BUY CALL, BUY PUT, or NO TRADE.
        """
        self.reset_daily_counters()
        
        if df is None or len(df) < 50:
            return self._no_trade_signal(
                reason=["Insufficient data for analysis"]
            )
        
        # Extract current data
        spot = round(df['close'].iloc[-1], 2)
        open_price = round(df['open'].iloc[-1], 2)
        high = round(df['high'].iloc[-1], 2)
        low = round(df['low'].iloc[-1], 2)
        
        # Calculate indicators
        close_prices = df['close'].values
        ema_20 = round(self.calculate_ema(close_prices, 20), 2)
        ema_50 = round(self.calculate_ema(close_prices, 50), 2)
        rsi = self.calculate_rsi(close_prices, 14)
        vwap = self.calculate_vwap(df)
        volume_status = self.analyze_volume(df)
        atr = self.calculate_atr(df)
        
        # Analyze candle patterns
        current_candle, prev_candle = self.analyze_candle_pattern(df)
        
        # Determine market trend
        trend = self.determine_market_trend(spot, ema_20, ema_50, rsi, vwap)
        
        # Check trading constraints
        reasons_no_trade = []
        
        # 1. Check trading hours
        if not self.is_trading_hours():
            reasons_no_trade.append("Outside trading hours (9:20 AM - 2:45 PM)")
        
        # 2. Check max signals per day
        if self.signals_today >= self.max_signals_per_day:
            reasons_no_trade.append(f"Max signals reached ({self.max_signals_per_day}/day)")
        
        # 3. Check for sideways market
        if trend == MarketTrend.SIDEWAYS:
            reasons_no_trade.append("Market is SIDEWAYS - No clear trend")
        
        # 4. Check volume
        if volume_status == "Low":
            reasons_no_trade.append("Volume too low - Avoid low liquidity")
        
        # 5. Check RSI extremes (avoid chasing)
        if 40 <= rsi <= 60:
            reasons_no_trade.append(f"RSI neutral ({rsi}) - No momentum confirmation")
        
        # Generate signal based on conditions
        signal_type = SignalType.NO_TRADE
        confidence = Confidence.LOW
        reasoning = []
        
        # ═══════════════════════════════════════════════════════════
        # BULLISH SIGNAL CONDITIONS
        # ═══════════════════════════════════════════════════════════
        if trend == MarketTrend.BULLISH and len(reasons_no_trade) == 0:
            bullish_score = 0
            
            # Condition 1: Price above both EMAs
            if spot > ema_20 > ema_50:
                bullish_score += 2
                reasoning.append(f"Price ({spot}) > EMA20 ({ema_20}) > EMA50 ({ema_50}) - Strong uptrend")
            
            # Condition 2: RSI showing strength but not overbought
            if self.rsi_extreme_oversold < rsi < self.rsi_overbought:
                if rsi > 55:
                    bullish_score += 1
                    reasoning.append(f"RSI ({rsi}) showing bullish momentum")
            elif rsi <= self.rsi_oversold:
                bullish_score += 2
                reasoning.append(f"RSI ({rsi}) oversold - Potential reversal")
            
            # Condition 3: Price above VWAP
            if spot > vwap:
                bullish_score += 1
                reasoning.append(f"Price above VWAP ({vwap}) - Institutional buying")
            
            # Condition 4: Volume confirmation
            if volume_status == "High":
                bullish_score += 1
                reasoning.append("High volume confirming move")
            
            # Condition 5: Bullish candle pattern
            if "Bullish" in current_candle and "Strong" in current_candle:
                bullish_score += 1
                reasoning.append(f"Strong bullish candle pattern")
            
            # Determine confidence and generate signal
            if bullish_score >= 5:
                signal_type = SignalType.BUY_CALL
                confidence = Confidence.HIGH
            elif bullish_score >= 4:
                signal_type = SignalType.BUY_CALL
                confidence = Confidence.MEDIUM
            elif bullish_score >= 3:
                signal_type = SignalType.BUY_CALL
                confidence = Confidence.LOW
        
        # ═══════════════════════════════════════════════════════════
        # BEARISH SIGNAL CONDITIONS
        # ═══════════════════════════════════════════════════════════
        elif trend == MarketTrend.BEARISH and len(reasons_no_trade) == 0:
            bearish_score = 0
            
            # Condition 1: Price below both EMAs
            if spot < ema_20 < ema_50:
                bearish_score += 2
                reasoning.append(f"Price ({spot}) < EMA20 ({ema_20}) < EMA50 ({ema_50}) - Strong downtrend")
            
            # Condition 2: RSI showing weakness but not oversold
            if self.rsi_oversold < rsi < self.rsi_extreme_overbought:
                if rsi < 45:
                    bearish_score += 1
                    reasoning.append(f"RSI ({rsi}) showing bearish momentum")
            elif rsi >= self.rsi_overbought:
                bearish_score += 2
                reasoning.append(f"RSI ({rsi}) overbought - Potential reversal")
            
            # Condition 3: Price below VWAP
            if spot < vwap:
                bearish_score += 1
                reasoning.append(f"Price below VWAP ({vwap}) - Institutional selling")
            
            # Condition 4: Volume confirmation
            if volume_status == "High":
                bearish_score += 1
                reasoning.append("High volume confirming move")
            
            # Condition 5: Bearish candle pattern
            if "Bearish" in current_candle and "Strong" in current_candle:
                bearish_score += 1
                reasoning.append(f"Strong bearish candle pattern")
            
            # Determine confidence and generate signal
            if bearish_score >= 5:
                signal_type = SignalType.BUY_PUT
                confidence = Confidence.HIGH
            elif bearish_score >= 4:
                signal_type = SignalType.BUY_PUT
                confidence = Confidence.MEDIUM
            elif bearish_score >= 3:
                signal_type = SignalType.BUY_PUT
                confidence = Confidence.LOW
        
        # ═══════════════════════════════════════════════════════════
        # NO TRADE - Conditions not met
        # ═══════════════════════════════════════════════════════════
        if signal_type == SignalType.NO_TRADE:
            if len(reasons_no_trade) > 0:
                reasoning = reasons_no_trade[:3]
            else:
                reasoning = [
                    "Multiple indicators not aligned",
                    "Waiting for clearer setup",
                    "Capital protection - No trade is better than weak trade"
                ]
            
            return TradingSignal(
                market_trend=trend,
                signal=SignalType.NO_TRADE,
                option_type=None,
                strike_price=None,
                strike_type=None,
                entry=None,
                stop_loss=None,
                target=None,
                risk_reward_ratio=None,
                confidence=Confidence.LOW,
                reasoning=reasoning[:3],
                timestamp=datetime.now(),
                spot_price=spot,
                ema_20=ema_20,
                ema_50=ema_50,
                rsi=rsi,
                vwap=vwap,
                volume_status=volume_status
            )
        
        # Generate trade details
        strike, strike_type = self.select_strike_price(spot, signal_type)
        stop_loss, target, rr_ratio = self.calculate_targets(spot, signal_type, atr)
        
        # Entry logic
        if signal_type == SignalType.BUY_CALL:
            entry = f"Enter on breakout above {round(high + 5, 0)} or pullback to EMA20"
            option_type = "CE (Call Option)"
        else:
            entry = f"Enter on breakdown below {round(low - 5, 0)} or pullback to EMA20"
            option_type = "PE (Put Option)"
        
        # Increment signal counter
        self.signals_today += 1
        
        return TradingSignal(
            market_trend=trend,
            signal=signal_type,
            option_type=option_type,
            strike_price=strike,
            strike_type=strike_type,
            entry=entry,
            stop_loss=stop_loss,
            target=target,
            risk_reward_ratio=rr_ratio,
            confidence=confidence,
            reasoning=reasoning[:3],
            timestamp=datetime.now(),
            spot_price=spot,
            ema_20=ema_20,
            ema_50=ema_50,
            rsi=rsi,
            vwap=vwap,
            volume_status=volume_status
        )
    
    def _no_trade_signal(self, reason: List[str]) -> TradingSignal:
        """Generate a NO TRADE signal."""
        return TradingSignal(
            market_trend=MarketTrend.SIDEWAYS,
            signal=SignalType.NO_TRADE,
            option_type=None,
            strike_price=None,
            strike_type=None,
            entry=None,
            stop_loss=None,
            target=None,
            risk_reward_ratio=None,
            confidence=Confidence.LOW,
            reasoning=reason,
            timestamp=datetime.now(),
            spot_price=0,
            ema_20=0,
            ema_50=0,
            rsi=50,
            vwap=0,
            volume_status="Unknown"
        )


# Export
__all__ = ['NiftyFnOSignalGenerator', 'TradingSignal', 'MarketTrend', 'SignalType', 'Confidence']

