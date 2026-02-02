"""
Professional NIFTY 50 F&O Signal Generator - Enhanced Version
==============================================================

Expert Indian equity derivatives trader logic for
NIFTY 50 intraday options trading with advanced indicators.

Enhanced Features:
- Multi-indicator confluence (MACD, BB, ADX, SuperTrend)
- Market regime detection
- Volatility-adjusted targets
- Time-decay awareness for options
- Support/Resistance detection
- Scoring system (0-100)

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
from datetime import datetime, time, timedelta
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class MarketTrend(Enum):
    STRONG_BULLISH = "STRONG BULLISH"
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    STRONG_BEARISH = "STRONG BEARISH"
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


class MarketRegime(Enum):
    """Market regime classification."""
    TRENDING_UP = "Trending Up"
    TRENDING_DOWN = "Trending Down"
    RANGING = "Ranging"
    VOLATILE = "Volatile"
    BREAKOUT = "Breakout"


@dataclass
class MarketIndicators:
    """Container for all calculated indicators."""
    spot: float
    open: float
    high: float
    low: float
    
    # Moving Averages
    ema_9: float
    ema_20: float
    ema_50: float
    sma_200: float
    
    # Momentum
    rsi: float
    rsi_sma: float
    macd: float
    macd_signal: float
    macd_histogram: float
    
    # Volatility
    atr: float
    atr_percent: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_width: float
    bb_position: float
    
    # Trend
    adx: float
    plus_di: float
    minus_di: float
    supertrend: float
    supertrend_direction: int
    
    # Volume
    vwap: float
    volume_sma: float
    volume_ratio: float
    obv_trend: str
    
    # Support/Resistance
    pivot: float
    support_1: float
    support_2: float
    resistance_1: float
    resistance_2: float
    
    # Pattern
    current_candle: str
    prev_candle: str
    
    # Regime
    regime: MarketRegime


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
    target_2: Optional[float] = None
    risk_reward_ratio: Optional[str] = None
    confidence: Confidence = Confidence.LOW
    reasoning: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Indicator data
    spot_price: float = 0.0
    ema_20: float = 0.0
    ema_50: float = 0.0
    rsi: float = 50.0
    vwap: float = 0.0
    volume_status: str = "Normal"
    
    # Enhanced data
    macd_signal_val: float = 0.0
    adx: float = 0.0
    bb_position: float = 50.0
    supertrend_dir: int = 0
    atr: float = 0.0
    regime: str = "Unknown"
    score: int = 0
    
    # NEW: Scalping & VWAP data
    scalping_signal: str = "WAIT"
    scalping_entry: float = 0.0
    scalping_sl: float = 0.0
    scalping_target: float = 0.0
    vwap_signal: str = "NEUTRAL"
    candle_pattern: str = "None"
    risk_level: str = "Medium"
    trade_type: str = "Swing"  # Swing or Scalp
    
    # NEW: Ahuja Rafale Indicator
    ahuja_rafale: dict = field(default_factory=lambda: {
        'signal': 'WAIT', 'momentum': 0, 'trend': 'Neutral', 
        'power': 0, 'strength': 0, 'color': 'gray', 'description': 'Loading...'
    })
    
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
            'target_2': self.target_2,
            'risk_reward_ratio': self.risk_reward_ratio,
            'confidence': self.confidence.value,
            'reasoning': self.reasoning,
            'timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'spot_price': self.spot_price,
            'ema_20': self.ema_20,
            'ema_50': self.ema_50,
            'rsi': self.rsi,
            'vwap': self.vwap,
            'volume_status': self.volume_status,
            'macd_signal': self.macd_signal_val,
            'adx': self.adx,
            'bb_position': self.bb_position,
            'supertrend_direction': self.supertrend_dir,
            'atr': self.atr,
            'regime': self.regime,
            'score': self.score,
            # NEW fields
            'scalping_signal': self.scalping_signal,
            'scalping_entry': self.scalping_entry,
            'scalping_sl': self.scalping_sl,
            'scalping_target': self.scalping_target,
            'vwap_signal': self.vwap_signal,
            'candle_pattern': self.candle_pattern,
            'risk_level': self.risk_level,
            'trade_type': self.trade_type,
            # Ahuja Rafale
            'ahuja_rafale': self.ahuja_rafale
        }


class TechnicalIndicators:
    """Optimized technical indicator calculations using numpy."""
    
    @staticmethod
    def ema(prices: np.ndarray, period: int) -> np.ndarray:
        """Exponential Moving Average."""
        if len(prices) < period:
            return np.full(len(prices), prices[-1] if len(prices) > 0 else 0)
        
        multiplier = 2 / (period + 1)
        ema_values = np.zeros(len(prices))
        ema_values[0] = prices[0]
        
        for i in range(1, len(prices)):
            ema_values[i] = (prices[i] - ema_values[i-1]) * multiplier + ema_values[i-1]
        
        return ema_values
    
    @staticmethod
    def rsi(prices: np.ndarray, period: int = 14) -> Tuple[float, float]:
        """RSI with smoothed RSI using Wilder's smoothing."""
        if len(prices) < period + 1:
            return 50.0, 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.zeros(len(gains))
        avg_loss = np.zeros(len(gains))
        
        avg_gain[period-1] = np.mean(gains[:period])
        avg_loss[period-1] = np.mean(losses[:period])
        
        for i in range(period, len(gains)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gains[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + losses[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi_values = 100 - (100 / (1 + rs))
        
        current_rsi = round(rsi_values[-1], 2)
        rsi_sma = round(np.mean(rsi_values[-5:]), 2) if len(rsi_values) >= 5 else current_rsi
        
        return current_rsi, rsi_sma
    
    @staticmethod
    def macd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
        """MACD with signal line and histogram."""
        if len(prices) < slow + signal:
            return 0.0, 0.0, 0.0
        
        ema_fast = TechnicalIndicators.ema(prices, fast)
        ema_slow = TechnicalIndicators.ema(prices, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return round(macd_line[-1], 2), round(signal_line[-1], 2), round(histogram[-1], 2)
    
    @staticmethod
    def bollinger_bands(prices: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Tuple[float, float, float, float, float]:
        """Bollinger Bands with width and position."""
        if len(prices) < period:
            return prices[-1], prices[-1], prices[-1], 0, 50
        
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        
        width = ((upper - lower) / sma) * 100 if sma > 0 else 0
        
        current_price = prices[-1]
        if upper != lower:
            position = ((current_price - lower) / (upper - lower)) * 100
            position = max(0, min(100, position))
        else:
            position = 50
        
        return round(upper, 2), round(sma, 2), round(lower, 2), round(width, 2), round(position, 2)
    
    @staticmethod
    def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> Tuple[float, float]:
        """Average True Range with percentage."""
        if len(high) < period + 1:
            atr_val = high[-1] - low[-1]
            return atr_val, (atr_val / close[-1]) * 100 if close[-1] > 0 else 0
        
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        
        atr_values = np.zeros(len(tr))
        atr_values[period-1] = np.mean(tr[:period])
        
        for i in range(period, len(tr)):
            atr_values[i] = (atr_values[i-1] * (period-1) + tr[i]) / period
        
        atr_val = round(atr_values[-1], 2)
        atr_pct = round((atr_val / close[-1]) * 100, 2) if close[-1] > 0 else 0
        
        return atr_val, atr_pct
    
    @staticmethod
    def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> Tuple[float, float, float]:
        """ADX with +DI and -DI."""
        if len(high) < period + 1:
            return 25.0, 25.0, 25.0
        
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        
        atr_smooth = TechnicalIndicators.ema(tr, period)
        plus_dm_smooth = TechnicalIndicators.ema(plus_dm, period)
        minus_dm_smooth = TechnicalIndicators.ema(minus_dm, period)
        
        plus_di = np.where(atr_smooth != 0, (plus_dm_smooth / atr_smooth) * 100, 0)
        minus_di = np.where(atr_smooth != 0, (minus_dm_smooth / atr_smooth) * 100, 0)
        
        di_sum = plus_di + minus_di
        dx = np.where(di_sum != 0, (np.abs(plus_di - minus_di) / di_sum) * 100, 0)
        adx_values = TechnicalIndicators.ema(dx, period)
        
        return round(adx_values[-1], 2), round(plus_di[-1], 2), round(minus_di[-1], 2)
    
    @staticmethod
    def supertrend(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                   period: int = 10, multiplier: float = 3.0) -> Tuple[float, int]:
        """SuperTrend indicator."""
        if len(high) < period + 1:
            return close[-1], 0
        
        atr_val, _ = TechnicalIndicators.atr(high, low, close, period)
        
        hl2 = (high + low) / 2
        
        upper_band = hl2[-1] + (multiplier * atr_val)
        lower_band = hl2[-1] - (multiplier * atr_val)
        
        if close[-1] > upper_band:
            direction = 1
            supertrend = lower_band
        elif close[-1] < lower_band:
            direction = -1
            supertrend = upper_band
        else:
            direction = 1 if close[-1] > close[-2] else -1 if len(close) >= 2 else 0
            supertrend = lower_band if direction == 1 else upper_band
        
        return round(supertrend, 2), direction
    
    @staticmethod
    def ahuja_rafale(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                     volume: np.ndarray, period: int = 14) -> dict:
        """
        Ahuja Rafale Indicator - Multi-factor momentum and trend indicator.
        
        Components:
        1. Rafale Momentum (RM) - Rate of change with volatility adjustment
        2. Rafale Trend (RT) - Adaptive moving average crossover
        3. Rafale Power (RP) - Volume-weighted strength indicator
        4. Rafale Signal - Combined buy/sell signal
        
        Returns dict with all components and final signal.
        """
        if len(close) < period * 2:
            return {
                'signal': 'WAIT',
                'momentum': 0,
                'trend': 'Neutral',
                'power': 0,
                'strength': 0,
                'color': 'gray',
                'description': 'Insufficient data'
            }
        
        # 1. Rafale Momentum (RM) - Smoothed ROC with ATR normalization
        roc = ((close[-1] - close[-period]) / close[-period]) * 100
        
        # ATR for normalization
        tr = np.maximum(high[1:] - low[1:], 
                       np.maximum(np.abs(high[1:] - close[:-1]), 
                                 np.abs(low[1:] - close[:-1])))
        atr = np.mean(tr[-period:])
        atr_pct = (atr / close[-1]) * 100
        
        # Normalized momentum
        rafale_momentum = roc / (atr_pct + 0.01)  # Avoid division by zero
        
        # 2. Rafale Trend (RT) - Triple EMA comparison
        ema_fast = TechnicalIndicators.ema(close, 8)[-1]
        ema_mid = TechnicalIndicators.ema(close, 21)[-1]
        ema_slow = TechnicalIndicators.ema(close, 55)[-1]
        
        if ema_fast > ema_mid > ema_slow:
            rafale_trend = "Strong Bullish"
            trend_score = 2
        elif ema_fast > ema_mid:
            rafale_trend = "Bullish"
            trend_score = 1
        elif ema_fast < ema_mid < ema_slow:
            rafale_trend = "Strong Bearish"
            trend_score = -2
        elif ema_fast < ema_mid:
            rafale_trend = "Bearish"
            trend_score = -1
        else:
            rafale_trend = "Neutral"
            trend_score = 0
        
        # 3. Rafale Power (RP) - Volume-weighted directional strength
        vol_sma = np.mean(volume[-20:]) if len(volume) >= 20 else volume[-1]
        vol_ratio = volume[-1] / vol_sma if vol_sma > 0 else 1
        
        # Directional movement
        plus_dm = np.maximum(high[1:] - high[:-1], 0)
        minus_dm = np.maximum(low[:-1] - low[1:], 0)
        
        plus_di = np.mean(plus_dm[-period:])
        minus_di = np.mean(minus_dm[-period:])
        
        di_diff = plus_di - minus_di
        rafale_power = di_diff * vol_ratio
        
        # 4. Calculate overall strength (0-100)
        momentum_component = min(max(rafale_momentum * 10, -50), 50) + 50  # 0-100
        trend_component = (trend_score + 2) * 25  # 0-100
        power_component = min(max(rafale_power * 100, 0), 100)  # 0-100
        
        rafale_strength = (momentum_component * 0.4 + trend_component * 0.35 + power_component * 0.25)
        
        # 5. Generate Signal
        if rafale_strength >= 70 and rafale_momentum > 1 and trend_score > 0:
            signal = "STRONG BUY"
            color = "green"
            desc = f"? Rafale BUY! Momentum: {rafale_momentum:.1f}, Trend: {rafale_trend}"
        elif rafale_strength >= 55 and rafale_momentum > 0.5 and trend_score >= 0:
            signal = "BUY"
            color = "lightgreen"
            desc = f"? Rafale Bullish. Strength: {rafale_strength:.0f}%"
        elif rafale_strength <= 30 and rafale_momentum < -1 and trend_score < 0:
            signal = "STRONG SELL"
            color = "red"
            desc = f"? Rafale SELL! Momentum: {rafale_momentum:.1f}, Trend: {rafale_trend}"
        elif rafale_strength <= 45 and rafale_momentum < -0.5 and trend_score <= 0:
            signal = "SELL"
            color = "lightcoral"
            desc = f"? Rafale Bearish. Strength: {rafale_strength:.0f}%"
        else:
            signal = "WAIT"
            color = "gray"
            desc = f"⏸️ Rafale Neutral. Waiting for setup..."
        
        return {
            'signal': signal,
            'momentum': round(rafale_momentum, 2),
            'trend': rafale_trend,
            'power': round(rafale_power, 2),
            'strength': round(rafale_strength, 1),
            'color': color,
            'description': desc,
            'ema_fast': round(ema_fast, 2),
            'ema_mid': round(ema_mid, 2),
            'ema_slow': round(ema_slow, 2),
            'vol_ratio': round(vol_ratio, 2)
        }
    
    @staticmethod
    def vwap(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> float:
        """Volume Weighted Average Price."""
        if volume.sum() == 0:
            return close[-1]
        
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).sum() / volume.sum()
        
        return round(vwap, 2)
    
    @staticmethod
    def obv_trend(close: np.ndarray, volume: np.ndarray, period: int = 10) -> str:
        """On-Balance Volume trend."""
        if len(close) < period + 1 or volume.sum() == 0:
            return "Neutral"
        
        obv = np.zeros(len(close))
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]
        
        obv_sma = np.mean(obv[-period:])
        
        if obv[-1] > obv_sma * 1.05:
            return "Bullish"
        elif obv[-1] < obv_sma * 0.95:
            return "Bearish"
        return "Neutral"
    
    @staticmethod
    def pivot_points(high: float, low: float, close: float) -> Tuple[float, float, float, float, float]:
        """Classic pivot points."""
        pivot = (high + low + close) / 3
        s1 = (2 * pivot) - high
        s2 = pivot - (high - low)
        r1 = (2 * pivot) - low
        r2 = pivot + (high - low)
        
        return round(pivot, 2), round(s1, 2), round(s2, 2), round(r1, 2), round(r2, 2)


class NiftyFnOSignalGenerator:
    """
    Enhanced Professional NIFTY 50 F&O Signal Generator.
    
    Features:
    - Multi-indicator confluence scoring (0-100)
    - Market regime detection
    - Volatility-adjusted risk management
    - Time-decay awareness
    """
    
    def __init__(self):
        # Trading constraints
        self.market_open = time(9, 20)
        self.market_close = time(14, 45)
        self.max_signals_per_day = 2
        self.risk_per_trade = 0.01
        
        # Signal tracking
        self.signals_today = 0
        self.last_signal_date = None
        self.last_signal_time = None
        self.min_signal_gap = timedelta(minutes=30)
        
        # Scoring thresholds (lowered for more frequent signals)
        self.high_confidence_threshold = 55   # Strong signals
        self.medium_confidence_threshold = 40  # Valid signals
        self.min_trade_threshold = 25          # Minimum score to consider
        
        # RSI thresholds
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.rsi_extreme_oversold = 20
        self.rsi_extreme_overbought = 80
        
        # ADX thresholds
        self.adx_strong_trend = 25
        self.adx_very_strong = 40
        
        # Cache
        self._indicator_cache = {}
        self._cache_timestamp = None
        self._cache_ttl = timedelta(seconds=30)
    
    def is_trading_hours(self) -> bool:
        """Check if within trading window."""
        now = datetime.now().time()
        return self.market_open <= now <= self.market_close
    
    def is_optimal_trading_time(self) -> Tuple[bool, str]:
        """Check if optimal for trading."""
        now = datetime.now().time()
        
        if time(9, 20) <= now <= time(11, 30):
            return True, "Prime trading hours"
        elif time(11, 30) < now <= time(13, 0):
            return True, "Mid-day trading"
        elif time(13, 0) < now <= time(14, 0):
            return False, "Lunch hours - Low activity"
        elif time(14, 0) < now <= time(14, 45):
            return False, "Late session - High time decay"
        return False, "Outside trading hours"
    
    def reset_daily_counters(self):
        """Reset daily signal counter."""
        today = datetime.now().date()
        if self.last_signal_date != today:
            self.signals_today = 0
            self.last_signal_date = today
            self.last_signal_time = None
    
    def can_generate_signal(self) -> Tuple[bool, str]:
        """Check if we can generate a new signal."""
        self.reset_daily_counters()
        
        if self.signals_today >= self.max_signals_per_day:
            return False, f"Max signals ({self.max_signals_per_day}) reached"
        
        if self.last_signal_time:
            elapsed = datetime.now() - self.last_signal_time
            if elapsed < self.min_signal_gap:
                remaining = (self.min_signal_gap - elapsed).seconds // 60
                return False, f"Wait {remaining} min"
        
        return True, "OK"
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> Optional[MarketIndicators]:
        """Calculate all technical indicators."""
        if df is None or len(df) < 50:
            return None
        
        # Check cache
        if self._indicator_cache and self._cache_timestamp:
            if datetime.now() - self._cache_timestamp < self._cache_ttl:
                return self._indicator_cache.get('indicators')
        
        try:
            close = df['close'].values.astype(float)
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)
            open_price = df['open'].values.astype(float)
            volume = df['volume'].values.astype(float) if 'volume' in df.columns else np.zeros(len(df))
            
            spot = round(close[-1], 2)
            
            # Moving Averages
            ema_9 = round(TechnicalIndicators.ema(close, 9)[-1], 2)
            ema_20 = round(TechnicalIndicators.ema(close, 20)[-1], 2)
            ema_50 = round(TechnicalIndicators.ema(close, 50)[-1], 2)
            sma_200 = round(np.mean(close[-200:]) if len(close) >= 200 else np.mean(close), 2)
            
            # Momentum
            rsi, rsi_sma = TechnicalIndicators.rsi(close, 14)
            macd, macd_signal, macd_hist = TechnicalIndicators.macd(close)
            
            # Volatility
            atr, atr_pct = TechnicalIndicators.atr(high, low, close, 14)
            bb_upper, bb_mid, bb_lower, bb_width, bb_pos = TechnicalIndicators.bollinger_bands(close)
            
            # Trend
            adx, plus_di, minus_di = TechnicalIndicators.adx(high, low, close, 14)
            supertrend, st_dir = TechnicalIndicators.supertrend(high, low, close)
            
            # Volume
            vwap = TechnicalIndicators.vwap(high, low, close, volume)
            vol_sma = round(np.mean(volume[-20:]), 0) if len(volume) >= 20 else volume[-1]
            vol_ratio = round(volume[-1] / vol_sma, 2) if vol_sma > 0 else 1.0
            obv = TechnicalIndicators.obv_trend(close, volume)
            
            # Pivot Points
            prev_high = high[-2] if len(high) >= 2 else high[-1]
            prev_low = low[-2] if len(low) >= 2 else low[-1]
            prev_close = close[-2] if len(close) >= 2 else close[-1]
            pivot, s1, s2, r1, r2 = TechnicalIndicators.pivot_points(prev_high, prev_low, prev_close)
            
            # Candle Patterns
            current_candle, prev_candle = self._analyze_candle_pattern(df)
            
            # Market Regime
            regime = self._detect_market_regime(adx, bb_width, atr_pct, vol_ratio)
            
            indicators = MarketIndicators(
                spot=spot, open=round(open_price[-1], 2),
                high=round(high[-1], 2), low=round(low[-1], 2),
                ema_9=ema_9, ema_20=ema_20, ema_50=ema_50, sma_200=sma_200,
                rsi=rsi, rsi_sma=rsi_sma,
                macd=macd, macd_signal=macd_signal, macd_histogram=macd_hist,
                atr=atr, atr_percent=atr_pct,
                bb_upper=bb_upper, bb_middle=bb_mid, bb_lower=bb_lower,
                bb_width=bb_width, bb_position=bb_pos,
                adx=adx, plus_di=plus_di, minus_di=minus_di,
                supertrend=supertrend, supertrend_direction=st_dir,
                vwap=vwap, volume_sma=vol_sma, volume_ratio=vol_ratio, obv_trend=obv,
                pivot=pivot, support_1=s1, support_2=s2, resistance_1=r1, resistance_2=r2,
                current_candle=current_candle, prev_candle=prev_candle,
                regime=regime
            )
            
            self._indicator_cache['indicators'] = indicators
            self._cache_timestamp = datetime.now()
            
            return indicators
            
        except Exception as e:
            print(f"Error calculating indicators: {e}")
            return None
    
    def _analyze_candle_pattern(self, df: pd.DataFrame) -> Tuple[str, str]:
        """Analyze candle patterns."""
        if len(df) < 3:
            return "Unknown", "Unknown"
        
        def describe_candle(row):
            body = row['close'] - row['open']
            range_size = row['high'] - row['low']
            body_pct = abs(body) / range_size * 100 if range_size > 0 else 0
            
            upper_wick = row['high'] - max(row['open'], row['close'])
            lower_wick = min(row['open'], row['close']) - row['low']
            
            direction = "Bullish" if body > 0 else "Bearish" if body < 0 else "Doji"
            
            if body_pct < 10:
                return "Doji"
            elif lower_wick > abs(body) * 2 and upper_wick < abs(body) * 0.5:
                return f"Hammer ({direction})"
            elif upper_wick > abs(body) * 2 and lower_wick < abs(body) * 0.5:
                return f"Shooting Star ({direction})"
            elif body_pct > 70:
                return f"Marubozu ({direction})"
            elif body_pct > 50:
                return f"Strong {direction}"
            return f"Weak {direction}"
        
        return describe_candle(df.iloc[-1]), describe_candle(df.iloc[-2])
    
    def _detect_market_regime(self, adx: float, bb_width: float, 
                               atr_pct: float, vol_ratio: float) -> MarketRegime:
        """Detect current market regime."""
        if adx > self.adx_very_strong:
            if vol_ratio > 1.5:
                return MarketRegime.BREAKOUT
            return MarketRegime.TRENDING_UP
        elif adx > self.adx_strong_trend:
            return MarketRegime.TRENDING_UP
        elif bb_width > 3.0 or atr_pct > 0.8:
            return MarketRegime.VOLATILE
        return MarketRegime.RANGING
    
    def calculate_signal_score(self, indicators: MarketIndicators) -> Tuple[int, int, List[str]]:
        """Calculate comprehensive signal score (0-100)."""
        bullish_score = 0
        bearish_score = 0
        reasoning = []
        
        # 1. TREND INDICATORS (Max 30 points)
        # EMA Alignment (10 points)
        if indicators.spot > indicators.ema_9 > indicators.ema_20 > indicators.ema_50:
            bullish_score += 10
            reasoning.append(f"✓ Perfect bullish EMA alignment")
        elif indicators.spot > indicators.ema_20 > indicators.ema_50:
            bullish_score += 7
        elif indicators.spot < indicators.ema_9 < indicators.ema_20 < indicators.ema_50:
            bearish_score += 10
            reasoning.append(f"✓ Perfect bearish EMA alignment")
        elif indicators.spot < indicators.ema_20 < indicators.ema_50:
            bearish_score += 7
        
        # SuperTrend (10 points)
        if indicators.supertrend_direction == 1:
            bullish_score += 10
            reasoning.append(f"✓ SuperTrend bullish @ {indicators.supertrend}")
        elif indicators.supertrend_direction == -1:
            bearish_score += 10
            reasoning.append(f"✓ SuperTrend bearish @ {indicators.supertrend}")
        
        # ADX (10 points)
        if indicators.adx > self.adx_very_strong:
            if indicators.plus_di > indicators.minus_di:
                bullish_score += 10
                reasoning.append(f"✓ Strong trend ADX={indicators.adx}")
            else:
                bearish_score += 10
        elif indicators.adx > self.adx_strong_trend:
            if indicators.plus_di > indicators.minus_di:
                bullish_score += 6
            else:
                bearish_score += 6
        
        # 2. MOMENTUM INDICATORS (Max 30 points)
        # RSI (10 points)
        if indicators.rsi <= self.rsi_extreme_oversold:
            bullish_score += 10
            reasoning.append(f"✓ RSI extreme oversold ({indicators.rsi})")
        elif indicators.rsi <= self.rsi_oversold:
            bullish_score += 7
        elif indicators.rsi >= self.rsi_extreme_overbought:
            bearish_score += 10
            reasoning.append(f"✓ RSI extreme overbought ({indicators.rsi})")
        elif indicators.rsi >= self.rsi_overbought:
            bearish_score += 7
        elif 55 <= indicators.rsi < 70:
            bullish_score += 4
        elif 30 < indicators.rsi <= 45:
            bearish_score += 4
        
        # MACD (10 points)
        if indicators.macd > indicators.macd_signal and indicators.macd_histogram > 0:
            bullish_score += 8 if indicators.macd_histogram > abs(indicators.macd) * 0.1 else 5
            reasoning.append(f"✓ MACD bullish, histogram expanding")
        elif indicators.macd < indicators.macd_signal and indicators.macd_histogram < 0:
            bearish_score += 8 if abs(indicators.macd_histogram) > abs(indicators.macd) * 0.1 else 5
            reasoning.append(f"✓ MACD bearish, histogram expanding")
        
        # VWAP (10 points)
        vwap_diff = ((indicators.spot - indicators.vwap) / indicators.vwap) * 100
        if vwap_diff > 0.3:
            bullish_score += 8
            reasoning.append(f"✓ Price above VWAP - Institutional buying")
        elif vwap_diff > 0:
            bullish_score += 4
        elif vwap_diff < -0.3:
            bearish_score += 8
            reasoning.append(f"✓ Price below VWAP - Institutional selling")
        elif vwap_diff < 0:
            bearish_score += 4
        
        # 3. VOLATILITY (Max 20 points)
        # BB Position (10 points)
        if indicators.bb_position > 90:
            if indicators.adx > self.adx_strong_trend:
                bullish_score += 7
            else:
                bearish_score += 5
        elif indicators.bb_position < 10:
            if indicators.adx > self.adx_strong_trend:
                bearish_score += 7
            else:
                bullish_score += 5
        elif 60 <= indicators.bb_position <= 80:
            bullish_score += 4
        elif 20 <= indicators.bb_position <= 40:
            bearish_score += 4
        
        # BB Squeeze (10 points)
        if indicators.bb_width < 1.5:
            if indicators.spot > indicators.bb_middle:
                bullish_score += 5
                reasoning.append("BB squeeze - Bullish breakout potential")
            else:
                bearish_score += 5
        
        # 4. VOLUME (Max 10 points)
        if indicators.volume_ratio > 2.0:
            if bullish_score > bearish_score:
                bullish_score += 10
                reasoning.append(f"✓ High volume confirms bullish move")
            else:
                bearish_score += 10
        elif indicators.volume_ratio > 1.3:
            if bullish_score > bearish_score:
                bullish_score += 5
            else:
                bearish_score += 5
        elif indicators.volume_ratio < 0.5:
            bullish_score = max(0, bullish_score - 5)
            bearish_score = max(0, bearish_score - 5)
            reasoning.append("⚠ Low volume")
        
        # OBV bonus
        if indicators.obv_trend == "Bullish":
            bullish_score += 3
        elif indicators.obv_trend == "Bearish":
            bearish_score += 3
        
        # 5. S/R LEVELS (Max 10 points)
        if indicators.spot <= indicators.support_1 * 1.003:
            bullish_score += 8
            reasoning.append(f"✓ At support S1={indicators.support_1}")
        elif indicators.spot >= indicators.resistance_1 * 0.997:
            bearish_score += 8
            reasoning.append(f"✓ At resistance R1={indicators.resistance_1}")
        
        # 6. CANDLE PATTERN (Bonus)
        if "Hammer" in indicators.current_candle or "Marubozu (Bullish)" in indicators.current_candle:
            bullish_score += 5
            reasoning.append(f"✓ {indicators.current_candle}")
        elif "Shooting Star" in indicators.current_candle or "Marubozu (Bearish)" in indicators.current_candle:
            bearish_score += 5
            reasoning.append(f"✓ {indicators.current_candle}")
        
        return bullish_score, bearish_score, reasoning
    
    def determine_market_trend(self, bullish_score: int, bearish_score: int,
                               indicators: MarketIndicators) -> MarketTrend:
        """Determine market trend from scores."""
        score_diff = bullish_score - bearish_score
        
        if score_diff > 30 and indicators.adx > self.adx_strong_trend:
            return MarketTrend.STRONG_BULLISH
        elif score_diff > 15:
            return MarketTrend.BULLISH
        elif score_diff < -30 and indicators.adx > self.adx_strong_trend:
            return MarketTrend.STRONG_BEARISH
        elif score_diff < -15:
            return MarketTrend.BEARISH
        return MarketTrend.SIDEWAYS
    
    def calculate_targets(self, spot: float, signal_type: SignalType,
                          indicators: MarketIndicators) -> Tuple[float, float, float, str]:
        """Calculate SL and targets using ATR and S/R."""
        atr = indicators.atr
        
        if signal_type == SignalType.BUY_CALL:
            sl_level = max(spot - atr * 1.5, indicators.support_1)
            target_1 = min(spot + atr * 2, indicators.resistance_1)
            target_2 = spot + atr * 3
            
            risk = spot - sl_level
            if risk <= 0:
                sl_level = spot - atr * 1.2
                risk = atr * 1.2
            
            if (target_1 - spot) / risk < 1.5:
                target_1 = spot + risk * 2
        else:
            sl_level = min(spot + atr * 1.5, indicators.resistance_1)
            target_1 = max(spot - atr * 2, indicators.support_1)
            target_2 = spot - atr * 3
            
            risk = sl_level - spot
            if risk <= 0:
                sl_level = spot + atr * 1.2
                risk = atr * 1.2
            
            if (spot - target_1) / risk < 1.5:
                target_1 = spot - risk * 2
        
        risk = abs(spot - sl_level)
        reward = abs(target_1 - spot)
        rr_ratio = round(reward / risk, 2) if risk > 0 else 0
        
        return round(sl_level, 2), round(target_1, 2), round(target_2, 2), f"1:{rr_ratio}"
    
    def select_strike_price(self, spot: float, signal_type: SignalType,
                            confidence: Confidence) -> Tuple[int, StrikeType]:
        """Select optimal strike."""
        atm_strike = round(spot / 50) * 50
        
        if confidence == Confidence.HIGH or confidence == Confidence.MEDIUM:
            return atm_strike, StrikeType.ATM
        else:
            if signal_type == SignalType.BUY_CALL:
                return atm_strike + 50, StrikeType.OTM
            return atm_strike - 50, StrikeType.OTM
    
    def detect_vwap_reversal(self, df: pd.DataFrame, indicators: MarketIndicators) -> Tuple[str, str]:
        """Detect VWAP reversal signals - institutional level indicator."""
        if len(df) < 10:
            return "NEUTRAL", "Insufficient data"
        
        close = df['close'].values[-10:]
        vwap = indicators.vwap
        spot = indicators.spot
        
        # Track recent VWAP crossings
        above_vwap = [c > vwap for c in close]
        below_vwap = [c < vwap for c in close]
        
        # Count candles above/below VWAP
        recent_above = sum(above_vwap[-5:])
        recent_below = sum(below_vwap[-5:])
        
        vwap_diff_pct = ((spot - vwap) / vwap) * 100
        
        signal = "NEUTRAL"
        reason = ""
        
        # BULLISH VWAP Reversal: Price was below VWAP and now crossing above
        if recent_below >= 3 and spot > vwap and vwap_diff_pct > 0.1:
            if close[-2] < vwap and close[-1] > vwap:
                signal = "BULLISH_REVERSAL"
                reason = f"VWAP Bullish Reversal! Price crossed above VWAP ({vwap_diff_pct:.2f}%)"
            elif vwap_diff_pct > 0.3:
                signal = "BULLISH"
                reason = f"Price sustained above VWAP (+{vwap_diff_pct:.2f}%)"
        
        # BEARISH VWAP Reversal: Price was above VWAP and now crossing below
        elif recent_above >= 3 and spot < vwap and vwap_diff_pct < -0.1:
            if close[-2] > vwap and close[-1] < vwap:
                signal = "BEARISH_REVERSAL"
                reason = f"VWAP Bearish Reversal! Price crossed below VWAP ({vwap_diff_pct:.2f}%)"
            elif vwap_diff_pct < -0.3:
                signal = "BEARISH"
                reason = f"Price sustained below VWAP ({vwap_diff_pct:.2f}%)"
        
        # Near VWAP - potential bounce
        elif abs(vwap_diff_pct) < 0.1:
            signal = "AT_VWAP"
            reason = "Price at VWAP - Watch for bounce or breakdown"
        
        return signal, reason
    
    def detect_scalping_opportunity(self, df: pd.DataFrame, indicators: MarketIndicators) -> dict:
        """Detect quick scalping opportunities (5-15 min trades)."""
        scalp = {
            'signal': 'WAIT',
            'entry': 0.0,
            'stop_loss': 0.0,
            'target': 0.0,
            'reason': 'No scalping setup'
        }
        
        if len(df) < 20:
            return scalp
        
        spot = indicators.spot
        atr = indicators.atr
        rsi = indicators.rsi
        bb_pos = indicators.bb_position
        ema_9 = indicators.ema_9
        
        # Scalping parameters (tighter than swing)
        scalp_sl_multiplier = 0.5  # Tighter SL
        scalp_target_multiplier = 1.0  # Quick target
        
        # SCALP BUY CONDITIONS
        # 1. RSI oversold bounce + Price near lower BB + Price above EMA9
        if rsi < 35 and bb_pos < 20 and spot > ema_9:
            scalp['signal'] = 'SCALP_BUY'
            scalp['entry'] = spot
            scalp['stop_loss'] = round(spot - atr * scalp_sl_multiplier, 2)
            scalp['target'] = round(spot + atr * scalp_target_multiplier, 2)
            scalp['reason'] = f"RSI oversold ({rsi:.0f}) + BB squeeze bounce"
        
        # 2. EMA9 bounce in uptrend
        elif spot > indicators.ema_20 and abs(spot - ema_9) < atr * 0.3 and indicators.macd > indicators.macd_signal:
            scalp['signal'] = 'SCALP_BUY'
            scalp['entry'] = spot
            scalp['stop_loss'] = round(ema_9 - atr * 0.3, 2)
            scalp['target'] = round(spot + atr * scalp_target_multiplier, 2)
            scalp['reason'] = "EMA9 bounce + MACD bullish"
        
        # 3. VWAP bounce in uptrend
        elif spot > indicators.vwap and abs(spot - indicators.vwap) < atr * 0.2:
            scalp['signal'] = 'SCALP_BUY'
            scalp['entry'] = spot
            scalp['stop_loss'] = round(indicators.vwap - atr * 0.3, 2)
            scalp['target'] = round(spot + atr * scalp_target_multiplier, 2)
            scalp['reason'] = "VWAP bounce support"
        
        # SCALP SELL CONDITIONS
        # 1. RSI overbought rejection + Price near upper BB + Price below EMA9
        elif rsi > 65 and bb_pos > 80 and spot < ema_9:
            scalp['signal'] = 'SCALP_SELL'
            scalp['entry'] = spot
            scalp['stop_loss'] = round(spot + atr * scalp_sl_multiplier, 2)
            scalp['target'] = round(spot - atr * scalp_target_multiplier, 2)
            scalp['reason'] = f"RSI overbought ({rsi:.0f}) + BB rejection"
        
        # 2. EMA9 rejection in downtrend
        elif spot < indicators.ema_20 and abs(spot - ema_9) < atr * 0.3 and indicators.macd < indicators.macd_signal:
            scalp['signal'] = 'SCALP_SELL'
            scalp['entry'] = spot
            scalp['stop_loss'] = round(ema_9 + atr * 0.3, 2)
            scalp['target'] = round(spot - atr * scalp_target_multiplier, 2)
            scalp['reason'] = "EMA9 rejection + MACD bearish"
        
        # 3. VWAP rejection in downtrend
        elif spot < indicators.vwap and abs(spot - indicators.vwap) < atr * 0.2:
            scalp['signal'] = 'SCALP_SELL'
            scalp['entry'] = spot
            scalp['stop_loss'] = round(indicators.vwap + atr * 0.3, 2)
            scalp['target'] = round(spot - atr * scalp_target_multiplier, 2)
            scalp['reason'] = "VWAP rejection resistance"
        
        return scalp
    
    def analyze_candle_trend(self, df: pd.DataFrame) -> Tuple[str, List[str]]:
        """Analyze recent candle patterns for trend confirmation."""
        if len(df) < 5:
            return "Unknown", ["Insufficient data"]
        
        patterns = []
        trend = "Neutral"
        
        # Last 5 candles analysis
        last_5 = df.iloc[-5:]
        bullish_count = 0
        bearish_count = 0
        
        for i, (_, row) in enumerate(last_5.iterrows()):
            body = row['close'] - row['open']
            if body > 0:
                bullish_count += 1
            elif body < 0:
                bearish_count += 1
        
        # Determine trend from candle colors
        if bullish_count >= 4:
            trend = "Strong Bullish"
            patterns.append(f"✓ {bullish_count}/5 bullish candles")
        elif bullish_count >= 3:
            trend = "Bullish"
            patterns.append(f"✓ {bullish_count}/5 bullish candles")
        elif bearish_count >= 4:
            trend = "Strong Bearish"
            patterns.append(f"✓ {bearish_count}/5 bearish candles")
        elif bearish_count >= 3:
            trend = "Bearish"
            patterns.append(f"✓ {bearish_count}/5 bearish candles")
        
        # Check for specific patterns
        last = df.iloc[-1]
        prev = df.iloc[-2]
        prev2 = df.iloc[-3]
        
        # Engulfing patterns
        last_body = last['close'] - last['open']
        prev_body = prev['close'] - prev['open']
        
        if last_body > 0 and prev_body < 0 and abs(last_body) > abs(prev_body) * 1.5:
            patterns.append("? Bullish Engulfing")
            trend = "Bullish Reversal"
        elif last_body < 0 and prev_body > 0 and abs(last_body) > abs(prev_body) * 1.5:
            patterns.append("? Bearish Engulfing")
            trend = "Bearish Reversal"
        
        # Three white soldiers / Three black crows
        if len(df) >= 3:
            candle_1 = df.iloc[-3]['close'] - df.iloc[-3]['open']
            candle_2 = df.iloc[-2]['close'] - df.iloc[-2]['open']
            candle_3 = df.iloc[-1]['close'] - df.iloc[-1]['open']
            
            if candle_1 > 0 and candle_2 > 0 and candle_3 > 0:
                if df.iloc[-1]['close'] > df.iloc[-2]['close'] > df.iloc[-3]['close']:
                    patterns.append("? Three White Soldiers (Strong Buy)")
                    trend = "Strong Bullish"
            elif candle_1 < 0 and candle_2 < 0 and candle_3 < 0:
                if df.iloc[-1]['close'] < df.iloc[-2]['close'] < df.iloc[-3]['close']:
                    patterns.append("? Three Black Crows (Strong Sell)")
                    trend = "Strong Bearish"
        
        # Hammer/Shooting star (from existing analysis)
        if "Hammer" in str(df.iloc[-1].get('pattern', '')):
            patterns.append("? Hammer (Bullish reversal)")
        elif "Shooting Star" in str(df.iloc[-1].get('pattern', '')):
            patterns.append("⭐ Shooting Star (Bearish reversal)")
        
        # Include existing candle analysis
        patterns.append(f"Current: {self._analyze_candle_pattern(df)[0]}")
        
        return trend, patterns
    
    def calculate_risk_level(self, indicators: MarketIndicators, signal_type: SignalType) -> Tuple[str, List[str]]:
        """Calculate comprehensive risk level for the trade."""
        risk_factors = []
        risk_score = 0  # 0-100, higher = more risk
        
        # 1. Volatility Risk
        if indicators.atr_percent > 1.0:
            risk_score += 25
            risk_factors.append(f"⚠ High volatility (ATR: {indicators.atr_percent:.2f}%)")
        elif indicators.atr_percent > 0.7:
            risk_score += 15
            risk_factors.append(f"⚡ Moderate volatility (ATR: {indicators.atr_percent:.2f}%)")
        
        # 2. ADX / Trend Strength Risk
        if indicators.adx < 20:
            risk_score += 20
            risk_factors.append(f"⚠ Weak trend (ADX: {indicators.adx:.0f})")
        elif indicators.adx < 25:
            risk_score += 10
        
        # 3. RSI Extreme Risk
        if indicators.rsi > 75 and signal_type == SignalType.BUY_CALL:
            risk_score += 25
            risk_factors.append(f"⚠ RSI overbought ({indicators.rsi:.0f}) - risk of reversal")
        elif indicators.rsi < 25 and signal_type == SignalType.BUY_PUT:
            risk_score += 25
            risk_factors.append(f"⚠ RSI oversold ({indicators.rsi:.0f}) - risk of bounce")
        
        # 4. BB Position Risk
        if indicators.bb_position > 95 and signal_type == SignalType.BUY_CALL:
            risk_score += 15
            risk_factors.append("⚠ Price at upper BB - extended")
        elif indicators.bb_position < 5 and signal_type == SignalType.BUY_PUT:
            risk_score += 15
            risk_factors.append("⚠ Price at lower BB - extended")
        
        # 5. Volume Risk
        if indicators.volume_ratio < 0.7:
            risk_score += 15
            risk_factors.append(f"⚠ Low volume ({indicators.volume_ratio:.1f}x) - weak conviction")
        
        # 6. Regime Risk
        if indicators.regime == MarketRegime.VOLATILE:
            risk_score += 20
            risk_factors.append("⚠ Volatile market regime")
        elif indicators.regime == MarketRegime.RANGING:
            risk_score += 10
            risk_factors.append("⚡ Ranging market - limited move potential")
        
        # Determine risk level
        if risk_score >= 60:
            risk_level = "HIGH"
        elif risk_score >= 35:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        if not risk_factors:
            risk_factors.append("✓ No major risk factors identified")
        
        return risk_level, risk_factors

    def generate_signal(self, df: pd.DataFrame) -> TradingSignal:
        """Generate trading signal with multi-indicator analysis."""
        self.reset_daily_counters()
        
        if df is None or len(df) < 50:
            return self._no_trade_signal(["Insufficient data"])
        
        indicators = self.calculate_all_indicators(df)
        if indicators is None:
            return self._no_trade_signal(["Failed to calculate indicators"])
        
        # Pre-trade checks
        rejection_reasons = []
        
        is_optimal, time_msg = self.is_optimal_trading_time()
        # Note: Trading hours check relaxed to allow signal generation for analysis
        # Actual trading should still follow market hours
        if not self.is_trading_hours():
            pass  # Don't reject, just note it in reasoning
            # rejection_reasons.append("Outside trading hours")
        elif not is_optimal:
            pass  # Don't reject for non-optimal hours
            # rejection_reasons.append(time_msg)
        
        can_signal, signal_msg = self.can_generate_signal()
        if not can_signal:
            rejection_reasons.append(signal_msg)
        
        if indicators.regime == MarketRegime.VOLATILE:
            rejection_reasons.append("Market too volatile")
        elif indicators.regime == MarketRegime.RANGING and indicators.adx < 20:
            rejection_reasons.append("Weak trend (ADX < 20)")
        
        # Calculate scores
        bullish_score, bearish_score, reasoning = self.calculate_signal_score(indicators)
        max_score = max(bullish_score, bearish_score)
        
        trend = self.determine_market_trend(bullish_score, bearish_score, indicators)
        volume_status = "High" if indicators.volume_ratio > 1.3 else "Low" if indicators.volume_ratio < 0.7 else "Normal"
        
        # NEW: Calculate scalping, VWAP reversal, candle trend and risk
        scalp_data = self.detect_scalping_opportunity(df, indicators)
        vwap_signal, vwap_reason = self.detect_vwap_reversal(df, indicators)
        candle_trend, candle_patterns = self.analyze_candle_trend(df)
        
        # NEW: Calculate Ahuja Rafale Indicator
        close = df['close'].values.astype(float)
        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)
        volume = df['volume'].values.astype(float) if 'volume' in df.columns else np.zeros(len(df))
        ahuja_rafale_data = TechnicalIndicators.ahuja_rafale(high, low, close, volume)
        
        # Too many rejection reasons
        if len(rejection_reasons) >= 2:
            # Still calculate risk for display
            risk_level, risk_factors = self.calculate_risk_level(indicators, SignalType.NO_TRADE)
            
            return TradingSignal(
                market_trend=trend, signal=SignalType.NO_TRADE,
                option_type=None, strike_price=None, strike_type=None,
                entry=None, stop_loss=None, target=None,
                confidence=Confidence.LOW, reasoning=rejection_reasons[:3],
                timestamp=datetime.now(),
                spot_price=indicators.spot, ema_20=indicators.ema_20,
                ema_50=indicators.ema_50, rsi=indicators.rsi,
                vwap=indicators.vwap, volume_status=volume_status,
                macd_signal_val=indicators.macd_signal, adx=indicators.adx,
                bb_position=indicators.bb_position, supertrend_dir=indicators.supertrend_direction,
                atr=indicators.atr, regime=indicators.regime.value, score=max_score,
                # NEW fields
                scalping_signal=scalp_data['signal'],
                scalping_entry=scalp_data['entry'],
                scalping_sl=scalp_data['stop_loss'],
                scalping_target=scalp_data['target'],
                vwap_signal=vwap_signal,
                candle_pattern=indicators.current_candle,
                risk_level=risk_level,
                trade_type="Swing",
                ahuja_rafale=ahuja_rafale_data
            )
        
        # Determine signal
        signal_type = SignalType.NO_TRADE
        confidence = Confidence.LOW
        
        if bullish_score >= self.high_confidence_threshold and bullish_score > bearish_score + 10:
            signal_type = SignalType.BUY_CALL
            confidence = Confidence.HIGH
        elif bullish_score >= self.medium_confidence_threshold and bullish_score > bearish_score + 5:
            signal_type = SignalType.BUY_CALL
            confidence = Confidence.MEDIUM
        elif bullish_score >= self.min_trade_threshold and bullish_score > bearish_score:
            signal_type = SignalType.BUY_CALL
            confidence = Confidence.LOW
        elif bearish_score >= self.high_confidence_threshold and bearish_score > bullish_score + 10:
            signal_type = SignalType.BUY_PUT
            confidence = Confidence.HIGH
        elif bearish_score >= self.medium_confidence_threshold and bearish_score > bullish_score + 5:
            signal_type = SignalType.BUY_PUT
            confidence = Confidence.MEDIUM
        elif bearish_score >= self.min_trade_threshold and bearish_score > bullish_score:
            signal_type = SignalType.BUY_PUT
            confidence = Confidence.LOW
        
        # NO TRADE if conditions not met
        if signal_type == SignalType.NO_TRADE or trend == MarketTrend.SIDEWAYS:
            if not rejection_reasons:
                rejection_reasons = [
                    f"Score below threshold (B:{bullish_score} S:{bearish_score})",
                    "Indicators not aligned",
                    "Waiting for stronger setup"
                ]
            
            risk_level, risk_factors = self.calculate_risk_level(indicators, signal_type)
            
            return TradingSignal(
                market_trend=trend, signal=SignalType.NO_TRADE,
                option_type=None, strike_price=None, strike_type=None,
                entry=None, stop_loss=None, target=None,
                confidence=Confidence.LOW, reasoning=rejection_reasons[:3],
                timestamp=datetime.now(),
                spot_price=indicators.spot, ema_20=indicators.ema_20,
                ema_50=indicators.ema_50, rsi=indicators.rsi,
                vwap=indicators.vwap, volume_status=volume_status,
                macd_signal_val=indicators.macd_signal, adx=indicators.adx,
                bb_position=indicators.bb_position, supertrend_dir=indicators.supertrend_direction,
                atr=indicators.atr, regime=indicators.regime.value, score=max_score,
                # NEW fields
                scalping_signal=scalp_data['signal'],
                scalping_entry=scalp_data['entry'],
                scalping_sl=scalp_data['stop_loss'],
                scalping_target=scalp_data['target'],
                vwap_signal=vwap_signal,
                candle_pattern=indicators.current_candle,
                risk_level=risk_level,
                trade_type="Swing",
                ahuja_rafale=ahuja_rafale_data
            )
        
        # Generate trade details
        strike, strike_type = self.select_strike_price(indicators.spot, signal_type, confidence)
        stop_loss, target_1, target_2, rr_ratio = self.calculate_targets(
            indicators.spot, signal_type, indicators
        )
        
        if signal_type == SignalType.BUY_CALL:
            entry = f"Enter on breakout above {round(indicators.high + 5, 0)} or pullback to EMA9"
            option_type = "CE (Call Option)"
        else:
            entry = f"Enter on breakdown below {round(indicators.low - 5, 0)} or pullback to EMA9"
            option_type = "PE (Put Option)"
        
        self.signals_today += 1
        self.last_signal_time = datetime.now()
        
        # Calculate risk level for the trade
        risk_level, risk_factors = self.calculate_risk_level(indicators, signal_type)
        
        # Add risk factors to reasoning
        reasoning.insert(0, f"Signal Score: Bull={bullish_score}, Bear={bearish_score}")
        if vwap_signal in ['BULLISH_REVERSAL', 'BEARISH_REVERSAL']:
            reasoning.append(f"VWAP: {vwap_signal}")
        if scalp_data['signal'] != 'WAIT':
            reasoning.append(f"Scalp: {scalp_data['reason']}")
        if candle_patterns:
            reasoning.append(f"Candle: {candle_trend}")
        
        # Determine trade type based on conditions
        trade_type = "Scalp" if scalp_data['signal'] != 'WAIT' else "Swing"
        
        return TradingSignal(
            market_trend=trend, signal=signal_type,
            option_type=option_type, strike_price=strike, strike_type=strike_type,
            entry=entry, stop_loss=stop_loss, target=target_1, target_2=target_2,
            risk_reward_ratio=rr_ratio, confidence=confidence,
            reasoning=reasoning[:6], timestamp=datetime.now(),
            spot_price=indicators.spot, ema_20=indicators.ema_20,
            ema_50=indicators.ema_50, rsi=indicators.rsi,
            vwap=indicators.vwap, volume_status=volume_status,
            macd_signal_val=indicators.macd_signal, adx=indicators.adx,
            bb_position=indicators.bb_position, supertrend_dir=indicators.supertrend_direction,
            atr=indicators.atr, regime=indicators.regime.value, score=max_score,
            # NEW fields
            scalping_signal=scalp_data['signal'],
            scalping_entry=scalp_data['entry'],
            scalping_sl=scalp_data['stop_loss'],
            scalping_target=scalp_data['target'],
            vwap_signal=vwap_signal,
            candle_pattern=indicators.current_candle,
            risk_level=risk_level,
            trade_type=trade_type,
            ahuja_rafale=ahuja_rafale_data
        )
    
    def _no_trade_signal(self, reason: List[str]) -> TradingSignal:
        """Generate a NO TRADE signal."""
        return TradingSignal(
            market_trend=MarketTrend.SIDEWAYS, signal=SignalType.NO_TRADE,
            option_type=None, strike_price=None, strike_type=None,
            entry=None, stop_loss=None, target=None,
            confidence=Confidence.LOW, reasoning=reason,
            timestamp=datetime.now(),
            spot_price=0, ema_20=0, ema_50=0, rsi=50, vwap=0,
            volume_status="Unknown"
        )


__all__ = [
    'NiftyFnOSignalGenerator', 'TradingSignal', 'MarketTrend', 
    'SignalType', 'Confidence', 'MarketIndicators', 'MarketRegime',
    'TechnicalIndicators'
]
