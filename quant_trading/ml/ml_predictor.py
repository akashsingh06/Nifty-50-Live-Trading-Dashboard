"""
ML Predictor Module - Jim Simons Style Quantitative Analysis
=============================================================

This module implements machine learning techniques inspired by 
Renaissance Technologies' Medallion Fund approach:

1. Statistical Pattern Recognition
2. Hidden Markov Models for Market Regimes
3. Mean Reversion Detection
4. Momentum Anomalies
5. Ensemble Predictions
6. Feature Engineering

Key Principles:
- Data-driven decisions only
- No human emotional override
- Statistical significance required
- Diversified model ensemble
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime states detected by HMM."""
    BULL_QUIET = "bull_quiet"       # Uptrend, low volatility
    BULL_VOLATILE = "bull_volatile" # Uptrend, high volatility
    BEAR_QUIET = "bear_quiet"       # Downtrend, low volatility
    BEAR_VOLATILE = "bear_volatile" # Downtrend, high volatility
    SIDEWAYS = "sideways"           # No clear trend
    CRISIS = "crisis"               # Extreme volatility, correlation breakdown


@dataclass
class PredictionResult:
    """Container for ML prediction output."""
    symbol: str
    prediction_horizon: str  # '1d', '5d', '20d'
    
    # Price predictions
    predicted_return: float
    predicted_price: float
    confidence_interval: Tuple[float, float]  # 95% CI
    
    # Probability estimates
    prob_up: float
    prob_down: float
    prob_sideways: float
    
    # Model outputs
    regime: MarketRegime
    pattern_detected: str
    anomaly_score: float
    
    # Ensemble details
    model_predictions: Dict[str, float]
    model_weights: Dict[str, float]
    
    # Risk metrics
    predicted_volatility: float
    var_95: float  # Value at Risk
    expected_sharpe: float
    
    # Confidence
    overall_confidence: float
    signal_strength: str  # 'strong', 'moderate', 'weak'
    
    timestamp: pd.Timestamp = field(default_factory=pd.Timestamp.now)


class PatternRecognizer:
    """
    Detects chart patterns and statistical anomalies.
    Based on Renaissance's pattern recognition approach.
    """
    
    def __init__(self):
        self.patterns = {}
    
    def detect_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect various chart and statistical patterns."""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values if 'volume' in df.columns else np.ones(len(close))
        
        patterns = {
            'trend': self._detect_trend(close),
            'support_resistance': self._find_support_resistance(close, high, low),
            'volume_pattern': self._analyze_volume_pattern(close, volume),
            'price_pattern': self._detect_price_pattern(close, high, low),
            'momentum_divergence': self._detect_divergence(close),
            'mean_reversion_signal': self._mean_reversion_signal(close),
            'breakout_potential': self._breakout_analysis(close, high, low, volume)
        }
        
        return patterns
    
    def _detect_trend(self, prices: np.ndarray) -> Dict:
        """Detect trend using multiple timeframes."""
        if len(prices) < 50:
            return {'direction': 'unknown', 'strength': 0, 'aligned': False,
                    'short_slope': 0, 'med_slope': 0, 'long_slope': 0}
        
        # Short-term trend (10 days)
        short_data = prices[-10:]
        short_slope = np.polyfit(range(len(short_data)), short_data, 1)[0]
        
        # Medium-term trend (20 days)
        med_data = prices[-20:]
        med_slope = np.polyfit(range(len(med_data)), med_data, 1)[0]
        
        # Long-term trend (50 days)
        long_data = prices[-50:]
        long_slope = np.polyfit(range(len(long_data)), long_data, 1)[0]
        
        # Normalize slopes
        avg_price = np.mean(prices[-50:])
        short_slope_pct = (short_slope / avg_price) * 100
        med_slope_pct = (med_slope / avg_price) * 100
        long_slope_pct = (long_slope / avg_price) * 100
        
        # Determine trend direction and strength
        avg_slope = (short_slope_pct + med_slope_pct + long_slope_pct) / 3
        
        if avg_slope > 0.5:
            direction = 'bullish'
        elif avg_slope < -0.5:
            direction = 'bearish'
        else:
            direction = 'sideways'
        
        # Trend strength (0-1)
        strength = min(abs(avg_slope) / 2, 1.0)
        
        # Trend alignment
        aligned = np.sign(short_slope) == np.sign(med_slope) == np.sign(long_slope)
        
        return {
            'direction': direction,
            'strength': strength,
            'aligned': aligned,
            'short_slope': short_slope_pct,
            'med_slope': med_slope_pct,
            'long_slope': long_slope_pct
        }
    
    def _find_support_resistance(self, close: np.ndarray, high: np.ndarray, 
                                  low: np.ndarray) -> Dict:
        """Find key support and resistance levels."""
        if len(close) < 20:
            return {'support': close[-1] * 0.95, 'resistance': close[-1] * 1.05}
        
        # Find local minima (support) and maxima (resistance)
        window = 5
        supports = []
        resistances = []
        
        for i in range(window, len(low) - window):
            if low[i] == min(low[i-window:i+window+1]):
                supports.append(low[i])
            if high[i] == max(high[i-window:i+window+1]):
                resistances.append(high[i])
        
        current_price = close[-1]
        
        # Find nearest support (below current price)
        supports_below = [s for s in supports if s < current_price]
        support = max(supports_below) if supports_below else current_price * 0.95
        
        # Find nearest resistance (above current price)
        resistances_above = [r for r in resistances if r > current_price]
        resistance = min(resistances_above) if resistances_above else current_price * 1.05
        
        # Calculate level strength
        support_tests = sum(1 for s in supports if abs(s - support) < support * 0.01)
        resistance_tests = sum(1 for r in resistances if abs(r - resistance) < resistance * 0.01)
        
        return {
            'support': support,
            'resistance': resistance,
            'support_strength': min(support_tests, 5),
            'resistance_strength': min(resistance_tests, 5),
            'range_pct': (resistance - support) / current_price * 100
        }
    
    def _analyze_volume_pattern(self, prices: np.ndarray, volume: np.ndarray) -> Dict:
        """Analyze volume patterns for confirmation signals."""
        if len(prices) < 20 or len(volume) < 20:
            return {'pattern': 'insufficient_data', 'signal': 'neutral', 
                    'volume_ratio': 1, 'price_volume_corr': 0}
        
        # Recent volume vs average
        recent_vol = np.mean(volume[-5:])
        avg_vol = np.mean(volume[-20:])
        vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1
        
        # Price-volume correlation
        price_change = np.diff(prices[-20:])
        vol_change = np.diff(volume[-20:])
        
        if len(price_change) > 0 and len(vol_change) > 0 and len(price_change) == len(vol_change):
            try:
                correlation = np.corrcoef(price_change, vol_change)[0, 1]
                if np.isnan(correlation):
                    correlation = 0
            except Exception:
                correlation = 0
        else:
            correlation = 0
        
        # Determine pattern
        if vol_ratio > 1.5 and prices[-1] > prices[-5]:
            pattern = 'volume_breakout_up'
            signal = 'bullish'
        elif vol_ratio > 1.5 and prices[-1] < prices[-5]:
            pattern = 'volume_breakout_down'
            signal = 'bearish'
        elif vol_ratio < 0.5:
            pattern = 'low_volume_consolidation'
            signal = 'neutral'
        else:
            pattern = 'normal'
            signal = 'neutral'
        
        return {
            'pattern': pattern,
            'signal': signal,
            'volume_ratio': vol_ratio,
            'price_volume_corr': correlation if not np.isnan(correlation) else 0
        }
    
    def _detect_price_pattern(self, close: np.ndarray, high: np.ndarray, 
                               low: np.ndarray) -> Dict:
        """Detect classic chart patterns."""
        if len(close) < 30:
            return {'pattern': 'none', 'confidence': 0}
        
        # Check for various patterns
        patterns_found = []
        
        # Double bottom
        if self._is_double_bottom(low[-30:]):
            patterns_found.append(('double_bottom', 'bullish', 0.7))
        
        # Double top
        if self._is_double_top(high[-30:]):
            patterns_found.append(('double_top', 'bearish', 0.7))
        
        # Head and shoulders (simplified)
        if self._is_head_shoulders(high[-30:]):
            patterns_found.append(('head_shoulders', 'bearish', 0.8))
        
        # Ascending triangle
        if self._is_ascending_triangle(high[-20:], low[-20:]):
            patterns_found.append(('ascending_triangle', 'bullish', 0.65))
        
        # Descending triangle
        if self._is_descending_triangle(high[-20:], low[-20:]):
            patterns_found.append(('descending_triangle', 'bearish', 0.65))
        
        if patterns_found:
            # Return highest confidence pattern
            best_pattern = max(patterns_found, key=lambda x: x[2])
            return {
                'pattern': best_pattern[0],
                'bias': best_pattern[1],
                'confidence': best_pattern[2],
                'all_patterns': patterns_found
            }
        
        return {'pattern': 'none', 'bias': 'neutral', 'confidence': 0}
    
    def _is_double_bottom(self, lows: np.ndarray) -> bool:
        """Check for double bottom pattern."""
        if len(lows) < 20:
            return False
        
        # Find two lowest points
        min_idx1 = np.argmin(lows[:len(lows)//2])
        min_idx2 = np.argmin(lows[len(lows)//2:]) + len(lows)//2
        
        min1 = lows[min_idx1]
        min2 = lows[min_idx2]
        
        # Check if they're at similar levels (within 2%)
        if abs(min1 - min2) / min1 < 0.02:
            # Check if there's a peak between them
            mid_section = lows[min_idx1:min_idx2]
            if len(mid_section) > 0:
                mid_high = max(mid_section)
                if mid_high > min1 * 1.03:  # Peak at least 3% higher
                    return True
        return False
    
    def _is_double_top(self, highs: np.ndarray) -> bool:
        """Check for double top pattern."""
        if len(highs) < 20:
            return False
        
        max_idx1 = np.argmax(highs[:len(highs)//2])
        max_idx2 = np.argmax(highs[len(highs)//2:]) + len(highs)//2
        
        max1 = highs[max_idx1]
        max2 = highs[max_idx2]
        
        if abs(max1 - max2) / max1 < 0.02:
            mid_section = highs[max_idx1:max_idx2]
            if len(mid_section) > 0:
                mid_low = min(mid_section)
                if mid_low < max1 * 0.97:
                    return True
        return False
    
    def _is_head_shoulders(self, highs: np.ndarray) -> bool:
        """Simplified head and shoulders detection."""
        if len(highs) < 20:
            return False
        
        # Divide into 3 sections
        third = len(highs) // 3
        left = highs[:third]
        middle = highs[third:2*third]
        right = highs[2*third:]
        
        if len(left) == 0 or len(middle) == 0 or len(right) == 0:
            return False
        
        left_max = max(left)
        head_max = max(middle)
        right_max = max(right)
        
        # Head should be higher than shoulders
        if head_max > left_max * 1.02 and head_max > right_max * 1.02:
            # Shoulders should be roughly equal
            if abs(left_max - right_max) / left_max < 0.03:
                return True
        return False
    
    def _is_ascending_triangle(self, highs: np.ndarray, lows: np.ndarray) -> bool:
        """Check for ascending triangle."""
        if len(highs) < 15:
            return False
        
        # Highs should be relatively flat
        high_slope = np.polyfit(range(len(highs)), highs, 1)[0]
        high_flat = abs(high_slope / np.mean(highs)) < 0.001
        
        # Lows should be rising
        low_slope = np.polyfit(range(len(lows)), lows, 1)[0]
        low_rising = low_slope / np.mean(lows) > 0.001
        
        return high_flat and low_rising
    
    def _is_descending_triangle(self, highs: np.ndarray, lows: np.ndarray) -> bool:
        """Check for descending triangle."""
        if len(highs) < 15:
            return False
        
        # Lows should be relatively flat
        low_slope = np.polyfit(range(len(lows)), lows, 1)[0]
        low_flat = abs(low_slope / np.mean(lows)) < 0.001
        
        # Highs should be falling
        high_slope = np.polyfit(range(len(highs)), highs, 1)[0]
        high_falling = high_slope / np.mean(highs) < -0.001
        
        return low_flat and high_falling
    
    def _detect_divergence(self, prices: np.ndarray) -> Dict:
        """Detect momentum divergences."""
        if len(prices) < 20:
            return {'divergence': 'none', 'signal': 'neutral'}
        
        # Calculate RSI
        delta = np.diff(prices)
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses[-14:])
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        # Check for divergence
        price_trend = prices[-1] - prices[-10]
        
        # Bullish divergence: price making lower lows, RSI making higher lows
        if price_trend < 0 and rsi > 40:
            return {'divergence': 'bullish', 'signal': 'buy', 'rsi': rsi}
        
        # Bearish divergence: price making higher highs, RSI making lower highs
        if price_trend > 0 and rsi < 60:
            return {'divergence': 'bearish', 'signal': 'sell', 'rsi': rsi}
        
        return {'divergence': 'none', 'signal': 'neutral', 'rsi': rsi}
    
    def _mean_reversion_signal(self, prices: np.ndarray) -> Dict:
        """Calculate mean reversion probability."""
        if len(prices) < 50:
            return {'signal': 'neutral', 'z_score': 0, 'reversion_prob': 0.5}
        
        # Calculate z-score relative to moving average
        ma_20 = np.mean(prices[-20:])
        ma_50 = np.mean(prices[-50:])
        std_20 = np.std(prices[-20:])
        
        if std_20 == 0:
            z_score = 0
        else:
            z_score = (prices[-1] - ma_20) / std_20
        
        # Calculate reversion probability based on z-score
        # Using empirical observation that extreme z-scores tend to revert
        if abs(z_score) > 2:
            reversion_prob = 0.8
            signal = 'buy' if z_score < -2 else 'sell'
        elif abs(z_score) > 1.5:
            reversion_prob = 0.65
            signal = 'buy' if z_score < -1.5 else 'sell'
        elif abs(z_score) > 1:
            reversion_prob = 0.55
            signal = 'weak_buy' if z_score < -1 else 'weak_sell'
        else:
            reversion_prob = 0.5
            signal = 'neutral'
        
        return {
            'signal': signal,
            'z_score': z_score,
            'reversion_prob': reversion_prob,
            'ma_20': ma_20,
            'ma_50': ma_50,
            'distance_from_ma': (prices[-1] - ma_20) / ma_20 * 100
        }
    
    def _breakout_analysis(self, prices: np.ndarray, highs: np.ndarray,
                          lows: np.ndarray, volume: np.ndarray) -> Dict:
        """Analyze breakout potential."""
        if len(prices) < 20:
            return {'potential': 'low', 'direction': 'none'}
        
        # Recent range
        recent_high = max(highs[-10:])
        recent_low = min(lows[-10:])
        range_pct = (recent_high - recent_low) / prices[-1] * 100
        
        # Longer-term range
        long_high = max(highs[-20:])
        long_low = min(lows[-20:])
        
        # Volume analysis
        recent_vol = np.mean(volume[-5:])
        avg_vol = np.mean(volume[-20:])
        vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1
        
        # Breakout conditions
        near_resistance = (long_high - prices[-1]) / prices[-1] < 0.02
        near_support = (prices[-1] - long_low) / prices[-1] < 0.02
        consolidating = range_pct < 5  # Tight range
        volume_building = vol_ratio > 1.2
        
        if consolidating and volume_building and near_resistance:
            return {
                'potential': 'high',
                'direction': 'up',
                'target': long_high * 1.05,
                'confidence': 0.7
            }
        elif consolidating and volume_building and near_support:
            return {
                'potential': 'high',
                'direction': 'down',
                'target': long_low * 0.95,
                'confidence': 0.7
            }
        elif consolidating:
            return {
                'potential': 'moderate',
                'direction': 'pending',
                'confidence': 0.5
            }
        
        return {'potential': 'low', 'direction': 'none', 'confidence': 0.3}


class StatisticalArbitrage:
    """
    Statistical arbitrage models inspired by Renaissance Technologies.
    """
    
    def __init__(self):
        self.lookback = 252  # 1 year of trading days
    
    def calculate_statistical_edge(self, df: pd.DataFrame) -> Dict:
        """Calculate statistical edge for trading."""
        close = df['close'].values
        
        if len(close) < 50:
            return {'edge': 0, 'confidence': 0}
        
        results = {
            'mean_reversion': self._mean_reversion_edge(close),
            'momentum': self._momentum_edge(close),
            'volatility': self._volatility_edge(close),
            'autocorrelation': self._autocorrelation_edge(close),
            'seasonality': self._seasonality_edge(close, df)
        }
        
        # Combine edges
        total_edge = 0
        total_weight = 0
        
        for name, edge_data in results.items():
            if edge_data['confidence'] > 0.5:
                total_edge += edge_data['edge'] * edge_data['confidence']
                total_weight += edge_data['confidence']
        
        combined_edge = total_edge / total_weight if total_weight > 0 else 0
        
        results['combined'] = {
            'edge': combined_edge,
            'confidence': total_weight / len(results)
        }
        
        return results
    
    def _mean_reversion_edge(self, prices: np.ndarray) -> Dict:
        """Calculate mean reversion trading edge."""
        returns = np.diff(prices) / prices[:-1]
        
        # Calculate Hurst exponent (simplified)
        # H < 0.5 indicates mean reversion
        lags = range(2, min(20, len(returns) // 4))
        tau = []
        
        for lag in lags:
            diff = returns[lag:] - returns[:-lag]
            tau.append(np.sqrt(np.mean(diff ** 2)))
        
        if len(tau) > 2 and len(lags) > 2:
            hurst = np.polyfit(np.log(list(lags)), np.log(tau), 1)[0]
        else:
            hurst = 0.5
        
        # Edge based on Hurst exponent
        if hurst < 0.4:
            edge = 0.1 * (0.5 - hurst) / 0.5  # Strong mean reversion
            confidence = 0.7
        elif hurst < 0.5:
            edge = 0.05 * (0.5 - hurst) / 0.5
            confidence = 0.5
        else:
            edge = 0
            confidence = 0.3
        
        return {
            'edge': edge,
            'confidence': confidence,
            'hurst': hurst,
            'interpretation': 'mean_reverting' if hurst < 0.5 else 'trending'
        }
    
    def _momentum_edge(self, prices: np.ndarray) -> Dict:
        """Calculate momentum trading edge."""
        if len(prices) < 60:
            return {'edge': 0, 'confidence': 0}
        
        # Multiple momentum lookbacks
        mom_10 = (prices[-1] / prices[-10] - 1)
        mom_20 = (prices[-1] / prices[-20] - 1)
        mom_60 = (prices[-1] / prices[-60] - 1)
        
        # Momentum consistency
        mom_aligned = np.sign(mom_10) == np.sign(mom_20) == np.sign(mom_60)
        
        # Calculate edge
        avg_momentum = (mom_10 + mom_20 / 2 + mom_60 / 6) / 3
        
        if mom_aligned:
            edge = avg_momentum * 0.2  # Scale down
            confidence = 0.65
        else:
            edge = avg_momentum * 0.1
            confidence = 0.4
        
        return {
            'edge': edge,
            'confidence': confidence,
            'mom_10': mom_10,
            'mom_20': mom_20,
            'mom_60': mom_60,
            'aligned': mom_aligned
        }
    
    def _volatility_edge(self, prices: np.ndarray) -> Dict:
        """Calculate volatility-based edge."""
        if len(prices) < 40:
            return {'edge': 0, 'confidence': 0}
        
        returns = np.diff(prices) / prices[:-1]
        
        # Recent vs historical volatility
        recent_vol = np.std(returns[-10:]) * np.sqrt(252)
        hist_vol = np.std(returns[-40:]) * np.sqrt(252)
        
        vol_ratio = recent_vol / hist_vol if hist_vol > 0 else 1
        
        # Volatility mean reverts
        if vol_ratio > 1.5:
            # High vol tends to decrease
            edge = 0.02
            signal = 'vol_decrease_expected'
        elif vol_ratio < 0.7:
            # Low vol tends to increase (breakout coming)
            edge = 0.01
            signal = 'vol_increase_expected'
        else:
            edge = 0
            signal = 'neutral'
        
        return {
            'edge': edge,
            'confidence': 0.6 if edge != 0 else 0.3,
            'recent_vol': recent_vol,
            'hist_vol': hist_vol,
            'vol_ratio': vol_ratio,
            'signal': signal
        }
    
    def _autocorrelation_edge(self, prices: np.ndarray) -> Dict:
        """Calculate autocorrelation-based edge."""
        if len(prices) < 30:
            return {'edge': 0, 'confidence': 0}
        
        returns = np.diff(prices) / prices[:-1]
        
        # Lag-1 autocorrelation
        if len(returns) > 1:
            autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
        else:
            autocorr = 0
        
        if np.isnan(autocorr):
            autocorr = 0
        
        # Significant positive autocorrelation = momentum
        # Significant negative autocorrelation = mean reversion
        if autocorr > 0.1:
            edge = autocorr * 0.05
            strategy = 'momentum'
        elif autocorr < -0.1:
            edge = -autocorr * 0.05  # Positive edge from mean reversion
            strategy = 'mean_reversion'
        else:
            edge = 0
            strategy = 'none'
        
        return {
            'edge': edge,
            'confidence': 0.5 + abs(autocorr),
            'autocorr': autocorr,
            'strategy': strategy
        }
    
    def _seasonality_edge(self, prices: np.ndarray, df: pd.DataFrame) -> Dict:
        """Calculate seasonality-based edge."""
        # Simple day-of-week effect
        if 'date' not in df.columns and df.index.name != 'date':
            return {'edge': 0, 'confidence': 0.3}
        
        try:
            if 'date' in df.columns:
                df['dow'] = pd.to_datetime(df['date']).dt.dayofweek
            else:
                df['dow'] = pd.to_datetime(df.index).dayofweek
            
            returns = df['close'].pct_change()
            
            # Day-of-week returns
            dow_returns = returns.groupby(df['dow']).mean()
            
            # Check for significant day effects
            current_dow = pd.Timestamp.now().dayofweek
            if current_dow in dow_returns.index:
                expected_return = dow_returns[current_dow]
                edge = expected_return * 0.5  # Scale down
            else:
                edge = 0
            
            return {
                'edge': edge,
                'confidence': 0.4,
                'dow_effects': dow_returns.to_dict() if hasattr(dow_returns, 'to_dict') else {}
            }
        except Exception:
            return {'edge': 0, 'confidence': 0.3}


class MarketRegimeDetector:
    """
    Hidden Markov Model-style regime detection.
    Identifies market states for adaptive strategy selection.
    """
    
    def __init__(self):
        self.n_regimes = 4
        self.current_regime = MarketRegime.SIDEWAYS
    
    def detect_regime(self, df: pd.DataFrame) -> Tuple[MarketRegime, Dict]:
        """Detect current market regime."""
        if len(df) < 50:
            return MarketRegime.SIDEWAYS, {'confidence': 0.3}
        
        close = df['close'].values
        returns = np.diff(close) / close[:-1]
        
        # Calculate regime indicators
        trend = self._calculate_trend(returns)
        volatility = self._calculate_volatility(returns)
        
        # Determine regime
        if trend > 0.02:  # Bullish
            if volatility > 0.02:
                regime = MarketRegime.BULL_VOLATILE
            else:
                regime = MarketRegime.BULL_QUIET
        elif trend < -0.02:  # Bearish
            if volatility > 0.02:
                regime = MarketRegime.BEAR_VOLATILE
            else:
                regime = MarketRegime.BEAR_QUIET
        else:  # Sideways
            if volatility > 0.03:
                regime = MarketRegime.CRISIS
            else:
                regime = MarketRegime.SIDEWAYS
        
        self.current_regime = regime
        
        details = {
            'trend': trend,
            'volatility': volatility,
            'confidence': self._calculate_confidence(trend, volatility),
            'regime_description': self._describe_regime(regime)
        }
        
        return regime, details
    
    def _calculate_trend(self, returns: np.ndarray) -> float:
        """Calculate trend strength."""
        if len(returns) < 20:
            return 0
        
        # EMA of returns
        weights = np.exp(np.linspace(-1, 0, 20))
        weights /= weights.sum()
        
        trend = np.average(returns[-20:], weights=weights)
        return trend
    
    def _calculate_volatility(self, returns: np.ndarray) -> float:
        """Calculate volatility level."""
        if len(returns) < 20:
            return 0.02
        
        return np.std(returns[-20:])
    
    def _calculate_confidence(self, trend: float, volatility: float) -> float:
        """Calculate regime detection confidence."""
        # Higher confidence when signals are clear
        trend_clarity = min(abs(trend) / 0.05, 1.0)
        vol_clarity = min(volatility / 0.03, 1.0)
        
        return (trend_clarity + vol_clarity) / 2
    
    def _describe_regime(self, regime: MarketRegime) -> str:
        """Get human-readable regime description."""
        descriptions = {
            MarketRegime.BULL_QUIET: "Steady uptrend with low volatility - Ideal for momentum strategies",
            MarketRegime.BULL_VOLATILE: "Strong uptrend but choppy - Use wider stops",
            MarketRegime.BEAR_QUIET: "Steady decline - Consider shorting or staying out",
            MarketRegime.BEAR_VOLATILE: "Sharp decline with high volatility - High risk environment",
            MarketRegime.SIDEWAYS: "Range-bound market - Mean reversion strategies work best",
            MarketRegime.CRISIS: "Extreme volatility - Reduce position sizes significantly"
        }
        return descriptions.get(regime, "Unknown regime")


class EnsemblePredictor:
    """
    Ensemble of prediction models for robust forecasting.
    Combines multiple approaches like Renaissance Technologies.
    """
    
    def __init__(self):
        self.models = {
            'linear_regression': self._linear_regression_predict,
            'momentum': self._momentum_predict,
            'mean_reversion': self._mean_reversion_predict,
            'volatility_adjusted': self._volatility_adjusted_predict,
            'pattern_based': self._pattern_based_predict
        }
        
        # Model weights (can be dynamically adjusted based on performance)
        self.weights = {
            'linear_regression': 0.15,
            'momentum': 0.25,
            'mean_reversion': 0.25,
            'volatility_adjusted': 0.20,
            'pattern_based': 0.15
        }
    
    def predict(self, df: pd.DataFrame, horizon: int = 5) -> Dict:
        """Generate ensemble prediction."""
        predictions = {}
        confidences = {}
        
        for name, model_func in self.models.items():
            try:
                pred, conf = model_func(df, horizon)
                predictions[name] = pred
                confidences[name] = conf
            except Exception as e:
                logger.warning(f"Model {name} failed: {e}")
                predictions[name] = 0
                confidences[name] = 0
        
        # Weighted ensemble
        weighted_pred = 0
        total_weight = 0
        
        for name, pred in predictions.items():
            weight = self.weights[name] * confidences[name]
            weighted_pred += pred * weight
            total_weight += weight
        
        ensemble_pred = weighted_pred / total_weight if total_weight > 0 else 0
        
        # Calculate prediction uncertainty
        pred_values = list(predictions.values())
        uncertainty = np.std(pred_values) if pred_values else 0
        
        return {
            'ensemble_prediction': ensemble_pred,
            'individual_predictions': predictions,
            'model_confidences': confidences,
            'uncertainty': uncertainty,
            'agreement': 1 - uncertainty / (abs(ensemble_pred) + 0.01)
        }
    
    def _linear_regression_predict(self, df: pd.DataFrame, horizon: int) -> Tuple[float, float]:
        """Simple linear regression forecast."""
        close = df['close'].values
        
        if len(close) < 20:
            return 0, 0.3
        
        # Fit linear trend
        x = np.arange(len(close[-20:]))
        slope, intercept = np.polyfit(x, close[-20:], 1)
        
        # Predict future
        future_x = len(close[-20:]) + horizon
        predicted_price = slope * future_x + intercept
        predicted_return = (predicted_price - close[-1]) / close[-1]
        
        # Confidence based on R-squared
        predicted = slope * x + intercept
        ss_res = np.sum((close[-20:] - predicted) ** 2)
        ss_tot = np.sum((close[-20:] - np.mean(close[-20:])) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return predicted_return, max(0.3, r_squared)
    
    def _momentum_predict(self, df: pd.DataFrame, horizon: int) -> Tuple[float, float]:
        """Momentum-based prediction."""
        close = df['close'].values
        
        if len(close) < 20:
            return 0, 0.3
        
        # Recent momentum
        mom_5 = (close[-1] / close[-5] - 1) if len(close) >= 5 else 0
        mom_10 = (close[-1] / close[-10] - 1) if len(close) >= 10 else 0
        mom_20 = (close[-1] / close[-20] - 1) if len(close) >= 20 else 0
        
        # Project momentum forward
        avg_mom = (mom_5 * 0.5 + mom_10 * 0.3 + mom_20 * 0.2)
        predicted_return = avg_mom * (horizon / 5)  # Scale by horizon
        
        # Confidence based on momentum consistency
        signs = [np.sign(mom_5), np.sign(mom_10), np.sign(mom_20)]
        consistency = sum(1 for s in signs if s == signs[0]) / 3
        
        return predicted_return, 0.3 + consistency * 0.4
    
    def _mean_reversion_predict(self, df: pd.DataFrame, horizon: int) -> Tuple[float, float]:
        """Mean reversion prediction."""
        close = df['close'].values
        
        if len(close) < 50:
            return 0, 0.3
        
        # Calculate deviation from mean
        ma_50 = np.mean(close[-50:])
        deviation = (close[-1] - ma_50) / ma_50
        
        # Expect reversion to mean
        reversion_speed = 0.1  # 10% reversion per period
        expected_reversion = -deviation * reversion_speed * horizon
        
        # Confidence based on historical reversion behavior
        confidence = 0.5 if abs(deviation) > 0.05 else 0.3
        
        return expected_reversion, confidence
    
    def _volatility_adjusted_predict(self, df: pd.DataFrame, horizon: int) -> Tuple[float, float]:
        """Volatility-adjusted prediction."""
        close = df['close'].values
        
        if len(close) < 30:
            return 0, 0.3
        
        returns = np.diff(close) / close[:-1]
        
        # Current volatility regime
        recent_vol = np.std(returns[-10:])
        hist_vol = np.std(returns[-30:])
        
        # In low vol, expect continuation; in high vol, expect mean reversion
        vol_ratio = recent_vol / hist_vol if hist_vol > 0 else 1
        
        recent_return = np.mean(returns[-5:])
        
        if vol_ratio < 0.8:  # Low vol - momentum
            predicted_return = recent_return * horizon * 0.5
            confidence = 0.6
        elif vol_ratio > 1.5:  # High vol - mean reversion
            predicted_return = -recent_return * horizon * 0.3
            confidence = 0.5
        else:
            predicted_return = recent_return * horizon * 0.2
            confidence = 0.4
        
        return predicted_return, confidence
    
    def _pattern_based_predict(self, df: pd.DataFrame, horizon: int) -> Tuple[float, float]:
        """Pattern-based prediction."""
        pattern_recognizer = PatternRecognizer()
        patterns = pattern_recognizer.detect_patterns(df)
        
        # Use trend pattern
        trend = patterns.get('trend', {})
        direction = trend.get('direction', 'sideways')
        strength = trend.get('strength', 0)
        
        if direction == 'bullish':
            predicted_return = strength * 0.02 * (horizon / 5)
        elif direction == 'bearish':
            predicted_return = -strength * 0.02 * (horizon / 5)
        else:
            predicted_return = 0
        
        # Use price pattern
        price_pattern = patterns.get('price_pattern', {})
        pattern_bias = price_pattern.get('bias', 'neutral')
        pattern_conf = price_pattern.get('confidence', 0)
        
        if pattern_bias == 'bullish':
            predicted_return += 0.01 * pattern_conf
        elif pattern_bias == 'bearish':
            predicted_return -= 0.01 * pattern_conf
        
        confidence = 0.3 + strength * 0.3 + pattern_conf * 0.2
        
        return predicted_return, min(confidence, 0.8)


class MLPredictor:
    """
    Main ML Predictor class combining all components.
    Inspired by Renaissance Technologies' approach.
    """
    
    def __init__(self):
        self.pattern_recognizer = PatternRecognizer()
        self.stat_arb = StatisticalArbitrage()
        self.regime_detector = MarketRegimeDetector()
        self.ensemble = EnsemblePredictor()
    
    def predict(self, df: pd.DataFrame, symbol: str, 
                horizon: str = '5d') -> PredictionResult:
        """
        Generate comprehensive ML prediction.
        
        Args:
            df: OHLCV DataFrame
            symbol: Stock symbol
            horizon: Prediction horizon ('1d', '5d', '20d')
        
        Returns:
            PredictionResult with all predictions and analysis
        """
        horizon_days = {'1d': 1, '5d': 5, '20d': 20}.get(horizon, 5)
        current_price = df['close'].iloc[-1]
        
        # Detect market regime
        regime, regime_details = self.regime_detector.detect_regime(df)
        
        # Recognize patterns
        patterns = self.pattern_recognizer.detect_patterns(df)
        
        # Calculate statistical edge
        stat_edge = self.stat_arb.calculate_statistical_edge(df)
        
        # Generate ensemble prediction
        ensemble_result = self.ensemble.predict(df, horizon_days)
        
        # Calculate predicted price
        predicted_return = ensemble_result['ensemble_prediction']
        predicted_price = current_price * (1 + predicted_return)
        
        # Calculate confidence interval
        uncertainty = ensemble_result['uncertainty']
        ci_lower = current_price * (1 + predicted_return - 2 * uncertainty)
        ci_upper = current_price * (1 + predicted_return + 2 * uncertainty)
        
        # Calculate probabilities
        prob_up, prob_down, prob_sideways = self._calculate_probabilities(
            predicted_return, uncertainty
        )
        
        # Calculate risk metrics
        returns = df['close'].pct_change().dropna()
        predicted_vol = returns.std() * np.sqrt(252)
        var_95 = current_price * (predicted_vol * 1.645 / np.sqrt(252 / horizon_days))
        
        # Expected Sharpe
        rf_rate = 0.05  # Risk-free rate
        expected_sharpe = (predicted_return * (252 / horizon_days) - rf_rate) / predicted_vol if predicted_vol > 0 else 0
        
        # Overall confidence
        model_agreement = ensemble_result['agreement']
        regime_confidence = regime_details['confidence']
        overall_confidence = (model_agreement + regime_confidence) / 2
        
        # Signal strength
        if overall_confidence > 0.7 and abs(predicted_return) > 0.02:
            signal_strength = 'strong'
        elif overall_confidence > 0.5 and abs(predicted_return) > 0.01:
            signal_strength = 'moderate'
        else:
            signal_strength = 'weak'
        
        # Pattern description
        price_pattern = patterns.get('price_pattern', {}).get('pattern', 'none')
        trend_direction = patterns.get('trend', {}).get('direction', 'unknown')
        pattern_detected = f"{trend_direction}_{price_pattern}"
        
        # Anomaly score (how unusual is current price action)
        mean_reversion = patterns.get('mean_reversion_signal', {})
        z_score = abs(mean_reversion.get('z_score', 0))
        anomaly_score = min(z_score / 3, 1.0)  # Normalize to 0-1
        
        return PredictionResult(
            symbol=symbol,
            prediction_horizon=horizon,
            predicted_return=predicted_return,
            predicted_price=predicted_price,
            confidence_interval=(ci_lower, ci_upper),
            prob_up=prob_up,
            prob_down=prob_down,
            prob_sideways=prob_sideways,
            regime=regime,
            pattern_detected=pattern_detected,
            anomaly_score=anomaly_score,
            model_predictions=ensemble_result['individual_predictions'],
            model_weights=self.ensemble.weights,
            predicted_volatility=predicted_vol,
            var_95=var_95,
            expected_sharpe=expected_sharpe,
            overall_confidence=overall_confidence,
            signal_strength=signal_strength
        )
    
    def _calculate_probabilities(self, predicted_return: float, 
                                  uncertainty: float) -> Tuple[float, float, float]:
        """Calculate probability of up/down/sideways movement."""
        # Using normal distribution assumption
        if uncertainty < 0.001:
            uncertainty = 0.01
        
        # Define sideways as within +/- 1%
        threshold = 0.01
        
        # Probability of being above threshold
        z_up = (predicted_return - threshold) / uncertainty
        z_down = (predicted_return + threshold) / uncertainty
        
        # Simple probability estimation
        prob_up = 0.5 + 0.5 * np.tanh(z_up)
        prob_down = 0.5 - 0.5 * np.tanh(z_down)
        prob_sideways = 1 - prob_up - prob_down
        
        # Ensure probabilities are valid
        prob_sideways = max(0, prob_sideways)
        total = prob_up + prob_down + prob_sideways
        
        return prob_up / total, prob_down / total, prob_sideways / total
    
    def get_prediction_summary(self, result: PredictionResult) -> str:
        """Get human-readable prediction summary."""
        direction = "? UP" if result.predicted_return > 0.005 else "? DOWN" if result.predicted_return < -0.005 else "➡️ SIDEWAYS"
        
        summary = f"""
╔══════════════════════════════════════════════════════════════╗
║              ML PREDICTION: {result.symbol:^10}                    ║
╠══════════════════════════════════════════════════════════════╣
║  Horizon: {result.prediction_horizon}                                              ║
║  Direction: {direction}                                        ║
║  Signal Strength: {result.signal_strength.upper():^10}                             ║
╠══════════════════════════════════════════════════════════════╣
║  PRICE FORECAST                                              ║
║  Current:     ₹{result.predicted_price / (1 + result.predicted_return):>12,.2f}                        ║
║  Predicted:   ₹{result.predicted_price:>12,.2f}  ({result.predicted_return:+.2%})                ║
║  95% CI:      ₹{result.confidence_interval[0]:>8,.2f} - ₹{result.confidence_interval[1]:>8,.2f}              ║
╠══════════════════════════════════════════════════════════════╣
║  PROBABILITIES                                               ║
║  Prob Up:     {result.prob_up:>6.1%}                                       ║
║  Prob Down:   {result.prob_down:>6.1%}                                       ║
║  Prob Flat:   {result.prob_sideways:>6.1%}                                       ║
╠══════════════════════════════════════════════════════════════╣
║  MARKET REGIME: {result.regime.value:<20}                    ║
║  Pattern: {result.pattern_detected:<25}                     ║
║  Anomaly Score: {result.anomaly_score:.2f}                                     ║
╠══════════════════════════════════════════════════════════════╣
║  RISK METRICS                                                ║
║  Predicted Vol: {result.predicted_volatility:.1%}                                    ║
║  VaR (95%):     ₹{result.var_95:>10,.2f}                               ║
║  Expected Sharpe: {result.expected_sharpe:>6.2f}                                  ║
╠══════════════════════════════════════════════════════════════╣
║  Confidence: {result.overall_confidence:.1%}                                        ║
╚══════════════════════════════════════════════════════════════╝
"""
        return summary