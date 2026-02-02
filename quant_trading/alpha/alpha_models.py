"""
Alpha Models Module
==================
Signal generation, probability models, and diversified strategies.

Alpha models generate trading signals based on features.
Multiple strategies are combined for diversification.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Trading signal types."""
    STRONG_BUY = 2
    BUY = 1
    NEUTRAL = 0
    SELL = -1
    STRONG_SELL = -2


@dataclass
class Signal:
    """Trading signal from an alpha model."""
    symbol: str
    signal_type: SignalType
    confidence: float  # 0 to 1
    strategy_name: str
    timestamp: pd.Timestamp
    metadata: Dict = field(default_factory=dict)
    
    @property
    def strength(self) -> float:
        """Signal strength (-2 to 2) weighted by confidence."""
        return self.signal_type.value * self.confidence
    
    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'signal': self.signal_type.name,
            'signal_value': self.signal_type.value,
            'confidence': self.confidence,
            'strength': self.strength,
            'strategy': self.strategy_name,
            'timestamp': self.timestamp,
            **self.metadata
        }


@dataclass 
class AlphaOutput:
    """Combined output from all alpha models."""
    symbol: str
    signals: List[Signal]
    combined_signal: float  # -1 to 1
    combined_confidence: float  # 0 to 1
    position_suggestion: str  # 'long', 'short', 'flat'
    timestamp: pd.Timestamp
    
    @property
    def should_trade(self) -> bool:
        """Whether the combined signal is strong enough to trade."""
        return abs(self.combined_signal) >= 0.3 and self.combined_confidence >= 0.5


class AlphaModel(ABC):
    """Abstract base class for alpha models."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def generate_signal(self, features: pd.DataFrame, symbol: str) -> Signal:
        """Generate trading signal from features."""
        pass
    
    def _create_signal(self, symbol: str, signal_value: float, 
                       confidence: float, **metadata) -> Signal:
        """Helper to create signal with proper type."""
        # Map continuous signal to discrete type
        if signal_value >= 0.7:
            signal_type = SignalType.STRONG_BUY
        elif signal_value >= 0.3:
            signal_type = SignalType.BUY
        elif signal_value <= -0.7:
            signal_type = SignalType.STRONG_SELL
        elif signal_value <= -0.3:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.NEUTRAL
        
        return Signal(
            symbol=symbol,
            signal_type=signal_type,
            confidence=min(max(confidence, 0), 1),  # Clamp to [0, 1]
            strategy_name=self.name,
            timestamp=pd.Timestamp.now(),
            metadata=metadata
        )


class TrendFollowingAlpha(AlphaModel):
    """
    Trend Following Strategy
    
    Core principle: "The trend is your friend"
    - Uses multiple moving average crossovers
    - Confirms with ADX for trend strength
    - Higher weight when volatility is moderate
    """
    
    def __init__(self, lookback: int = 20):
        super().__init__("TrendFollowing")
        self.lookback = lookback
    
    def generate_signal(self, features: pd.DataFrame, symbol: str) -> Signal:
        """Generate trend following signal."""
        latest = features.iloc[-1]
        
        # Component signals
        signals = []
        weights = []
        
        # 1. Moving Average Crossovers
        if 'close_to_sma_20' in features.columns and 'close_to_sma_50' in features.columns:
            # Price above/below MAs
            ma_signal = 0
            if latest['close_to_sma_20'] > 0 and latest['close_to_sma_50'] > 0:
                ma_signal = 1  # Both bullish
            elif latest['close_to_sma_20'] < 0 and latest['close_to_sma_50'] < 0:
                ma_signal = -1  # Both bearish
            elif latest['close_to_sma_20'] > latest['close_to_sma_50']:
                ma_signal = 0.5  # Short-term bullish
            else:
                ma_signal = -0.5  # Short-term bearish
            
            signals.append(ma_signal)
            weights.append(0.3)
        
        # 2. MACD Signal
        if 'macd_hist' in features.columns:
            macd_hist = latest['macd_hist']
            macd_signal = np.tanh(macd_hist / 10)  # Normalize
            signals.append(macd_signal)
            weights.append(0.25)
        
        # 3. ADX Trend Strength
        if 'adx' in features.columns and 'plus_di' in features.columns:
            adx = latest['adx']
            plus_di = latest['plus_di']
            minus_di = latest['minus_di']
            
            # ADX > 25 indicates strong trend
            if adx > 25:
                trend_signal = 1 if plus_di > minus_di else -1
                trend_strength = min(adx / 50, 1)  # Scale ADX
            else:
                trend_signal = 0
                trend_strength = 0.3
            
            signals.append(trend_signal * trend_strength)
            weights.append(0.25)
        
        # 4. Higher Highs / Lower Lows
        if 'trend_score' in features.columns:
            trend_score = latest['trend_score']
            signals.append(trend_score / 2)  # Scale to [-0.5, 0.5]
            weights.append(0.2)
        
        # Combine signals
        if signals:
            weights = np.array(weights) / sum(weights)  # Normalize weights
            combined = sum(s * w for s, w in zip(signals, weights))
        else:
            combined = 0
        
        # Calculate confidence based on signal agreement
        if signals:
            agreement = 1 - np.std([s for s in signals if s != 0]) if any(s != 0 for s in signals) else 0.5
            confidence = agreement * 0.7 + 0.3  # Base confidence of 0.3
        else:
            confidence = 0.3
        
        # Reduce confidence in choppy markets (low ADX)
        if 'adx' in features.columns and latest['adx'] < 20:
            confidence *= 0.7
        
        return self._create_signal(
            symbol=symbol,
            signal_value=combined,
            confidence=confidence,
            ma_position=latest.get('close_to_sma_20', 0),
            adx=latest.get('adx', 0)
        )


class MeanReversionAlpha(AlphaModel):
    """
    Mean Reversion Strategy
    
    Core principle: "Prices revert to their mean"
    - Uses Bollinger Bands and Z-scores
    - Better in ranging/choppy markets
    - Contrarian approach
    """
    
    def __init__(self, zscore_threshold: float = 2.0):
        super().__init__("MeanReversion")
        self.zscore_threshold = zscore_threshold
    
    def generate_signal(self, features: pd.DataFrame, symbol: str) -> Signal:
        """Generate mean reversion signal."""
        latest = features.iloc[-1]
        
        signals = []
        weights = []
        
        # 1. Bollinger Band Position
        if 'bb_pct_b' in features.columns:
            pct_b = latest['bb_pct_b']
            
            # Below lower band = oversold (buy signal)
            # Above upper band = overbought (sell signal)
            if pct_b < 0:
                bb_signal = 1  # Below lower band
            elif pct_b > 1:
                bb_signal = -1  # Above upper band
            else:
                # Linear interpolation between bands
                bb_signal = 1 - 2 * pct_b
            
            signals.append(bb_signal)
            weights.append(0.3)
        
        # 2. RSI Extremes
        if 'rsi' in features.columns:
            rsi = latest['rsi']
            
            if rsi < 30:
                rsi_signal = 1  # Oversold
            elif rsi > 70:
                rsi_signal = -1  # Overbought
            else:
                rsi_signal = (50 - rsi) / 50  # Linear scale
            
            signals.append(rsi_signal)
            weights.append(0.25)
        
        # 3. Price Z-Score
        if 'close_zscore' in features.columns:
            zscore = latest['close_zscore']
            
            # Contrarian: high z-score = sell, low z-score = buy
            zscore_signal = -np.tanh(zscore / self.zscore_threshold)
            signals.append(zscore_signal)
            weights.append(0.25)
        
        # 4. Stochastic Oversold/Overbought
        if 'stoch_k' in features.columns:
            stoch = latest['stoch_k']
            
            if stoch < 20:
                stoch_signal = 1  # Oversold
            elif stoch > 80:
                stoch_signal = -1  # Overbought
            else:
                stoch_signal = (50 - stoch) / 50
            
            signals.append(stoch_signal)
            weights.append(0.2)
        
        # Combine signals
        if signals:
            weights = np.array(weights) / sum(weights)
            combined = sum(s * w for s, w in zip(signals, weights))
        else:
            combined = 0
        
        # Confidence based on extremity of indicators
        extremity = 0
        if 'rsi' in features.columns:
            extremity += abs(latest['rsi'] - 50) / 50
        if 'close_zscore' in features.columns:
            extremity += min(abs(latest['close_zscore']) / 2, 1)
        
        confidence = 0.3 + 0.5 * (extremity / 2) if extremity > 0 else 0.3
        
        # Mean reversion works better in low-trend environments
        if 'adx' in features.columns and latest['adx'] > 30:
            confidence *= 0.6  # Reduce confidence in strong trends
        
        # Check Hurst exponent - mean reversion works when Hurst < 0.5
        if 'hurst' in features.columns:
            hurst = latest['hurst']
            if hurst < 0.45:
                confidence *= 1.2  # Boost for mean-reverting regime
            elif hurst > 0.55:
                confidence *= 0.7  # Reduce for trending regime
        
        return self._create_signal(
            symbol=symbol,
            signal_value=combined,
            confidence=min(confidence, 1),
            rsi=latest.get('rsi', 50),
            zscore=latest.get('close_zscore', 0)
        )


class MomentumAlpha(AlphaModel):
    """
    Momentum Strategy
    
    Core principle: "Winners keep winning, losers keep losing"
    - Uses price momentum over multiple timeframes
    - Confirms with volume momentum
    - Time-series and cross-sectional momentum
    """
    
    def __init__(self, lookback: int = 10):
        super().__init__("Momentum")
        self.lookback = lookback
    
    def generate_signal(self, features: pd.DataFrame, symbol: str) -> Signal:
        """Generate momentum signal."""
        latest = features.iloc[-1]
        
        signals = []
        weights = []
        
        # 1. Price Momentum (multiple timeframes)
        momentum_cols = [col for col in features.columns if col.startswith('momentum_')]
        for col in momentum_cols:
            mom = latest[col]
            mom_signal = np.tanh(mom * 10)  # Scale and bound
            signals.append(mom_signal)
            weights.append(0.15)
        
        # 2. Rate of Change
        if 'roc_10' in features.columns:
            roc = latest['roc_10']
            roc_signal = np.tanh(roc / 5)
            signals.append(roc_signal)
            weights.append(0.2)
        
        # 3. MACD Momentum
        if 'macd_hist' in features.columns:
            # Check if MACD histogram is accelerating
            hist_recent = features['macd_hist'].tail(5)
            if len(hist_recent) >= 3:
                accel = hist_recent.iloc[-1] - hist_recent.iloc[0]
                accel_signal = np.tanh(accel / 5)
                signals.append(accel_signal)
                weights.append(0.2)
        
        # 4. Volume Confirmation
        if 'volume_ma_ratio' in features.columns:
            vol_ratio = latest['volume_ma_ratio']
            
            # Strong momentum should have above-average volume
            if 'momentum_10' in features.columns:
                mom = latest['momentum_10']
                if vol_ratio > 1.2 and mom > 0:
                    vol_signal = 0.5  # Bullish volume confirmation
                elif vol_ratio > 1.2 and mom < 0:
                    vol_signal = -0.5  # Bearish volume confirmation
                else:
                    vol_signal = 0
                
                signals.append(vol_signal)
                weights.append(0.15)
        
        # 5. Money Flow
        if 'mfi' in features.columns:
            mfi = latest['mfi']
            mfi_signal = (mfi - 50) / 50
            signals.append(mfi_signal)
            weights.append(0.15)
        
        # Combine signals
        if signals:
            weights = np.array(weights) / sum(weights)
            combined = sum(s * w for s, w in zip(signals, weights))
        else:
            combined = 0
        
        # Confidence based on momentum consistency
        if signals:
            signs = [np.sign(s) for s in signals if s != 0]
            if signs:
                consistency = abs(sum(signs)) / len(signs)
                confidence = 0.4 + 0.5 * consistency
            else:
                confidence = 0.4
        else:
            confidence = 0.3
        
        return self._create_signal(
            symbol=symbol,
            signal_value=combined,
            confidence=confidence,
            momentum_10d=latest.get('momentum_10', 0)
        )


class VolatilityAlpha(AlphaModel):
    """
    Volatility Strategy
    
    Core principle: "Volatility is cyclical and mean-reverts"
    - Sells volatility when it's high
    - Buys volatility when it's low
    - Uses VIX and realized volatility
    """
    
    def __init__(self):
        super().__init__("Volatility")
    
    def generate_signal(self, features: pd.DataFrame, symbol: str) -> Signal:
        """Generate volatility-based signal."""
        latest = features.iloc[-1]
        
        signals = []
        weights = []
        
        # 1. VIX Signal (if available)
        if 'vix' in features.columns and 'vix_zscore' in features.columns:
            vix = latest['vix']
            vix_zscore = latest['vix_zscore']
            
            # High VIX = fear = potential buying opportunity
            # Low VIX = complacency = potential risk
            if vix_zscore > 1.5:
                vix_signal = 0.7  # High fear, potential buy
            elif vix_zscore < -1.5:
                vix_signal = -0.5  # Low fear, potential caution
            else:
                vix_signal = -vix_zscore / 3
            
            signals.append(vix_signal)
            weights.append(0.35)
        
        # 2. Realized vs Implied Volatility
        if 'volatility' in features.columns:
            vol = latest['volatility']
            
            # Look at vol regime
            vol_series = features['volatility'].tail(60)
            vol_percentile = (vol_series < vol).mean()
            
            if vol_percentile > 0.8:
                # High volatility percentile - expect mean reversion
                vol_signal = 0.3  # Slight bullish (vol to decrease)
            elif vol_percentile < 0.2:
                # Low volatility - expect increase
                vol_signal = -0.3  # Slight bearish (vol to increase)
            else:
                vol_signal = 0
            
            signals.append(vol_signal)
            weights.append(0.3)
        
        # 3. Bollinger Bandwidth (volatility expansion/contraction)
        if 'bb_bandwidth' in features.columns:
            bw = latest['bb_bandwidth']
            bw_series = features['bb_bandwidth'].tail(60)
            bw_percentile = (bw_series < bw).mean()
            
            # Squeeze (low bandwidth) often precedes big moves
            if bw_percentile < 0.2:
                # Volatility squeeze - direction uncertain but move coming
                # Use momentum to determine direction
                if 'momentum_10' in features.columns:
                    squeeze_signal = np.sign(latest['momentum_10']) * 0.4
                else:
                    squeeze_signal = 0
            else:
                squeeze_signal = 0
            
            signals.append(squeeze_signal)
            weights.append(0.2)
        
        # 4. ATR Regime
        if 'atr_pct' in features.columns:
            atr_pct = latest['atr_pct']
            atr_series = features['atr_pct'].tail(60)
            atr_percentile = (atr_series < atr_pct).mean()
            
            # Very high ATR might indicate panic selling
            if atr_percentile > 0.9:
                atr_signal = 0.3  # Potential capitulation
            else:
                atr_signal = 0
            
            signals.append(atr_signal)
            weights.append(0.15)
        
        # Combine signals
        if signals:
            weights = np.array(weights) / sum(weights)
            combined = sum(s * w for s, w in zip(signals, weights))
        else:
            combined = 0
        
        # Confidence is moderate for volatility signals
        confidence = 0.5 if signals else 0.3
        
        # Boost confidence in extreme volatility regimes
        if 'vix' in features.columns:
            vix = latest['vix']
            if vix > 25 or vix < 12:
                confidence += 0.15
        
        return self._create_signal(
            symbol=symbol,
            signal_value=combined,
            confidence=min(confidence, 1),
            vix=latest.get('vix', 0),
            volatility=latest.get('volatility', 0)
        )


class AlphaEngine:
    """
    Main alpha engine combining multiple strategies.
    
    Responsibilities:
    - Run multiple alpha models
    - Combine signals using ensemble methods
    - Apply regime detection
    - Output final trading signals
    """
    
    def __init__(self, config=None):
        from ..config import AlphaConfig
        self.config = config or AlphaConfig()
        
        # Initialize alpha models
        self.models: Dict[str, AlphaModel] = {
            'trend_following': TrendFollowingAlpha(lookback=self.config.trend_lookback),
            'mean_reversion': MeanReversionAlpha(zscore_threshold=self.config.mean_reversion_zscore),
            'momentum': MomentumAlpha(lookback=self.config.momentum_lookback),
            'volatility': VolatilityAlpha()
        }
        
        # Strategy weights from config
        self.weights = self.config.strategy_weights
    
    def generate_signals(self, features: pd.DataFrame, symbol: str) -> AlphaOutput:
        """
        Generate combined trading signal for a symbol.
        
        Args:
            features: DataFrame with computed features
            symbol: Symbol name
            
        Returns:
            AlphaOutput with combined signal and individual signals
        """
        signals = []
        
        # Generate signal from each model
        for name, model in self.models.items():
            try:
                signal = model.generate_signal(features, symbol)
                signals.append(signal)
                logger.debug(f"{name} signal for {symbol}: {signal.signal_type.name} "
                            f"(conf: {signal.confidence:.2f})")
            except Exception as e:
                logger.warning(f"Error in {name} model for {symbol}: {e}")
        
        # Combine signals based on ensemble method
        if self.config.ensemble_method == 'weighted_average':
            combined_signal, combined_confidence = self._weighted_average_ensemble(signals)
        elif self.config.ensemble_method == 'voting':
            combined_signal, combined_confidence = self._voting_ensemble(signals)
        else:
            combined_signal, combined_confidence = self._weighted_average_ensemble(signals)
        
        # Determine position suggestion
        if combined_signal > self.config.position_threshold and combined_confidence > 0.5:
            position = 'long'
        elif combined_signal < -self.config.position_threshold and combined_confidence > 0.5:
            position = 'short'
        else:
            position = 'flat'
        
        return AlphaOutput(
            symbol=symbol,
            signals=signals,
            combined_signal=combined_signal,
            combined_confidence=combined_confidence,
            position_suggestion=position,
            timestamp=pd.Timestamp.now()
        )
    
    def _weighted_average_ensemble(self, signals: List[Signal]) -> Tuple[float, float]:
        """Combine signals using weighted average."""
        if not signals:
            return 0.0, 0.0
        
        total_weight = 0
        weighted_signal = 0
        weighted_confidence = 0
        
        for signal in signals:
            weight = self.weights.get(signal.strategy_name.lower().replace(' ', '_'), 0.25)
            weighted_signal += signal.strength * weight
            weighted_confidence += signal.confidence * weight
            total_weight += weight
        
        if total_weight > 0:
            combined_signal = weighted_signal / total_weight
            combined_confidence = weighted_confidence / total_weight
        else:
            combined_signal = 0
            combined_confidence = 0
        
        # Clamp signal to [-1, 1]
        combined_signal = max(-1, min(1, combined_signal))
        
        return combined_signal, combined_confidence
    
    def _voting_ensemble(self, signals: List[Signal]) -> Tuple[float, float]:
        """Combine signals using majority voting."""
        if not signals:
            return 0.0, 0.0
        
        # Count votes weighted by confidence
        buy_votes = sum(s.confidence for s in signals if s.signal_type.value > 0)
        sell_votes = sum(s.confidence for s in signals if s.signal_type.value < 0)
        total_conf = sum(s.confidence for s in signals)
        
        if total_conf > 0:
            signal = (buy_votes - sell_votes) / total_conf
            confidence = max(buy_votes, sell_votes) / total_conf
        else:
            signal = 0
            confidence = 0
        
        return signal, confidence
    
    def get_regime(self, features: pd.DataFrame) -> str:
        """
        Detect market regime for dynamic strategy weighting.
        
        Returns:
            'trending', 'mean_reverting', or 'volatile'
        """
        latest = features.iloc[-1]
        
        # Check ADX for trend
        adx = latest.get('adx', 0)
        
        # Check Hurst for mean reversion vs trending
        hurst = latest.get('hurst', 0.5)
        
        # Check VIX/volatility for volatile regime
        vix = latest.get('vix', 15)
        vol = latest.get('volatility', 0.2)
        
        if vix > 25 or vol > 0.3:
            return 'volatile'
        elif adx > 30 and hurst > 0.5:
            return 'trending'
        elif hurst < 0.45:
            return 'mean_reverting'
        else:
            return 'neutral'
    
    def adjust_weights_for_regime(self, regime: str):
        """Dynamically adjust strategy weights based on regime."""
        if regime == 'trending':
            self.weights = {
                'trend_following': 0.45,
                'mean_reversion': 0.10,
                'momentum': 0.35,
                'volatility': 0.10
            }
        elif regime == 'mean_reverting':
            self.weights = {
                'trend_following': 0.15,
                'mean_reversion': 0.45,
                'momentum': 0.20,
                'volatility': 0.20
            }
        elif regime == 'volatile':
            self.weights = {
                'trend_following': 0.20,
                'mean_reversion': 0.20,
                'momentum': 0.15,
                'volatility': 0.45
            }
        else:
            # Reset to config defaults
            self.weights = self.config.strategy_weights
