"""
Feature Engineering Module
==========================
Mathematical transforms and technical indicators for alpha generation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FeatureSet:
    """Container for computed features."""
    symbol: str
    features: pd.DataFrame
    feature_names: List[str]
    timestamp: pd.Timestamp


class TechnicalIndicators:
    """Technical analysis indicators."""
    
    @staticmethod
    def sma(prices: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average."""
        return prices.rolling(window=period, min_periods=1).mean()
    
    @staticmethod
    def ema(prices: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average."""
        return prices.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    @staticmethod
    def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Moving Average Convergence Divergence."""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'macd_signal': signal_line,
            'macd_hist': histogram
        }
    
    @staticmethod
    def bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """Bollinger Bands."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        
        # Percent B: where price is relative to bands
        pct_b = (prices - lower) / (upper - lower)
        
        # Bandwidth: volatility measure
        bandwidth = (upper - lower) / sma
        
        return {
            'bb_upper': upper,
            'bb_middle': sma,
            'bb_lower': lower,
            'bb_pct_b': pct_b,
            'bb_bandwidth': bandwidth
        }
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range."""
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Dict[str, pd.Series]:
        """Average Directional Index."""
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        atr = TechnicalIndicators.atr(high, low, close, period)
        
        plus_di = 100 * (plus_dm.ewm(span=period).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=period).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1)
        adx = dx.ewm(span=period).mean()
        
        return {
            'adx': adx,
            'plus_di': plus_di,
            'minus_di': minus_di
        }
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                   k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """Stochastic Oscillator."""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, 1)
        stoch_d = stoch_k.rolling(window=d_period).mean()
        
        return {
            'stoch_k': stoch_k,
            'stoch_d': stoch_d
        }
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume."""
        direction = np.sign(close.diff())
        obv = (direction * volume).cumsum()
        return obv
    
    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Volume Weighted Average Price."""
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        return vwap


class StatisticalFeatures:
    """Statistical and mathematical features."""
    
    @staticmethod
    def returns(prices: pd.Series, period: int = 1) -> pd.Series:
        """Calculate returns over period."""
        return prices.pct_change(periods=period)
    
    @staticmethod
    def log_returns(prices: pd.Series, period: int = 1) -> pd.Series:
        """Calculate log returns over period."""
        return np.log(prices / prices.shift(period))
    
    @staticmethod
    def volatility(returns: pd.Series, window: int = 20, annualize: bool = True) -> pd.Series:
        """Rolling volatility (standard deviation of returns)."""
        vol = returns.rolling(window=window).std()
        if annualize:
            vol = vol * np.sqrt(252)  # Annualize
        return vol
    
    @staticmethod
    def realized_volatility(high: pd.Series, low: pd.Series, close: pd.Series, 
                           window: int = 20) -> pd.Series:
        """Parkinson realized volatility estimator."""
        log_hl = np.log(high / low)
        parkinson = np.sqrt((1 / (4 * np.log(2))) * (log_hl ** 2).rolling(window).mean())
        return parkinson * np.sqrt(252)  # Annualize
    
    @staticmethod
    def zscore(series: pd.Series, window: int = 20) -> pd.Series:
        """Rolling Z-score."""
        mean = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        return (series - mean) / std.replace(0, 1)
    
    @staticmethod
    def skewness(returns: pd.Series, window: int = 60) -> pd.Series:
        """Rolling skewness."""
        return returns.rolling(window=window).skew()
    
    @staticmethod
    def kurtosis(returns: pd.Series, window: int = 60) -> pd.Series:
        """Rolling kurtosis."""
        return returns.rolling(window=window).kurt()
    
    @staticmethod
    def drawdown(prices: pd.Series) -> Dict[str, pd.Series]:
        """Calculate drawdown from peak."""
        running_max = prices.cummax()
        drawdown = (prices - running_max) / running_max
        
        return {
            'drawdown': drawdown,
            'drawdown_pct': drawdown * 100
        }
    
    @staticmethod
    def autocorrelation(returns: pd.Series, lag: int = 1, window: int = 60) -> pd.Series:
        """Rolling autocorrelation."""
        return returns.rolling(window=window).apply(
            lambda x: x.autocorr(lag=lag) if len(x) > lag else 0,
            raw=False
        )
    
    @staticmethod
    def hurst_exponent(prices: pd.Series, window: int = 100) -> pd.Series:
        """Rolling Hurst exponent (trend vs mean-reversion indicator)."""
        def calc_hurst(x):
            if len(x) < 20:
                return 0.5
            
            lags = range(2, min(20, len(x) // 2))
            tau = [np.sqrt(np.std(np.subtract(x[lag:], x[:-lag]))) for lag in lags]
            
            if len(tau) < 2 or min(tau) == 0:
                return 0.5
            
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        
        return prices.rolling(window=window).apply(calc_hurst, raw=True)


class MomentumFeatures:
    """Momentum and trend features."""
    
    @staticmethod
    def momentum(prices: pd.Series, period: int = 10) -> pd.Series:
        """Price momentum over period."""
        return prices / prices.shift(period) - 1
    
    @staticmethod
    def rate_of_change(prices: pd.Series, period: int = 10) -> pd.Series:
        """Rate of change."""
        return ((prices - prices.shift(period)) / prices.shift(period)) * 100
    
    @staticmethod
    def trend_strength(prices: pd.Series, period: int = 20) -> pd.Series:
        """Trend strength indicator (-1 to 1)."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        # Distance from SMA normalized by volatility
        return (prices - sma) / (std * 2).replace(0, 1)
    
    @staticmethod
    def crossover_signal(fast: pd.Series, slow: pd.Series) -> pd.Series:
        """Generate crossover signals: 1 for bullish, -1 for bearish, 0 for no signal."""
        signal = pd.Series(0, index=fast.index)
        
        # Current position relative to each other
        position = (fast > slow).astype(int)
        
        # Detect crossovers
        signal = position.diff()
        
        return signal
    
    @staticmethod
    def higher_highs_lower_lows(high: pd.Series, low: pd.Series, window: int = 5) -> Dict[str, pd.Series]:
        """Detect higher highs and lower lows patterns."""
        rolling_high = high.rolling(window=window).max()
        rolling_low = low.rolling(window=window).min()
        
        higher_high = (rolling_high > rolling_high.shift(window)).astype(int)
        lower_low = (rolling_low < rolling_low.shift(window)).astype(int)
        
        # Trend score: positive for uptrend, negative for downtrend
        trend_score = higher_high - lower_low
        
        return {
            'higher_high': higher_high,
            'lower_low': lower_low,
            'trend_score': trend_score
        }


class VolumeFeatures:
    """Volume-based features."""
    
    @staticmethod
    def volume_ma_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
        """Volume relative to moving average."""
        vol_ma = volume.rolling(window=period).mean()
        return volume / vol_ma.replace(0, 1)
    
    @staticmethod
    def volume_trend(volume: pd.Series, period: int = 10) -> pd.Series:
        """Volume trend (increasing or decreasing)."""
        return volume.rolling(window=period).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0],
            raw=True
        )
    
    @staticmethod
    def price_volume_trend(close: pd.Series, volume: pd.Series) -> pd.Series:
        """Price Volume Trend indicator."""
        return ((close.diff() / close.shift(1)) * volume).cumsum()
    
    @staticmethod
    def accumulation_distribution(high: pd.Series, low: pd.Series, 
                                  close: pd.Series, volume: pd.Series) -> pd.Series:
        """Accumulation/Distribution Line."""
        clv = ((close - low) - (high - close)) / (high - low).replace(0, 1)
        ad = (clv * volume).cumsum()
        return ad
    
    @staticmethod
    def money_flow_index(high: pd.Series, low: pd.Series, close: pd.Series, 
                         volume: pd.Series, period: int = 14) -> pd.Series:
        """Money Flow Index (volume-weighted RSI)."""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        delta = typical_price.diff()
        positive_flow = money_flow.where(delta > 0, 0).rolling(window=period).sum()
        negative_flow = money_flow.where(delta < 0, 0).abs().rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + positive_flow / negative_flow.replace(0, 1)))
        return mfi


class FeatureEngine:
    """
    Main feature engineering class.
    
    Combines all indicators and features into a comprehensive feature set
    for alpha model consumption.
    """
    
    def __init__(self, config=None):
        from ..config import FeatureConfig
        self.config = config or FeatureConfig()
        
        self.technical = TechnicalIndicators()
        self.statistical = StatisticalFeatures()
        self.momentum = MomentumFeatures()
        self.volume = VolumeFeatures()
    
    def compute_features(self, df: pd.DataFrame, symbol: str = "") -> FeatureSet:
        """
        Compute all features for a given OHLCV dataframe.
        
        Args:
            df: DataFrame with columns: open, high, low, close, volume
            symbol: Symbol name for reference
            
        Returns:
            FeatureSet with all computed features
        """
        features = df.copy()
        feature_names = []
        
        # =====================
        # Price Features
        # =====================
        
        # Returns
        for period in self.config.returns_periods:
            col_name = f'returns_{period}d'
            features[col_name] = self.statistical.returns(df['close'], period)
            feature_names.append(col_name)
            
            col_name = f'log_returns_{period}d'
            features[col_name] = self.statistical.log_returns(df['close'], period)
            feature_names.append(col_name)
        
        # =====================
        # Moving Averages
        # =====================
        
        # SMAs
        for period in self.config.sma_periods:
            col_name = f'sma_{period}'
            features[col_name] = self.technical.sma(df['close'], period)
            feature_names.append(col_name)
            
            # Price relative to SMA
            col_name = f'close_to_sma_{period}'
            features[col_name] = df['close'] / features[f'sma_{period}'] - 1
            feature_names.append(col_name)
        
        # EMAs
        for period in self.config.ema_periods:
            col_name = f'ema_{period}'
            features[col_name] = self.technical.ema(df['close'], period)
            feature_names.append(col_name)
        
        # =====================
        # Oscillators
        # =====================
        
        # RSI
        features['rsi'] = self.technical.rsi(df['close'], self.config.rsi_period)
        feature_names.append('rsi')
        
        # MACD
        macd = self.technical.macd(
            df['close'], 
            self.config.macd_fast, 
            self.config.macd_slow, 
            self.config.macd_signal
        )
        for name, series in macd.items():
            features[name] = series
            feature_names.append(name)
        
        # Stochastic
        stoch = self.technical.stochastic(df['high'], df['low'], df['close'])
        for name, series in stoch.items():
            features[name] = series
            feature_names.append(name)
        
        # =====================
        # Volatility
        # =====================
        
        # Bollinger Bands
        bb = self.technical.bollinger_bands(
            df['close'], 
            self.config.bollinger_period, 
            self.config.bollinger_std
        )
        for name, series in bb.items():
            features[name] = series
            feature_names.append(name)
        
        # ATR
        features['atr'] = self.technical.atr(
            df['high'], df['low'], df['close'], 
            self.config.atr_period
        )
        feature_names.append('atr')
        
        # ATR as percentage of price
        features['atr_pct'] = features['atr'] / df['close']
        feature_names.append('atr_pct')
        
        # Rolling volatility
        features['volatility'] = self.statistical.volatility(
            features['returns_1d'], 
            self.config.volatility_window
        )
        feature_names.append('volatility')
        
        # Realized volatility
        features['realized_vol'] = self.statistical.realized_volatility(
            df['high'], df['low'], df['close'], 
            self.config.volatility_window
        )
        feature_names.append('realized_vol')
        
        # =====================
        # Trend Indicators
        # =====================
        
        # ADX
        adx = self.technical.adx(df['high'], df['low'], df['close'])
        for name, series in adx.items():
            features[name] = series
            feature_names.append(name)
        
        # Trend strength
        features['trend_strength'] = self.momentum.trend_strength(df['close'], 20)
        feature_names.append('trend_strength')
        
        # Higher highs / lower lows
        hh_ll = self.momentum.higher_highs_lower_lows(df['high'], df['low'])
        for name, series in hh_ll.items():
            features[name] = series
            feature_names.append(name)
        
        # =====================
        # Momentum
        # =====================
        
        for period in self.config.momentum_periods:
            col_name = f'momentum_{period}'
            features[col_name] = self.momentum.momentum(df['close'], period)
            feature_names.append(col_name)
            
            col_name = f'roc_{period}'
            features[col_name] = self.momentum.rate_of_change(df['close'], period)
            feature_names.append(col_name)
        
        # MA Crossover signals
        if len(self.config.ema_periods) >= 2:
            fast_ema = features[f'ema_{self.config.ema_periods[0]}']
            slow_ema = features[f'ema_{self.config.ema_periods[1]}']
            features['ema_crossover'] = self.momentum.crossover_signal(fast_ema, slow_ema)
            feature_names.append('ema_crossover')
        
        # =====================
        # Volume Features
        # =====================
        
        # Volume MA ratio
        features['volume_ma_ratio'] = self.volume.volume_ma_ratio(
            df['volume'], 
            self.config.volume_ma_period
        )
        feature_names.append('volume_ma_ratio')
        
        # OBV
        features['obv'] = self.technical.obv(df['close'], df['volume'])
        feature_names.append('obv')
        
        # VWAP
        features['vwap'] = self.technical.vwap(
            df['high'], df['low'], df['close'], df['volume']
        )
        feature_names.append('vwap')
        
        # Price to VWAP
        features['close_to_vwap'] = df['close'] / features['vwap'] - 1
        feature_names.append('close_to_vwap')
        
        # MFI
        features['mfi'] = self.volume.money_flow_index(
            df['high'], df['low'], df['close'], df['volume']
        )
        feature_names.append('mfi')
        
        # A/D Line
        features['ad_line'] = self.volume.accumulation_distribution(
            df['high'], df['low'], df['close'], df['volume']
        )
        feature_names.append('ad_line')
        
        # =====================
        # Statistical Features
        # =====================
        
        # Z-scores
        features['close_zscore'] = self.statistical.zscore(df['close'], 20)
        feature_names.append('close_zscore')
        
        features['volume_zscore'] = self.statistical.zscore(df['volume'], 20)
        feature_names.append('volume_zscore')
        
        # Skewness and Kurtosis
        features['returns_skew'] = self.statistical.skewness(features['returns_1d'], 60)
        feature_names.append('returns_skew')
        
        features['returns_kurt'] = self.statistical.kurtosis(features['returns_1d'], 60)
        feature_names.append('returns_kurt')
        
        # Drawdown
        dd = self.statistical.drawdown(df['close'])
        features['drawdown'] = dd['drawdown']
        feature_names.append('drawdown')
        
        # Hurst exponent
        features['hurst'] = self.statistical.hurst_exponent(df['close'], 100)
        feature_names.append('hurst')
        
        # Autocorrelation
        features['autocorr_1'] = self.statistical.autocorrelation(
            features['returns_1d'], lag=1, window=60
        )
        feature_names.append('autocorr_1')
        
        # =====================
        # VIX Features (if available)
        # =====================
        
        if 'vix' in df.columns:
            features['vix'] = df['vix']
            feature_names.append('vix')
            
            features['vix_zscore'] = self.statistical.zscore(df['vix'], 20)
            feature_names.append('vix_zscore')
            
            features['vix_change'] = df['vix'].pct_change()
            feature_names.append('vix_change')
        
        # =====================
        # Clean up
        # =====================
        
        # Replace infinities with NaN
        features = features.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill NaN values
        features = features.ffill()
        
        # Fill remaining NaN with 0
        features = features.fillna(0)
        
        logger.info(f"Computed {len(feature_names)} features for {symbol}")
        
        return FeatureSet(
            symbol=symbol,
            features=features,
            feature_names=feature_names,
            timestamp=pd.Timestamp.now()
        )
    
    def get_feature_importance(self, feature_set: FeatureSet) -> pd.Series:
        """Calculate feature importance based on correlation with forward returns."""
        df = feature_set.features.copy()
        
        # Forward returns (next day)
        df['forward_return'] = df['close'].pct_change().shift(-1)
        
        # Calculate correlation of each feature with forward returns
        correlations = {}
        for col in feature_set.feature_names:
            if col in df.columns:
                corr = df[col].corr(df['forward_return'])
                correlations[col] = abs(corr) if not np.isnan(corr) else 0
        
        importance = pd.Series(correlations).sort_values(ascending=False)
        return importance
