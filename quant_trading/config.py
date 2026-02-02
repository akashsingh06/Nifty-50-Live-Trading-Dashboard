"""
Configuration Management
========================
Central configuration for the entire trading system.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
import json
import os


class TradingMode(Enum):
    """Trading operation modes."""
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


class AssetClass(Enum):
    """Supported asset classes."""
    EQUITY = "equity"
    FUTURES = "futures"
    OPTIONS = "options"
    FOREX = "forex"
    CRYPTO = "crypto"


@dataclass
class DataConfig:
    """Data module configuration."""
    # Data sources
    primary_source: str = "yfinance"  # yfinance, alpha_vantage, polygon, etc.
    backup_source: Optional[str] = None
    
    # Symbols to track
    symbols: List[str] = field(default_factory=lambda: ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS", "INFY"])
    
    # Data parameters
    lookback_days: int = 252  # 1 year of trading days
    update_frequency_seconds: int = 60
    
    # Market data types
    fetch_prices: bool = True
    fetch_volume: bool = True
    fetch_open_interest: bool = True
    fetch_vix: bool = True
    
    # Storage
    data_dir: str = "./data"
    use_cache: bool = True
    cache_expiry_hours: int = 24


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""
    # Technical indicators
    sma_periods: List[int] = field(default_factory=lambda: [10, 20, 50, 200])
    ema_periods: List[int] = field(default_factory=lambda: [12, 26, 50])
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    atr_period: int = 14
    
    # Statistical features
    returns_periods: List[int] = field(default_factory=lambda: [1, 5, 10, 20])
    volatility_window: int = 20
    correlation_window: int = 60
    
    # Volume features
    volume_ma_period: int = 20
    
    # Momentum features
    momentum_periods: List[int] = field(default_factory=lambda: [5, 10, 20])


@dataclass
class AlphaConfig:
    """Alpha models configuration."""
    # Strategy weights (must sum to 1.0)
    strategy_weights: Dict[str, float] = field(default_factory=lambda: {
        "trend_following": 0.30,
        "mean_reversion": 0.25,
        "momentum": 0.25,
        "volatility": 0.20
    })
    
    # Signal thresholds
    signal_threshold: float = 0.6  # Minimum confidence to generate signal
    position_threshold: float = 0.7  # Minimum for position change
    
    # Model parameters
    trend_lookback: int = 20
    mean_reversion_zscore: float = 2.0
    momentum_lookback: int = 10
    
    # Ensemble method: 'weighted_average', 'voting', 'stacking'
    ensemble_method: str = "weighted_average"


@dataclass
class RiskConfig:
    """Risk engine configuration."""
    # Position sizing
    max_position_size_pct: float = 0.10  # Max 10% of portfolio per position
    max_sector_exposure_pct: float = 0.30  # Max 30% in one sector
    max_total_exposure_pct: float = 0.95  # Max 95% invested
    
    # Risk limits
    max_drawdown_pct: float = 0.15  # Kill switch at 15% drawdown
    daily_loss_limit_pct: float = 0.03  # Stop trading at 3% daily loss
    max_var_pct: float = 0.05  # Maximum Value at Risk
    
    # Position limits
    max_positions: int = 20
    min_position_value: float = 10000  # Minimum position size in currency
    
    # Volatility scaling
    target_volatility: float = 0.15  # 15% annual target vol
    vol_scaling_enabled: bool = True
    
    # Correlation limits
    max_correlation: float = 0.7  # Max correlation between positions


@dataclass
class ExecutionConfig:
    """Execution module configuration."""
    # Broker settings
    broker: str = "zerodha"  # zerodha, upstox, interactive_brokers
    api_key: str = ""
    api_secret: str = ""
    
    # Order parameters
    default_order_type: str = "LIMIT"  # MARKET, LIMIT
    slippage_tolerance_pct: float = 0.001  # 0.1% slippage tolerance
    max_order_retries: int = 3
    
    # Execution timing
    market_open: str = "09:15"
    market_close: str = "15:30"
    no_trade_last_minutes: int = 15  # Don't trade last 15 mins
    
    # Order splitting
    split_large_orders: bool = True
    max_order_value: float = 500000  # Split orders above this


@dataclass
class MonitoringConfig:
    """Monitoring and alerting configuration."""
    # Monitoring intervals
    check_interval_seconds: int = 30
    
    # Kill switches
    enable_kill_switch: bool = True
    drawdown_kill_threshold: float = 0.15
    daily_loss_kill_threshold: float = 0.05
    
    # Alerts
    enable_alerts: bool = True
    alert_channels: List[str] = field(default_factory=lambda: ["console", "email"])
    email_recipients: List[str] = field(default_factory=list)
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "./logs/trading.log"
    
    # Performance tracking
    track_sharpe: bool = True
    track_sortino: bool = True
    track_calmar: bool = True


@dataclass
class SystemConfig:
    """Master system configuration."""
    # Operating mode
    mode: TradingMode = TradingMode.PAPER
    asset_class: AssetClass = AssetClass.EQUITY
    
    # Capital
    initial_capital: float = 1000000  # 10 Lakhs
    
    # Component configs
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    alpha: AlphaConfig = field(default_factory=AlphaConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    def save(self, filepath: str):
        """Save configuration to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self._to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'SystemConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls._from_dict(data)
    
    def _to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'mode': self.mode.value,
            'asset_class': self.asset_class.value,
            'initial_capital': self.initial_capital,
            # Add nested configs as needed
        }
    
    @classmethod
    def _from_dict(cls, data: dict) -> 'SystemConfig':
        """Create from dictionary."""
        config = cls()
        config.mode = TradingMode(data.get('mode', 'paper'))
        config.asset_class = AssetClass(data.get('asset_class', 'equity'))
        config.initial_capital = data.get('initial_capital', 1000000)
        return config


# Default configuration instance
DEFAULT_CONFIG = SystemConfig()