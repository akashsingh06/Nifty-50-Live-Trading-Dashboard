"""
Quantitative & Systematic Trading System - Enhanced
====================================================

A complete algorithmic trading framework implementing:

ENHANCED FEATURES (v2.0):
- Multi-indicator confluence scoring (MACD, BB, ADX, SuperTrend)
- Market regime detection (Trending, Ranging, Volatile)
- Data caching with TTL for performance
- Retry logic with exponential backoff
- Comprehensive logging and metrics

PRINCIPLES:
- Quantitative models: Uses math/statistics to analyze markets
- Algorithmic execution: Computers execute trades at scale
- Data-driven decisions: Rejects gut, relies on patterns in data
- No emotional overrides: Humans can't override models once live
- Diversified strategies: Multiple quantitative strategies combined

PIPELINE:
    ┌─────────┐
    │  DATA   │  ← prices, volume, OI, VIX (CACHED)
    └────┬────┘
         ↓
    ┌──────────────┐
    │ FEATURE ENG. │  ← math transforms (EMA, RSI, MACD, BB, ADX)
    └────┬─────────┘
         ↓
    ┌──────────────┐
    │ ALPHA MODELS │  ← signals & probabilities
    └────┬─────────┘
         ↓
    ┌──────────────┐
    │ RISK ENGINE  │  ← position sizing, limits
    └────┬─────────┘
         ↓
    ┌──────────────┐
    │ EXECUTION    │  ← broker API
    └────┬─────────┘
         ↓
    ┌──────────────┐
    │ MONITORING   │  ← drawdown, kill-switch
    └──────────────┘

USAGE:
    # Paper trading
    python -m quant_trading.orchestrator --mode paper --capital 1000000
    
    # Backtesting
    python -m quant_trading.orchestrator --mode backtest --start-date 2024-01-01 --end-date 2024-12-31
    
    # Programmatic usage
    from quant_trading import TradingSystem, SystemConfig
    
    config = SystemConfig()
    config.initial_capital = 1000000
    
    system = TradingSystem(config)
    system.initialize()
    system.run()

MODULES:
    - data: Market data acquisition (prices, volume, OI, VIX)
    - features: Feature engineering (technical indicators, stats)
    - alpha: Alpha models (trend, mean-reversion, momentum, volatility)
    - risk: Risk management (position sizing, limits, VaR)
    - execution: Order execution (broker APIs)
    - monitoring: Performance tracking, alerts, kill-switch
"""

from .config import SystemConfig, TradingMode, AssetClass
from .orchestrator import TradingSystem, main
from .data import DataManager, MarketData
from .features import FeatureEngine, FeatureSet
from .alpha import AlphaEngine, AlphaOutput, Signal, SignalType
from .risk import RiskEngine, RiskMetrics, Position, OrderSizing
from .execution import ExecutionEngine, Order, OrderType, OrderSide
from .monitoring import MonitoringSystem, PerformanceMetrics, AlertSeverity
from .ml import MLPredictor, PredictionResult, MarketRegime, PatternRecognizer

__version__ = "1.0.0"
__all__ = [
    # Main
    'TradingSystem',
    'SystemConfig',
    'TradingMode',
    'AssetClass',
    'main',
    
    # Data
    'DataManager',
    'MarketData',
    
    # Features
    'FeatureEngine',
    'FeatureSet',
    
    # Alpha
    'AlphaEngine',
    'AlphaOutput',
    'Signal',
    'SignalType',
    
    # Risk
    'RiskEngine',
    'RiskMetrics',
    'Position',
    'OrderSizing',
    
    # Execution
    'ExecutionEngine',
    'Order',
    'OrderType',
    'OrderSide',
    
    # Monitoring
    'MonitoringSystem',
    'PerformanceMetrics',
    'AlertSeverity',
    
    # ML
    'MLPredictor',
    'PredictionResult',
    'MarketRegime',
    'PatternRecognizer'
]
