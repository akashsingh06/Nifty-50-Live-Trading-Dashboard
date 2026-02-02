"""
Data Module
===========
Handles all data acquisition: prices, volume, open interest, VIX
Supports multiple data sources with caching and failover.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import os
import pickle

logger = logging.getLogger(__name__)


@dataclass
class MarketData:
    """Container for market data."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    open_interest: Optional[float] = None
    vix: Optional[float] = None
    
    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'open_interest': self.open_interest,
            'vix': self.vix
        }


class DataSource(ABC):
    """Abstract base class for data sources."""
    
    @abstractmethod
    def fetch_ohlcv(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Fetch OHLCV data for a symbol."""
        pass
    
    @abstractmethod
    def fetch_realtime(self, symbol: str) -> MarketData:
        """Fetch real-time data for a symbol."""
        pass
    
    @abstractmethod
    def fetch_vix(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Fetch VIX/India VIX data."""
        pass


class YFinanceSource(DataSource):
    """Yahoo Finance data source."""
    
    def __init__(self):
        try:
            import yfinance as yf
            self.yf = yf
        except ImportError:
            logger.warning("yfinance not installed. Install with: pip install yfinance")
            self.yf = None
    
    def fetch_ohlcv(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Fetch OHLCV data from Yahoo Finance."""
        if self.yf is None:
            raise ImportError("yfinance not installed")
        
        # Convert Indian symbols to Yahoo format
        yahoo_symbol = self._convert_symbol(symbol)
        
        ticker = self.yf.Ticker(yahoo_symbol)
        df = ticker.history(start=start, end=end)
        
        # Standardize column names
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        df['symbol'] = symbol
        return df[['symbol', 'open', 'high', 'low', 'close', 'volume']]
    
    def fetch_realtime(self, symbol: str) -> MarketData:
        """Fetch real-time quote."""
        if self.yf is None:
            raise ImportError("yfinance not installed")
        
        yahoo_symbol = self._convert_symbol(symbol)
        ticker = self.yf.Ticker(yahoo_symbol)
        info = ticker.info
        
        return MarketData(
            symbol=symbol,
            timestamp=datetime.now(),
            open=info.get('open', 0),
            high=info.get('dayHigh', 0),
            low=info.get('dayLow', 0),
            close=info.get('currentPrice', info.get('regularMarketPrice', 0)),
            volume=info.get('volume', 0)
        )
    
    def fetch_vix(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Fetch India VIX data."""
        if self.yf is None:
            raise ImportError("yfinance not installed")
        
        # India VIX symbol
        ticker = self.yf.Ticker("^INDIAVIX")
        df = ticker.history(start=start, end=end)
        
        df = df.rename(columns={'Close': 'vix'})
        return df[['vix']]
    
    def _convert_symbol(self, symbol: str) -> str:
        """Convert symbol to Yahoo Finance format."""
        # Map common Indian symbols
        symbol_map = {
            'NIFTY': '^NSEI',
            'BANKNIFTY': '^NSEBANK',
            'NIFTY50': '^NSEI',
            'SENSEX': '^BSESN'
        }
        
        if symbol in symbol_map:
            return symbol_map[symbol]
        
        # Add .NS suffix for NSE stocks
        if not symbol.endswith('.NS') and not symbol.startswith('^'):
            return f"{symbol}.NS"
        
        return symbol


class MockDataSource(DataSource):
    """Mock data source for testing and paper trading."""
    
    def __init__(self, volatility: float = 0.02):
        self.volatility = volatility
        self.base_prices = {
            'NIFTY': 22000,
            'BANKNIFTY': 47000,
            'RELIANCE': 2500,
            'TCS': 3800,
            'INFY': 1500,
            'HDFCBANK': 1600,
            'ICICIBANK': 1000
        }
    
    def fetch_ohlcv(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Generate synthetic OHLCV data."""
        dates = pd.date_range(start=start, end=end, freq='D')
        dates = dates[dates.dayofweek < 5]  # Remove weekends
        
        base_price = self.base_prices.get(symbol, 1000)
        
        # Generate random walk prices
        n = len(dates)
        returns = np.random.normal(0.0005, self.volatility, n)
        prices = base_price * np.cumprod(1 + returns)
        
        # Generate OHLCV
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            daily_vol = self.volatility * close
            high = close + abs(np.random.normal(0, daily_vol))
            low = close - abs(np.random.normal(0, daily_vol))
            open_price = low + np.random.random() * (high - low)
            volume = int(np.random.normal(1000000, 200000))
            
            data.append({
                'symbol': symbol,
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close, 2),
                'volume': max(volume, 100000)
            })
        
        df = pd.DataFrame(data, index=dates)
        return df
    
    def fetch_realtime(self, symbol: str) -> MarketData:
        """Generate synthetic real-time data."""
        base_price = self.base_prices.get(symbol, 1000)
        price = base_price * (1 + np.random.normal(0, self.volatility))
        
        return MarketData(
            symbol=symbol,
            timestamp=datetime.now(),
            open=round(price * 0.998, 2),
            high=round(price * 1.01, 2),
            low=round(price * 0.99, 2),
            close=round(price, 2),
            volume=int(np.random.normal(1000000, 200000))
        )
    
    def fetch_vix(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Generate synthetic VIX data."""
        dates = pd.date_range(start=start, end=end, freq='D')
        dates = dates[dates.dayofweek < 5]
        
        # VIX mean-reverts around 15-20
        n = len(dates)
        vix_values = 15 + 5 * np.sin(np.linspace(0, 4*np.pi, n)) + np.random.normal(0, 2, n)
        vix_values = np.clip(vix_values, 10, 40)
        
        return pd.DataFrame({'vix': vix_values}, index=dates)


class DataCache:
    """Local cache for market data."""
    
    def __init__(self, cache_dir: str = "./data/cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get(self, key: str, max_age_hours: int = 24) -> Optional[pd.DataFrame]:
        """Get data from cache if fresh enough."""
        filepath = os.path.join(self.cache_dir, f"{key}.pkl")
        
        if not os.path.exists(filepath):
            return None
        
        # Check age
        mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
        if datetime.now() - mtime > timedelta(hours=max_age_hours):
            return None
        
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Cache read error for {key}: {e}")
            return None
    
    def set(self, key: str, data: pd.DataFrame):
        """Store data in cache."""
        filepath = os.path.join(self.cache_dir, f"{key}.pkl")
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"Cache write error for {key}: {e}")
    
    def clear(self):
        """Clear all cached data."""
        for f in os.listdir(self.cache_dir):
            os.remove(os.path.join(self.cache_dir, f))


class DataManager:
    """
    Main data manager coordinating all data operations.
    
    Responsibilities:
    - Fetch data from multiple sources with failover
    - Cache management
    - Data validation and cleaning
    - Real-time and historical data handling
    """
    
    def __init__(self, config=None, use_mock: bool = False):
        from ..config import DataConfig
        self.config = config or DataConfig()
        
        # Initialize data sources
        if use_mock:
            self.primary_source = MockDataSource()
        else:
            self.primary_source = YFinanceSource()
        
        self.backup_source = MockDataSource()  # Fallback to mock
        self.cache = DataCache(self.config.data_dir)
        
        # Data storage
        self.historical_data: Dict[str, pd.DataFrame] = {}
        self.realtime_data: Dict[str, MarketData] = {}
        self.vix_data: Optional[pd.DataFrame] = None
    
    def initialize(self):
        """Initialize data manager and load historical data."""
        logger.info("Initializing DataManager...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.config.lookback_days)
        
        # Load data for all symbols
        for symbol in self.config.symbols:
            self.load_historical(symbol, start_date, end_date)
        
        # Load VIX data
        if self.config.fetch_vix:
            self.load_vix(start_date, end_date)
        
        logger.info(f"DataManager initialized with {len(self.historical_data)} symbols")
    
    def load_historical(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Load historical data for a symbol."""
        cache_key = f"{symbol}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"
        
        # Try cache first
        if self.config.use_cache:
            cached = self.cache.get(cache_key, self.config.cache_expiry_hours)
            if cached is not None:
                logger.debug(f"Cache hit for {symbol}")
                self.historical_data[symbol] = cached
                return cached
        
        # Fetch from primary source
        try:
            df = self.primary_source.fetch_ohlcv(symbol, start, end)
            logger.info(f"Fetched {len(df)} rows for {symbol} from primary source")
        except Exception as e:
            logger.warning(f"Primary source failed for {symbol}: {e}")
            # Fallback to backup source
            try:
                df = self.backup_source.fetch_ohlcv(symbol, start, end)
                logger.info(f"Fetched {len(df)} rows for {symbol} from backup source")
            except Exception as e2:
                logger.error(f"All sources failed for {symbol}: {e2}")
                raise
        
        # Validate and clean data
        df = self._clean_data(df)
        
        # Cache the data
        if self.config.use_cache:
            self.cache.set(cache_key, df)
        
        self.historical_data[symbol] = df
        return df
    
    def load_vix(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Load VIX data."""
        try:
            self.vix_data = self.primary_source.fetch_vix(start, end)
        except Exception as e:
            logger.warning(f"VIX fetch failed: {e}")
            self.vix_data = self.backup_source.fetch_vix(start, end)
        
        return self.vix_data
    
    def get_realtime(self, symbol: str) -> MarketData:
        """Get real-time data for a symbol."""
        try:
            data = self.primary_source.fetch_realtime(symbol)
        except Exception as e:
            logger.warning(f"Realtime fetch failed for {symbol}: {e}")
            data = self.backup_source.fetch_realtime(symbol)
        
        self.realtime_data[symbol] = data
        return data
    
    def get_latest_prices(self) -> Dict[str, float]:
        """Get latest prices for all tracked symbols."""
        prices = {}
        for symbol in self.config.symbols:
            if symbol in self.realtime_data:
                prices[symbol] = self.realtime_data[symbol].close
            elif symbol in self.historical_data:
                prices[symbol] = self.historical_data[symbol]['close'].iloc[-1]
        return prices
    
    def get_combined_data(self, symbol: str) -> pd.DataFrame:
        """Get combined historical data with VIX."""
        if symbol not in self.historical_data:
            raise ValueError(f"No data for symbol: {symbol}")
        
        df = self.historical_data[symbol].copy()
        
        # Add VIX data if available
        if self.vix_data is not None:
            df = df.join(self.vix_data, how='left')
            df['vix'] = df['vix'].ffill()  # Forward fill missing VIX
        
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate market data."""
        # Remove duplicates
        df = df[~df.index.duplicated(keep='last')]
        
        # Sort by date
        df = df.sort_index()
        
        # Handle missing values
        df = df.ffill()  # Forward fill
        
        # Validate OHLC relationships
        # High should be >= Open, Close, Low
        # Low should be <= Open, Close, High
        df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
        df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
        
        # Remove rows with zero or negative prices
        df = df[(df['close'] > 0) & (df['volume'] >= 0)]
        
        return df
    
    def get_universe(self) -> List[str]:
        """Get list of all tracked symbols."""
        return list(self.historical_data.keys())
