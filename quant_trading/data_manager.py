"""
Data Manager with Caching and Retry Logic
==========================================

Features:
- In-memory caching with TTL
- Automatic retry with exponential backoff
- Multiple data source fallback
- Rate limiting protection
- Performance metrics
"""

import yfinance as yf
import pandas as pd
# numpy imported when needed
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from dataclasses import dataclass
import threading
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with expiration."""
    data: Any
    timestamp: datetime
    ttl_seconds: int
    
    def is_expired(self) -> bool:
        return datetime.now() - self.timestamp > timedelta(seconds=self.ttl_seconds)


@dataclass
class DataMetrics:
    """Track data fetch performance."""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    failed_requests: int = 0
    avg_fetch_time_ms: float = 0.0
    last_fetch_time: Optional[datetime] = None
    
    @property
    def cache_hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return (self.cache_hits / self.total_requests) * 100


class DataCache:
    """Thread-safe in-memory cache."""
    
    def __init__(self):
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            if entry.is_expired():
                del self._cache[key]
                return None
            return entry.data
    
    def set(self, key: str, data: Any, ttl_seconds: int = 60):
        with self._lock:
            self._cache[key] = CacheEntry(
                data=data,
                timestamp=datetime.now(),
                ttl_seconds=ttl_seconds
            )
    
    def clear(self):
        with self._lock:
            self._cache.clear()
    
    def cleanup_expired(self):
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]
            for key in expired_keys:
                del self._cache[key]


class RateLimiter:
    """Simple rate limiter."""
    
    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: list = []
        self._lock = threading.Lock()
    
    def can_request(self) -> bool:
        with self._lock:
            now = datetime.now()
            # Remove old requests
            self.requests = [
                req_time for req_time in self.requests
                if now - req_time < timedelta(seconds=self.window_seconds)
            ]
            return len(self.requests) < self.max_requests
    
    def record_request(self):
        with self._lock:
            self.requests.append(datetime.now())
    
    def wait_time(self) -> float:
        """Returns seconds to wait before next request."""
        with self._lock:
            if len(self.requests) < self.max_requests:
                return 0
            oldest = min(self.requests)
            wait = (oldest + timedelta(seconds=self.window_seconds) - datetime.now()).total_seconds()
            return max(0, wait)


class MarketDataManager:
    """
    Robust market data manager with caching and error handling.
    """
    
    def __init__(self):
        self.cache = DataCache()
        self.rate_limiter = RateLimiter(max_requests=30, window_seconds=60)
        self.metrics = DataMetrics()
        
        # Cache TTLs (seconds)
        self.NIFTY_DATA_TTL = 30  # 30 seconds for live data
        self.VIX_DATA_TTL = 60    # 1 minute for VIX
        self.HISTORICAL_TTL = 300  # 5 minutes for historical
        
        # Retry settings
        self.MAX_RETRIES = 3
        self.BASE_DELAY = 1  # seconds
        
        # NIFTY ticker
        self.NIFTY_TICKER = "^NSEI"
        self.VIX_TICKER = "^INDIAVIX"
    
    def _fetch_with_retry(self, fetch_func, *args, **kwargs) -> Optional[Any]:
        """Execute fetch with exponential backoff retry."""
        last_error = None
        
        for attempt in range(self.MAX_RETRIES):
            try:
                # Check rate limit
                if not self.rate_limiter.can_request():
                    wait_time = self.rate_limiter.wait_time()
                    logger.warning(f"Rate limit hit, waiting {wait_time:.1f}s")
                    time.sleep(wait_time)
                
                start_time = time.time()
                result = fetch_func(*args, **kwargs)
                fetch_time = (time.time() - start_time) * 1000
                
                # Update metrics
                self.metrics.last_fetch_time = datetime.now()
                self._update_fetch_time(fetch_time)
                self.rate_limiter.record_request()
                
                return result
                
            except Exception as e:
                last_error = e
                delay = self.BASE_DELAY * (2 ** attempt)
                logger.warning(f"Fetch attempt {attempt + 1} failed: {e}. Retrying in {delay}s")
                time.sleep(delay)
        
        self.metrics.failed_requests += 1
        logger.error(f"All retry attempts failed: {last_error}")
        return None
    
    def _update_fetch_time(self, fetch_time_ms: float):
        """Update rolling average fetch time."""
        if self.metrics.avg_fetch_time_ms == 0:
            self.metrics.avg_fetch_time_ms = fetch_time_ms
        else:
            # Exponential moving average
            self.metrics.avg_fetch_time_ms = (
                0.9 * self.metrics.avg_fetch_time_ms + 0.1 * fetch_time_ms
            )
    
    def get_nifty_data(self, period: str = '5d', interval: str = '5m') -> Optional[pd.DataFrame]:
        """
        Fetch NIFTY 50 data with caching.
        
        Args:
            period: Data period ('1d', '5d', '1mo', etc.)
            interval: Candle interval ('1m', '5m', '15m', etc.)
        
        Returns:
            DataFrame with OHLCV data or None
        """
        cache_key = f"nifty_{period}_{interval}"
        self.metrics.total_requests += 1
        
        # Check cache
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            self.metrics.cache_hits += 1
            logger.debug(f"Cache hit for {cache_key}")
            return cached_data
        
        self.metrics.cache_misses += 1
        logger.debug(f"Cache miss for {cache_key}, fetching...")
        
        def fetch():
            ticker = yf.Ticker(self.NIFTY_TICKER)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                raise ValueError("Empty data returned from yfinance")
            
            # Standardize column names
            df.columns = df.columns.str.lower()
            df = df.reset_index()
            
            # Validate data
            required_cols = ['close', 'high', 'low', 'open']
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Missing column: {col}")
            
            # Clean data - remove NaN
            df = df.dropna(subset=['close', 'high', 'low', 'open'])
            
            return df
        
        data = self._fetch_with_retry(fetch)
        
        if data is not None:
            self.cache.set(cache_key, data, self.NIFTY_DATA_TTL)
        
        return data
    
    def get_india_vix(self) -> Optional[float]:
        """Fetch India VIX with caching."""
        cache_key = "india_vix"
        self.metrics.total_requests += 1
        
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            self.metrics.cache_hits += 1
            return cached_data
        
        self.metrics.cache_misses += 1
        
        def fetch():
            ticker = yf.Ticker(self.VIX_TICKER)
            data = ticker.history(period='1d')
            
            if data.empty:
                raise ValueError("Empty VIX data")
            
            return round(data['Close'].iloc[-1], 2)
        
        vix = self._fetch_with_retry(fetch)
        
        if vix is not None:
            self.cache.set(cache_key, vix, self.VIX_DATA_TTL)
        
        return vix
    
    def get_multiple_timeframes(self) -> Dict[str, Optional[pd.DataFrame]]:
        """
        Fetch data for multiple timeframes for MTF analysis.
        
        Returns:
            Dict with '5m', '15m', '1h' DataFrames
        """
        timeframes = {
            '5m': ('5d', '5m'),
            '15m': ('5d', '15m'),
            '1h': ('1mo', '1h')
        }
        
        results = {}
        
        for tf_name, (period, interval) in timeframes.items():
            cache_key = f"nifty_{period}_{interval}"
            
            cached = self.cache.get(cache_key)
            if cached is not None:
                results[tf_name] = cached
                continue
            
            def fetch(p=period, i=interval):
                ticker = yf.Ticker(self.NIFTY_TICKER)
                df = ticker.history(period=p, interval=i)
                if not df.empty:
                    df.columns = df.columns.str.lower()
                    df = df.reset_index()
                return df if not df.empty else None
            
            data = self._fetch_with_retry(fetch)
            if data is not None:
                self.cache.set(cache_key, data, self.NIFTY_DATA_TTL)
            results[tf_name] = data
        
        return results
    
    def get_option_chain_data(self) -> Optional[Dict[str, Any]]:
        """
        Fetch NIFTY option chain (basic info).
        Note: Full option chain requires NSE API or broker API.
        """
        cache_key = "nifty_options"
        
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        
        def fetch():
            ticker = yf.Ticker(self.NIFTY_TICKER)
            info = ticker.info
            
            # Get expiration dates if available
            try:
                options = ticker.options
            except Exception:
                options = []
            
            return {
                'spot': info.get('regularMarketPrice', info.get('previousClose', 0)),
                'expiry_dates': list(options)[:4] if options else [],
                'timestamp': datetime.now().isoformat()
            }
        
        data = self._fetch_with_retry(fetch)
        
        if data is not None:
            self.cache.set(cache_key, data, 300)  # 5 min cache
        
        return data
    
    def get_market_status(self) -> Dict[str, Any]:
        """Get current market status."""
        now = datetime.now()
        current_time = now.time()
        
        # NSE trading hours
        pre_open_start = datetime.strptime("09:00", "%H:%M").time()
        market_open = datetime.strptime("09:15", "%H:%M").time()
        market_close = datetime.strptime("15:30", "%H:%M").time()
        post_close = datetime.strptime("16:00", "%H:%M").time()
        
        # Check if weekend
        if now.weekday() >= 5:
            status = "CLOSED"
            message = "Weekend - Market Closed"
        elif current_time < pre_open_start:
            status = "PRE_MARKET"
            message = "Pre-market hours"
        elif pre_open_start <= current_time < market_open:
            status = "PRE_OPEN"
            message = "Pre-open session"
        elif market_open <= current_time <= market_close:
            status = "OPEN"
            message = "Market is open"
        elif market_close < current_time <= post_close:
            status = "POST_CLOSE"
            message = "Post-market hours"
        else:
            status = "CLOSED"
            message = "Market closed"
        
        return {
            'status': status,
            'message': message,
            'current_time': now.strftime('%H:%M:%S'),
            'is_trading': status == "OPEN"
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get data manager metrics."""
        return {
            'total_requests': self.metrics.total_requests,
            'cache_hits': self.metrics.cache_hits,
            'cache_misses': self.metrics.cache_misses,
            'cache_hit_rate': f"{self.metrics.cache_hit_rate:.1f}%",
            'failed_requests': self.metrics.failed_requests,
            'avg_fetch_time_ms': f"{self.metrics.avg_fetch_time_ms:.1f}",
            'last_fetch': self.metrics.last_fetch_time.isoformat() if self.metrics.last_fetch_time else None
        }
    
    def clear_cache(self):
        """Clear all cached data."""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def cleanup(self):
        """Cleanup expired cache entries."""
        self.cache.cleanup_expired()


# Singleton instance
_data_manager: Optional[MarketDataManager] = None


def get_data_manager() -> MarketDataManager:
    """Get singleton data manager instance."""
    global _data_manager
    if _data_manager is None:
        _data_manager = MarketDataManager()
    return _data_manager


__all__ = ['MarketDataManager', 'get_data_manager', 'DataCache', 'DataMetrics']
