"""
Execution Module
================
Broker API integration and order management.

Core principle: "Algorithmic execution at scale"
Orders are executed systematically without human intervention.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging
import time as time_module
import uuid

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Represents a trading order."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: Optional[float] = None  # For limit orders
    stop_price: Optional[float] = None  # For stop orders
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    filled_price: float = 0.0
    created_at: pd.Timestamp = field(default_factory=pd.Timestamp.now)
    updated_at: pd.Timestamp = field(default_factory=pd.Timestamp.now)
    broker_order_id: Optional[str] = None
    notes: str = ""
    
    @property
    def is_active(self) -> bool:
        """Check if order is still active."""
        return self.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL]
    
    @property
    def remaining_quantity(self) -> int:
        """Remaining quantity to be filled."""
        return self.quantity - self.filled_quantity
    
    def to_dict(self) -> dict:
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'type': self.order_type.value,
            'quantity': self.quantity,
            'price': self.price,
            'stop_price': self.stop_price,
            'status': self.status.value,
            'filled_qty': self.filled_quantity,
            'filled_price': self.filled_price,
            'created': self.created_at,
            'updated': self.updated_at
        }


@dataclass
class Fill:
    """Represents an order fill/execution."""
    fill_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    price: float
    timestamp: pd.Timestamp
    commission: float = 0.0
    
    @property
    def total_value(self) -> float:
        return self.quantity * self.price + self.commission


class BrokerAPI(ABC):
    """Abstract base class for broker API integration."""
    
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to broker."""
        pass
    
    @abstractmethod
    def disconnect(self):
        """Disconnect from broker."""
        pass
    
    @abstractmethod
    def submit_order(self, order: Order) -> Tuple[bool, str]:
        """Submit order to broker. Returns (success, message)."""
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> Tuple[bool, str]:
        """Cancel an order. Returns (success, message)."""
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> Order:
        """Get current order status."""
        pass
    
    @abstractmethod
    def get_positions(self) -> Dict[str, dict]:
        """Get current positions."""
        pass
    
    @abstractmethod
    def get_account_info(self) -> dict:
        """Get account information."""
        pass


class MockBroker(BrokerAPI):
    """
    Mock broker for paper trading and testing.
    
    Simulates order execution with realistic fills.
    """
    
    def __init__(self, initial_capital: float = 1000000, slippage: float = 0.001):
        self.capital = initial_capital
        self.cash = initial_capital
        self.slippage = slippage
        self.connected = False
        
        self.positions: Dict[str, dict] = {}
        self.orders: Dict[str, Order] = {}
        self.fills: List[Fill] = []
        
        # Simulated market prices
        self.market_prices: Dict[str, float] = {}
    
    def connect(self) -> bool:
        """Simulate connection."""
        self.connected = True
        logger.info("MockBroker connected")
        return True
    
    def disconnect(self):
        """Simulate disconnection."""
        self.connected = False
        logger.info("MockBroker disconnected")
    
    def set_market_prices(self, prices: Dict[str, float]):
        """Update simulated market prices."""
        self.market_prices.update(prices)
    
    def submit_order(self, order: Order) -> Tuple[bool, str]:
        """
        Submit and immediately execute order (for paper trading).
        
        In live trading, this would submit to broker and wait for fill.
        """
        if not self.connected:
            return False, "Not connected to broker"
        
        # Validate order
        if order.quantity <= 0:
            order.status = OrderStatus.REJECTED
            return False, "Invalid quantity"
        
        # Get market price
        if order.symbol not in self.market_prices:
            order.status = OrderStatus.REJECTED
            return False, f"No market price for {order.symbol}"
        
        market_price = self.market_prices[order.symbol]
        
        # Simulate slippage
        if order.side == OrderSide.BUY:
            fill_price = market_price * (1 + self.slippage)
        else:
            fill_price = market_price * (1 - self.slippage)
        
        # Check for limit orders
        if order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY and order.price < fill_price:
                order.status = OrderStatus.SUBMITTED
                self.orders[order.order_id] = order
                return True, "Limit order submitted, awaiting fill"
            elif order.side == OrderSide.SELL and order.price > fill_price:
                order.status = OrderStatus.SUBMITTED
                self.orders[order.order_id] = order
                return True, "Limit order submitted, awaiting fill"
        
        # Execute the order
        total_cost = order.quantity * fill_price
        
        if order.side == OrderSide.BUY:
            if total_cost > self.cash:
                order.status = OrderStatus.REJECTED
                return False, f"Insufficient cash: {self.cash:,.2f} < {total_cost:,.2f}"
            
            self.cash -= total_cost
            
            # Update position
            if order.symbol in self.positions:
                pos = self.positions[order.symbol]
                total_qty = pos['quantity'] + order.quantity
                total_cost = pos['quantity'] * pos['avg_price'] + order.quantity * fill_price
                pos['quantity'] = total_qty
                pos['avg_price'] = total_cost / total_qty
            else:
                self.positions[order.symbol] = {
                    'quantity': order.quantity,
                    'avg_price': fill_price,
                    'side': 'long'
                }
        
        else:  # SELL
            if order.symbol in self.positions:
                pos = self.positions[order.symbol]
                if pos['quantity'] >= order.quantity:
                    pos['quantity'] -= order.quantity
                    self.cash += total_cost
                    
                    if pos['quantity'] == 0:
                        del self.positions[order.symbol]
                else:
                    order.status = OrderStatus.REJECTED
                    return False, f"Insufficient position: {pos['quantity']} < {order.quantity}"
            else:
                # Allow short selling
                self.positions[order.symbol] = {
                    'quantity': -order.quantity,
                    'avg_price': fill_price,
                    'side': 'short'
                }
                self.cash += total_cost
        
        # Record fill
        fill = Fill(
            fill_id=str(uuid.uuid4()),
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            timestamp=pd.Timestamp.now()
        )
        self.fills.append(fill)
        
        # Update order
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.filled_price = fill_price
        order.updated_at = pd.Timestamp.now()
        self.orders[order.order_id] = order
        
        logger.info(f"Order filled: {order.symbol} {order.side.value} {order.quantity} @ {fill_price:.2f}")
        
        return True, f"Filled at {fill_price:.2f}"
    
    def cancel_order(self, order_id: str) -> Tuple[bool, str]:
        """Cancel a pending order."""
        if order_id not in self.orders:
            return False, "Order not found"
        
        order = self.orders[order_id]
        if not order.is_active:
            return False, f"Order already {order.status.value}"
        
        order.status = OrderStatus.CANCELLED
        order.updated_at = pd.Timestamp.now()
        
        return True, "Order cancelled"
    
    def get_order_status(self, order_id: str) -> Order:
        """Get order by ID."""
        return self.orders.get(order_id)
    
    def get_positions(self) -> Dict[str, dict]:
        """Get all positions."""
        return self.positions.copy()
    
    def get_account_info(self) -> dict:
        """Get account information."""
        # Calculate total equity
        position_value = sum(
            abs(pos['quantity']) * self.market_prices.get(symbol, pos['avg_price'])
            for symbol, pos in self.positions.items()
        )
        
        return {
            'cash': self.cash,
            'position_value': position_value,
            'total_equity': self.cash + position_value,
            'margin_used': 0,
            'margin_available': self.cash
        }


class ZerodhaAPI(BrokerAPI):
    """
    Zerodha Kite Connect API integration.
    
    Requires kiteconnect package: pip install kiteconnect
    """
    
    def __init__(self, api_key: str, api_secret: str, access_token: str = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.access_token = access_token
        self.kite = None
        self.connected = False
    
    def connect(self) -> bool:
        """Connect to Zerodha API."""
        try:
            from kiteconnect import KiteConnect
            
            self.kite = KiteConnect(api_key=self.api_key)
            
            if self.access_token:
                self.kite.set_access_token(self.access_token)
                self.connected = True
                logger.info("Connected to Zerodha")
                return True
            else:
                # Generate login URL for user
                login_url = self.kite.login_url()
                logger.info(f"Please login at: {login_url}")
                return False
                
        except ImportError:
            logger.error("kiteconnect not installed. Install with: pip install kiteconnect")
            return False
        except Exception as e:
            logger.error(f"Zerodha connection error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from Zerodha."""
        self.connected = False
        self.kite = None
    
    def submit_order(self, order: Order) -> Tuple[bool, str]:
        """Submit order to Zerodha."""
        if not self.connected or not self.kite:
            return False, "Not connected"
        
        try:
            # Map order type
            order_type_map = {
                OrderType.MARKET: "MARKET",
                OrderType.LIMIT: "LIMIT",
                OrderType.STOP: "SL",
                OrderType.STOP_LIMIT: "SL"
            }
            
            variety = "regular"
            
            zerodha_order = {
                'tradingsymbol': order.symbol,
                'exchange': 'NSE',
                'transaction_type': 'BUY' if order.side == OrderSide.BUY else 'SELL',
                'quantity': order.quantity,
                'order_type': order_type_map.get(order.order_type, 'MARKET'),
                'product': 'CNC',  # Delivery
                'variety': variety
            }
            
            if order.price:
                zerodha_order['price'] = order.price
            if order.stop_price:
                zerodha_order['trigger_price'] = order.stop_price
            
            broker_order_id = self.kite.place_order(**zerodha_order)
            
            order.broker_order_id = str(broker_order_id)
            order.status = OrderStatus.SUBMITTED
            
            return True, f"Order submitted: {broker_order_id}"
            
        except Exception as e:
            order.status = OrderStatus.REJECTED
            return False, str(e)
    
    def cancel_order(self, order_id: str) -> Tuple[bool, str]:
        """Cancel order on Zerodha."""
        if not self.connected or not self.kite:
            return False, "Not connected"
        
        try:
            self.kite.cancel_order(variety="regular", order_id=order_id)
            return True, "Order cancelled"
        except Exception as e:
            return False, str(e)
    
    def get_order_status(self, order_id: str) -> Order:
        """Get order status from Zerodha."""
        if not self.connected or not self.kite:
            return None
        
        try:
            orders = self.kite.orders()
            for o in orders:
                if str(o['order_id']) == order_id:
                    status_map = {
                        'COMPLETE': OrderStatus.FILLED,
                        'REJECTED': OrderStatus.REJECTED,
                        'CANCELLED': OrderStatus.CANCELLED,
                        'PENDING': OrderStatus.PENDING,
                        'OPEN': OrderStatus.SUBMITTED
                    }
                    return Order(
                        order_id=order_id,
                        symbol=o['tradingsymbol'],
                        side=OrderSide.BUY if o['transaction_type'] == 'BUY' else OrderSide.SELL,
                        order_type=OrderType.MARKET,
                        quantity=o['quantity'],
                        status=status_map.get(o['status'], OrderStatus.PENDING),
                        filled_quantity=o.get('filled_quantity', 0),
                        filled_price=o.get('average_price', 0),
                        broker_order_id=order_id
                    )
            return None
        except Exception as e:
            logger.error(f"Error getting order status: {e}")
            return None
    
    def get_positions(self) -> Dict[str, dict]:
        """Get positions from Zerodha."""
        if not self.connected or not self.kite:
            return {}
        
        try:
            positions = self.kite.positions()
            result = {}
            
            for pos in positions.get('net', []):
                if pos['quantity'] != 0:
                    result[pos['tradingsymbol']] = {
                        'quantity': pos['quantity'],
                        'avg_price': pos['average_price'],
                        'side': 'long' if pos['quantity'] > 0 else 'short',
                        'pnl': pos.get('pnl', 0)
                    }
            
            return result
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return {}
    
    def get_account_info(self) -> dict:
        """Get account info from Zerodha."""
        if not self.connected or not self.kite:
            return {}
        
        try:
            margins = self.kite.margins()
            equity = margins.get('equity', {})
            
            return {
                'cash': equity.get('available', {}).get('cash', 0),
                'margin_used': equity.get('utilised', {}).get('debits', 0),
                'margin_available': equity.get('available', {}).get('live_balance', 0),
                'total_equity': equity.get('net', 0)
            }
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {}


class AngelOneAPI(BrokerAPI):
    """
    Angel One (AngelBroking) SmartAPI integration.
    
    Requires smartapi-python package: pip install smartapi-python
    
    Authentication flow:
    1. Generate TOTP using authenticator app linked to Angel One account
    2. Pass TOTP along with credentials to connect
    """
    
    def __init__(self, api_key: str, secret_key: str, client_id: str = None,
                 password: str = None, totp: str = None):
        self.api_key = api_key
        self.secret_key = secret_key
        self.client_id = client_id  # Angel One User ID
        self.password = password
        self.totp = totp
        self.smart_api = None
        self.connected = False
        self.auth_token = None
        self.feed_token = None
        
        # Symbol token cache (Angel One uses numeric tokens)
        self.symbol_tokens: Dict[str, str] = {}
    
    def connect(self) -> bool:
        """Connect to Angel One SmartAPI."""
        try:
            from SmartApi import SmartConnect
            import pyotp
            
            self.smart_api = SmartConnect(api_key=self.api_key)
            
            if self.client_id and self.password and self.totp:
                # Generate TOTP if secret provided
                if len(self.totp) > 6:  # It's a TOTP secret, not a code
                    totp_obj = pyotp.TOTP(self.totp)
                    totp_code = totp_obj.now()
                else:
                    totp_code = self.totp
                
                # Login
                data = self.smart_api.generateSession(
                    clientCode=self.client_id,
                    password=self.password,
                    totp=totp_code
                )
                
                if data['status']:
                    self.auth_token = data['data']['jwtToken']
                    self.feed_token = self.smart_api.getfeedToken()
                    self.connected = True
                    logger.info(f"Connected to Angel One as {self.client_id}")
                    return True
                else:
                    logger.error(f"Angel One login failed: {data.get('message', 'Unknown error')}")
                    return False
            else:
                logger.warning("Angel One credentials incomplete. Provide client_id, password, and totp.")
                return False
                
        except ImportError:
            logger.error("smartapi-python not installed. Install with: pip install smartapi-python pyotp")
            return False
        except Exception as e:
            logger.error(f"Angel One connection error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from Angel One."""
        if self.smart_api and self.connected:
            try:
                self.smart_api.terminateSession(self.client_id)
            except:
                pass
        self.connected = False
        self.smart_api = None
        logger.info("Disconnected from Angel One")
    
    def _get_symbol_token(self, symbol: str, exchange: str = "NSE") -> str:
        """Get Angel One symbol token for a trading symbol."""
        cache_key = f"{exchange}:{symbol}"
        if cache_key in self.symbol_tokens:
            return self.symbol_tokens[cache_key]
        
        # Common NSE symbols token mapping
        # In production, fetch from Angel One's instrument master
        nse_tokens = {
            'RELIANCE': '2885',
            'TCS': '11536',
            'INFY': '1594',
            'HDFCBANK': '1333',
            'ICICIBANK': '4963',
            'KOTAKBANK': '1922',
            'SBIN': '3045',
            'BHARTIARTL': '10604',
            'ITC': '1660',
            'HINDUNILVR': '1394',
            'LT': '11483',
            'AXISBANK': '5900',
            'BAJFINANCE': '317',
            'MARUTI': '10999',
            'ASIANPAINT': '236',
            'NIFTY': '99926000',  # NIFTY 50 Index
            'BANKNIFTY': '99926009',  # Bank Nifty Index
        }
        
        token = nse_tokens.get(symbol.upper(), symbol)
        self.symbol_tokens[cache_key] = token
        return token
    
    def submit_order(self, order: Order) -> Tuple[bool, str]:
        """Submit order to Angel One."""
        if not self.connected or not self.smart_api:
            return False, "Not connected"
        
        try:
            # Map order type
            order_type_map = {
                OrderType.MARKET: "MARKET",
                OrderType.LIMIT: "LIMIT",
                OrderType.STOP: "STOPLOSS_LIMIT",
                OrderType.STOP_LIMIT: "STOPLOSS_LIMIT"
            }
            
            # Get symbol token
            symbol_token = self._get_symbol_token(order.symbol)
            
            order_params = {
                'variety': 'NORMAL',
                'tradingsymbol': order.symbol,
                'symboltoken': symbol_token,
                'transactiontype': 'BUY' if order.side == OrderSide.BUY else 'SELL',
                'exchange': 'NSE',
                'ordertype': order_type_map.get(order.order_type, 'MARKET'),
                'producttype': 'DELIVERY',  # CNC equivalent
                'duration': 'DAY',
                'quantity': str(order.quantity)
            }
            
            # Add price for limit orders
            if order.price and order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
                order_params['price'] = str(order.price)
            else:
                order_params['price'] = '0'
            
            # Add trigger price for stop orders
            if order.stop_price and order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
                order_params['triggerprice'] = str(order.stop_price)
            else:
                order_params['triggerprice'] = '0'
            
            # Place order
            response = self.smart_api.placeOrder(order_params)
            
            if response and response.get('status'):
                order.broker_order_id = response['data']['orderid']
                order.status = OrderStatus.SUBMITTED
                logger.info(f"Angel One order submitted: {order.broker_order_id}")
                return True, f"Order submitted: {order.broker_order_id}"
            else:
                error_msg = response.get('message', 'Order failed') if response else 'No response'
                order.status = OrderStatus.REJECTED
                return False, error_msg
            
        except Exception as e:
            order.status = OrderStatus.REJECTED
            logger.error(f"Angel One order error: {e}")
            return False, str(e)
    
    def cancel_order(self, order_id: str) -> Tuple[bool, str]:
        """Cancel order on Angel One."""
        if not self.connected or not self.smart_api:
            return False, "Not connected"
        
        try:
            response = self.smart_api.cancelOrder(order_id, 'NORMAL')
            
            if response and response.get('status'):
                return True, "Order cancelled"
            else:
                return False, response.get('message', 'Cancel failed') if response else 'No response'
                
        except Exception as e:
            logger.error(f"Angel One cancel error: {e}")
            return False, str(e)
    
    def get_order_status(self, order_id: str) -> Order:
        """Get order status from Angel One."""
        if not self.connected or not self.smart_api:
            return None
        
        try:
            orders = self.smart_api.orderBook()
            
            if orders and orders.get('status') and orders.get('data'):
                for o in orders['data']:
                    if o.get('orderid') == order_id:
                        status_map = {
                            'complete': OrderStatus.FILLED,
                            'rejected': OrderStatus.REJECTED,
                            'cancelled': OrderStatus.CANCELLED,
                            'pending': OrderStatus.PENDING,
                            'open': OrderStatus.SUBMITTED,
                            'trigger pending': OrderStatus.SUBMITTED
                        }
                        
                        return Order(
                            order_id=order_id,
                            symbol=o.get('tradingsymbol', ''),
                            side=OrderSide.BUY if o.get('transactiontype') == 'BUY' else OrderSide.SELL,
                            order_type=OrderType.MARKET,
                            quantity=int(o.get('quantity', 0)),
                            status=status_map.get(o.get('status', '').lower(), OrderStatus.PENDING),
                            filled_quantity=int(o.get('filledshares', 0)),
                            filled_price=float(o.get('averageprice', 0)),
                            broker_order_id=order_id
                        )
            return None
            
        except Exception as e:
            logger.error(f"Angel One order status error: {e}")
            return None
    
    def get_positions(self) -> Dict[str, dict]:
        """Get positions from Angel One."""
        if not self.connected or not self.smart_api:
            return {}
        
        try:
            positions = self.smart_api.position()
            result = {}
            
            if positions and positions.get('status') and positions.get('data'):
                for pos in positions['data']:
                    qty = int(pos.get('netqty', 0))
                    if qty != 0:
                        result[pos.get('tradingsymbol', '')] = {
                            'quantity': abs(qty),
                            'avg_price': float(pos.get('averageprice', 0)),
                            'side': 'long' if qty > 0 else 'short',
                            'pnl': float(pos.get('pnl', 0)),
                            'ltp': float(pos.get('ltp', 0))
                        }
            
            return result
            
        except Exception as e:
            logger.error(f"Angel One positions error: {e}")
            return {}
    
    def get_account_info(self) -> dict:
        """Get account info from Angel One."""
        if not self.connected or not self.smart_api:
            return {}
        
        try:
            # Get RMS (Risk Management System) limits
            rms = self.smart_api.rmsLimit()
            
            if rms and rms.get('status') and rms.get('data'):
                data = rms['data']
                return {
                    'cash': float(data.get('availablecash', 0)),
                    'margin_used': float(data.get('utiliseddebits', 0)),
                    'margin_available': float(data.get('availableintradaypayin', 0)),
                    'total_equity': float(data.get('net', 0)),
                    'collateral': float(data.get('collateral', 0))
                }
            return {}
            
        except Exception as e:
            logger.error(f"Angel One account info error: {e}")
            return {}
    
    def get_ltp(self, symbol: str, exchange: str = "NSE") -> float:
        """Get Last Traded Price for a symbol."""
        if not self.connected or not self.smart_api:
            return 0.0
        
        try:
            token = self._get_symbol_token(symbol, exchange)
            data = self.smart_api.ltpData(exchange, symbol, token)
            
            if data and data.get('status') and data.get('data'):
                return float(data['data'].get('ltp', 0))
            return 0.0
            
        except Exception as e:
            logger.error(f"Angel One LTP error: {e}")
            return 0.0


class ExecutionEngine:
    """
    Main execution engine coordinating order management.
    
    Responsibilities:
    - Order creation and validation
    - Broker API interaction
    - Order splitting for large orders
    - Execution timing management
    - Slippage and fill tracking
    """
    
    def __init__(self, config=None, broker: BrokerAPI = None, use_mock: bool = True):
        from ..config import ExecutionConfig
        self.config = config or ExecutionConfig()
        
        # Initialize broker
        if broker:
            self.broker = broker
        elif use_mock:
            self.broker = MockBroker()
        else:
            self.broker = MockBroker()  # Default to mock
        
        # Order tracking
        self.pending_orders: Dict[str, Order] = {}
        self.completed_orders: List[Order] = []
        
        # Execution state
        self.trading_enabled = True
        self.last_order_time: Dict[str, pd.Timestamp] = {}
    
    def initialize(self):
        """Initialize execution engine."""
        success = self.broker.connect()
        if success:
            logger.info("ExecutionEngine initialized")
        else:
            logger.warning("ExecutionEngine started in limited mode (broker connection failed)")
        return success
    
    def shutdown(self):
        """Shutdown execution engine."""
        # Cancel all pending orders
        for order_id, order in list(self.pending_orders.items()):
            if order.is_active:
                self.cancel_order(order_id)
        
        self.broker.disconnect()
        logger.info("ExecutionEngine shutdown")
    
    def create_order(self, symbol: str, side: str, quantity: int,
                     order_type: str = 'MARKET', price: float = None,
                     stop_price: float = None) -> Order:
        """
        Create a new order.
        
        Args:
            symbol: Symbol to trade
            side: 'buy' or 'sell'
            quantity: Quantity to trade
            order_type: 'MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT'
            price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            
        Returns:
            Created Order object
        """
        order = Order(
            order_id=str(uuid.uuid4()),
            symbol=symbol,
            side=OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL,
            order_type=OrderType[order_type.upper()],
            quantity=quantity,
            price=price,
            stop_price=stop_price
        )
        
        return order
    
    def execute_order(self, order: Order) -> Tuple[bool, str]:
        """
        Execute an order through the broker.
        
        Handles order validation, splitting, and submission.
        """
        # Check if trading is enabled
        if not self.trading_enabled:
            return False, "Trading is disabled"
        
        # Check trading hours
        if not self._is_trading_time():
            return False, "Outside trading hours"
        
        # Validate order
        valid, message = self._validate_order(order)
        if not valid:
            order.status = OrderStatus.REJECTED
            order.notes = message
            return False, message
        
        # Check if order needs splitting
        if self._should_split_order(order):
            return self._execute_split_order(order)
        
        # Submit to broker
        success, message = self.broker.submit_order(order)
        
        if success:
            if order.status == OrderStatus.FILLED:
                self.completed_orders.append(order)
            else:
                self.pending_orders[order.order_id] = order
            
            self.last_order_time[order.symbol] = pd.Timestamp.now()
        
        return success, message
    
    def cancel_order(self, order_id: str) -> Tuple[bool, str]:
        """Cancel a pending order."""
        if order_id not in self.pending_orders:
            return False, "Order not found"
        
        success, message = self.broker.cancel_order(order_id)
        
        if success:
            order = self.pending_orders.pop(order_id)
            order.status = OrderStatus.CANCELLED
            self.completed_orders.append(order)
        
        return success, message
    
    def cancel_all_orders(self, symbol: str = None):
        """Cancel all pending orders, optionally for a specific symbol."""
        cancelled = 0
        
        for order_id, order in list(self.pending_orders.items()):
            if symbol is None or order.symbol == symbol:
                success, _ = self.cancel_order(order_id)
                if success:
                    cancelled += 1
        
        logger.info(f"Cancelled {cancelled} orders")
        return cancelled
    
    def update_market_prices(self, prices: Dict[str, float]):
        """Update market prices (for mock broker)."""
        if isinstance(self.broker, MockBroker):
            self.broker.set_market_prices(prices)
    
    def get_positions(self) -> Dict[str, dict]:
        """Get current positions from broker."""
        return self.broker.get_positions()
    
    def get_account_info(self) -> dict:
        """Get account information from broker."""
        return self.broker.get_account_info()
    
    def disable_trading(self, reason: str = ""):
        """Disable trading (kill switch)."""
        self.trading_enabled = False
        logger.warning(f"Trading DISABLED: {reason}")
    
    def enable_trading(self):
        """Enable trading."""
        self.trading_enabled = True
        logger.info("Trading enabled")
    
    def _validate_order(self, order: Order) -> Tuple[bool, str]:
        """Validate order before submission."""
        # Check quantity
        if order.quantity <= 0:
            return False, "Invalid quantity"
        
        # Check limit price for limit orders
        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            if order.price is None or order.price <= 0:
                return False, "Limit price required"
        
        # Check stop price for stop orders
        if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
            if order.stop_price is None or order.stop_price <= 0:
                return False, "Stop price required"
        
        return True, "Valid"
    
    def _is_trading_time(self) -> bool:
        """Check if current time is within trading hours."""
        now = datetime.now().time()
        
        market_open = time(*map(int, self.config.market_open.split(':')))
        market_close = time(*map(int, self.config.market_close.split(':')))
        
        # Adjust for no-trade period at end of day
        close_adjusted = time(
            market_close.hour,
            market_close.minute - self.config.no_trade_last_minutes
        )
        
        return market_open <= now <= close_adjusted
    
    def _should_split_order(self, order: Order) -> bool:
        """Check if order should be split into smaller orders."""
        if not self.config.split_large_orders:
            return False
        
        # Get approximate order value
        if order.price:
            order_value = order.quantity * order.price
        else:
            # Use last known price or estimate
            prices = self.broker.get_positions() if hasattr(self.broker, 'market_prices') else {}
            price = getattr(self.broker, 'market_prices', {}).get(order.symbol, 0)
            order_value = order.quantity * price if price else 0
        
        return order_value > self.config.max_order_value
    
    def _execute_split_order(self, order: Order) -> Tuple[bool, str]:
        """Split and execute a large order."""
        # Get price for splitting
        price = order.price or getattr(self.broker, 'market_prices', {}).get(order.symbol, 0)
        if not price:
            return False, "Cannot determine price for order splitting"
        
        # Calculate split quantity
        max_qty_per_order = int(self.config.max_order_value / price)
        remaining = order.quantity
        child_orders = []
        
        while remaining > 0:
            split_qty = min(remaining, max_qty_per_order)
            
            child_order = Order(
                order_id=str(uuid.uuid4()),
                symbol=order.symbol,
                side=order.side,
                order_type=order.order_type,
                quantity=split_qty,
                price=order.price,
                stop_price=order.stop_price,
                notes=f"Split from {order.order_id}"
            )
            child_orders.append(child_order)
            remaining -= split_qty
        
        # Execute child orders with small delay between them
        total_filled = 0
        errors = []
        
        for i, child in enumerate(child_orders):
            if i > 0:
                time_module.sleep(0.1)  # Small delay between orders
            
            success, message = self.broker.submit_order(child)
            
            if success and child.status == OrderStatus.FILLED:
                total_filled += child.filled_quantity
            elif not success:
                errors.append(message)
        
        # Update parent order
        order.filled_quantity = total_filled
        order.status = OrderStatus.FILLED if total_filled == order.quantity else OrderStatus.PARTIAL
        self.completed_orders.append(order)
        
        if errors:
            return False, f"Partial fill: {total_filled}/{order.quantity}. Errors: {'; '.join(errors)}"
        
        return True, f"Split order filled: {total_filled} shares"
