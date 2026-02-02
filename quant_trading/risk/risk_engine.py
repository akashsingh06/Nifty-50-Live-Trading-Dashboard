"""
Risk Engine Module
==================
Position sizing, risk limits, and portfolio risk management.

Core principle: "No emotional overrides once live"
Risk rules are enforced algorithmically without human intervention.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk levels for the portfolio."""
    LOW = "low"
    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Position:
    """Represents a trading position."""
    symbol: str
    quantity: int
    entry_price: float
    current_price: float
    side: str  # 'long' or 'short'
    entry_time: pd.Timestamp
    
    @property
    def market_value(self) -> float:
        """Current market value of position."""
        return self.quantity * self.current_price
    
    @property
    def cost_basis(self) -> float:
        """Total cost of position."""
        return self.quantity * self.entry_price
    
    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss."""
        if self.side == 'long':
            return (self.current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - self.current_price) * self.quantity
    
    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized P&L as percentage."""
        return self.unrealized_pnl / self.cost_basis if self.cost_basis > 0 else 0


@dataclass
class RiskMetrics:
    """Portfolio risk metrics."""
    total_exposure: float
    exposure_pct: float
    cash: float
    unrealized_pnl: float
    realized_pnl: float
    current_drawdown: float
    max_drawdown: float
    daily_pnl: float
    var_95: float  # Value at Risk at 95%
    var_99: float  # Value at Risk at 99%
    sharpe_ratio: float
    risk_level: RiskLevel
    positions_count: int
    largest_position_pct: float


@dataclass
class OrderSizing:
    """Order sizing result from risk engine."""
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: int
    max_quantity: int
    position_value: float
    risk_adjusted: bool
    sizing_method: str
    notes: List[str] = field(default_factory=list)


class PositionSizer:
    """Position sizing algorithms."""
    
    @staticmethod
    def fixed_fractional(capital: float, risk_pct: float, entry_price: float, 
                         stop_loss_price: float) -> int:
        """
        Fixed Fractional position sizing.
        
        Risk a fixed percentage of capital per trade.
        """
        if entry_price <= 0 or stop_loss_price <= 0:
            return 0
        
        risk_per_share = abs(entry_price - stop_loss_price)
        if risk_per_share <= 0:
            return 0
        
        risk_amount = capital * risk_pct
        quantity = int(risk_amount / risk_per_share)
        
        return max(quantity, 0)
    
    @staticmethod
    def volatility_scaled(capital: float, target_vol: float, 
                         asset_vol: float, price: float) -> int:
        """
        Volatility-scaled position sizing.
        
        Size positions inversely proportional to volatility.
        """
        if asset_vol <= 0 or price <= 0:
            return 0
        
        # Position size that achieves target portfolio volatility contribution
        position_value = (capital * target_vol) / asset_vol
        quantity = int(position_value / price)
        
        return max(quantity, 0)
    
    @staticmethod
    def kelly_criterion(win_rate: float, avg_win: float, avg_loss: float,
                       capital: float, price: float, kelly_fraction: float = 0.25) -> int:
        """
        Kelly Criterion position sizing.
        
        Optimal bet sizing based on edge and odds.
        kelly_fraction reduces the full Kelly for safety.
        """
        if avg_loss <= 0 or price <= 0:
            return 0
        
        # Calculate Kelly percentage
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        kelly_pct = win_rate - ((1 - win_rate) / win_loss_ratio) if win_loss_ratio > 0 else 0
        
        # Apply fraction (typically 25% of full Kelly)
        kelly_pct = max(0, kelly_pct * kelly_fraction)
        
        # Convert to position size
        position_value = capital * kelly_pct
        quantity = int(position_value / price)
        
        return max(quantity, 0)
    
    @staticmethod
    def equal_weight(capital: float, num_positions: int, price: float) -> int:
        """
        Equal weight position sizing.
        
        Divide capital equally among positions.
        """
        if num_positions <= 0 or price <= 0:
            return 0
        
        position_value = capital / num_positions
        quantity = int(position_value / price)
        
        return max(quantity, 0)


class RiskEngine:
    """
    Main risk engine for position sizing and risk management.
    
    Responsibilities:
    - Position sizing based on multiple algorithms
    - Portfolio-level risk limits
    - Drawdown management
    - Exposure limits
    - Correlation-based diversification
    """
    
    def __init__(self, config=None, initial_capital: float = 1000000):
        from ..config import RiskConfig
        self.config = config or RiskConfig()
        
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.cash = initial_capital
        
        self.positions: Dict[str, Position] = {}
        self.realized_pnl = 0.0
        self.peak_equity = initial_capital
        self.daily_starting_equity = initial_capital
        
        # Historical tracking
        self.equity_history: List[Tuple[pd.Timestamp, float]] = []
        self.trade_history: List[Dict] = []
        
        self.sizer = PositionSizer()
    
    def calculate_position_size(self, symbol: str, signal_strength: float,
                                 price: float, volatility: float = 0.02,
                                 stop_loss_pct: float = 0.02) -> OrderSizing:
        """
        Calculate appropriate position size given signal and constraints.
        
        Args:
            symbol: Symbol to trade
            signal_strength: Signal strength from alpha (-1 to 1)
            price: Current price
            volatility: Asset volatility (annualized)
            stop_loss_pct: Stop loss percentage from entry
            
        Returns:
            OrderSizing with recommended size
        """
        notes = []
        
        # Determine side
        side = 'buy' if signal_strength > 0 else 'sell'
        abs_signal = abs(signal_strength)
        
        # Check if we can trade
        if not self._can_open_position(symbol, side):
            return OrderSizing(
                symbol=symbol,
                side=side,
                quantity=0,
                max_quantity=0,
                position_value=0,
                risk_adjusted=True,
                sizing_method='blocked',
                notes=['Position blocked by risk limits']
            )
        
        # Calculate base size using multiple methods
        
        # 1. Fixed fractional (risk-based)
        stop_price = price * (1 - stop_loss_pct) if side == 'buy' else price * (1 + stop_loss_pct)
        size_fixed = self.sizer.fixed_fractional(
            self.cash, 
            self.config.max_position_size_pct * 0.5,  # Risk half of max position
            price,
            stop_price
        )
        
        # 2. Volatility-scaled
        if self.config.vol_scaling_enabled:
            size_vol = self.sizer.volatility_scaled(
                self.cash,
                self.config.target_volatility / np.sqrt(252),  # Daily target vol
                volatility / np.sqrt(252),  # Daily asset vol
                price
            )
        else:
            size_vol = size_fixed
        
        # 3. Equal weight fallback
        max_positions = self.config.max_positions
        current_positions = len(self.positions)
        remaining_slots = max(max_positions - current_positions, 1)
        size_equal = self.sizer.equal_weight(self.cash, remaining_slots, price)
        
        # Combine sizing methods - use minimum for safety
        base_size = min(size_fixed, size_vol, size_equal)
        
        # Scale by signal strength
        scaled_size = int(base_size * abs_signal)
        
        # Apply position limits
        max_position_value = self.current_capital * self.config.max_position_size_pct
        max_size = int(max_position_value / price)
        
        # Apply minimum position value
        min_size = int(self.config.min_position_value / price) if price > 0 else 0
        
        # Final size
        final_size = min(scaled_size, max_size)
        
        if final_size < min_size:
            notes.append(f'Position too small (min: {min_size})')
            final_size = 0
        
        # Check cash availability
        required_cash = final_size * price
        if required_cash > self.cash * self.config.max_total_exposure_pct:
            available_size = int((self.cash * self.config.max_total_exposure_pct) / price)
            final_size = min(final_size, available_size)
            notes.append('Size reduced due to cash constraints')
        
        # Apply exposure limits
        total_exposure_after = self._calculate_exposure() + final_size * price
        max_exposure = self.current_capital * self.config.max_total_exposure_pct
        if total_exposure_after > max_exposure:
            excess = total_exposure_after - max_exposure
            reduce_by = int(excess / price) + 1
            final_size = max(0, final_size - reduce_by)
            notes.append('Size reduced due to exposure limits')
        
        return OrderSizing(
            symbol=symbol,
            side=side,
            quantity=final_size,
            max_quantity=max_size,
            position_value=final_size * price,
            risk_adjusted=True,
            sizing_method='hybrid',
            notes=notes
        )
    
    def check_risk_limits(self) -> Tuple[bool, List[str]]:
        """
        Check all portfolio-level risk limits.
        
        Returns:
            (can_trade: bool, violations: List[str])
        """
        violations = []
        
        metrics = self.get_risk_metrics()
        
        # Check drawdown limit
        if abs(metrics.current_drawdown) >= self.config.max_drawdown_pct:
            violations.append(
                f'Max drawdown exceeded: {metrics.current_drawdown:.1%} >= {self.config.max_drawdown_pct:.1%}'
            )
        
        # Check daily loss limit
        daily_pnl_pct = metrics.daily_pnl / self.daily_starting_equity if self.daily_starting_equity > 0 else 0
        if daily_pnl_pct <= -self.config.daily_loss_limit_pct:
            violations.append(
                f'Daily loss limit: {daily_pnl_pct:.1%} <= -{self.config.daily_loss_limit_pct:.1%}'
            )
        
        # Check exposure limit
        if metrics.exposure_pct > self.config.max_total_exposure_pct:
            violations.append(
                f'Exposure limit: {metrics.exposure_pct:.1%} > {self.config.max_total_exposure_pct:.1%}'
            )
        
        # Check position count
        if metrics.positions_count >= self.config.max_positions:
            violations.append(
                f'Max positions reached: {metrics.positions_count} >= {self.config.max_positions}'
            )
        
        # Check VaR limit
        if abs(metrics.var_95) > self.current_capital * self.config.max_var_pct:
            violations.append(
                f'VaR limit exceeded: {metrics.var_95:,.0f} > {self.current_capital * self.config.max_var_pct:,.0f}'
            )
        
        can_trade = len(violations) == 0
        return can_trade, violations
    
    def get_risk_metrics(self) -> RiskMetrics:
        """Calculate current portfolio risk metrics."""
        # Calculate totals
        total_exposure = self._calculate_exposure()
        unrealized_pnl = sum(p.unrealized_pnl for p in self.positions.values())
        
        # Current equity
        current_equity = self.cash + total_exposure
        self.current_capital = current_equity
        
        # Update peak equity
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        # Drawdown
        current_drawdown = (current_equity - self.peak_equity) / self.peak_equity if self.peak_equity > 0 else 0
        
        # Max historical drawdown
        max_drawdown = self._calculate_max_drawdown()
        
        # Daily P&L
        daily_pnl = current_equity - self.daily_starting_equity
        
        # VaR calculation (parametric)
        var_95, var_99 = self._calculate_var()
        
        # Sharpe ratio (annualized)
        sharpe = self._calculate_sharpe()
        
        # Largest position
        if self.positions:
            largest_pos = max(self.positions.values(), key=lambda p: p.market_value)
            largest_pct = largest_pos.market_value / current_equity if current_equity > 0 else 0
        else:
            largest_pct = 0
        
        # Determine risk level
        risk_level = self._determine_risk_level(current_drawdown, daily_pnl)
        
        return RiskMetrics(
            total_exposure=total_exposure,
            exposure_pct=total_exposure / current_equity if current_equity > 0 else 0,
            cash=self.cash,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=self.realized_pnl,
            current_drawdown=current_drawdown,
            max_drawdown=max_drawdown,
            daily_pnl=daily_pnl,
            var_95=var_95,
            var_99=var_99,
            sharpe_ratio=sharpe,
            risk_level=risk_level,
            positions_count=len(self.positions),
            largest_position_pct=largest_pct
        )
    
    def update_position(self, symbol: str, quantity: int, price: float, 
                        side: str, is_close: bool = False):
        """
        Update position after a trade.
        
        Args:
            symbol: Symbol traded
            quantity: Quantity traded
            price: Execution price
            side: 'buy' or 'sell'
            is_close: Whether this closes a position
        """
        if is_close and symbol in self.positions:
            # Close position
            pos = self.positions[symbol]
            realized = pos.unrealized_pnl
            self.realized_pnl += realized
            self.cash += pos.market_value + realized
            del self.positions[symbol]
            
            self.trade_history.append({
                'symbol': symbol,
                'action': 'close',
                'quantity': quantity,
                'price': price,
                'pnl': realized,
                'timestamp': pd.Timestamp.now()
            })
            
            logger.info(f"Closed {symbol} position. Realized P&L: {realized:,.2f}")
        
        elif symbol in self.positions:
            # Update existing position
            pos = self.positions[symbol]
            pos.current_price = price
            
            if (side == 'buy' and pos.side == 'long') or (side == 'sell' and pos.side == 'short'):
                # Adding to position
                total_cost = pos.cost_basis + quantity * price
                pos.quantity += quantity
                pos.entry_price = total_cost / pos.quantity
                self.cash -= quantity * price
            else:
                # Reducing position
                reduce_qty = min(quantity, pos.quantity)
                pos.quantity -= reduce_qty
                self.cash += reduce_qty * price
                
                if pos.quantity <= 0:
                    del self.positions[symbol]
        
        else:
            # New position
            position_side = 'long' if side == 'buy' else 'short'
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                entry_price=price,
                current_price=price,
                side=position_side,
                entry_time=pd.Timestamp.now()
            )
            self.cash -= quantity * price
            
            logger.info(f"Opened {position_side} position in {symbol}: {quantity} @ {price}")
        
        # Update equity history
        equity = self.cash + self._calculate_exposure()
        self.equity_history.append((pd.Timestamp.now(), equity))
    
    def update_prices(self, prices: Dict[str, float]):
        """Update current prices for all positions."""
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].current_price = price
    
    def reset_daily(self):
        """Reset daily tracking (call at market open)."""
        self.daily_starting_equity = self.current_capital
    
    def _can_open_position(self, symbol: str, side: str) -> bool:
        """Check if a new position can be opened."""
        can_trade, violations = self.check_risk_limits()
        
        if not can_trade:
            # Only block if it's not just max positions limit
            if len(violations) == 1 and 'Max positions' in violations[0]:
                if symbol in self.positions:
                    return True  # Can add to existing position
            return False
        
        return True
    
    def _calculate_exposure(self) -> float:
        """Calculate total portfolio exposure."""
        return sum(p.market_value for p in self.positions.values())
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum historical drawdown."""
        if len(self.equity_history) < 2:
            return 0
        
        equities = [e[1] for e in self.equity_history]
        peak = equities[0]
        max_dd = 0
        
        for eq in equities:
            if eq > peak:
                peak = eq
            dd = (eq - peak) / peak if peak > 0 else 0
            max_dd = min(max_dd, dd)
        
        return max_dd
    
    def _calculate_var(self) -> Tuple[float, float]:
        """Calculate Value at Risk (parametric method)."""
        if len(self.equity_history) < 20:
            return 0, 0
        
        equities = pd.Series([e[1] for e in self.equity_history])
        returns = equities.pct_change().dropna()
        
        if len(returns) < 2:
            return 0, 0
        
        mean_ret = returns.mean()
        std_ret = returns.std()
        
        # VaR at 95% and 99% confidence
        var_95 = self.current_capital * (mean_ret - 1.645 * std_ret)
        var_99 = self.current_capital * (mean_ret - 2.326 * std_ret)
        
        return var_95, var_99
    
    def _calculate_sharpe(self, risk_free_rate: float = 0.05) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(self.equity_history) < 20:
            return 0
        
        equities = pd.Series([e[1] for e in self.equity_history])
        returns = equities.pct_change().dropna()
        
        if len(returns) < 2 or returns.std() == 0:
            return 0
        
        excess_return = returns.mean() - risk_free_rate / 252
        sharpe = (excess_return / returns.std()) * np.sqrt(252)
        
        return sharpe
    
    def _determine_risk_level(self, drawdown: float, daily_pnl: float) -> RiskLevel:
        """Determine current risk level."""
        daily_pnl_pct = daily_pnl / self.daily_starting_equity if self.daily_starting_equity > 0 else 0
        
        # Critical: near kill switch levels
        if abs(drawdown) > self.config.max_drawdown_pct * 0.9:
            return RiskLevel.CRITICAL
        if daily_pnl_pct < -self.config.daily_loss_limit_pct * 0.9:
            return RiskLevel.CRITICAL
        
        # High: significant losses
        if abs(drawdown) > self.config.max_drawdown_pct * 0.7:
            return RiskLevel.HIGH
        if daily_pnl_pct < -self.config.daily_loss_limit_pct * 0.7:
            return RiskLevel.HIGH
        
        # Elevated: moderate losses
        if abs(drawdown) > self.config.max_drawdown_pct * 0.5:
            return RiskLevel.ELEVATED
        
        # Low: no significant drawdown
        if abs(drawdown) < self.config.max_drawdown_pct * 0.2:
            return RiskLevel.LOW
        
        return RiskLevel.NORMAL
    
    def should_reduce_exposure(self) -> Tuple[bool, float]:
        """
        Check if exposure should be reduced based on risk.
        
        Returns:
            (should_reduce: bool, target_reduction_pct: float)
        """
        metrics = self.get_risk_metrics()
        
        if metrics.risk_level == RiskLevel.CRITICAL:
            return True, 0.5  # Reduce 50%
        elif metrics.risk_level == RiskLevel.HIGH:
            return True, 0.3  # Reduce 30%
        elif metrics.risk_level == RiskLevel.ELEVATED:
            return True, 0.15  # Reduce 15%
        
        return False, 0