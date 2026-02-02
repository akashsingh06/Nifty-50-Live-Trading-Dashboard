"""
Paper Trading Engine for NIFTY 50 F&O
=====================================

Virtual portfolio management with:
- Position tracking
- P&L calculation
- Trade history
- Performance analytics
"""

import json
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Optional
from enum import Enum
import threading


class TradeStatus(Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    SL_HIT = "SL HIT"
    TARGET_HIT = "TARGET HIT"
    EXPIRED = "EXPIRED"


@dataclass
class PaperTrade:
    """Individual paper trade record with enhanced tracking."""
    trade_id: str
    timestamp: datetime
    signal_type: str  # BUY CALL / BUY PUT
    
    # Underlying details
    spot_price: float
    strike_price: int
    option_type: str  # CE / PE
    
    # Entry details
    entry_premium: float
    lot_size: int
    quantity: int  # Number of lots
    entry_value: float  # Total entry value
    
    # Exit targets
    stop_loss_premium: float
    target_1_premium: float
    target_2_premium: float
    
    # Spot levels
    spot_stop_loss: float
    spot_target: float
    
    # Time management
    recommended_hold_time: str
    max_hold_time: str
    entry_time: str
    exit_by: str
    
    # Status
    status: TradeStatus
    exit_premium: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_value: Optional[float] = None
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    exit_reason: Optional[str] = None
    
    # Enhanced: Trailing stop loss
    trailing_sl_enabled: bool = False
    trailing_sl_trigger: float = 0.0  # Trigger when profit reaches this
    trailing_sl_distance: float = 0.0  # Distance from peak
    highest_premium: float = 0.0
    current_trailing_sl: float = 0.0
    
    # Enhanced: Partial profit booking
    partial_exit_done: bool = False
    partial_exit_premium: float = 0.0
    partial_exit_pnl: float = 0.0
    remaining_quantity: int = 0
    
    def to_dict(self) -> dict:
        data = asdict(self)
        data['timestamp'] = self.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        data['status'] = self.status.value
        if self.exit_time:
            data['exit_time'] = self.exit_time.strftime('%Y-%m-%d %H:%M:%S')
        return data


@dataclass 
class TradeRecommendation:
    """Detailed trade recommendation with specific prices."""
    # Signal info
    signal_type: str
    market_trend: str
    confidence: str
    
    # SPOT Details
    spot_price: float
    spot_stop_loss: float
    spot_target_1: float
    spot_target_2: float
    
    # OPTION Details
    strike_price: int
    option_type: str  # CE / PE
    strike_type: str  # ATM / ITM / OTM
    
    # Premium prices (Option buying/selling prices)
    entry_premium_range: str  # e.g., "₹180-200"
    entry_premium_ideal: float
    stop_loss_premium: float
    target_1_premium: float
    target_2_premium: float
    
    # Trade sizing
    lot_size: int
    recommended_lots: int
    capital_required: float
    max_risk: float
    
    # Time management
    entry_window: str  # e.g., "9:20 AM - 9:45 AM"
    recommended_hold: str  # e.g., "30-60 minutes"
    max_hold: str  # e.g., "Until 2:30 PM"
    exit_by: str  # Mandatory exit time
    
    # Risk-Reward
    risk_per_lot: float
    reward_per_lot: float
    risk_reward_ratio: str
    
    # Reasoning
    reasoning: List[str]
    
    def to_dict(self) -> dict:
        return asdict(self)


class PaperTradingEngine:
    """
    Enhanced Paper Trading Engine for virtual F&O trading.
    
    Features:
    - Virtual capital management
    - Position tracking with trailing stop loss
    - Automatic partial profit booking
    - Automatic P&L calculation
    - Trade history with analytics
    - Risk management alerts
    """
    
    NIFTY_LOT_SIZE = 65  # Current NIFTY lot size (1 lot = 65 units)
    
    # Trailing SL settings
    TRAILING_SL_TRIGGER_PCT = 30  # Activate trailing SL after 30% profit
    TRAILING_SL_DISTANCE_PCT = 15  # Trail 15% behind peak
    
    # Partial profit settings
    PARTIAL_EXIT_AT_T1 = True  # Exit 50% at Target 1
    PARTIAL_EXIT_RATIO = 0.5  # Exit 50% of position
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.open_positions: List[PaperTrade] = []
        self.closed_trades: List[PaperTrade] = []
        self.trade_counter = 0
        self.data_file = "paper_trades.json"
        self.lock = threading.Lock()
        
        # Performance tracking
        self.max_drawdown = 0.0
        self.peak_capital = initial_capital
        self.consecutive_losses = 0
        self.max_consecutive_losses = 0
        
        # Load existing data
        self._load_data()
    
    def _load_data(self):
        """Load paper trading data from file."""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    self.current_capital = data.get('current_capital', self.initial_capital)
                    self.trade_counter = data.get('trade_counter', 0)
                    # Load closed trades count for stats
        except Exception as e:
            print(f"Error loading paper trade data: {e}")
    
    def _save_data(self):
        """Save paper trading data to file."""
        try:
            with open(self.data_file, 'w') as f:
                data = {
                    'current_capital': self.current_capital,
                    'trade_counter': self.trade_counter,
                    'open_positions': [t.to_dict() for t in self.open_positions],
                    'closed_trades': [t.to_dict() for t in self.closed_trades[-50:]]  # Keep last 50
                }
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving paper trade data: {e}")
    
    def generate_trade_recommendation(self, signal_data: dict) -> Optional[TradeRecommendation]:
        """
        Generate detailed trade recommendation with specific prices.
        
        Args:
            signal_data: Signal from NiftyFnOSignalGenerator
            
        Returns:
            TradeRecommendation with all trade details
        """
        if signal_data['signal'] == 'NO TRADE':
            return None
        
        spot_price = signal_data['spot_price']
        is_call = 'CALL' in signal_data['signal']
        
        # Calculate strike price (ATM)
        strike_price = round(spot_price / 50) * 50
        option_type = "CE" if is_call else "PE"
        
        # Estimate option premium based on moneyness and volatility
        # Using approximate Black-Scholes-like estimation
        atm_premium = self._estimate_premium(spot_price, strike_price, is_call)
        
        # Calculate spot levels
        atr = abs(signal_data.get('ema_20', spot_price) - signal_data.get('ema_50', spot_price)) * 0.5
        if atr < 30:
            atr = 50  # Minimum ATR for NIFTY
        
        if is_call:
            spot_sl = round(spot_price - atr * 1.5, 0)
            spot_t1 = round(spot_price + atr * 2, 0)
            spot_t2 = round(spot_price + atr * 3.5, 0)
        else:
            spot_sl = round(spot_price + atr * 1.5, 0)
            spot_t1 = round(spot_price - atr * 2, 0)
            spot_t2 = round(spot_price - atr * 3.5, 0)
        
        # Calculate premium targets based on delta approximation
        # ATM options have ~0.5 delta
        delta = 0.50
        spot_move_to_sl = abs(spot_price - spot_sl)
        spot_move_to_t1 = abs(spot_t1 - spot_price)
        spot_move_to_t2 = abs(spot_t2 - spot_price)
        
        sl_premium = max(atm_premium - (spot_move_to_sl * delta), atm_premium * 0.5)
        t1_premium = atm_premium + (spot_move_to_t1 * delta)
        t2_premium = atm_premium + (spot_move_to_t2 * delta)
        
        # Round premiums
        atm_premium = round(atm_premium, 1)
        sl_premium = round(sl_premium, 1)
        t1_premium = round(t1_premium, 1)
        t2_premium = round(t2_premium, 1)
        
        # Entry range (market fluctuation buffer)
        entry_low = round(atm_premium * 0.97, 1)
        entry_high = round(atm_premium * 1.03, 1)
        
        # Calculate risk-reward
        risk_per_lot = (atm_premium - sl_premium) * self.NIFTY_LOT_SIZE
        reward_per_lot = (t1_premium - atm_premium) * self.NIFTY_LOT_SIZE
        rr_ratio = round(reward_per_lot / risk_per_lot, 2) if risk_per_lot > 0 else 0
        
        # Capital and lot calculation
        capital_per_lot = atm_premium * self.NIFTY_LOT_SIZE
        max_risk_amount = self.current_capital * 0.01  # 1% risk
        recommended_lots = max(1, int(max_risk_amount / risk_per_lot)) if risk_per_lot > 0 else 1
        recommended_lots = min(recommended_lots, 2)  # Max 2 lots for safety
        
        # Time management
        now = datetime.now()
        entry_end = now + timedelta(minutes=25)
        
        # Hold time based on trend strength
        if signal_data['confidence'] == 'High':
            hold_time = "45-90 minutes"
            max_hold = "Until 2:30 PM"
        elif signal_data['confidence'] == 'Medium':
            hold_time = "30-60 minutes"
            max_hold = "Until 2:00 PM"
        else:
            hold_time = "20-40 minutes"
            max_hold = "Until 1:30 PM"
        
        return TradeRecommendation(
            signal_type=signal_data['signal'],
            market_trend=signal_data['market_trend'],
            confidence=signal_data['confidence'],
            
            spot_price=spot_price,
            spot_stop_loss=spot_sl,
            spot_target_1=spot_t1,
            spot_target_2=spot_t2,
            
            strike_price=strike_price,
            option_type=option_type,
            strike_type="ATM",
            
            entry_premium_range=f"₹{entry_low}-{entry_high}",
            entry_premium_ideal=atm_premium,
            stop_loss_premium=sl_premium,
            target_1_premium=t1_premium,
            target_2_premium=t2_premium,
            
            lot_size=self.NIFTY_LOT_SIZE,
            recommended_lots=recommended_lots,
            capital_required=round(capital_per_lot * recommended_lots, 0),
            max_risk=round(risk_per_lot * recommended_lots, 0),
            
            entry_window=f"{now.strftime('%I:%M %p')} - {entry_end.strftime('%I:%M %p')}",
            recommended_hold=hold_time,
            max_hold=max_hold,
            exit_by="2:45 PM (Mandatory)",
            
            risk_per_lot=round(risk_per_lot, 0),
            reward_per_lot=round(reward_per_lot, 0),
            risk_reward_ratio=f"1:{rr_ratio}",
            
            reasoning=signal_data['reasoning'][:3]
        )
    
    def _estimate_premium(self, spot: float, strike: float, is_call: bool) -> float:
        """
        Estimate option premium for ATM options.
        
        Uses simplified estimation based on:
        - Typical NIFTY ATM premiums
        - Time value approximation
        """
        # Base ATM premium (typical range for NIFTY weekly options)
        # Varies based on VIX and time to expiry
        now = datetime.now()
        
        # Estimate days to weekly expiry (Thursday)
        days_to_expiry = (3 - now.weekday()) % 7
        if days_to_expiry == 0 and now.hour >= 15:
            days_to_expiry = 7
        
        # Base premium calculation
        # ATM options typically trade at 0.3-0.5% of spot for weekly
        base_premium_pct = 0.004  # 0.4% of spot
        time_decay_factor = max(0.3, days_to_expiry / 5)  # Decay as expiry approaches
        
        base_premium = spot * base_premium_pct * time_decay_factor
        
        # Adjust for moneyness
        moneyness = (spot - strike) / spot
        if is_call:
            if moneyness > 0:  # ITM
                intrinsic = spot - strike
                base_premium += intrinsic
            # OTM calls have lower premium
            elif moneyness < -0.005:
                base_premium *= 0.7
        else:  # PUT
            if moneyness < 0:  # ITM
                intrinsic = strike - spot
                base_premium += intrinsic
            # OTM puts have lower premium
            elif moneyness > 0.005:
                base_premium *= 0.7
        
        # Minimum premium
        return max(base_premium, 50)
    
    def execute_paper_trade(self, recommendation: TradeRecommendation) -> PaperTrade:
        """
        Execute a paper trade based on recommendation.
        
        Args:
            recommendation: TradeRecommendation object
            
        Returns:
            PaperTrade object
        """
        with self.lock:
            self.trade_counter += 1
            trade_id = f"PT{self.trade_counter:04d}"
            
            now = datetime.now()
            entry_value = recommendation.entry_premium_ideal * recommendation.lot_size * recommendation.recommended_lots
            
            trade = PaperTrade(
                trade_id=trade_id,
                timestamp=now,
                signal_type=recommendation.signal_type,
                
                spot_price=recommendation.spot_price,
                strike_price=recommendation.strike_price,
                option_type=recommendation.option_type,
                
                entry_premium=recommendation.entry_premium_ideal,
                lot_size=recommendation.lot_size,
                quantity=recommendation.recommended_lots,
                entry_value=entry_value,
                
                stop_loss_premium=recommendation.stop_loss_premium,
                target_1_premium=recommendation.target_1_premium,
                target_2_premium=recommendation.target_2_premium,
                
                spot_stop_loss=recommendation.spot_stop_loss,
                spot_target=recommendation.spot_target_1,
                
                recommended_hold_time=recommendation.recommended_hold,
                max_hold_time=recommendation.max_hold,
                entry_time=now.strftime('%I:%M %p'),
                exit_by=recommendation.exit_by,
                
                status=TradeStatus.OPEN
            )
            
            self.open_positions.append(trade)
            self._save_data()
            
            return trade
    
    def close_paper_trade(self, trade_id: str, exit_premium: float, 
                          exit_reason: str = "Manual Exit") -> Optional[PaperTrade]:
        """
        Close an open paper trade.
        
        Args:
            trade_id: Trade ID to close
            exit_premium: Exit premium price
            exit_reason: Reason for exit
            
        Returns:
            Closed PaperTrade object
        """
        with self.lock:
            for i, trade in enumerate(self.open_positions):
                if trade.trade_id == trade_id:
                    trade.exit_premium = exit_premium
                    trade.exit_time = datetime.now()
                    trade.exit_value = exit_premium * trade.lot_size * trade.quantity
                    trade.pnl = trade.exit_value - trade.entry_value
                    trade.pnl_percent = (trade.pnl / trade.entry_value) * 100
                    trade.exit_reason = exit_reason
                    
                    # Determine status
                    if exit_premium <= trade.stop_loss_premium:
                        trade.status = TradeStatus.SL_HIT
                    elif exit_premium >= trade.target_1_premium:
                        trade.status = TradeStatus.TARGET_HIT
                    else:
                        trade.status = TradeStatus.CLOSED
                    
                    # Update capital
                    self.current_capital += trade.pnl
                    
                    # Move to closed trades
                    self.closed_trades.append(trade)
                    self.open_positions.pop(i)
                    
                    self._save_data()
                    return trade
            
            return None
    
    def check_and_update_positions(self, current_spot: float) -> List[dict]:
        """
        Check open positions with enhanced tracking.
        
        Features:
        - Stop loss monitoring
        - Target hit detection  
        - Trailing stop loss updates
        - Partial profit alerts
        
        Args:
            current_spot: Current NIFTY spot price
            
        Returns:
            List of alert dictionaries
        """
        alerts = []
        
        with self.lock:
            for trade in self.open_positions:
                is_call = trade.option_type == "CE"
                
                # Estimate current premium based on spot movement
                spot_change = current_spot - trade.spot_price
                delta = 0.5  # ATM delta
                estimated_premium = trade.entry_premium + (spot_change * delta if is_call else -spot_change * delta)
                estimated_premium = max(estimated_premium, 5)  # Min premium
                
                # Update highest premium for trailing SL
                if estimated_premium > trade.highest_premium:
                    trade.highest_premium = estimated_premium
                
                # Calculate current P&L
                current_pnl_pct = ((estimated_premium - trade.entry_premium) / trade.entry_premium) * 100
                
                # Check and update trailing SL
                if current_pnl_pct >= self.TRAILING_SL_TRIGGER_PCT and not trade.trailing_sl_enabled:
                    trade.trailing_sl_enabled = True
                    trade.trailing_sl_trigger = trade.entry_premium * (1 + self.TRAILING_SL_TRIGGER_PCT/100)
                    trade.trailing_sl_distance = trade.highest_premium * (self.TRAILING_SL_DISTANCE_PCT/100)
                    trade.current_trailing_sl = trade.highest_premium - trade.trailing_sl_distance
                    alerts.append({
                        'trade_id': trade.trade_id,
                        'type': 'TRAILING_SL_ACTIVATED',
                        'message': f"? {trade.trade_id}: Trailing SL activated @ ₹{trade.current_trailing_sl:.1f}"
                    })
                
                # Update trailing SL level if enabled
                if trade.trailing_sl_enabled:
                    new_trailing_sl = trade.highest_premium - trade.trailing_sl_distance
                    if new_trailing_sl > trade.current_trailing_sl:
                        trade.current_trailing_sl = new_trailing_sl
                        alerts.append({
                            'trade_id': trade.trade_id,
                            'type': 'TRAILING_SL_UPDATED',
                            'message': f"? {trade.trade_id}: Trailing SL raised to ₹{trade.current_trailing_sl:.1f}"
                        })
                    
                    # Check if trailing SL hit
                    if estimated_premium <= trade.current_trailing_sl:
                        alerts.append({
                            'trade_id': trade.trade_id,
                            'type': 'TRAILING_SL_HIT',
                            'message': f"? {trade.trade_id}: Trailing SL hit! Exit @ ₹{estimated_premium:.1f}"
                        })
                
                # Check for partial profit booking at T1
                if (self.PARTIAL_EXIT_AT_T1 and 
                    not trade.partial_exit_done and 
                    estimated_premium >= trade.target_1_premium):
                    
                    trade.partial_exit_done = True
                    partial_qty = int(trade.quantity * self.PARTIAL_EXIT_RATIO)
                    trade.partial_exit_premium = estimated_premium
                    trade.partial_exit_pnl = (estimated_premium - trade.entry_premium) * trade.lot_size * partial_qty
                    trade.remaining_quantity = trade.quantity - partial_qty
                    
                    alerts.append({
                        'trade_id': trade.trade_id,
                        'type': 'PARTIAL_EXIT',
                        'message': f"✅ {trade.trade_id}: Partial profit booked! {partial_qty} lots @ ₹{estimated_premium:.1f}, P&L: ₹{trade.partial_exit_pnl:.0f}"
                    })
                
                # Standard SL check
                if is_call and current_spot <= trade.spot_stop_loss:
                    alerts.append({
                        'trade_id': trade.trade_id,
                        'type': 'SL_WARNING',
                        'message': f"⚠️ {trade.trade_id}: SL level breached! Spot: {current_spot:.0f}, SL: {trade.spot_stop_loss:.0f}"
                    })
                elif not is_call and current_spot >= trade.spot_stop_loss:
                    alerts.append({
                        'trade_id': trade.trade_id,
                        'type': 'SL_WARNING',
                        'message': f"⚠️ {trade.trade_id}: SL level breached! Spot: {current_spot:.0f}, SL: {trade.spot_stop_loss:.0f}"
                    })
                
                # Check Target hit
                if is_call and current_spot >= trade.spot_target:
                    alerts.append({
                        'trade_id': trade.trade_id,
                        'type': 'TARGET_HIT',
                        'message': f"? {trade.trade_id}: Target reached! Spot: {current_spot}, Target: {trade.spot_target}"
                    })
                elif not is_call and current_spot <= trade.spot_target:
                    alerts.append({
                        'trade_id': trade.trade_id,
                        'type': 'TARGET_HIT',
                        'message': f"? {trade.trade_id}: Target reached! Spot: {current_spot}, Target: {trade.spot_target}"
                    })
        
        return alerts
    
    def get_portfolio_summary(self) -> dict:
        """
        Get comprehensive portfolio summary with advanced metrics.
        
        Returns:
            Dictionary with portfolio statistics including:
            - Capital metrics
            - Win/loss statistics
            - Risk metrics (max drawdown, profit factor, Sharpe-like ratio)
            - Streak analysis
        """
        total_pnl = sum(t.pnl or 0 for t in self.closed_trades)
        winning_trades = sum(1 for t in self.closed_trades if (t.pnl or 0) > 0)
        losing_trades = sum(1 for t in self.closed_trades if (t.pnl or 0) < 0)
        total_trades = len(self.closed_trades)
        
        open_value = sum(t.entry_value for t in self.open_positions)
        
        # Calculate gross profit and gross loss
        gross_profit = sum(t.pnl for t in self.closed_trades if (t.pnl or 0) > 0)
        gross_loss = abs(sum(t.pnl for t in self.closed_trades if (t.pnl or 0) < 0))
        
        # Profit factor (gross profit / gross loss)
        profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else float('inf')
        if profit_factor == float('inf'):
            profit_factor = 999.99  # Cap for JSON serialization
        
        # Calculate max drawdown
        max_drawdown = 0
        peak_capital = self.initial_capital
        running_capital = self.initial_capital
        
        for trade in self.closed_trades:
            running_capital += (trade.pnl or 0)
            if running_capital > peak_capital:
                peak_capital = running_capital
            drawdown = (peak_capital - running_capital) / peak_capital * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Calculate consecutive wins/losses
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_consecutive_wins = 0
        current_consecutive_losses = 0
        
        for trade in self.closed_trades:
            if (trade.pnl or 0) > 0:
                current_consecutive_wins += 1
                current_consecutive_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_consecutive_wins)
            elif (trade.pnl or 0) < 0:
                current_consecutive_losses += 1
                current_consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_consecutive_losses)
        
        # Calculate average holding time
        holding_times = []
        for trade in self.closed_trades:
            if trade.entry_time and trade.exit_time:
                try:
                    entry = datetime.fromisoformat(trade.entry_time) if isinstance(trade.entry_time, str) else trade.entry_time
                    exit = datetime.fromisoformat(trade.exit_time) if isinstance(trade.exit_time, str) else trade.exit_time
                    holding_times.append((exit - entry).total_seconds() / 60)  # in minutes
                except Exception:
                    pass
        
        avg_holding_time_mins = round(sum(holding_times) / len(holding_times), 1) if holding_times else 0
        
        # Calculate expectancy (average profit per trade)
        expectancy = round(total_pnl / total_trades, 2) if total_trades > 0 else 0
        
        # Risk-reward ratio (avg win / avg loss)
        avg_profit = gross_profit / winning_trades if winning_trades > 0 else 0
        avg_loss = gross_loss / losing_trades if losing_trades > 0 else 0
        risk_reward_ratio = round(avg_profit / avg_loss, 2) if avg_loss > 0 else 0
        
        # Calculate partial profit booking stats
        partial_exits = sum(1 for t in self.closed_trades if t.partial_exit_done)
        total_partial_pnl = sum(t.partial_exit_pnl or 0 for t in self.closed_trades if t.partial_exit_done)
        
        # Trailing SL effectiveness
        trailing_sl_exits = sum(1 for t in self.closed_trades if t.trailing_sl_enabled)
        
        return {
            # Capital metrics
            'initial_capital': self.initial_capital,
            'current_capital': round(self.current_capital, 2),
            'total_pnl': round(total_pnl, 2),
            'total_pnl_percent': round((total_pnl / self.initial_capital) * 100, 2),
            'open_positions': len(self.open_positions),
            'open_value': round(open_value, 2),
            
            # Trade statistics
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': round((winning_trades / total_trades) * 100, 2) if total_trades > 0 else 0,
            
            # Profit/Loss metrics
            'gross_profit': round(gross_profit, 2),
            'gross_loss': round(gross_loss, 2),
            'avg_profit': round(avg_profit, 2),
            'avg_loss': round(avg_loss, 2),
            'largest_win': round(max((t.pnl or 0) for t in self.closed_trades), 2) if self.closed_trades else 0,
            'largest_loss': round(min((t.pnl or 0) for t in self.closed_trades), 2) if self.closed_trades else 0,
            
            # Risk metrics
            'profit_factor': profit_factor,
            'risk_reward_ratio': risk_reward_ratio,
            'max_drawdown': round(max_drawdown, 2),
            'expectancy': expectancy,
            
            # Streak analysis
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'current_streak': current_consecutive_wins if current_consecutive_wins > 0 else -current_consecutive_losses,
            
            # Time metrics
            'avg_holding_time_mins': avg_holding_time_mins,
            
            # Advanced features
            'partial_exits_count': partial_exits,
            'partial_exits_pnl': round(total_partial_pnl, 2),
            'trailing_sl_exits': trailing_sl_exits
        }
    
    def get_open_positions(self) -> List[dict]:
        """Get all open positions."""
        return [t.to_dict() for t in self.open_positions]
    
    def get_trade_history(self, limit: int = 20) -> List[dict]:
        """Get recent trade history."""
        return [t.to_dict() for t in self.closed_trades[-limit:]]
    
    def reset_paper_trading(self):
        """Reset paper trading to initial state."""
        with self.lock:
            self.current_capital = self.initial_capital
            self.open_positions = []
            self.closed_trades = []
            self.trade_counter = 0
            self._save_data()
    
    def reset(self):
        """Alias for reset_paper_trading."""
        self.reset_paper_trading()


# Export
__all__ = ['PaperTradingEngine', 'PaperTrade', 'TradeRecommendation', 'TradeStatus']
