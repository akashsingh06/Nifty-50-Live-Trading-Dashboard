"""
Monitoring Module
=================
Drawdown tracking, kill-switch, performance monitoring, and alerts.

Core principle: "No emotional overrides - rules are enforced automatically"
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
import logging
import threading
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class SystemState(Enum):
    """System operation states."""
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    KILLED = "killed"  # Kill switch activated


@dataclass
class Alert:
    """Alert notification."""
    severity: AlertSeverity
    title: str
    message: str
    timestamp: pd.Timestamp = field(default_factory=pd.Timestamp.now)
    acknowledged: bool = False
    source: str = ""
    
    def to_dict(self) -> dict:
        return {
            'severity': self.severity.value,
            'title': self.title,
            'message': self.message,
            'timestamp': self.timestamp,
            'acknowledged': self.acknowledged,
            'source': self.source
        }


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    # Returns
    total_return: float = 0.0
    total_return_pct: float = 0.0
    daily_return: float = 0.0
    mtd_return: float = 0.0
    ytd_return: float = 0.0
    
    # Risk-adjusted returns
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # Drawdown
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    drawdown_duration_days: int = 0
    
    # Win/Loss
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    
    # Exposure
    avg_exposure: float = 0.0
    max_exposure: float = 0.0
    current_exposure: float = 0.0
    
    # Volatility
    volatility_realized: float = 0.0
    volatility_target: float = 0.0


class AlertManager:
    """Manages alerts and notifications."""
    
    def __init__(self, config=None):
        from ..config import MonitoringConfig
        self.config = config or MonitoringConfig()
        
        self.alerts: List[Alert] = []
        self.alert_handlers: List[Callable[[Alert], None]] = []
        
        # Email configuration
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        self.email_user = ""
        self.email_password = ""
    
    def add_handler(self, handler: Callable[[Alert], None]):
        """Add custom alert handler."""
        self.alert_handlers.append(handler)
    
    def send_alert(self, severity: AlertSeverity, title: str, message: str, source: str = ""):
        """Create and dispatch an alert."""
        alert = Alert(
            severity=severity,
            title=title,
            message=message,
            source=source
        )
        
        self.alerts.append(alert)
        
        # Log the alert
        log_method = getattr(logger, severity.value if severity != AlertSeverity.EMERGENCY else 'critical')
        log_method(f"[ALERT] {title}: {message}")
        
        # Dispatch to channels
        for channel in self.config.alert_channels:
            try:
                if channel == 'console':
                    self._console_alert(alert)
                elif channel == 'email':
                    self._email_alert(alert)
            except Exception as e:
                logger.error(f"Alert dispatch to {channel} failed: {e}")
        
        # Call custom handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")
    
    def _console_alert(self, alert: Alert):
        """Print alert to console with formatting."""
        colors = {
            AlertSeverity.INFO: '\033[94m',      # Blue
            AlertSeverity.WARNING: '\033[93m',   # Yellow
            AlertSeverity.CRITICAL: '\033[91m',  # Red
            AlertSeverity.EMERGENCY: '\033[95m'  # Magenta
        }
        reset = '\033[0m'
        
        color = colors.get(alert.severity, '')
        print(f"{color}[{alert.severity.value.upper()}] {alert.title}{reset}")
        print(f"  {alert.message}")
        print(f"  Time: {alert.timestamp}")
    
    def _email_alert(self, alert: Alert):
        """Send alert via email."""
        if not self.config.email_recipients or not self.email_user:
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_user
            msg['To'] = ', '.join(self.config.email_recipients)
            msg['Subject'] = f"[{alert.severity.value.upper()}] Trading Alert: {alert.title}"
            
            body = f"""
            Trading System Alert
            ====================
            
            Severity: {alert.severity.value.upper()}
            Title: {alert.title}
            Time: {alert.timestamp}
            Source: {alert.source}
            
            Message:
            {alert.message}
            
            --
            Quantitative Trading System
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.email_user, self.email_password)
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            logger.error(f"Email send failed: {e}")
    
    def get_recent_alerts(self, hours: int = 24, severity: AlertSeverity = None) -> List[Alert]:
        """Get recent alerts filtered by time and severity."""
        cutoff = pd.Timestamp.now() - timedelta(hours=hours)
        
        alerts = [a for a in self.alerts if a.timestamp >= cutoff]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return alerts


class KillSwitch:
    """
    Automatic kill switch for emergency situations.
    
    Monitors for dangerous conditions and automatically stops trading.
    No human override allowed once triggered.
    """
    
    def __init__(self, config=None):
        from ..config import MonitoringConfig
        self.config = config or MonitoringConfig()
        
        self.triggered = False
        self.trigger_reason = ""
        self.trigger_time: Optional[pd.Timestamp] = None
        
        # Callbacks to execute when kill switch triggers
        self.callbacks: List[Callable] = []
    
    def add_callback(self, callback: Callable):
        """Add callback to execute when kill switch triggers."""
        self.callbacks.append(callback)
    
    def check_conditions(self, drawdown: float, daily_loss_pct: float, 
                        system_errors: int = 0) -> bool:
        """
        Check kill switch conditions.
        
        Returns True if kill switch should trigger.
        """
        if not self.config.enable_kill_switch:
            return False
        
        if self.triggered:
            return True  # Already triggered
        
        reasons = []
        
        # Check drawdown threshold
        if abs(drawdown) >= self.config.drawdown_kill_threshold:
            reasons.append(f"Drawdown {drawdown:.1%} >= {self.config.drawdown_kill_threshold:.1%}")
        
        # Check daily loss threshold
        if daily_loss_pct <= -self.config.daily_loss_kill_threshold:
            reasons.append(f"Daily loss {daily_loss_pct:.1%} <= -{self.config.daily_loss_kill_threshold:.1%}")
        
        # Check system errors
        if system_errors >= 10:
            reasons.append(f"System errors: {system_errors}")
        
        if reasons:
            self.trigger("; ".join(reasons))
            return True
        
        return False
    
    def trigger(self, reason: str):
        """Trigger the kill switch."""
        if self.triggered:
            return
        
        self.triggered = True
        self.trigger_reason = reason
        self.trigger_time = pd.Timestamp.now()
        
        logger.critical(f"KILL SWITCH TRIGGERED: {reason}")
        
        # Execute callbacks
        for callback in self.callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Kill switch callback error: {e}")
    
    def is_triggered(self) -> bool:
        """Check if kill switch is triggered."""
        return self.triggered
    
    def reset(self, override_key: str = None):
        """
        Reset kill switch (requires manual override).
        
        In production, this should require additional authentication.
        """
        if override_key != "MANUAL_RESET_CONFIRMED":
            logger.warning("Kill switch reset requires proper authorization")
            return False
        
        self.triggered = False
        self.trigger_reason = ""
        self.trigger_time = None
        
        logger.warning("Kill switch reset - trading can resume")
        return True


class PerformanceTracker:
    """Tracks and calculates trading performance metrics."""
    
    def __init__(self, initial_capital: float = 1000000):
        self.initial_capital = initial_capital
        
        # Time series data
        self.equity_curve: List[Tuple[pd.Timestamp, float]] = []
        self.returns_series: List[float] = []
        self.exposure_series: List[float] = []
        
        # Trade tracking
        self.trades: List[Dict] = []
        
        # Peak tracking for drawdown
        self.peak_equity = initial_capital
        self.drawdown_start: Optional[pd.Timestamp] = None
    
    def update_equity(self, timestamp: pd.Timestamp, equity: float, exposure: float = 0):
        """Record equity point."""
        self.equity_curve.append((timestamp, equity))
        self.exposure_series.append(exposure)
        
        # Calculate return
        if len(self.equity_curve) >= 2:
            prev_equity = self.equity_curve[-2][1]
            if prev_equity > 0:
                ret = (equity - prev_equity) / prev_equity
                self.returns_series.append(ret)
        
        # Update peak for drawdown
        if equity > self.peak_equity:
            self.peak_equity = equity
            self.drawdown_start = None
        elif self.drawdown_start is None:
            self.drawdown_start = timestamp
    
    def record_trade(self, symbol: str, side: str, quantity: int, 
                     entry_price: float, exit_price: float, pnl: float):
        """Record a completed trade."""
        self.trades.append({
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'pnl_pct': pnl / (quantity * entry_price) if quantity * entry_price > 0 else 0,
            'timestamp': pd.Timestamp.now()
        })
    
    def get_metrics(self) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        metrics = PerformanceMetrics()
        
        if not self.equity_curve:
            return metrics
        
        # Current equity
        current_equity = self.equity_curve[-1][1]
        
        # Total return
        metrics.total_return = current_equity - self.initial_capital
        metrics.total_return_pct = metrics.total_return / self.initial_capital
        
        # Returns series
        if self.returns_series:
            returns = pd.Series(self.returns_series)
            
            # Daily return (most recent)
            metrics.daily_return = returns.iloc[-1] if len(returns) > 0 else 0
            
            # Volatility
            metrics.volatility_realized = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
            
            # Sharpe Ratio (assuming 5% risk-free rate)
            excess_return = returns.mean() - 0.05/252
            if returns.std() > 0:
                metrics.sharpe_ratio = (excess_return / returns.std()) * np.sqrt(252)
            
            # Sortino Ratio (downside deviation)
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0 and downside_returns.std() > 0:
                metrics.sortino_ratio = (excess_return / downside_returns.std()) * np.sqrt(252)
        
        # Drawdown
        metrics.current_drawdown = (current_equity - self.peak_equity) / self.peak_equity
        metrics.max_drawdown = self._calculate_max_drawdown()
        
        if self.drawdown_start:
            metrics.drawdown_duration_days = (pd.Timestamp.now() - self.drawdown_start).days
        
        # Calmar Ratio
        if abs(metrics.max_drawdown) > 0:
            annualized_return = metrics.total_return_pct  # Simplified
            metrics.calmar_ratio = annualized_return / abs(metrics.max_drawdown)
        
        # Trade statistics
        if self.trades:
            metrics.total_trades = len(self.trades)
            winning = [t for t in self.trades if t['pnl'] > 0]
            losing = [t for t in self.trades if t['pnl'] < 0]
            
            metrics.winning_trades = len(winning)
            metrics.losing_trades = len(losing)
            metrics.win_rate = len(winning) / len(self.trades) if self.trades else 0
            
            if winning:
                metrics.avg_win = np.mean([t['pnl'] for t in winning])
            if losing:
                metrics.avg_loss = abs(np.mean([t['pnl'] for t in losing]))
            
            total_wins = sum(t['pnl'] for t in winning)
            total_losses = abs(sum(t['pnl'] for t in losing))
            if total_losses > 0:
                metrics.profit_factor = total_wins / total_losses
        
        # Exposure
        if self.exposure_series:
            metrics.avg_exposure = np.mean(self.exposure_series)
            metrics.max_exposure = max(self.exposure_series)
            metrics.current_exposure = self.exposure_series[-1]
        
        return metrics
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum historical drawdown."""
        if len(self.equity_curve) < 2:
            return 0
        
        equities = [e[1] for e in self.equity_curve]
        peak = equities[0]
        max_dd = 0
        
        for eq in equities:
            if eq > peak:
                peak = eq
            dd = (eq - peak) / peak if peak > 0 else 0
            max_dd = min(max_dd, dd)
        
        return max_dd
    
    def get_equity_series(self) -> pd.Series:
        """Get equity curve as pandas Series."""
        if not self.equity_curve:
            return pd.Series()
        
        timestamps = [e[0] for e in self.equity_curve]
        values = [e[1] for e in self.equity_curve]
        
        return pd.Series(values, index=timestamps)


class MonitoringSystem:
    """
    Main monitoring system coordinating all monitoring components.
    
    Responsibilities:
    - Real-time performance tracking
    - Kill switch monitoring
    - Alert management
    - System health checks
    """
    
    def __init__(self, config=None, initial_capital: float = 1000000):
        from ..config import MonitoringConfig
        self.config = config or MonitoringConfig()
        
        self.state = SystemState.STOPPED
        self.initial_capital = initial_capital
        
        # Components
        self.alert_manager = AlertManager(self.config)
        self.kill_switch = KillSwitch(self.config)
        self.performance = PerformanceTracker(initial_capital)
        
        # Monitoring state
        self.last_check_time: Optional[pd.Timestamp] = None
        self.error_count = 0
        self.monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        
        # Callbacks
        self._on_kill_switch: List[Callable] = []
    
    def start(self):
        """Start the monitoring system."""
        self.state = SystemState.RUNNING
        self._stop_monitoring.clear()
        
        # Start background monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Monitoring system started")
        self.alert_manager.send_alert(
            AlertSeverity.INFO,
            "System Started",
            "Trading system monitoring is now active",
            source="MonitoringSystem"
        )
    
    def stop(self):
        """Stop the monitoring system."""
        self.state = SystemState.STOPPED
        self._stop_monitoring.set()
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("Monitoring system stopped")
    
    def update(self, equity: float, exposure: float, drawdown: float, daily_pnl_pct: float):
        """
        Update monitoring with latest state.
        
        Should be called regularly with current portfolio state.
        """
        timestamp = pd.Timestamp.now()
        
        # Update performance tracker
        self.performance.update_equity(timestamp, equity, exposure)
        
        # Check kill switch conditions
        if self.kill_switch.check_conditions(drawdown, daily_pnl_pct, self.error_count):
            self.state = SystemState.KILLED
            
            self.alert_manager.send_alert(
                AlertSeverity.EMERGENCY,
                "KILL SWITCH TRIGGERED",
                f"Trading halted: {self.kill_switch.trigger_reason}",
                source="KillSwitch"
            )
            
            # Execute callbacks
            for callback in self._on_kill_switch:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Kill switch callback error: {e}")
        
        # Check for warnings
        self._check_warnings(drawdown, daily_pnl_pct, exposure)
        
        self.last_check_time = timestamp
    
    def _check_warnings(self, drawdown: float, daily_pnl_pct: float, exposure: float):
        """Check for warning conditions and send alerts."""
        # Drawdown warnings
        if abs(drawdown) > self.config.drawdown_kill_threshold * 0.7:
            self.alert_manager.send_alert(
                AlertSeverity.CRITICAL,
                "High Drawdown Warning",
                f"Current drawdown: {drawdown:.1%} (threshold: {self.config.drawdown_kill_threshold:.1%})",
                source="RiskMonitor"
            )
        elif abs(drawdown) > self.config.drawdown_kill_threshold * 0.5:
            self.alert_manager.send_alert(
                AlertSeverity.WARNING,
                "Elevated Drawdown",
                f"Current drawdown: {drawdown:.1%}",
                source="RiskMonitor"
            )
        
        # Daily loss warnings
        if daily_pnl_pct < -self.config.daily_loss_kill_threshold * 0.7:
            self.alert_manager.send_alert(
                AlertSeverity.CRITICAL,
                "High Daily Loss Warning",
                f"Daily P&L: {daily_pnl_pct:.1%} (threshold: -{self.config.daily_loss_kill_threshold:.1%})",
                source="RiskMonitor"
            )
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while not self._stop_monitoring.is_set():
            try:
                # Perform periodic health checks
                self._health_check()
                
                time.sleep(self.config.check_interval_seconds)
                
            except Exception as e:
                self.error_count += 1
                logger.error(f"Monitoring error: {e}")
                
                if self.error_count >= 10:
                    self.alert_manager.send_alert(
                        AlertSeverity.CRITICAL,
                        "System Errors",
                        f"Multiple system errors detected: {self.error_count}",
                        source="HealthCheck"
                    )
    
    def _health_check(self):
        """Perform system health check."""
        # Check if updates are being received
        if self.last_check_time:
            time_since_update = (pd.Timestamp.now() - self.last_check_time).total_seconds()
            
            if time_since_update > 300:  # 5 minutes
                self.alert_manager.send_alert(
                    AlertSeverity.WARNING,
                    "Stale Data",
                    f"No updates received for {time_since_update:.0f} seconds",
                    source="HealthCheck"
                )
    
    def on_kill_switch(self, callback: Callable):
        """Register callback for kill switch trigger."""
        self._on_kill_switch.append(callback)
        self.kill_switch.add_callback(callback)
    
    def record_trade(self, symbol: str, side: str, quantity: int,
                    entry_price: float, exit_price: float, pnl: float):
        """Record a completed trade."""
        self.performance.record_trade(symbol, side, quantity, entry_price, exit_price, pnl)
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        return self.performance.get_metrics()
    
    def get_status(self) -> Dict:
        """Get current monitoring status."""
        metrics = self.performance.get_metrics()
        
        return {
            'state': self.state.value,
            'kill_switch_triggered': self.kill_switch.is_triggered(),
            'kill_switch_reason': self.kill_switch.trigger_reason,
            'error_count': self.error_count,
            'last_update': self.last_check_time,
            'total_return_pct': metrics.total_return_pct,
            'current_drawdown': metrics.current_drawdown,
            'max_drawdown': metrics.max_drawdown,
            'sharpe_ratio': metrics.sharpe_ratio,
            'win_rate': metrics.win_rate,
            'total_trades': metrics.total_trades,
            'recent_alerts': len(self.alert_manager.get_recent_alerts(hours=1))
        }
    
    def generate_report(self) -> str:
        """Generate a text performance report."""
        metrics = self.performance.get_metrics()
        status = self.get_status()
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║              QUANTITATIVE TRADING SYSTEM REPORT              ║
╠══════════════════════════════════════════════════════════════╣
║ Status: {status['state'].upper():53s} ║
║ Time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'):55s} ║
╠══════════════════════════════════════════════════════════════╣
║ PERFORMANCE                                                  ║
╟──────────────────────────────────────────────────────────────╢
║ Total Return:       {metrics.total_return:>15,.2f} ({metrics.total_return_pct:>6.2%})         ║
║ Daily Return:       {metrics.daily_return:>22.2%}                   ║
║ Current Drawdown:   {metrics.current_drawdown:>22.2%}                   ║
║ Max Drawdown:       {metrics.max_drawdown:>22.2%}                   ║
╠══════════════════════════════════════════════════════════════╣
║ RISK-ADJUSTED METRICS                                        ║
╟──────────────────────────────────────────────────────────────╢
║ Sharpe Ratio:       {metrics.sharpe_ratio:>22.2f}                   ║
║ Sortino Ratio:      {metrics.sortino_ratio:>22.2f}                   ║
║ Calmar Ratio:       {metrics.calmar_ratio:>22.2f}                   ║
║ Realized Vol:       {metrics.volatility_realized:>22.2%}                   ║
╠══════════════════════════════════════════════════════════════╣
║ TRADING STATISTICS                                           ║
╟──────────────────────────────────────────────────────────────╢
║ Total Trades:       {metrics.total_trades:>22d}                   ║
║ Win Rate:           {metrics.win_rate:>22.2%}                   ║
║ Avg Win:            {metrics.avg_win:>22,.2f}                   ║
║ Avg Loss:           {metrics.avg_loss:>22,.2f}                   ║
║ Profit Factor:      {metrics.profit_factor:>22.2f}                   ║
╠══════════════════════════════════════════════════════════════╣
║ EXPOSURE                                                     ║
╟──────────────────────────────────────────────────────────────╢
║ Current Exposure:   {metrics.current_exposure:>22.2%}                   ║
║ Avg Exposure:       {metrics.avg_exposure:>22.2%}                   ║
║ Max Exposure:       {metrics.max_exposure:>22.2%}                   ║
╚══════════════════════════════════════════════════════════════╝
"""
        return report