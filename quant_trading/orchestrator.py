"""
Trading System Orchestrator
===========================
Main pipeline orchestrating all components:
    DATA → FEATURE ENG. → ALPHA MODELS → RISK ENGINE → EXECUTION → MONITORING

Core principles enforced:
- Quantitative models (math/statistics)
- Algorithmic execution (computers execute at scale)
- Data-driven decisions (no gut feelings)
- No emotional overrides (humans can't override once live)
- Diversified strategies (multiple quant strategies combined)
"""

import pandas as pd
import numpy as np
from datetime import datetime, time
from typing import Dict, List, Optional, Tuple
import logging
import time as time_module
import threading

from .config import SystemConfig, TradingMode
from .data import DataManager
from .features import FeatureEngine, FeatureSet
from .alpha import AlphaEngine, AlphaOutput
from .risk import RiskEngine, RiskMetrics
from .execution import ExecutionEngine, Order, OrderSide
from .monitoring import MonitoringSystem, SystemState

logger = logging.getLogger(__name__)


class TradingSystem:
    """
    Main trading system orchestrator.
    
    Coordinates the complete trading pipeline:
    1. DATA: Fetch prices, volume, OI, VIX
    2. FEATURE ENG.: Apply math transforms to create features
    3. ALPHA MODELS: Generate signals & probabilities
    4. RISK ENGINE: Apply position sizing and limits
    5. EXECUTION: Submit orders via broker API
    6. MONITORING: Track performance, drawdown, kill-switch
    """
    
    def __init__(self, config: SystemConfig = None):
        self.config = config or SystemConfig()
        
        # Initialize all components
        self.data_manager = DataManager(
            config=self.config.data,
            use_mock=(self.config.mode != TradingMode.LIVE)
        )
        
        self.feature_engine = FeatureEngine(config=self.config.features)
        
        self.alpha_engine = AlphaEngine(config=self.config.alpha)
        
        self.risk_engine = RiskEngine(
            config=self.config.risk,
            initial_capital=self.config.initial_capital
        )
        
        self.execution_engine = ExecutionEngine(
            config=self.config.execution,
            use_mock=(self.config.mode != TradingMode.LIVE)
        )
        
        self.monitoring = MonitoringSystem(
            config=self.config.monitoring,
            initial_capital=self.config.initial_capital
        )
        
        # System state
        self.running = False
        self.iteration = 0
        self._stop_event = threading.Event()
        
        # Feature cache
        self.feature_cache: Dict[str, FeatureSet] = {}
        
        # Connect kill switch to execution
        self.monitoring.on_kill_switch(self._on_kill_switch)
        
        logger.info(f"TradingSystem initialized in {self.config.mode.value} mode")
    
    def initialize(self):
        """Initialize all system components."""
        logger.info("Initializing trading system...")
        
        # Initialize data manager (load historical data)
        self.data_manager.initialize()
        
        # Initialize execution engine (connect to broker)
        self.execution_engine.initialize()
        
        # Start monitoring
        self.monitoring.start()
        
        # Compute initial features for all symbols
        for symbol in self.data_manager.get_universe():
            self._update_features(symbol)
        
        # Reset daily tracking
        self.risk_engine.reset_daily()
        
        logger.info("Trading system initialized successfully")
    
    def shutdown(self):
        """Gracefully shutdown the system."""
        logger.info("Shutting down trading system...")
        
        self.running = False
        self._stop_event.set()
        
        # Cancel all pending orders
        self.execution_engine.cancel_all_orders()
        
        # Stop monitoring
        self.monitoring.stop()
        
        # Shutdown execution
        self.execution_engine.shutdown()
        
        # Generate final report
        report = self.monitoring.generate_report()
        print(report)
        
        logger.info("Trading system shutdown complete")
    
    def run(self):
        """
        Main trading loop.
        
        Continuously runs the pipeline until stopped.
        """
        self.running = True
        self._stop_event.clear()
        
        logger.info("Starting main trading loop...")
        
        while self.running and not self._stop_event.is_set():
            try:
                self.iteration += 1
                
                # Check if we should trade
                if not self._should_trade():
                    time_module.sleep(60)
                    continue
                
                # Run one iteration of the pipeline
                self._run_iteration()
                
                # Wait for next iteration
                time_module.sleep(self.config.data.update_frequency_seconds)
                
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                self.monitoring.error_count += 1
                time_module.sleep(10)  # Brief pause before retry
        
        self.shutdown()
    
    def _run_iteration(self):
        """Run one complete iteration of the trading pipeline."""
        logger.debug(f"=== Iteration {self.iteration} ===")
        
        # 1. DATA: Update market data
        prices = self._update_data()
        
        # 2. FEATURE ENG.: Update features
        for symbol in self.data_manager.get_universe():
            self._update_features(symbol)
        
        # 3. Update prices in execution and risk
        self.execution_engine.update_market_prices(prices)
        self.risk_engine.update_prices(prices)
        
        # 4. ALPHA MODELS: Generate signals
        signals = self._generate_signals()
        
        # 5. RISK ENGINE: Check limits and size positions
        can_trade, violations = self.risk_engine.check_risk_limits()
        
        if not can_trade:
            logger.warning(f"Risk limits violated: {violations}")
            # Check if we need to reduce exposure
            should_reduce, reduction_pct = self.risk_engine.should_reduce_exposure()
            if should_reduce:
                self._reduce_exposure(reduction_pct)
            return
        
        # 6. EXECUTION: Execute trades based on signals
        for symbol, alpha_output in signals.items():
            if alpha_output.should_trade:
                self._process_signal(symbol, alpha_output, prices.get(symbol, 0))
        
        # 7. MONITORING: Update monitoring with current state
        self._update_monitoring()
    
    def _update_data(self) -> Dict[str, float]:
        """Update market data for all symbols."""
        prices = {}
        
        for symbol in self.config.data.symbols:
            try:
                data = self.data_manager.get_realtime(symbol)
                prices[symbol] = data.close
            except Exception as e:
                logger.warning(f"Failed to get data for {symbol}: {e}")
                # Use last known price
                if symbol in self.data_manager.historical_data:
                    prices[symbol] = self.data_manager.historical_data[symbol]['close'].iloc[-1]
        
        return prices
    
    def _update_features(self, symbol: str):
        """Update features for a symbol."""
        try:
            df = self.data_manager.get_combined_data(symbol)
            feature_set = self.feature_engine.compute_features(df, symbol)
            self.feature_cache[symbol] = feature_set
        except Exception as e:
            logger.error(f"Feature computation failed for {symbol}: {e}")
    
    def _generate_signals(self) -> Dict[str, AlphaOutput]:
        """Generate trading signals for all symbols."""
        signals = {}
        
        for symbol, feature_set in self.feature_cache.items():
            try:
                # Detect market regime and adjust weights
                regime = self.alpha_engine.get_regime(feature_set.features)
                self.alpha_engine.adjust_weights_for_regime(regime)
                
                # Generate signal
                alpha_output = self.alpha_engine.generate_signals(
                    feature_set.features, 
                    symbol
                )
                signals[symbol] = alpha_output
                
                logger.debug(
                    f"{symbol} signal: {alpha_output.combined_signal:.2f} "
                    f"(conf: {alpha_output.combined_confidence:.2f}, "
                    f"suggest: {alpha_output.position_suggestion})"
                )
                
            except Exception as e:
                logger.error(f"Signal generation failed for {symbol}: {e}")
        
        return signals
    
    def _process_signal(self, symbol: str, alpha: AlphaOutput, current_price: float):
        """Process a trading signal and execute if appropriate."""
        # Get feature data for volatility
        volatility = 0.20  # Default
        if symbol in self.feature_cache:
            features = self.feature_cache[symbol].features
            if 'volatility' in features.columns:
                volatility = features['volatility'].iloc[-1]
        
        # Check current position
        current_pos = self.risk_engine.positions.get(symbol)
        
        # Determine action
        if alpha.position_suggestion == 'long':
            if current_pos and current_pos.side == 'long':
                logger.debug(f"Already long {symbol}, skipping")
                return
            
            # Close short position if exists
            if current_pos and current_pos.side == 'short':
                self._close_position(symbol, current_price)
            
            # Calculate position size
            sizing = self.risk_engine.calculate_position_size(
                symbol=symbol,
                signal_strength=alpha.combined_signal,
                price=current_price,
                volatility=volatility
            )
            
            if sizing.quantity > 0:
                self._execute_trade(symbol, 'buy', sizing.quantity, current_price)
        
        elif alpha.position_suggestion == 'short':
            if current_pos and current_pos.side == 'short':
                logger.debug(f"Already short {symbol}, skipping")
                return
            
            # Close long position if exists
            if current_pos and current_pos.side == 'long':
                self._close_position(symbol, current_price)
            
            # Calculate position size for short
            sizing = self.risk_engine.calculate_position_size(
                symbol=symbol,
                signal_strength=alpha.combined_signal,
                price=current_price,
                volatility=volatility
            )
            
            if sizing.quantity > 0:
                self._execute_trade(symbol, 'sell', sizing.quantity, current_price)
        
        elif alpha.position_suggestion == 'flat':
            # Close any existing position
            if current_pos:
                self._close_position(symbol, current_price)
    
    def _execute_trade(self, symbol: str, side: str, quantity: int, price: float):
        """Execute a trade."""
        # Create order
        order = self.execution_engine.create_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=self.config.execution.default_order_type,
            price=price if self.config.execution.default_order_type == 'LIMIT' else None
        )
        
        # Execute
        success, message = self.execution_engine.execute_order(order)
        
        if success and order.filled_quantity > 0:
            # Update risk engine
            self.risk_engine.update_position(
                symbol=symbol,
                quantity=order.filled_quantity,
                price=order.filled_price,
                side=side
            )
            
            logger.info(
                f"Trade executed: {side.upper()} {order.filled_quantity} {symbol} "
                f"@ {order.filled_price:.2f}"
            )
        else:
            logger.warning(f"Trade failed: {message}")
    
    def _close_position(self, symbol: str, current_price: float):
        """Close an existing position."""
        pos = self.risk_engine.positions.get(symbol)
        if not pos:
            return
        
        # Determine closing side
        close_side = 'sell' if pos.side == 'long' else 'buy'
        
        # Create closing order
        order = self.execution_engine.create_order(
            symbol=symbol,
            side=close_side,
            quantity=pos.quantity,
            order_type='MARKET'
        )
        
        success, message = self.execution_engine.execute_order(order)
        
        if success:
            # Calculate P&L
            pnl = pos.unrealized_pnl
            
            # Update risk engine
            self.risk_engine.update_position(
                symbol=symbol,
                quantity=pos.quantity,
                price=order.filled_price,
                side=close_side,
                is_close=True
            )
            
            # Record trade in monitoring
            self.monitoring.record_trade(
                symbol=symbol,
                side=pos.side,
                quantity=pos.quantity,
                entry_price=pos.entry_price,
                exit_price=order.filled_price,
                pnl=pnl
            )
            
            logger.info(f"Closed {symbol} position. P&L: {pnl:,.2f}")
    
    def _reduce_exposure(self, reduction_pct: float):
        """Reduce portfolio exposure by closing positions."""
        logger.warning(f"Reducing exposure by {reduction_pct:.0%}")
        
        # Sort positions by size (close largest first)
        positions = sorted(
            self.risk_engine.positions.items(),
            key=lambda x: x[1].market_value,
            reverse=True
        )
        
        total_exposure = sum(p.market_value for _, p in positions)
        target_reduction = total_exposure * reduction_pct
        reduced = 0
        
        for symbol, pos in positions:
            if reduced >= target_reduction:
                break
            
            price = pos.current_price
            self._close_position(symbol, price)
            reduced += pos.market_value
    
    def _update_monitoring(self):
        """Update monitoring with current portfolio state."""
        metrics = self.risk_engine.get_risk_metrics()
        
        # Calculate daily P&L percentage
        daily_pnl = metrics.daily_pnl
        daily_pnl_pct = daily_pnl / self.risk_engine.daily_starting_equity \
            if self.risk_engine.daily_starting_equity > 0 else 0
        
        # Update monitoring
        self.monitoring.update(
            equity=metrics.total_exposure + metrics.cash,
            exposure=metrics.exposure_pct,
            drawdown=metrics.current_drawdown,
            daily_pnl_pct=daily_pnl_pct
        )
    
    def _should_trade(self) -> bool:
        """Check if we should be trading right now."""
        # Check system state
        if self.monitoring.state == SystemState.KILLED:
            return False
        
        if self.monitoring.state == SystemState.PAUSED:
            return False
        
        # Check trading mode
        if self.config.mode == TradingMode.BACKTEST:
            return True  # Backtesting runs as fast as possible
        
        # Check trading hours
        now = datetime.now().time()
        market_open = time(*map(int, self.config.execution.market_open.split(':')))
        market_close = time(*map(int, self.config.execution.market_close.split(':')))
        
        if not (market_open <= now <= market_close):
            return False
        
        # Check if we're in the no-trade zone at end of day
        close_time = datetime.now().replace(
            hour=market_close.hour,
            minute=market_close.minute
        )
        time_to_close = (close_time - datetime.now()).total_seconds() / 60
        
        if time_to_close < self.config.execution.no_trade_last_minutes:
            return False
        
        return True
    
    def _on_kill_switch(self):
        """Handle kill switch trigger."""
        logger.critical("Kill switch triggered - stopping all trading")
        
        # Disable trading in execution engine
        self.execution_engine.disable_trading("Kill switch triggered")
        
        # Cancel all pending orders
        self.execution_engine.cancel_all_orders()
        
        # Stop the main loop
        self.running = False
    
    def get_status(self) -> Dict:
        """Get comprehensive system status."""
        risk_metrics = self.risk_engine.get_risk_metrics()
        monitoring_status = self.monitoring.get_status()
        
        return {
            'mode': self.config.mode.value,
            'running': self.running,
            'iteration': self.iteration,
            'capital': self.config.initial_capital,
            'current_equity': risk_metrics.total_exposure + risk_metrics.cash,
            'positions_count': risk_metrics.positions_count,
            'exposure_pct': risk_metrics.exposure_pct,
            'unrealized_pnl': risk_metrics.unrealized_pnl,
            'realized_pnl': risk_metrics.realized_pnl,
            'drawdown': risk_metrics.current_drawdown,
            'risk_level': risk_metrics.risk_level.value,
            'kill_switch': monitoring_status['kill_switch_triggered'],
            **monitoring_status
        }
    
    def run_backtest(self, start_date: str, end_date: str) -> Dict:
        """
        Run a backtest over historical data.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Backtest results dictionary
        """
        logger.info(f"Running backtest from {start_date} to {end_date}")
        
        # Set mode to backtest
        self.config.mode = TradingMode.BACKTEST
        
        # Initialize with historical data
        self.initialize()
        
        # Get all symbols' data
        all_dates = set()
        for symbol in self.data_manager.get_universe():
            if symbol in self.data_manager.historical_data:
                dates = self.data_manager.historical_data[symbol].index
                all_dates.update(dates)
        
        # Sort dates and filter to range
        all_dates = sorted([d for d in all_dates 
                          if start_date <= str(d.date()) <= end_date])
        
        # Run simulation day by day
        for date in all_dates:
            # Simulate this day
            prices = {}
            for symbol in self.data_manager.get_universe():
                if symbol in self.data_manager.historical_data:
                    df = self.data_manager.historical_data[symbol]
                    if date in df.index:
                        prices[symbol] = df.loc[date, 'close']
            
            if prices:
                self.execution_engine.update_market_prices(prices)
                self.risk_engine.update_prices(prices)
                
                # Update features and generate signals
                for symbol in prices.keys():
                    self._update_features(symbol)
                
                signals = self._generate_signals()
                
                # Execute trades
                can_trade, _ = self.risk_engine.check_risk_limits()
                if can_trade:
                    for symbol, alpha in signals.items():
                        if alpha.should_trade and symbol in prices:
                            self._process_signal(symbol, alpha, prices[symbol])
                
                self._update_monitoring()
        
        # Get final results
        metrics = self.monitoring.get_performance_metrics()
        
        return {
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': self.config.initial_capital,
            'final_equity': self.risk_engine.current_capital,
            'total_return_pct': metrics.total_return_pct,
            'sharpe_ratio': metrics.sharpe_ratio,
            'sortino_ratio': metrics.sortino_ratio,
            'max_drawdown': metrics.max_drawdown,
            'win_rate': metrics.win_rate,
            'profit_factor': metrics.profit_factor,
            'total_trades': metrics.total_trades,
            'avg_exposure': metrics.avg_exposure
        }


def main():
    """Main entry point for the trading system."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Quantitative Trading System')
    parser.add_argument('--mode', choices=['live', 'paper', 'backtest'], 
                        default='paper', help='Trading mode')
    parser.add_argument('--capital', type=float, default=1000000,
                        help='Initial capital')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--start-date', type=str, help='Backtest start date')
    parser.add_argument('--end-date', type=str, help='Backtest end date')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create configuration
    config = SystemConfig()
    config.mode = TradingMode(args.mode)
    config.initial_capital = args.capital
    
    if args.config:
        config = SystemConfig.load(args.config)
    
    # Create and run system
    system = TradingSystem(config)
    
    if args.mode == 'backtest':
        if not args.start_date or not args.end_date:
            print("Backtest requires --start-date and --end-date")
            return
        
        results = system.run_backtest(args.start_date, args.end_date)
        
        print("\n" + "="*50)
        print("BACKTEST RESULTS")
        print("="*50)
        for key, value in results.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
    else:
        try:
            system.initialize()
            system.run()
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            system.shutdown()


if __name__ == "__main__":
    main()
