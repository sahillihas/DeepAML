#!/usr/bin/env python3
"""
DeepAML Police Dashboard
------------------------
Real-time monitoring interface for law enforcement
"""

import argparse
import curses
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from aml_detector import AMLDetector
from models import PatternType

logger = logging.getLogger("DeepAML.Dashboard")

class DashboardUI:
    def __init__(self, detector: AMLDetector, screen):
        self.detector = detector
        self.screen = screen
        self.current_alerts = []
        self.high_priority_count = 0
        self.medium_priority_count = 0
        self.last_update = datetime.now()
        self.last_file_mtime = os.path.getmtime(detector.monitor_file) if os.path.exists(detector.monitor_file) else 0
        self.max_alerts = 20  # Maximum alerts to display
        self.processed_tx_ids = set()  # Track processed transactions
        
    def check_file_changed(self) -> bool:
        """Check if monitored file has been modified"""
        try:
            current_mtime = os.path.getmtime(self.detector.monitor_file)
            if current_mtime > self.last_file_mtime:
                self.last_file_mtime = current_mtime
                # Reload transactions when file changes
                self.detector.load_transactions(self.detector.monitor_file)
                return True
            return False
        except OSError:
            return False
            
    def setup_windows(self):
        """Initialize UI windows"""
        screen_height, screen_width = self.screen.getmaxyx()
        
        # Create windows for different sections
        self.header_window = curses.newwin(3, screen_width, 0, 0)
        self.alerts_window = curses.newwin(screen_height-6, screen_width, 3, 0)
        self.stats_window = curses.newwin(3, screen_width, screen_height-3, 0)
        
        # Enable scrolling for alerts window
        self.alerts_window.scrollok(True)
        self.alerts_window.idlok(True)
        
    def refresh_screen(self):
        """Update all UI elements"""
        try:
            # Check for file changes first
            if self.check_file_changed():
                self._process_new_transactions()
            
            self._update_header()
            self._update_alerts()
            self._update_stats()
            self.screen.refresh()
        except curses.error:
            pass  # Handle resize events gracefully
            
    def _process_new_transactions(self):
        """Process any new transactions after file change"""
        if not self.detector.transactions:
            return
            
        # Get new transactions
        new_txs = [tx for tx in self.detector.transactions 
                   if tx.get('transaction_id') not in self.processed_tx_ids]
                   
        for tx in new_txs:
            self.processed_tx_ids.add(tx.get('transaction_id'))
            
            # Handle simulator-injected pattern types
            if 'pattern_type' in tx:
                pattern_type = tx['pattern_type'].upper().replace('-', '_')
                alert = {
                    'id': f"ALT{len(self.current_alerts):06d}",
                    'timestamp': tx['timestamp'],
                    'pattern_type': pattern_type,
                    'risk_score': tx.get('risk_score', 0.95),
                    'transaction': tx,
                    'details': f"Detected {pattern_type} pattern"
                }
                self.current_alerts.append(alert)
                self.high_priority_count += 1
                continue
                
            # Process other transaction patterns
            patterns = self.detector._analyze_single_transaction(tx)
            if patterns:
                for pattern_type, pattern_info in patterns.items():
                    if pattern_info['risk_score'] > self.detector.risk_threshold:
                        alert = {
                            'id': f"ALT{len(self.current_alerts):06d}",
                            'timestamp': datetime.now().isoformat(),
                            'pattern_type': pattern_type.upper(),
                            'risk_score': pattern_info['risk_score'],
                            'transaction': tx,
                            'details': pattern_info['details']
                        }
                        self.current_alerts.append(alert)
                        if pattern_info['risk_score'] > 0.8:
                            self.high_priority_count += 1
                        else:
                            self.medium_priority_count += 1
                            
        # Keep only the most recent alerts
        self.current_alerts = self.current_alerts[-1000:]  # Keep last 1000 alerts
                
    def _update_header(self):
        """Update header window"""
        self.header_window.clear()
        self.header_window.box()
        title = "🚔 DeepAML Police Dashboard 👮"
        self.header_window.addstr(1, 2, title)
        self.header_window.refresh()

    def _update_alerts(self):
        """Update alerts window with latest alerts"""
        self.alerts_window.clear()
        self.alerts_window.box()
        
        # Add section title with stats
        title = f"Active Alerts ({len(self.current_alerts)} total)"
        self.alerts_window.addstr(1, 2, title)
        
        # Sort alerts by timestamp (most recent first)
        sorted_alerts = sorted(
            self.current_alerts[-self.max_alerts:],  # Keep only most recent alerts
            key=lambda x: datetime.fromisoformat(x['timestamp']),
            reverse=True
        )
        
        # Display alerts
        row = 2
        max_height = self.alerts_window.getmaxyx()[0] - 3
        
        for alert in sorted_alerts:
            if row >= max_height:
                break
                
            try:
                tx = alert['transaction']
                timestamp = datetime.fromisoformat(tx['timestamp']).strftime('%H:%M:%S')
                pattern_type = alert['pattern_type']
                risk_score = alert.get('risk_score', 0.0)
                
                # Format alert based on pattern type
                if pattern_type in ['FAN_IN', 'FAN_OUT']:
                    details = f"${tx['amount']:,.2f} ({pattern_type.replace('_', '-')})"
                    participants = "from multiple sources" if pattern_type == 'FAN_IN' else "to multiple destinations"
                    alert_text = (
                        f"[{timestamp}] {details}\n"
                        f"    {participants} → {tx['receiver'] if pattern_type == 'FAN_IN' else tx['sender']}"
                    )
                elif pattern_type == 'ROUND_TRIP':
                    alert_text = (
                        f"[{timestamp}] ROUND-TRIP ${tx['amount']:,.2f}\n"
                        f"    {tx['sender']} → ... → {tx['receiver']}"
                    )
                elif pattern_type == 'RAPID_MOVEMENT':
                    alert_text = (
                        f"[{timestamp}] RAPID-MOVE ${tx['amount']:,.2f}\n"
                        f"    Quick transfers: {tx['sender']} → {tx['receiver']}"
                    )
                else:
                    alert_text = (
                        f"[{timestamp}] {pattern_type} ${tx['amount']:,.2f}\n"
                        f"    From: {tx['sender']} → {tx['receiver']}"
                    )
                
                # Add risk indicator
                risk_indicator = "‼️ " if risk_score > 0.9 else "⚠️ "
                self.alerts_window.addstr(row, 4, risk_indicator + alert_text)
                row += 2
                
            except (curses.error, KeyError) as e:
                logger.error(f"Error displaying alert: {str(e)}")
                continue
                
        self.alerts_window.refresh()

    def _update_stats(self):
        """Update statistics window"""
        self.stats_window.clear()
        self.stats_window.box()
        
        stats_text = (
            f" Alerts: {len(self.current_alerts)} "
            f"(High: {self.high_priority_count} "
            f"Medium: {self.medium_priority_count}) | "
            f"Last Update: {self.last_update.strftime('%H:%M:%S')}"
        )
        
        self.stats_window.addstr(1, 2, stats_text)
        self.stats_window.refresh()

def run_dashboard(file_path: str, risk_threshold: float = 0.7, check_interval: int = 5):
    """Run the police dashboard"""
    def dashboard_thread(stdscr, detector: AMLDetector, ui: DashboardUI):
        # Initialize UI
        stdscr.clear()
        ui.setup_windows()
        
        # Initialize detector with file
        detector.monitor_file = file_path
        detector.risk_threshold = risk_threshold
        detector.load_transactions(file_path)  # Load existing transactions
        logger.info(f"Initial transactions loaded: {len(detector.transactions)}")
        
        # Keep track of processed transaction IDs per pattern type to avoid duplicates
        processed_tx_ids = {}  # pattern_type -> set(tx_ids)
        
        while True:
            try:
                # Check for new transactions
                new_transactions = detector._check_new_transactions()
                if new_transactions:
                    logger.info(f"Found {len(new_transactions)} new transactions")
                    for tx in new_transactions:
                        # Handle simulator-injected pattern types directly
                        if 'pattern_type' in tx:
                            pattern_type = tx['pattern_type'].upper().replace('-', '_')
                            # Skip if we've seen this transaction for this pattern
                            if pattern_type not in processed_tx_ids:
                                processed_tx_ids[pattern_type] = set()
                            if tx.get('transaction_id') in processed_tx_ids[pattern_type]:
                                continue
                                
                            processed_tx_ids[pattern_type].add(tx.get('transaction_id'))
                            logger.info(f"Processing injected pattern: {pattern_type}")
                            
                            alert = {
                                'id': f"ALT{len(ui.current_alerts):06d}",
                                'timestamp': tx['timestamp'],
                                'pattern_type': pattern_type,
                                'risk_score': tx.get('risk_score', 0.95),
                                'transaction': tx,
                                'details': f"Detected {pattern_type} pattern"
                            }
                            ui.current_alerts.append(alert)
                            ui.high_priority_count += 1
                            logger.info(f"Created alert for {pattern_type}")
                            continue
                        
                        # Process normal transaction patterns
                        patterns = detector._analyze_single_transaction(tx)
                        if patterns:
                            logger.info(f"Found patterns in transaction: {list(patterns.keys())}")
                            for pattern_type, pattern_info in patterns.items():
                                if pattern_info['risk_score'] > detector.risk_threshold:
                                    # Skip if we've seen this transaction for this pattern
                                    if pattern_type not in processed_tx_ids:
                                        processed_tx_ids[pattern_type] = set()
                                    if tx.get('transaction_id') in processed_tx_ids[pattern_type]:
                                        continue
                                        
                                    processed_tx_ids[pattern_type].add(tx.get('transaction_id'))
                                    
                                    alert = {
                                        'id': f"ALT{len(ui.current_alerts):06d}",
                                        'timestamp': datetime.now().isoformat(),
                                        'pattern_type': pattern_type.upper(),
                                        'risk_score': pattern_info['risk_score'],
                                        'transaction': tx,
                                        'details': pattern_info['details']
                                    }
                                    ui.current_alerts.append(alert)
                                    if pattern_info['risk_score'] > 0.8:
                                        ui.high_priority_count += 1
                                    else:
                                        ui.medium_priority_count += 1
                
                # Run periodic full pattern detection
                if detector.last_check_time is None or \
                   (datetime.now() - detector.last_check_time).seconds >= check_interval:
                    logger.info("Running periodic pattern detection...")
                    # Reload transactions to catch any new ones
                    detector.load_transactions(file_path)
                    patterns = detector.detect_patterns()
                    if patterns:
                        logger.info(f"Detected patterns: {list(patterns.keys())}")
                        for pattern_type, pattern_list in patterns.items():
                            pattern_type = pattern_type.upper().replace('-', '_')
                            for pattern in pattern_list:
                                if pattern.get('risk_score', 0) > detector.risk_threshold:
                                    # Get the last transaction in the pattern chain
                                    tx = pattern.get('transactions', [])[-1] if 'transactions' in pattern else \
                                         pattern.get('transaction', None)
                                    
                                    if tx:
                                        # Skip if we've seen this transaction for this pattern type
                                        if pattern_type not in processed_tx_ids:
                                            processed_tx_ids[pattern_type] = set()
                                        if tx.get('transaction_id') in processed_tx_ids[pattern_type]:
                                            continue
                                            
                                        processed_tx_ids[pattern_type].add(tx.get('transaction_id'))
                                        alert = {
                                            'id': f"ALT{len(ui.current_alerts):06d}",
                                            'timestamp': tx.get('timestamp', datetime.now().isoformat()),
                                            'pattern_type': pattern_type,
                                            'risk_score': pattern['risk_score'],
                                            'transaction': tx,
                                            'details': pattern.get('details', f"Complex {pattern_type} pattern detected")
                                        }
                                        ui.current_alerts.append(alert)
                                        if pattern['risk_score'] > 0.8:
                                            ui.high_priority_count += 1
                                        else:
                                            ui.medium_priority_count += 1
                                        logger.info(f"Created alert for complex {pattern_type}")
                    detector.last_check_time = datetime.now()

                # Limit alerts to most recent ones
                ui.current_alerts = ui.current_alerts[-1000:]  # Keep last 1000 alerts
                
                # Update screen
                ui.last_update = datetime.now()
                ui.refresh_screen()
                time.sleep(1)  # Check every second
                
            except KeyboardInterrupt:
                break
                
            except Exception as e:
                logger.error(f"Dashboard error: {str(e)}")
                time.sleep(check_interval)  # Wait before retrying
    
    # Initialize detector
    detector = AMLDetector(risk_threshold=risk_threshold)
    detector.monitor_file = file_path
    
    # Start curses application
    curses.wrapper(lambda stdscr: 
        dashboard_thread(
            stdscr,
            detector,
            DashboardUI(detector, stdscr)
        )
    )

def setup_logging(level: str = "INFO"):
    """Set up logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        filename="dashboard.log"
    )

def main():
    parser = argparse.ArgumentParser(description="DeepAML Police Dashboard")
    
    parser.add_argument('--file', default= 'data/raw/transactions.json',
                       help='Transaction file to monitor')
                       
    parser.add_argument('--risk-threshold', type=float, default=0.7,
                       help='Risk threshold for alerting (0.0-1.0)')
                       
    parser.add_argument('--check-interval', type=int, default=5,
                       help='Interval in seconds between checks')
                       
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       help="Logging level")
    
    args = parser.parse_args()
    setup_logging(args.log_level)
    
    try:
        run_dashboard(args.file, args.risk_threshold, args.check_interval)
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error running dashboard: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
