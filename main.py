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
            self._update_header()
            self._update_alerts()
            self._update_stats()
            self.screen.refresh()
        except curses.error:
            pass  # Handle resize events gracefully

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
        
        # Add section title
        self.alerts_window.addstr(1, 2, "Active Alerts")
        
        # Sort alerts by timestamp (most recent first)
        alerts = sorted(self.current_alerts, 
                      key=lambda x: x['timestamp'],
                      reverse=True)
        
        # Display alerts
        row = 2
        for alert in alerts[:20]:  # Show last 20 alerts
            if row >= self.alerts_window.getmaxyx()[0]-2:
                break
                
            try:
                tx = alert['transaction']
                timestamp = datetime.fromisoformat(tx['timestamp']).strftime('%H:%M:%S')
                pattern_type = alert['pattern_type']
                
                alert_text = (
                    f"[{timestamp}] {pattern_type:<20} ${tx['amount']:>9,.2f}\n"
                    f"    From: {tx['sender']:<12} To: {tx['receiver']}"
                )
                
                self.alerts_window.addstr(row, 4, alert_text)
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
        
        while True:
            try:
                new_transactions = detector._check_new_transactions()
                if new_transactions:
                    # Process each transaction
                    for tx in new_transactions:
                        patterns = detector._analyze_single_transaction(tx)
                        if patterns:
                            for pattern_type, pattern_info in patterns.items():
                                if pattern_info['risk_score'] > detector.risk_threshold:
                                    alert = {
                                        'id': f"ALT{len(ui.current_alerts):06d}",
                                        'timestamp': datetime.now().isoformat(),
                                        'pattern_type': pattern_type,
                                        'risk_score': pattern_info['risk_score'],
                                        'transaction': tx,
                                        'details': pattern_info['details']
                                    }
                                    ui.current_alerts.append(alert)
                                    
                                    if pattern_info['risk_score'] > 0.8:
                                        ui.high_priority_count += 1
                                    else:
                                        ui.medium_priority_count += 1
                
                # Update screen
                ui.last_update = datetime.now()
                ui.refresh_screen()
                time.sleep(check_interval)
                
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
    
    parser.add_argument('--file', required=True,
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
