#!/usr/bin/env python3
"""
DeepAML Police Dashboard
------------------------
Real-time monitoring interface for law enforcement
"""

import argparse
import curses
import logging
import os
import sys
import time
from datetime import datetime
from typing import List

from aml_detector import AMLDetector
from models import PatternType

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DeepAML.Dashboard")

class DashboardUI:
    def __init__(self, detector: AMLDetector, screen):
        self.detector = detector
        self.screen = screen
        self.current_alerts: List[dict] = []
        self.high_priority_count = 0
        self.medium_priority_count = 0
        self.last_update = datetime.now()
        self.last_file_mtime = self._get_file_mtime()
        self.max_alerts = 20
        self.processed_tx_ids = set()
        self.setup_windows()

    def _get_file_mtime(self) -> float:
        try:
            return os.path.getmtime(self.detector.monitor_file)
        except OSError:
            logger.warning("Monitored file not found.")
            return 0

    def check_file_changed(self) -> bool:
        current_mtime = self._get_file_mtime()
        if current_mtime > self.last_file_mtime:
            self.last_file_mtime = current_mtime
            self.detector.load_transactions(self.detector.monitor_file)
            return True
        return False

    def setup_windows(self):
        h, w = self.screen.getmaxyx()
        self.header_window = curses.newwin(3, w, 0, 0)
        self.alerts_window = curses.newwin(h - 6, w, 3, 0)
        self.stats_window = curses.newwin(3, w, h - 3, 0)
        self.alerts_window.scrollok(True)
        self.alerts_window.idlok(True)

    def refresh_screen(self):
        try:
            if self.check_file_changed():
                self._process_new_transactions()
            self._update_header()
            self._update_alerts()
            self._update_stats()
            self.screen.refresh()
        except curses.error as e:
            logger.warning(f"Curses error during refresh: {e}")

    def _process_new_transactions(self):
        new_txs = [
            tx for tx in self.detector.transactions
            if tx.get('transaction_id') not in self.processed_tx_ids
        ]

        for tx in new_txs:
            self.processed_tx_ids.add(tx.get('transaction_id'))

            pattern_type = tx.get('pattern_type')
            if pattern_type:
                self._add_alert(
                    pattern_type=pattern_type,
                    tx=tx,
                    risk_score=tx.get('risk_score', 0.95),
                    injected=True
                )
                continue

            patterns = self.detector._analyze_single_transaction(tx)
            if not patterns:
                continue

            for pt, info in patterns.items():
                if info['risk_score'] > self.detector.risk_threshold:
                    self._add_alert(
                        pattern_type=pt,
                        tx=tx,
                        risk_score=info['risk_score'],
                        details=info.get('details')
                    )

        self.current_alerts = self.current_alerts[-1000:]
        self.last_update = datetime.now()

    def _add_alert(self, pattern_type, tx, risk_score, details=None, injected=False):
        pattern_type = str(pattern_type).upper().replace('-', '_')
        alert_id = f"ALT{len(self.current_alerts):06d}"
        timestamp = tx.get('timestamp') or datetime.now().isoformat()
        details = details or f"Detected {pattern_type} pattern"

        alert = {
            'id': alert_id,
            'timestamp': timestamp,
            'pattern_type': pattern_type,
            'risk_score': risk_score,
            'transaction': tx,
            'details': details
        }

        self.current_alerts.append(alert)
        if risk_score > 0.8:
            self.high_priority_count += 1
        else:
            self.medium_priority_count += 1

    def _update_header(self):
        self.header_window.clear()
        self.header_window.box()
        self.header_window.addstr(1, 2, "🚔 DeepAML Police Dashboard 👮")
        self.header_window.refresh()

    def _update_alerts(self):
        self.alerts_window.clear()
        self.alerts_window.box()
        self.alerts_window.addstr(1, 2, f"Active Alerts ({len(self.current_alerts)} total)")

        max_display = self.alerts_window.getmaxyx()[0] - 3
        alerts_to_show = sorted(
            self.current_alerts[-self.max_alerts:],
            key=lambda a: datetime.fromisoformat(a['timestamp']),
            reverse=True
        )

        row = 2
        for alert in alerts_to_show:
            if row >= max_display:
                break
            try:
                self._draw_alert(alert, row)
                row += 2
            except (curses.error, KeyError) as e:
                logger.warning(f"Failed to draw alert: {e}")

        self.alerts_window.refresh()

    def _draw_alert(self, alert: dict, row: int):
        tx = alert['transaction']
        pattern = alert['pattern_type']
        score = alert.get('risk_score', 0.0)

        try:
            ts_raw = tx.get('timestamp') or alert['timestamp']
            ts = datetime.fromisoformat(ts_raw).strftime('%H:%M:%S')
        except Exception:
            ts = "??:??:??"

        sender = tx.get('sender', '?')
        receiver = tx.get('receiver', '?')
        amount = tx.get('amount', 0.0)

        base = f"[{ts}] {pattern.replace('_', '-')} ${amount:,.2f}"
        prefix = "‼️ " if score > 0.9 else "⚠️ "

        if pattern == 'FAN_IN':
            extra = f"\n    from multiple sources → {receiver}"
        elif pattern == 'FAN_OUT':
            extra = f"\n    to multiple destinations ← {sender}"
        elif pattern == 'ROUND_TRIP':
            extra = f"\n    {sender} → ... → {receiver}"
        elif pattern == 'RAPID_MOVEMENT':
            extra = f"\n    Quick transfers: {sender} → {receiver}"
        else:
            extra = f"\n    From: {sender} → {receiver}"

        self.alerts_window.addstr(row, 4, prefix + base + extra)

    def _update_stats(self):
        self.stats_window.clear()
        self.stats_window.box()
        self.stats_window.addstr(
            1, 2,
            f" Alerts: {len(self.current_alerts)} "
            f"(High: {self.high_priority_count} Medium: {self.medium_priority_count}) | "
            f"Last Update: {self.last_update.strftime('%H:%M:%S')}"
        )
        self.stats_window.refresh()

    def run_loop(self):
        """Main loop to refresh dashboard periodically."""
        while True:
            self.refresh_screen()
            time.sleep(2)

def main():
    parser = argparse.ArgumentParser(description="DeepAML Real-time Dashboard")
    parser.add_argument("--monitor_file", required=True, help="Path to transaction log file")
    args = parser.parse_args()

    detector = AMLDetector()
    detector.monitor_file = args.monitor_file
    detector.load_transactions(args.monitor_file)

    curses.wrapper(lambda stdscr: DashboardUI(detector, stdscr).run_loop())

if __name__ == "__main__":
    main()
