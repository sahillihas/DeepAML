from typing import List, Dict
from datetime import datetime, timedelta
from models import Transaction


class FeatureExtractor:
    def __init__(self):
        # Risk scores based on FATF, tax haven status, etc.
        self.country_risk_map = {
            # High-risk countries
            'IR': 1.0, 'KP': 1.0, 'SY': 1.0, 'MM': 1.0, 'PA': 1.0, 'BZ': 1.0, 'VG': 1.0,
            'AF': 1.0, 'CU': 1.0, 'CD': 1.0, 'SD': 1.0, 'YE': 1.0, 'ZW': 1.0, 'SO': 1.0,
            'AE': 1.0, 'MT': 1.0, 'CY': 1.0, 'LU': 1.0, 'KY': 1.0, 'LC': 1.0, 'AD': 1.0,
            # Medium-risk countries
            'CH': 0.6, 'SG': 0.6, 'HK': 0.6, 'MO': 0.6, 'BS': 0.6, 'AI': 0.6, 'MU': 0.6,
            'DO': 0.6, 'TR': 0.6, 'JO': 0.6, 'QA': 0.6, 'RU': 0.6, 'UA': 0.6, 'KZ': 0.6,
            'LB': 0.6, 'EG': 0.6, 'PK': 0.6, 'TN': 0.6
            # Others default to low risk
        }

    def extract_features(self, transaction: Transaction, history: List[Transaction]) -> Dict:
        """Extracts all features for a transaction based on historical context."""
        return {
            'frequency': self._frequency(transaction, history),
            'velocity': self._velocity(transaction, history),
            'amount_variance': self._amount_variance(transaction, history),
            'country_risk': self._country_risk(transaction),
            'cross_border': self._is_cross_border(transaction),
            'structuring_risk': self._structuring_risk(transaction, history),
            'network_risk': self._network_risk(transaction, history),
            'time_risk': self._time_risk(transaction),
            'burst_risk': self._burst_risk(transaction, history),
            'amount_risk': self._amount_risk(transaction),
            'round_number': self._is_round_number(transaction.amount),
        }

    # --- Country & Cross-border Features ---

    def _country_risk(self, transaction: Transaction) -> float:
        """Returns the highest risk score between sender and receiver country."""
        return max(
            self._country_score(transaction.sender_country),
            self._country_score(transaction.receiver_country)
        )

    def _country_score(self, country_code: str) -> float:
        """Gets risk score for a given country."""
        return self.country_risk_map.get(country_code.upper(), 0.2)

    def _is_cross_border(self, transaction: Transaction) -> bool:
        """Checks if transaction is international."""
        return transaction.sender_country != transaction.receiver_country

    # --- Frequency, Velocity, and Burst Features ---

    def _frequency(self, transaction: Transaction, history: List[Transaction]) -> float:
        """Average daily transaction count over past 30 days."""
        start_time = self._parse_time(transaction.timestamp) - timedelta(days=30)
        count = sum(1 for tx in history if tx.sender == transaction.sender and self._parse_time(tx.timestamp) >= start_time)
        return count / 30

    def _velocity(self, transaction: Transaction, history: List[Transaction]) -> float:
        """Average daily transaction volume over past 30 days."""
        start_time = self._parse_time(transaction.timestamp) - timedelta(days=30)
        relevant = [tx.amount for tx in history if tx.sender == transaction.sender and self._parse_time(tx.timestamp) >= start_time]
        total = sum(relevant) + transaction.amount
        return total / 30 if relevant else 0.0

    def _burst_risk(self, transaction: Transaction, history: List[Transaction]) -> float:
        """Detects sudden activity spikes within 24 hours."""
        start_time = self._parse_time(transaction.timestamp) - timedelta(hours=24)
        recent = [tx for tx in history if tx.sender == transaction.sender and self._parse_time(tx.timestamp) >= start_time]
        return min(len(recent) / 24, 1.0)

    # --- Risk Score Features ---

    def _structuring_risk(self, transaction: Transaction, history: List[Transaction]) -> float:
        """Flags attempts to avoid reporting thresholds via repeated smaller transactions."""
        window = self._parse_time(transaction.timestamp) - timedelta(hours=48)
        threshold = 10000
        relevant = [
            tx for tx in history
            if tx.sender == transaction.sender and tx.receiver == transaction.receiver
            and self._parse_time(tx.timestamp) >= window
        ]
        near_threshold_count = sum(1 for tx in relevant if threshold * 0.8 <= tx.amount < threshold)
        return 0.8 if near_threshold_count >= 2 else 0.4 if near_threshold_count == 1 else 0.0

    def _network_risk(self, transaction: Transaction, history: List[Transaction]) -> float:
        """Measures sender/receiver's exposure to large or complex networks."""
        sender_partners = {tx.receiver for tx in history if tx.sender == transaction.sender}
        receiver_partners = {tx.sender for tx in history if tx.receiver == transaction.receiver}
        return max(min(len(sender_partners) / 20, 1.0), min(len(receiver_partners) / 20, 1.0))

    def _time_risk(self, transaction: Transaction) -> float:
        """Flags transactions made at unusual hours (before 6am or after 6pm)."""
        hour = self._parse_time(transaction.timestamp).hour
        return 0.8 if hour < 6 or hour > 18 else 0.2

    def _amount_risk(self, transaction: Transaction) -> float:
        """Assigns risk based on absolute transaction amount."""
        if transaction.amount >= 50000:
            return 1.0
        elif transaction.amount >= 10000:
            return 0.6
        return 0.2

    def _amount_variance(self, transaction: Transaction, history: List[Transaction]) -> float:
        """Calculates variance in transaction amounts over time."""
        amounts = [tx.amount for tx in history if tx.sender == transaction.sender]
        if not amounts:
            return 0.0
        mean = sum(amounts) / len(amounts)
        return sum((amt - mean) ** 2 for amt in amounts) / len(amounts)

    def _is_round_number(self, amount: float) -> bool:
        """Checks if the transaction amount is a suspiciously round number."""
        return amount % 1000 == 0 or amount % 10000 == 0

    # --- Utility ---

    def _parse_time(self, ts: str) -> datetime:
        """Parses ISO timestamp string to datetime object."""
        return datetime.fromisoformat(ts)
