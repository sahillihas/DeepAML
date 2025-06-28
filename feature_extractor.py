from typing import List, Dict
from datetime import datetime, timedelta
from collections import defaultdict
from models import Transaction


class FeatureExtractor:
    def __init__(self):
        self.high_risk_countries = {'CY', 'AE', 'MT', 'LU', 'BZ', 'PA', 'VG'}
        self.medium_risk_countries = {'CH', 'HK', 'SG', 'MO', 'BS', 'AI'}

    def extract_features(self, transaction: Transaction, history: List[Transaction]) -> Dict:
        """Extract various features from a transaction and its history for risk analysis."""
        return {
            # Activity-based
            'frequency': self._calculate_frequency(transaction, history),
            'velocity': self._calculate_velocity(transaction, history),
            'amount_variance': self._calculate_amount_variance(transaction, history),

            # Risk-based
            'country_risk': self._calculate_country_risk(transaction),
            'cross_border': self._is_cross_border(transaction),
            'structuring_risk': self._calculate_structuring_risk(transaction, history),
            'network_risk': self._calculate_network_risk(transaction, history),

            # Time-based
            'time_risk': self._calculate_time_risk(transaction),
            'burst_risk': self._calculate_burst_risk(transaction, history),

            # Amount-based
            'amount_risk': self._calculate_amount_risk(transaction),
            'round_number': self._is_round_number(transaction.amount),
        }

    def _calculate_frequency(self, transaction: Transaction, history: List[Transaction]) -> float:
        """Compute frequency of transactions by sender in the past 30 days."""
        cutoff = datetime.fromisoformat(transaction.timestamp) - timedelta(days=30)
        count = sum(1 for tx in history if tx.sender == transaction.sender and datetime.fromisoformat(tx.timestamp) >= cutoff)
        return count / 30

    def _calculate_velocity(self, transaction: Transaction, history: List[Transaction]) -> float:
        """Compute total amount sent per day by sender in the past 30 days."""
        cutoff = datetime.fromisoformat(transaction.timestamp) - timedelta(days=30)
        relevant = [tx for tx in history if tx.sender == transaction.sender and datetime.fromisoformat(tx.timestamp) >= cutoff]
        total = sum(tx.amount for tx in relevant) + transaction.amount
        return total / 30 if relevant else 0.0

    def _calculate_amount_variance(self, transaction: Transaction, history: List[Transaction]) -> float:
        """Compute variance in the sender's transaction amounts."""
        amounts = [tx.amount for tx in history if tx.sender == transaction.sender]
        if not amounts:
            return 0.0
        mean = sum(amounts) / len(amounts)
        return sum((amt - mean) ** 2 for amt in amounts) / len(amounts)

    def _calculate_country_risk(self, transaction: Transaction) -> float:
        """Assess risk based on sender/receiver country."""
        def risk_level(country: str) -> float:
            if country in self.high_risk_countries:
                return 1.0
            elif country in self.medium_risk_countries:
                return 0.6
            return 0.2

        return max(risk_level(transaction.sender_country), risk_level(transaction.receiver_country))

    def _is_cross_border(self, transaction: Transaction) -> bool:
        """Determine if transaction is cross-border."""
        return transaction.sender_country != transaction.receiver_country

    def _calculate_structuring_risk(self, transaction: Transaction, history: List[Transaction]) -> float:
        """Estimate risk of structuring (smurfing) behavior."""
        cutoff = datetime.fromisoformat(transaction.timestamp) - timedelta(hours=48)
        threshold = 10000

        relevant = [
            tx for tx in history
            if tx.sender == transaction.sender and tx.receiver == transaction.receiver
            and datetime.fromisoformat(tx.timestamp) >= cutoff
        ]

        near_threshold = sum(1 for tx in relevant if threshold * 0.8 <= tx.amount < threshold)

        if near_threshold >= 2:
            return 0.8
        elif near_threshold == 1:
            return 0.4
        return 0.0

    def _calculate_network_risk(self, transaction: Transaction, history: List[Transaction]) -> float:
        """Assess risk based on number of unique counterparties involved."""
        sender_partners = {tx.receiver for tx in history if tx.sender == transaction.sender}
        receiver_partners = {tx.sender for tx in history if tx.receiver == transaction.receiver}
        return max(min(len(sender_partners) / 20, 1.0), min(len(receiver_partners) / 20, 1.0))

    def _calculate_time_risk(self, transaction: Transaction) -> float:
        """Evaluate risk based on time of transaction."""
        hour = datetime.fromisoformat(transaction.timestamp).hour
        return 0.8 if hour < 6 or hour > 18 else 0.2

    def _calculate_burst_risk(self, transaction: Transaction, history: List[Transaction]) -> float:
        """Detect burst patterns in recent activity."""
        cutoff = datetime.fromisoformat(transaction.timestamp) - timedelta(hours=24)
        recent = [tx for tx in history if tx.sender == transaction.sender and datetime.fromisoformat(tx.timestamp) >= cutoff]
        tx_per_hour = len(recent) / 24
        return min(tx_per_hour / 2, 1.0)

    def _calculate_amount_risk(self, transaction: Transaction) -> float:
        """Score risk based on the amount of transaction."""
        if transaction.amount >= 50000:
            return 1.0
        elif transaction.amount >= 10000:
            return 0.6
        return 0.2

    def _is_round_number(self, amount: float) -> bool:
        """Check if the amount is suspiciously round."""
        return amount % 1000 == 0 or amount % 10000 == 0
