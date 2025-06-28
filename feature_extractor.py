from typing import List, Dict
from datetime import datetime, timedelta
from models import Transaction


class FeatureExtractor:
    def __init__(self):
        # Country risk levels based on AML compliance, FATF, tax haven status, etc.
        self.country_risk_map = {
            # High-risk jurisdictions (FATF blacklist or greylist, offshore tax havens)
            'IR': 1.0, 'KP': 1.0, 'SY': 1.0, 'MM': 1.0, 'PA': 1.0, 'BZ': 1.0, 'VG': 1.0,
            'AF': 1.0, 'CU': 1.0, 'CD': 1.0, 'SD': 1.0, 'YE': 1.0, 'ZW': 1.0, 'SO': 1.0,
            'AE': 1.0, 'MT': 1.0, 'CY': 1.0, 'LU': 1.0, 'KY': 1.0, 'LC': 1.0, 'AD': 1.0,

            # Medium-risk (financial secrecy, less transparency, or in FATF follow-up)
            'CH': 0.6, 'SG': 0.6, 'HK': 0.6, 'MO': 0.6, 'BS': 0.6, 'AI': 0.6, 'MU': 0.6,
            'DO': 0.6, 'TR': 0.6, 'JO': 0.6, 'QA': 0.6, 'RU': 0.6, 'UA': 0.6, 'KZ': 0.6,
            'LB': 0.6, 'EG': 0.6, 'PK': 0.6, 'TN': 0.6,

            # Default low-risk for all others
        }

    def extract_features(self, transaction: Transaction, history: List[Transaction]) -> Dict:
        return {
            'frequency': self._calculate_frequency(transaction, history),
            'velocity': self._calculate_velocity(transaction, history),
            'amount_variance': self._calculate_amount_variance(transaction, history),
            'country_risk': self._calculate_country_risk(transaction),
            'cross_border': self._is_cross_border(transaction),
            'structuring_risk': self._calculate_structuring_risk(transaction, history),
            'network_risk': self._calculate_network_risk(transaction, history),
            'time_risk': self._calculate_time_risk(transaction),
            'burst_risk': self._calculate_burst_risk(transaction, history),
            'amount_risk': self._calculate_amount_risk(transaction),
            'round_number': self._is_round_number(transaction.amount),
        }

    def _get_country_risk_score(self, country_code: str) -> float:
        """Return predefined risk score for a country, defaulting to low risk."""
        return self.country_risk_map.get(country_code.upper(), 0.2)

    def _calculate_country_risk(self, transaction: Transaction) -> float:
        """Compute risk score based on sender and receiver countries."""
        sender_risk = self._get_country_risk_score(transaction.sender_country)
        receiver_risk = self._get_country_risk_score(transaction.receiver_country)
        return max(sender_risk, receiver_risk)

    def _is_cross_border(self, transaction: Transaction) -> bool:
        return transaction.sender_country != transaction.receiver_country

    def _calculate_frequency(self, transaction: Transaction, history: List[Transaction]) -> float:
        cutoff = datetime.fromisoformat(transaction.timestamp) - timedelta(days=30)
        count = sum(1 for tx in history if tx.sender == transaction.sender and datetime.fromisoformat(tx.timestamp) >= cutoff)
        return count / 30

    def _calculate_velocity(self, transaction: Transaction, history: List[Transaction]) -> float:
        cutoff = datetime.fromisoformat(transaction.timestamp) - timedelta(days=30)
        relevant = [tx for tx in history if tx.sender == transaction.sender and datetime.fromisoformat(tx.timestamp) >= cutoff]
        total = sum(tx.amount for tx in relevant) + transaction.amount
        return total / 30 if relevant else 0.0

    def _calculate_amount_variance(self, transaction: Transaction, history: List[Transaction]) -> float:
        amounts = [tx.amount for tx in history if tx.sender == transaction.sender]
        if not amounts:
            return 0.0
        mean = sum(amounts) / len(amounts)
        return sum((amt - mean) ** 2 for amt in amounts) / len(amounts)

    def _calculate_structuring_risk(self, transaction: Transaction, history: List[Transaction]) -> float:
        cutoff = datetime.fromisoformat(transaction.timestamp) - timedelta(hours=48)
        threshold = 10000
        recent = [
            tx for tx in history
            if tx.sender == transaction.sender and tx.receiver == transaction.receiver
            and datetime.fromisoformat(tx.timestamp) >= cutoff
        ]
        near_threshold = sum(1 for tx in recent if threshold * 0.8 <= tx.amount < threshold)
        return 0.8 if near_threshold >= 2 else 0.4 if near_threshold == 1 else 0.0

    def _calculate_network_risk(self, transaction: Transaction, history: List[Transaction]) -> float:
        sender_partners = {tx.receiver for tx in history if tx.sender == transaction.sender}
        receiver_partners = {tx.sender for tx in history if tx.receiver == transaction.receiver}
        return max(min(len(sender_partners) / 20, 1.0), min(len(receiver_partners) / 20, 1.0))

    def _calculate_time_risk(self, transaction: Transaction) -> float:
        hour = datetime.fromisoformat(transaction.timestamp).hour
        return 0.8 if hour < 6 or hour > 18 else 0.2

    def _calculate_burst_risk(self, transaction: Transaction, history: List[Transaction]) -> float:
        cutoff = datetime.fromisoformat(transaction.timestamp) - timedelta(hours=24)
        recent = [tx for tx in history if tx.sender == transaction.sender and datetime.fromisoformat(tx.timestamp) >= cutoff]
        return min(len(recent) / 24, 1.0)

    def _calculate_amount_risk(self, transaction: Transaction) -> float:
        if transaction.amount >= 50000:
            return 1.0
        elif transaction.amount >= 10000:
            return 0.6
        return 0.2

    def _is_round_number(self, amount: float) -> bool:
        return amount % 1000 == 0 or amount % 10000 == 0
