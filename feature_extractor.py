from models import Transaction
from typing import List, Dict
from collections import defaultdict

class FeatureExtractor:
    def __init__(self):
        self.high_risk_countries = {'CountryA', 'CountryB'}  # Add high-risk countries
        
    def extract_features(self, transaction: Transaction, history: List[Transaction]) -> Dict:
        features = {
            'amount': transaction.amount,
            'is_high_risk_country': transaction.country in self.high_risk_countries,
            'frequency': self._calculate_frequency(transaction, history),
            'velocity': self._calculate_velocity(transaction, history)
        }
        return features
        
    def _calculate_frequency(self, transaction: Transaction, history: List[Transaction]) -> int:
        return sum(1 for t in history if t.sender == transaction.sender)
        
    def _calculate_velocity(self, transaction: Transaction, history: List[Transaction]) -> float:
        sender_transactions = [t for t in history if t.sender == transaction.sender]
        if not sender_transactions:
            return 0.0
        total_amount = sum(t.amount for t in sender_transactions)
        return total_amount / len(sender_transactions)
