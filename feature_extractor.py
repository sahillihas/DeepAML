from typing import List, Dict, Set
from models import Transaction
from datetime import datetime, timedelta
from collections import defaultdict

class FeatureExtractor:
    def __init__(self):
        # Define high-risk countries based on AML risk factors
        self.high_risk_countries = {'CY', 'AE', 'MT', 'LU', 'BZ', 'PA', 'VG'}
        self.medium_risk_countries = {'CH', 'HK', 'SG', 'MO', 'BS', 'AI'}
        
    def extract_features(self, transaction: Transaction, history: List[Transaction]) -> Dict:
        """Extract features from a transaction and its history for risk analysis"""
        features = {
            # Activity-based features
            'frequency': self._calculate_frequency(transaction, history),
            'velocity': self._calculate_velocity(transaction, history),
            'amount_variance': self._calculate_amount_variance(transaction, history),
            
            # Risk-based features
            'country_risk': self._calculate_country_risk(transaction),
            'cross_border': self._is_cross_border(transaction),
            'structuring_risk': self._calculate_structuring_risk(transaction, history),
            'network_risk': self._calculate_network_risk(transaction, history),
            
            # Time-based features
            'time_risk': self._calculate_time_risk(transaction),
            'burst_risk': self._calculate_burst_risk(transaction, history),
            
            # Amount-based features
            'amount_risk': self._calculate_amount_risk(transaction),
            'round_number': self._is_round_number(transaction.amount)
        }
        return features
        
    def _calculate_frequency(self, transaction: Transaction, history: List[Transaction]) -> float:
        """Calculate how frequently this sender transacts"""
        lookback_days = 30
        cutoff = datetime.fromisoformat(transaction.timestamp) - timedelta(days=lookback_days)
        recent_history = [tx for tx in history 
                         if datetime.fromisoformat(tx.timestamp) >= cutoff
                         and tx.sender == transaction.sender]
        return len(recent_history) / lookback_days
        
    def _calculate_velocity(self, transaction: Transaction, history: List[Transaction]) -> float:
        """Calculate the velocity of money movement (amount per day)"""
        lookback_days = 30
        cutoff = datetime.fromisoformat(transaction.timestamp) - timedelta(days=lookback_days)
        recent_history = [tx for tx in history 
                         if datetime.fromisoformat(tx.timestamp) >= cutoff
                         and tx.sender == transaction.sender]
                         
        if not recent_history:
            return 0.0
            
        total_amount = sum(tx.amount for tx in recent_history) + transaction.amount
        return total_amount / lookback_days
        
    def _calculate_amount_variance(self, transaction: Transaction, history: List[Transaction]) -> float:
        """Calculate variance in transaction amounts"""
        sender_history = [tx.amount for tx in history if tx.sender == transaction.sender]
        if not sender_history:
            return 0.0
            
        mean = sum(sender_history) / len(sender_history)
        squared_diff_sum = sum((amt - mean) ** 2 for amt in sender_history)
        variance = squared_diff_sum / len(sender_history)
        return variance
        
    def _calculate_country_risk(self, transaction: Transaction) -> float:
        """Calculate risk based on countries involved"""
        sender_risk = (1.0 if transaction.sender_country in self.high_risk_countries else
                      0.6 if transaction.sender_country in self.medium_risk_countries else 0.2)
                      
        receiver_risk = (1.0 if transaction.receiver_country in self.high_risk_countries else
                        0.6 if transaction.receiver_country in self.medium_risk_countries else 0.2)
                        
        return max(sender_risk, receiver_risk)
        
    def _is_cross_border(self, transaction: Transaction) -> bool:
        """Check if transaction crosses national borders"""
        return transaction.sender_country != transaction.receiver_country
        
    def _calculate_structuring_risk(self, transaction: Transaction, history: List[Transaction]) -> float:
        """Calculate risk of structuring behavior"""
        lookback_hours = 48
        threshold = 10000  # Common structuring threshold
        cutoff = datetime.fromisoformat(transaction.timestamp) - timedelta(hours=lookback_hours)
        
        # Get recent transactions between same sender and receiver
        recent_txs = [tx for tx in history 
                     if datetime.fromisoformat(tx.timestamp) >= cutoff
                     and tx.sender == transaction.sender 
                     and tx.receiver == transaction.receiver]
        
        if not recent_txs:
            return 0.0
            
        # Check for multiple transactions just under threshold
        near_threshold_count = sum(1 for tx in recent_txs 
                                 if threshold * 0.8 <= tx.amount < threshold)
        
        if near_threshold_count >= 2:
            return 0.8
        elif near_threshold_count == 1:
            return 0.4
        return 0.0
        
    def _calculate_network_risk(self, transaction: Transaction, history: List[Transaction]) -> float:
        """Calculate risk based on transaction network characteristics"""
        # Count unique counterparties
        sender_counterparties = {tx.receiver for tx in history if tx.sender == transaction.sender}
        receiver_counterparties = {tx.sender for tx in history if tx.receiver == transaction.receiver}
        
        # More counterparties = higher risk
        sender_risk = min(1.0, len(sender_counterparties) / 20)
        receiver_risk = min(1.0, len(receiver_counterparties) / 20)
        
        return max(sender_risk, receiver_risk)
        
    def _calculate_time_risk(self, transaction: Transaction) -> float:
        """Calculate risk based on transaction timing"""
        timestamp = datetime.fromisoformat(transaction.timestamp)
        hour = timestamp.hour
        
        # Higher risk for transactions during non-business hours
        if hour < 6 or hour > 18:  # 6 AM to 6 PM considered normal
            return 0.8
        return 0.2
        
    def _calculate_burst_risk(self, transaction: Transaction, history: List[Transaction]) -> float:
        """Calculate risk of burst transaction activity"""
        lookback_hours = 24
        cutoff = datetime.fromisoformat(transaction.timestamp) - timedelta(hours=lookback_hours)
        
        # Get recent transactions from sender
        recent_txs = [tx for tx in history 
                     if datetime.fromisoformat(tx.timestamp) >= cutoff
                     and tx.sender == transaction.sender]
        
        # Calculate transactions per hour
        tx_count = len(recent_txs)
        tx_per_hour = tx_count / lookback_hours
        
        # More than 2 transactions per hour on average is suspicious
        return min(1.0, tx_per_hour / 2)
        
    def _calculate_amount_risk(self, transaction: Transaction) -> float:
        """Calculate risk based on transaction amount"""
        # Different thresholds for different risk levels
        high_risk_threshold = 50000
        medium_risk_threshold = 10000
        
        if transaction.amount >= high_risk_threshold:
            return 1.0
        elif transaction.amount >= medium_risk_threshold:
            return 0.6
        return 0.2
        
    def _is_round_number(self, amount: float) -> bool:
        """Check if amount is suspiciously round"""
        return amount % 1000 == 0 or amount % 10000 == 0
