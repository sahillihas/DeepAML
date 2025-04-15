from typing import List, Optional
from models import Transaction, AMLAlert
from feature_extractor import FeatureExtractor
from risk_scorer import RiskScorer
from alert_manager import AlertManager

class AMLDetector:
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.risk_scorer = RiskScorer()
        self.alert_manager = AlertManager()
        self.transaction_history: List[Transaction] = []
        
    def process_transaction(self, transaction: Transaction) -> Optional[AMLAlert]:
        # Extract features
        features = self.feature_extractor.extract_features(transaction, self.transaction_history)
        
        # Calculate risk score
        risk_score = self.risk_scorer.calculate_risk_score(features)
        
        # Generate alert if suspicious
        alert = None
        if self.risk_scorer.is_suspicious(risk_score):
            alert = self.alert_manager.generate_alert(transaction, risk_score)
            
        # Update history
        self.transaction_history.append(transaction)
        
        return alert
