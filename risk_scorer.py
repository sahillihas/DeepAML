from typing import Dict

class RiskScorer:
    def __init__(self):
        self.threshold = 0.7
        
    def calculate_risk_score(self, features: Dict) -> float:
        score = 0.0
        
        # Amount-based risk
        if features['amount'] > 10000:
            score += 0.4
        
        # Country risk
        if features['is_high_risk_country']:
            score += 0.3
            
        # Transaction frequency risk
        if features['frequency'] > 5:
            score += 0.2
            
        # Velocity risk
        if features['velocity'] > 5000:
            score += 0.1
            
        return min(score, 1.0)
        
    def is_suspicious(self, risk_score: float) -> bool:
        return risk_score >= self.threshold
