from typing import Dict

class RiskScorer:
    def __init__(self):
        # Risk thresholds
        self.high_risk_threshold = 0.7
        self.medium_risk_threshold = 0.4
        
        # Feature weights for risk calculation
        self.weights = {
            # Activity-based features
            'frequency': 0.1,
            'velocity': 0.15,
            'amount_variance': 0.05,
            
            # Risk-based features
            'country_risk': 0.2,
            'cross_border': 0.05,
            'structuring_risk': 0.15,
            'network_risk': 0.1,
            
            # Time-based features
            'time_risk': 0.05,
            'burst_risk': 0.05,
            
            # Amount-based features
            'amount_risk': 0.08,
            'round_number': 0.02
        }
        
    def calculate_risk_score(self, features: Dict) -> float:
        """Calculate risk score from transaction features"""
        score = 0.0
        
        # Calculate weighted sum of features
        for feature, weight in self.weights.items():
            if feature in features:
                # Convert boolean features to float
                value = float(features[feature]) if isinstance(features[feature], bool) else features[feature]
                score += value * weight
                
        # Ensure score is between 0 and 1
        return min(1.0, max(0.0, score))
        
    def get_risk_level(self, risk_score: float) -> str:
        """Get risk level category from risk score"""
        if risk_score >= self.high_risk_threshold:
            return "HIGH"
        elif risk_score >= self.medium_risk_threshold:
            return "MEDIUM"
        return "LOW"
        
    def is_suspicious(self, risk_score: float) -> bool:
        """Determine if transaction is suspicious based on risk score"""
        return risk_score >= self.high_risk_threshold
