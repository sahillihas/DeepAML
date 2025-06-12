from typing import Dict

class RiskScorer:
    def __init__(self):
        # Thresholds for categorizing risk levels
        self.thresholds = {
            "HIGH": 0.7,
            "MEDIUM": 0.5,
            "LOW": 0.3
        }

        # Feature weights used in risk score computation
        self.weights: Dict[str, float] = {
            # Activity-based
            'frequency': 0.1,
            'velocity': 0.15,
            'amount_variance': 0.05,

            # Risk-based
            'country_risk': 0.2,
            'cross_border': 0.05,
            'structuring_risk': 0.15,
            'network_risk': 0.1,

            # Time-based
            'time_risk': 0.05,
            'burst_risk': 0.05,

            # Amount-based
            'amount_risk': 0.08,
            'round_number': 0.02
        }

    def calculate_risk_score(self, features: Dict[str, float]) -> float:
        """
        Calculate a weighted risk score from transaction features.

        Args:
            features (Dict[str, float]): A dictionary of feature values.

        Returns:
            float: Risk score normalized between 0.0 and 1.0.
        """
        score = sum(
            float(features.get(feature, 0)) * weight
            for feature, weight in self.weights.items()
        )
        return min(1.0, max(0.0, score))

    def get_risk_level(self, risk_score: float) -> str:
        """
        Categorize risk score into a risk level.

        Args:
            risk_score (float): The calculated risk score.

        Returns:
            str: "HIGH", "MEDIUM", or "LOW".
        """
        if risk_score >= self.thresholds["HIGH"]:
            return "HIGH"
        elif risk_score >= self.thresholds["MEDIUM"]:
            return "MEDIUM"
        return "LOW"

    def is_suspicious(self, risk_score: float, threshold: float = None) -> bool:
        """
        Determine whether a transaction is suspicious.

        Args:
            risk_score (float): The calculated risk score.
            threshold (float, optional): Custom threshold for suspicion. 
                                         Defaults to HIGH risk threshold.

        Returns:
            bool: True if suspicious, False otherwise.
        """
        threshold = threshold if threshold is not None else self.thresholds["HIGH"]
        return risk_score >= threshold
