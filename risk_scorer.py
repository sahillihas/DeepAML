from typing import Dict, Optional
from enum import Enum


class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class RiskScorer:
    """
    Class to compute risk scores based on weighted transaction features 
    and determine the associated risk level.
    """

    # Thresholds for determining risk levels
    RISK_THRESHOLDS: Dict[RiskLevel, float] = {
        RiskLevel.HIGH: 0.7,
        RiskLevel.MEDIUM: 0.5,
        RiskLevel.LOW: 0.3
    }

    # Weights assigned to each transaction feature
    FEATURE_WEIGHTS: Dict[str, float] = {
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
        Calculates the weighted risk score from input features.

        Args:
            features: Dictionary of feature name to feature value.

        Returns:
            A float score between 0.0 and 1.0.
        """
        score = sum(
            features.get(feature, 0.0) * weight
            for feature, weight in self.FEATURE_WEIGHTS.items()
        )
        return max(0.0, min(1.0, score))  # Clamp between 0 and 1

    def get_risk_level(self, risk_score: float) -> RiskLevel:
        """
        Maps the risk score to a risk category.

        Args:
            risk_score: Risk score between 0.0 and 1.0.

        Returns:
            RiskLevel: LOW, MEDIUM, or HIGH.
        """
        if risk_score >= self.RISK_THRESHOLDS[RiskLevel.HIGH]:
            return RiskLevel.HIGH
        elif risk_score >= self.RISK_THRESHOLDS[RiskLevel.MEDIUM]:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def is_suspicious(self, risk_score: float, threshold: Optional[float] = None) -> bool:
        """
        Determines if the transaction is suspicious based on threshold.

        Args:
            risk_score: The calculated risk score.
            threshold: Optional custom threshold for suspicion.

        Returns:
            True if score is greater than or equal to the threshold.
        """
        effective_threshold = threshold if threshold is not None else self.RISK_THRESHOLDS[RiskLevel.HIGH]
        return risk_score >= effective_threshold
