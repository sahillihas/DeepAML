from typing import Dict, Optional
from enum import Enum, auto


class RiskLevel(Enum):
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()

    def __str__(self):
        return self.name


class RiskScorer:
    """
    RiskScorer evaluates the risk score of a transaction based on a set of features
    and categorizes it into LOW, MEDIUM, or HIGH risk levels.
    """

    _THRESHOLDS: Dict[RiskLevel, float] = {
        RiskLevel.HIGH: 0.7,
        RiskLevel.MEDIUM: 0.5,
        RiskLevel.LOW: 0.3
    }

    _FEATURE_WEIGHTS: Dict[str, float] = {
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
        Calculates a risk score by applying predefined feature weights.

        Args:
            features (Dict[str, float]): Feature values of a transaction.

        Returns:
            float: Risk score clamped between 0.0 and 1.0.
        """
        total_score = 0.0
        for feature, weight in self._FEATURE_WEIGHTS.items():
            value = features.get(feature, 0.0)
            if not isinstance(value, (int, float)):
                raise ValueError(f"Feature '{feature}' must be a number, got {type(value)}")
            total_score += value * weight

        return min(1.0, max(0.0, total_score))  # Ensure score is within [0, 1]

    def get_risk_level(self, risk_score: float) -> RiskLevel:
        """
        Classifies a numerical risk score into a RiskLevel category.

        Args:
            risk_score (float): Score between 0.0 and 1.0.

        Returns:
            RiskLevel: Enum category for risk.
        """
        if risk_score >= self._THRESHOLDS[RiskLevel.HIGH]:
            return RiskLevel.HIGH
        elif risk_score >= self._THRESHOLDS[RiskLevel.MEDIUM]:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def is_suspicious(self, risk_score: float, threshold: Optional[float] = None) -> bool:
        """
        Determines if a transaction is suspicious based on the risk score and threshold.

        Args:
            risk_score (float): Risk score from 0.0 to 1.0.
            threshold (Optional[float]): Custom threshold for suspicion.

        Returns:
            bool: True if risk_score >= threshold, else False.
        """
        effective_threshold = threshold if threshold is not None else self._THRESHOLDS[RiskLevel.HIGH]
        if not isinstance(effective_threshold, (int, float)):
            raise ValueError("Threshold must be a numeric value.")
        return risk_score >= effective_threshold
