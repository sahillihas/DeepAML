from typing import List, Optional
from models import Transaction, AMLAlert
from feature_extractor import FeatureExtractor
from risk_scorer import RiskScorer
from alert_manager import AlertManager


class AMLDetector:
    """
    Detects potential money laundering activities by analyzing financial transactions.
    """

    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.risk_scorer = RiskScorer()
        self.alert_manager = AlertManager()
        self.transaction_history: List[Transaction] = []

    def process_transaction(self, transaction: Transaction) -> Optional[AMLAlert]:
        """
        Analyzes a single transaction for suspicious activity.

        Steps:
            1. Extract features from the transaction and its history.
            2. Calculate the risk score based on these features.
            3. If the transaction is suspicious, generate an alert.

        Args:
            transaction (Transaction): The transaction to be analyzed.

        Returns:
            Optional[AMLAlert]: An alert if the transaction is suspicious; otherwise, None.
        """
        features = self.feature_extractor.extract_features(transaction, self.transaction_history)
        risk_score = self.risk_scorer.calculate_risk_score(features)

        alert = None
        if self.risk_scorer.is_suspicious(risk_score):
            alert = self._handle_suspicion(transaction, risk_score)

        self._record_transaction(transaction)
        return alert

    def _handle_suspicion(self, transaction: Transaction, risk_score: float) -> AMLAlert:
        """
        Generates an alert for a suspicious transaction.

        Args:
            transaction (Transaction): The suspicious transaction.
            risk_score (float): The associated risk score.

        Returns:
            AMLAlert: The generated alert.
        """
        return self.alert_manager.generate_alert(transaction, risk_score)

    def _record_transaction(self, transaction: Transaction) -> None:
        """
        Adds the transaction to history and manages memory usage.

        Args:
            transaction (Transaction): The transaction to record.
        """
        self.transaction_history.append(transaction)

        # Optional: cap history size to limit memory usage
        MAX_HISTORY_SIZE = 10000
        if len(self.transaction_history) > MAX_HISTORY_SIZE:
            self.transaction_history.pop(0)
