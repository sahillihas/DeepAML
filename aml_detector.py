from typing import List, Optional
from models import Transaction, AMLAlert
from feature_extractor import FeatureExtractor
from risk_scorer import RiskScorer
from alert_manager import AlertManager

class AMLDetector:
    """
    Main class for processing financial transactions and detecting potential money laundering activities.
    """

    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.risk_scorer = RiskScorer()
        self.alert_manager = AlertManager()
        self.transaction_history: List[Transaction] = []

    def process_transaction(self, transaction: Transaction) -> Optional[AMLAlert]:
        """
        Processes a single transaction: extracts features, scores risk, and optionally generates an alert.

        Args:
            transaction (Transaction): The transaction to be analyzed.

        Returns:
            Optional[AMLAlert]: Returns an AML alert if the transaction is deemed suspicious, otherwise None.
        """
        features = self.feature_extractor.extract_features(transaction, self.transaction_history)
        risk_score = self.risk_scorer.calculate_risk_score(features)

        if self.risk_scorer.is_suspicious(risk_score):
            alert = self._handle_suspicion(transaction, risk_score)
        else:
            alert = None

        self.transaction_history.append(transaction)
        # Optionally, maintain a bounded history to limit memory usage:
        # if len(self.transaction_history) > 10000:
        #     self.transaction_history.pop(0)

        return alert

    def _handle_suspicion(self, transaction: Transaction, risk_score: float) -> AMLAlert:
        """
        Handles suspicious transactions by generating an alert.

        Args:
            transaction (Transaction): The suspicious transaction.
            risk_score (float): The calculated risk score.

        Returns:
            AMLAlert: The generated AML alert.
        """
        return self.alert_manager.generate_alert(transaction, risk_score)
