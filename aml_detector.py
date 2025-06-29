from typing import List, Optional
from models import Transaction, AMLAlert
from feature_extractor import FeatureExtractor
from risk_scorer import RiskScorer
from alert_manager import AlertManager

class AMLDetector:
    """
    A system for detecting potential money laundering activities by analyzing 
    financial transactions using feature extraction, risk scoring, and alert generation.
    """

    MAX_HISTORY_SIZE = 10000  # Maximum number of transactions to retain in memory

    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.risk_scorer = RiskScorer()
        self.alert_manager = AlertManager()
        self.transaction_history: List[Transaction] = []

    def process_transaction(self, transaction: Transaction) -> Optional[AMLAlert]:
        """
        Processes and analyzes a transaction for suspicious activity.

        Workflow:
            1. Extract relevant features from the transaction and history.
            2. Compute a risk score based on the extracted features.
            3. Generate an alert if the transaction is deemed suspicious.
            4. Record the transaction in history for future analysis.

        Args:
            transaction (Transaction): The financial transaction to analyze.

        Returns:
            Optional[AMLAlert]: An AML alert if suspicious; otherwise, None.
        """
        features = self.feature_extractor.extract_features(transaction, self.transaction_history)
        risk_score = self.risk_scorer.calculate_risk_score(features)

        if self.risk_scorer.is_suspicious(risk_score):
            alert = self._generate_alert(transaction, risk_score)
        else:
            alert = None

        self._record_transaction(transaction)
        return alert

    def _generate_alert(self, transaction: Transaction, risk_score: float) -> AMLAlert:
        """
        Delegates the creation of an alert for a suspicious transaction.

        Args:
            transaction (Transaction): The transaction flagged as suspicious.
            risk_score (float): The associated calculated risk score.

        Returns:
            AMLAlert: The generated alert object.
        """
        return self.alert_manager.generate_alert(transaction, risk_score)

    def _record_transaction(self, transaction: Transaction) -> None:
        """
        Appends the transaction to the history and ensures memory is managed.

        Args:
            transaction (Transaction): The transaction to store.
        """
        self.transaction_history.append(transaction)

        if len(self.transaction_history) > self.MAX_HISTORY_SIZE:
            self.transaction_history.pop(0)
