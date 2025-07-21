from typing import Optional
from collections import deque

from models import Transaction, AMLAlert
from feature_extractor import FeatureExtractor
from risk_scorer import RiskScorer
from alert_manager import AlertManager

class AMLDetector:
    """
    AMLDetector is responsible for identifying suspicious financial transactions
    that may indicate money laundering. It uses feature extraction, risk scoring,
    and alert management in a streamlined pipeline.
    """

    def __init__(
        self, 
        max_history_size: int = 10000, 
        risk_threshold: float = 0.7
    ):
        """
        Initializes the AMLDetector system.

        Args:
            max_history_size (int): Max number of transactions to retain for context.
            risk_threshold (float): Threshold above which a transaction is considered suspicious.
        """
        self.feature_extractor = FeatureExtractor()
        self.risk_scorer = RiskScorer()
        self.alert_manager = AlertManager()
        self.transaction_history: deque[Transaction] = deque(maxlen=max_history_size)
        self.risk_threshold = risk_threshold

    def process_transaction(self, transaction: Transaction) -> Optional[AMLAlert]:
        """
        Analyzes a transaction and determines whether it should raise an AML alert.

        Args:
            transaction (Transaction): A single financial transaction.

        Returns:
            Optional[AMLAlert]: An alert object if suspicious; otherwise, None.
        """
        # Step 1: Extract features from transaction and history
        history = list(self.transaction_history)
        features = self.feature_extractor.extract_features(transaction, history)

        # Step 2: Calculate risk score from features
        risk_score = self.risk_scorer.calculate_risk_score(features)

        # Step 3: Determine if transaction is suspicious
        if risk_score >= self.risk_threshold:
            alert = self._generate_alert(transaction, risk_score)
        else:
            alert = None

        # Step 4: Record transaction for future reference
        self._record_transaction(transaction)
        return alert

    def _generate_alert(self, transaction: Transaction, risk_score: float) -> AMLAlert:
        """
        Creates an alert for a suspicious transaction.

        Args:
            transaction (Transaction): The transaction to be flagged.
            risk_score (float): Computed risk score for the transaction.

        Returns:
            AMLAlert: The generated alert object.
        """
        return self.alert_manager.generate_alert(transaction, risk_score)

    def _record_transaction(self, transaction: Transaction) -> None:
        """
        Stores the transaction in the bounded history.

        Args:
            transaction (Transaction): The transaction to store.
        """
        self.transaction_history.append(transaction)
