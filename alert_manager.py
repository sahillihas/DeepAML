from models import Transaction, AMLAlert
from datetime import datetime
import logging
import hashlib
from typing import Optional, Dict

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class AlertManager:
    """Handles generation and classification of AML alerts based on transaction risk scores."""

    def __init__(self):
        self.thresholds = {
            'default': {'HIGH_RISK': 0.9, 'MEDIUM_RISK': 0.7},
            'international': {'HIGH_RISK': 0.85, 'MEDIUM_RISK': 0.6},
        }
        self._seen_alerts = set()  # Used for duplicate alert detection

    def generate_alert(self, transaction: Transaction, risk_score: float, metadata: Optional[Dict] = None) -> Optional[AMLAlert]:
        """Creates an AML alert with classification, deduplication, and optional metadata."""
        alert_id = self._generate_alert_id(transaction, risk_score)

        if alert_id in self._seen_alerts:
            logger.info("Duplicate alert skipped for transaction ID: %s", transaction.transaction_id)
            return None

        alert_type = self._determine_alert_type(transaction, risk_score)
        description = self._generate_description(transaction, risk_score, alert_type)
        tags = self._generate_tags(transaction, risk_score)

        alert = AMLAlert(
            transaction=transaction,
            risk_score=risk_score,
            alert_type=alert_type,
            description=description,
            tags=tags,
            metadata=metadata or {},
            timestamp=datetime.now()
        )

        self._seen_alerts.add(alert_id)
        logger.info("Generated alert: %s", alert)

        return alert

    def _determine_alert_type(self, transaction: Transaction, risk_score: float) -> str:
        """Classifies alert severity based on transaction type-specific risk thresholds."""
        tx_type = getattr(transaction, "transaction_type", "default")
        thresholds = self.thresholds.get(tx_type, self.thresholds['default'])

        if risk_score >= thresholds['HIGH_RISK']:
            return "HIGH_RISK"
        elif risk_score >= thresholds['MEDIUM_RISK']:
            return "MEDIUM_RISK"
        return "LOW_RISK"

    def _generate_description(self, transaction: Transaction, risk_score: float, alert_type: str) -> str:
        """Creates a detailed and human-readable alert description."""
        return (
            f"[{alert_type}] Suspicious transaction of {transaction.amount} "
            f"from '{transaction.sender_account}' to '{transaction.recipient_account}' "
            f"on {transaction.timestamp.strftime('%Y-%m-%d')} with risk score {risk_score:.2f}."
        )

    def _generate_tags(self, transaction: Transaction, risk_score: float) -> list:
        """Generates tags based on transaction properties."""
        tags = []
        if risk_score > 0.9:
            tags.append("urgent_review")
        if getattr(transaction, "is_international", False):
            tags.append("international")
        if transaction.amount > 100000:
            tags.append("high_value")
        return tags

    def _generate_alert_id(self, transaction: Transaction, risk_score: float) -> str:
        """Generates a hash to detect duplicate alerts."""
        base_str = f"{transaction.transaction_id}|{transaction.timestamp.isoformat()}|{risk_score:.2f}"
        return hashlib.sha256(base_str.encode()).hexdigest()

    def export_alert_to_json(self, alert: AMLAlert) -> Dict:
        """Converts the alert object to a serializable JSON-like dictionary."""
        return {
            "transaction_id": alert.transaction.transaction_id,
            "sender": alert.transaction.sender_account,
            "recipient": alert.transaction.recipient_account,
            "amount": alert.transaction.amount,
            "timestamp": alert.timestamp.isoformat(),
            "risk_score": alert.risk_score,
            "alert_type": alert.alert_type,
            "description": alert.description,
            "tags": alert.tags,
            "metadata": alert.metadata
        }
