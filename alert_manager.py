from models import Transaction, AMLAlert
from datetime import datetime

class AlertManager:
    """Handles generation of AML alerts based on transaction risk scores."""

    def generate_alert(self, transaction: Transaction, risk_score: float) -> AMLAlert:
        """Creates an AML alert with appropriate classification and metadata."""
        alert_type = self._determine_alert_type(risk_score)
        description = self._generate_description(transaction, risk_score)

        return AMLAlert(
            transaction=transaction,
            risk_score=risk_score,
            alert_type=alert_type,
            description=description,
            timestamp=datetime.now()
        )

    def _determine_alert_type(self, risk_score: float) -> str:
        """Classifies alert severity based on risk score thresholds."""
        if risk_score >= 0.9:
            return "HIGH_RISK"
        elif risk_score >= 0.7:
            return "MEDIUM_RISK"
        return "LOW_RISK"

    def _generate_description(self, transaction: Transaction, risk_score: float) -> str:
        """Generates a human-readable description for the alert."""
        return (
            f"Suspicious transaction from '{transaction.sender_account}' to "
            f"'{transaction.recipient_account}' flagged with risk score {risk_score:.2f}."
        )
