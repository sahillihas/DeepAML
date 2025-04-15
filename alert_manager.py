from models import Transaction, AMLAlert
from datetime import datetime

class AlertManager:
    def generate_alert(self, transaction: Transaction, risk_score: float) -> AMLAlert:
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
        if risk_score >= 0.9:
            return "HIGH_RISK"
        elif risk_score >= 0.7:
            return "MEDIUM_RISK"
        return "LOW_RISK"
        
    def _generate_description(self, transaction: Transaction, risk_score: float) -> str:
        return f"Suspicious transaction detected with risk score {risk_score:.2f}"
