from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class Transaction:
    transaction_id: str
    amount: float
    sender: str
    receiver: str
    timestamp: datetime
    transaction_type: str
    country: str
    
@dataclass
class AMLAlert:
    transaction: Transaction
    risk_score: float
    alert_type: str
    description: str
    timestamp: datetime
