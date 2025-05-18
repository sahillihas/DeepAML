"""
Models and Enums for AML Detection System
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List, Union

class PatternType(Enum):
    """Types of money laundering patterns"""
    STRUCTURING = "structuring"
    LAYERING = "layering"
    RAPID_MOVEMENT = "rapid_movement"
    FAN_IN = "fan_in"
    FAN_OUT = "fan_out"

class RiskLevel(Enum):
    """Risk levels for accounts and transactions"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

@dataclass
class Transaction:
    """Represents a financial transaction"""
    transaction_id: str
    timestamp: str
    amount: float
    currency: str
    sender: str
    sender_name: str
    sender_bank: str
    sender_country: str
    receiver: str
    receiver_name: str
    receiver_bank: str
    receiver_country: str
    reference: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    risk_score: Optional[float] = None
    detected_patterns: Dict[str, Dict[str, Any]] = field(default_factory=dict)

@dataclass
class PatternDetail:
    """Represents details of a detected pattern"""
    pattern_type: PatternType
    risk_score: float
    severity: str
    details: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    detection_time: datetime = field(default_factory=datetime.now)

@dataclass
class AlertNote:
    """Represents a note on an alert"""
    timestamp: datetime
    author: str
    content: str
    note_type: str = "GENERAL"  # GENERAL, INVESTIGATION, RESOLUTION
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AMLAlert:
    """Represents an anti-money laundering alert"""
    alert_id: str
    transaction: Transaction
    risk_score: float
    risk_level: RiskLevel
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    status: str = "NEW"
    assigned_to: Optional[str] = None
    description: str = ""
    patterns: Dict[str, PatternDetail] = field(default_factory=dict)
    pattern_summary: Dict[str, Any] = field(default_factory=dict)
    notes: List[AlertNote] = field(default_factory=list)
    related_alerts: List[str] = field(default_factory=list)
    resolution: Optional[str] = None
    false_positive_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_pattern(self, pattern: PatternDetail):
        """Add a detected pattern to the alert"""
        self.patterns[pattern.pattern_type.value] = pattern
        self.updated_at = datetime.now()
        self._update_pattern_summary()

    def add_note(self, content: str, author: str, note_type: str = "GENERAL"):
        """Add a note to the alert"""
        note = AlertNote(
            timestamp=datetime.now(),
            author=author,
            content=content,
            note_type=note_type
        )
        self.notes.append(note)
        self.updated_at = datetime.now()

    def update_status(self, new_status: str, reason: str = None):
        """Update alert status"""
        self.status = new_status
        self.updated_at = datetime.now()
        if reason:
            self.add_note(
                f"Status changed to {new_status}: {reason}",
                "SYSTEM",
                "STATUS_CHANGE"
            )

    def _update_pattern_summary(self):
        """Update pattern summary based on detected patterns"""
        high_risk = sum(1 for p in self.patterns.values() if p.severity == "HIGH")
        medium_risk = sum(1 for p in self.patterns.values() if p.severity == "MEDIUM")
        
        self.pattern_summary = {
            'high_risk_patterns': high_risk,
            'medium_risk_patterns': medium_risk,
            'total_patterns': len(self.patterns),
            'primary_pattern': self._get_primary_pattern()
        }

    def _get_primary_pattern(self) -> str:
        """Determine the primary suspicious pattern"""
        if not self.patterns:
            return PatternType.UNKNOWN.value
            
        # Sort patterns by risk score
        sorted_patterns = sorted(
            self.patterns.items(),
            key=lambda x: x[1].risk_score,
            reverse=True
        )
        
        return sorted_patterns[0][0]

@dataclass
class RiskProfile:
    """Represents a risk profile for an entity"""
    entity_id: str
    entity_type: str  # INDIVIDUAL, ORGANIZATION
    risk_score: float
    risk_factors: Dict[str, float]
    last_updated: datetime
    review_frequency: int  # Days between reviews
    historical_scores: List[Dict[str, Union[float, datetime]]] = field(default_factory=list)
    last_review: Optional[datetime] = None
    next_review: Optional[datetime] = None
    review_notes: List[AlertNote] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_risk_score(self, new_score: float, reason: str = None):
        """Update risk score and maintain history"""
        self.historical_scores.append({
            'score': self.risk_score,
            'timestamp': self.last_updated
        })
        self.risk_score = new_score
        self.last_updated = datetime.now()
        if reason:
            self.review_notes.append(AlertNote(
                timestamp=datetime.now(),
                author="SYSTEM",
                content=f"Risk score updated to {new_score}: {reason}"
            ))

@dataclass
class AMLCase:
    """Represents a money laundering investigation case"""
    case_id: str
    priority: str  # HIGH, MEDIUM, LOW
    status: str  # OPEN, IN_PROGRESS, CLOSED
    risk_score: float
    description: str
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    assigned_to: Optional[str] = None
    related_transactions: List[Transaction] = field(default_factory=list)
    related_alerts: List[AMLAlert] = field(default_factory=list)
    investigation_notes: List[AlertNote] = field(default_factory=list)
    findings: Optional[str] = None
    resolution: Optional[str] = None
    resolution_time: Optional[datetime] = None
    escalation_history: List[Dict[str, Any]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_related_alert(self, alert: AMLAlert):
        """Add a related alert to the case"""
        if alert not in self.related_alerts:
            self.related_alerts.append(alert)
            self.updated_at = datetime.now()
            self._update_risk_score()

    def add_investigation_note(self, content: str, author: str):
        """Add an investigation note"""
        note = AlertNote(
            timestamp=datetime.now(),
            author=author,
            content=content,
            note_type="INVESTIGATION"
        )
        self.investigation_notes.append(note)
        self.updated_at = datetime.now()

    def escalate(self, reason: str, escalated_by: str):
        """Escalate the case"""
        self.escalation_history.append({
            'timestamp': datetime.now(),
            'reason': reason,
            'escalated_by': escalated_by,
            'previous_status': self.status
        })
        self.status = "ESCALATED"
        self.updated_at = datetime.now()
        self.add_investigation_note(
            f"Case escalated: {reason}",
            escalated_by
        )

    def resolve(self, resolution: str, findings: str, resolved_by: str):
        """Resolve the case"""
        self.status = "CLOSED"
        self.resolution = resolution
        self.findings = findings
        self.resolution_time = datetime.now()
        self.updated_at = datetime.now()
        self.add_investigation_note(
            f"Case resolved: {resolution}\nFindings: {findings}",
            resolved_by
        )

    def _update_risk_score(self):
        """Update case risk score based on related alerts"""
        if self.related_alerts:
            self.risk_score = max(alert.risk_score for alert in self.related_alerts)
