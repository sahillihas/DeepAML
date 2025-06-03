from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List, Union, Literal


class PatternType(Enum):
    """Types of money laundering patterns"""
    STRUCTURING = "structuring"
    LAYERING = "layering"
    RAPID_MOVEMENT = "rapid_movement"
    FAN_IN = "fan_in"
    FAN_OUT = "fan_out"
    UNKNOWN = "unknown"  # Added to handle missing patterns safely


class RiskLevel(Enum):
    """Risk levels for accounts and transactions"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


NoteType = Literal["GENERAL", "INVESTIGATION", "RESOLUTION", "STATUS_CHANGE"]
EntityType = Literal["INDIVIDUAL", "ORGANIZATION"]
CasePriority = Literal["LOW", "MEDIUM", "HIGH"]
CaseStatus = Literal["OPEN", "IN_PROGRESS", "CLOSED", "ESCALATED"]


@dataclass
class Transaction:
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
    pattern_type: PatternType
    risk_score: float
    severity: str  # You could enforce Literal["LOW", "MEDIUM", "HIGH"] here too
    details: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    detection_time: datetime = field(default_factory=datetime.now)


@dataclass
class AlertNote:
    timestamp: datetime
    author: str
    content: str
    note_type: NoteType = "GENERAL"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AMLAlert:
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
        self.patterns[pattern.pattern_type.value] = pattern
        self.updated_at = datetime.now()
        self._update_pattern_summary()

    def add_note(self, content: str, author: str, note_type: NoteType = "GENERAL"):
        self.notes.append(AlertNote(
            timestamp=datetime.now(),
            author=author,
            content=content,
            note_type=note_type
        ))
        self.updated_at = datetime.now()

    def update_status(self, new_status: str, reason: Optional[str] = None):
        self.status = new_status
        self.updated_at = datetime.now()
        if reason:
            self.add_note(
                f"Status changed to {new_status}: {reason}",
                "SYSTEM",
                "STATUS_CHANGE"
            )

    def _update_pattern_summary(self):
        high = sum(1 for p in self.patterns.values() if p.severity.upper() == "HIGH")
        medium = sum(1 for p in self.patterns.values() if p.severity.upper() == "MEDIUM")
        self.pattern_summary = {
            "high_risk_patterns": high,
            "medium_risk_patterns": medium,
            "total_patterns": len(self.patterns),
            "primary_pattern": self._get_primary_pattern()
        }

    def _get_primary_pattern(self) -> str:
        if not self.patterns:
            return PatternType.UNKNOWN.value
        sorted_patterns = sorted(
            self.patterns.items(),
            key=lambda x: x[1].risk_score,
            reverse=True
        )
        return sorted_patterns[0][0]


@dataclass
class RiskProfile:
    entity_id: str
    entity_type: EntityType
    risk_score: float
    risk_factors: Dict[str, float]
    last_updated: datetime
    review_frequency: int
    historical_scores: List[Dict[str, Union[float, datetime]]] = field(default_factory=list)
    last_review: Optional[datetime] = None
    next_review: Optional[datetime] = None
    review_notes: List[AlertNote] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_risk_score(self, new_score: float, reason: Optional[str] = None):
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
    case_id: str
    priority: CasePriority
    status: CaseStatus
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
    escalation_history: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_related_alert(self, alert: AMLAlert):
        if alert not in self.related_alerts:
            self.related_alerts.append(alert)
            self.updated_at = datetime.now()
            self._update_risk_score()

    def add_investigation_note(self, content: str, author: str):
        self.investigation_notes.append(AlertNote(
            timestamp=datetime.now(),
            author=author,
            content=content,
            note_type="INVESTIGATION"
        ))
        self.updated_at = datetime.now()

    def escalate(self, reason: str, escalated_by: str):
        self.escalation_history.append({
            'timestamp': datetime.now(),
            'reason': reason,
            'escalated_by': escalated_by,
            'previous_status': self.status
        })
        self.status = "ESCALATED"
        self.updated_at = datetime.now()
        self.add_investigation_note(f"Case escalated: {reason}", escalated_by)

    def resolve(self, resolution: str, findings: str, resolved_by: str):
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
        if self.related_alerts:
            self.risk_score = max(alert.risk_score for alert in self.related_alerts)
