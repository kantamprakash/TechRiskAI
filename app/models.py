"""Data models for documents and risks.

The validators are deliberately lenient: small local models often return
numbers as strings, lists as comma-separated strings, or slightly wrong
category names. We coerce rather than reject.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field, field_validator

CATEGORIES = [
    "Architecture & Scalability",
    "Security & Compliance",
    "Integration & Dependencies",
    "Data",
    "Technology Stack & Skills",
    "Requirements Quality",
    "Performance & Availability",
    "Delivery & Operations",
]

_CATEGORY_KEYWORDS = {
    "architecture": "Architecture & Scalability",
    "scalab": "Architecture & Scalability",
    "security": "Security & Compliance",
    "complian": "Security & Compliance",
    "privacy": "Security & Compliance",
    "integrat": "Integration & Dependencies",
    "dependen": "Integration & Dependencies",
    "vendor": "Integration & Dependencies",
    "data": "Data",
    "migrat": "Data",
    "stack": "Technology Stack & Skills",
    "skill": "Technology Stack & Skills",
    "technolog": "Technology Stack & Skills",
    "requirement": "Requirements Quality",
    "ambigu": "Requirements Quality",
    "performance": "Performance & Availability",
    "availab": "Performance & Availability",
    "latency": "Performance & Availability",
    "delivery": "Delivery & Operations",
    "operation": "Delivery & Operations",
    "deploy": "Delivery & Operations",
    "timeline": "Delivery & Operations",
}


def severity_for(score: int) -> str:
    """Severity band for likelihood x impact (1..25)."""
    if score >= 17:
        return "Critical"
    if score >= 10:
        return "High"
    if score >= 5:
        return "Medium"
    return "Low"


def _to_list(v) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        parts = [p.strip(" -•*\t") for p in v.replace(";", "\n").splitlines()]
        return [p for p in parts if p]
    if isinstance(v, (list, tuple)):
        return [str(x).strip() for x in v if str(x).strip()]
    return [str(v)]


def _to_score(v) -> int:
    if isinstance(v, str):
        word = v.strip().lower()
        words = {"very low": 1, "low": 2, "medium": 3, "moderate": 3, "high": 4, "very high": 5, "critical": 5}
        if word in words:
            return words[word]
        digits = "".join(ch for ch in word if ch.isdigit())
        v = int(digits[:1]) if digits else 3
    try:
        return min(5, max(1, int(round(float(v)))))
    except (TypeError, ValueError):
        return 3


class RiskItem(BaseModel):
    """One risk as produced by the LLM (after coercion)."""

    title: str
    category: str = "Requirements Quality"
    description: str = ""
    evidence: str = ""
    source_section: str = ""
    likelihood: int = 3
    impact: int = 3
    mitigation: str = ""
    open_questions: list[str] = Field(default_factory=list)
    affected_requirements: list[str] = Field(default_factory=list)
    reference: str = ""

    @field_validator("title", "description", "evidence", "source_section", "mitigation", "reference", mode="before")
    @classmethod
    def _str(cls, v):
        if v is None:
            return ""
        if isinstance(v, list):
            return "\n".join(str(x) for x in v)
        return str(v).strip()

    @field_validator("likelihood", "impact", mode="before")
    @classmethod
    def _score(cls, v):
        return _to_score(v)

    @field_validator("open_questions", "affected_requirements", mode="before")
    @classmethod
    def _list(cls, v):
        return _to_list(v)

    @field_validator("category", mode="before")
    @classmethod
    def _category(cls, v):
        text = str(v or "").strip()
        for c in CATEGORIES:
            if text.lower() == c.lower():
                return c
        low = text.lower()
        for key, cat in _CATEGORY_KEYWORDS.items():
            if key in low:
                return cat
        return "Requirements Quality"


class Risk(RiskItem):
    """A consolidated, scored risk ready for the report."""

    risk_id: str = ""
    source_document: str = ""
    evidence_verified: bool = False
    occurrences: int = 1

    @property
    def score(self) -> int:
        return self.likelihood * self.impact

    @property
    def severity(self) -> str:
        return severity_for(self.score)


@dataclass
class SourceDocument:
    name: str
    doc_type: str  # "BRD" | "SRD" | "Other"
    text: str


@dataclass
class Chunk:
    document: SourceDocument
    index: int
    section: str
    text: str


@dataclass
class ProjectProfile:
    project_name: str = ""
    domain: str = ""
    summary: str = ""
    key_systems: list[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    project_name: str
    profile: ProjectProfile
    risks: list[Risk]
    documents: list[SourceDocument]
    model: str
    provider: str
    chunks_total: int
    chunks_failed: list[str]
    started_at: str
    finished_at: str
    context_tokens: int = 0
    max_output_tokens: int = 0
    chunk_tokens: int = 0
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    models_used: dict = field(default_factory=dict)
    model_switches: list = field(default_factory=list)
    rag_status: str = "disabled"
    embedding_model: str = ""
    rag_pieces_indexed: int = 0
    embedding_calls: int = 0
    knowledge_base_files: int = 0
    raw_risks: int = 0
