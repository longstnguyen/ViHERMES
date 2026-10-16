"""Domain objects shared by storage, retrieval, agents, and evaluation."""

from dataclasses import dataclass, field
from typing import Literal

STRUCTURAL = frozenset({"HAS_ARTICLE", "HAS_CLAUSE"})
VALIDITY = frozenset({"AMENDS", "REPLACES", "SUPPLEMENTS"})
RELATIONS = STRUCTURAL | VALIDITY | {"REFERS_TO"}


@dataclass(frozen=True)
class Node:
    id: str
    document_id: str
    kind: Literal["Document", "Article", "Clause"] = "Clause"
    title: str = ""
    text: str = ""
    metadata: dict = field(default_factory=dict)
    placeholder: bool = False

    def __post_init__(self):
        if not self.id or not self.document_id:
            raise ValueError("Node identifiers must be non-empty")
        if self.kind not in {"Document", "Article", "Clause"}:
            raise ValueError(f"Unknown node kind: {self.kind}")


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    relation: str

    def __post_init__(self):
        if self.relation not in RELATIONS:
            raise ValueError(f"Unknown relation: {self.relation}")


@dataclass
class Evidence:
    node: Node
    score: float
    origin: str = "seed"
    terminal: bool = False


@dataclass
class Retrieval:
    evidence: list[Evidence] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class Plan:
    query: str
    regulatory: bool = True
    use_graph: bool = True
    document_ids: list[str] = field(default_factory=list)


@dataclass
class Claim:
    text: str
    citations: list[str]
    quotes: dict[str, str]


@dataclass
class Answer:
    question: str
    answer: str
    claims: list[Claim]
    sources: list[dict]
    abstained: bool = False
    reasons: list[str] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)


@dataclass
class QAExample:
    id: str
    question: str
    answer: str
    gold_context_ids: list[str]
    hop_level: int = 1
    metadata: dict = field(default_factory=dict)
