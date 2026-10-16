"""Deterministic legal-unit parsing and explicit, rule-based dependency extraction."""

import csv
import hashlib
import json
import re
import unicodedata
from pathlib import Path

from .models import Edge, Node
from .ports import GraphStore

DOCUMENT_CODE = r"\d{1,4}/\d{4}/[A-ZĐ][A-ZĐ0-9-]*"
ARTICLE = re.compile(r"(?m)^\s*Điều\s+(\d+[a-z]?)\s*[.:]\s*", re.I)
CLAUSE = re.compile(r"(?m)^\s*(\d+)\.\s+")
# Explicit cross-document references only. Unqualified references resolve within the source doc.
REFERENCE = re.compile(
    r"(?P<action>sửa đổi(?:,?\s*bổ sung)?|bổ sung|thay thế|căn cứ|theo|quy định tại)\s+"
    r"(?:(?:khoản\s+(?P<clause>\d+)\s+)?điều\s+(?P<article>\d+[a-z]?)\s*)?"
    r"(?:(?:của\s+)?(?:Luật|Nghị định|Thông tư|Quyết định|Nghị quyết)\s*(?:số\s*)?"
    r"(?P<doc>" + DOCUMENT_CODE + r"))?",
    re.I,
)


def normalize(text):
    return unicodedata.normalize("NFC", text).replace("\r\n", "\n").strip()


def document_id(title):
    match = re.search(DOCUMENT_CODE, title, re.I)
    return (
        match.group().upper()
        if match
        else "doc-" + hashlib.sha256(normalize(title).encode()).hexdigest()[:16]
    )


def references(source: Node):
    for match in REFERENCE.finditer(source.text):
        if not (match["article"] or match["doc"]):
            continue
        doc = (match["doc"] or source.document_id).upper()
        if not match["doc"]:
            doc = source.document_id
        target = doc
        if match["article"]:
            target += "::article:" + match["article"].lower()
        if match["clause"]:
            target += "::clause:" + match["clause"]
        action = match["action"].lower()
        relation = (
            "AMENDS"
            if action.startswith("sửa đổi")
            else "SUPPLEMENTS"
            if action == "bổ sung"
            else "REPLACES"
            if action == "thay thế"
            else "REFERS_TO"
        )
        if source.id == target:
            continue
        # Validity edges point from the affected unit to the updating unit.
        if relation == "REFERS_TO":
            yield Edge(source.id, target, relation)
        else:
            yield Edge(target, source.id, relation)


def parse_document(doc_id: str, title: str, text: str, metadata=None):
    if not doc_id or not title or not text.strip():
        raise ValueError("Document requires id, title, and non-empty text")
    text = normalize(text)
    metadata = dict(metadata or {})
    nodes = [Node(doc_id, doc_id, "Document", title, text, metadata)]
    edges = []
    articles = list(ARTICLE.finditer(text))
    if not articles:
        nodes.append(Node(doc_id + "::body", doc_id, "Clause", title, text, metadata))
        edges.append(Edge(doc_id, nodes[-1].id, "HAS_CLAUSE"))
    for i, match in enumerate(articles):
        end = articles[i + 1].start() if i + 1 < len(articles) else len(text)
        body = text[match.start() : end].strip()
        aid = doc_id + "::article:" + match.group(1).lower()
        nodes.append(Node(aid, doc_id, "Article", title, body, metadata))
        edges.append(Edge(doc_id, aid, "HAS_ARTICLE"))
        clauses = list(CLAUSE.finditer(body))
        for j, clause in enumerate(clauses):
            stop = clauses[j + 1].start() if j + 1 < len(clauses) else len(body)
            cid = aid + "::clause:" + clause.group(1)
            nodes.append(
                Node(cid, doc_id, "Clause", title, body[clause.start() : stop].strip(), metadata)
            )
            edges.append(Edge(aid, cid, "HAS_CLAUSE"))
    if len({n.id for n in nodes}) != len(nodes):
        raise ValueError(f"Duplicate article or clause numbers in {doc_id}")
    # Article text includes its clauses: extract at the most specific available level.
    parents = {e.source for e in edges}
    for node in nodes:
        if node.kind != "Document":
            scan = node
            if node.id in parents:
                first = CLAUSE.search(node.text)
                if first:
                    scan = Node(
                        node.id,
                        node.document_id,
                        node.kind,
                        node.title,
                        node.text[: first.start()],
                        node.metadata,
                    )
            edges.extend(references(scan))
    return nodes, list(dict.fromkeys(edges))


def ingest(path: str | Path, graph: GraphStore):
    """Accept context CSV (context_id,title,text) or document JSONL (id,title,text)."""
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = (
            list(csv.DictReader(f))
            if path.suffix == ".csv"
            else [json.loads(line) for line in f if line.strip()]
        )
    count = 0
    # Validate and parse everything before committing any record.
    parsed = []
    for row in rows:
        if "context_id" in row:
            title, text, key = row["title"], normalize(row["text"]), row["context_id"]
            if not title or not text or not key:
                raise ValueError("Context requires context_id, title, and text")
            did = document_id(title)
            n = Node(key, did, title=title, text=text)
            parsed.append(
                (
                    [Node(did, did, "Document", title), n],
                    [Edge(did, key, "HAS_CLAUSE"), *references(n)],
                )
            )
        else:
            parsed.append(parse_document(row["id"], row["title"], row["text"], row.get("metadata")))
    for nodes, edges in parsed:
        for node in nodes:
            # Context imports add provenance without erasing richer raw-document metadata.
            if node.kind == "Document" and not node.text:
                existing = graph.get(node.id)
                if existing is not None and not existing.placeholder:
                    continue
            graph.upsert_node(node)
        for edge in edges:
            graph.upsert_edge(edge)
        count += 1
    return {
        "records": count,
        "nodes": len(graph.nodes()),
        "placeholders": sum(n.placeholder for n in graph.nodes()),
    }
