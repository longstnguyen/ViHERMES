"""BM25 + dense seed ranking, followed by bounded relation-aware propagation."""

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict, deque
from dataclasses import asdict

import numpy as np

from .models import STRUCTURAL, VALIDITY, Evidence, Retrieval
from .ports import DenseSearch, Embedder, GraphStore


def tokens(text):
    return re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold())


def units(graph):
    return [n for n in graph.nodes() if not n.placeholder and n.kind != "Document" and n.text]


def fingerprint(nodes):
    return hashlib.sha256(
        json.dumps([asdict(n) for n in nodes], sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def normalize_vectors(vectors):
    a = np.asarray(vectors, dtype=np.float32)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError("Embeddings must be a finite 2D matrix")
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


class NumpyDenseIndex:
    def __init__(self, nodes, embedder: Embedder, path=None):
        self.embedder = embedder
        self.ids = [n.id for n in nodes]
        self.positions = {key: i for i, key in enumerate(self.ids)}
        signature = fingerprint(nodes)
        if path is not None and path.exists():
            with np.load(path, allow_pickle=False) as saved:
                if (
                    str(saved["fingerprint"]) != signature
                    or str(saved["embedder"]) != embedder.identity
                ):
                    raise ValueError(
                        "Dense index does not match this corpus/model; rebuild with index --force"
                    )
                if saved["ids"].tolist() != self.ids:
                    raise ValueError("Dense index IDs do not match the corpus")
                self.matrix = saved["vectors"]
        else:
            if not nodes:
                raise ValueError("No legal units to embed")
            self.matrix = normalize_vectors(embedder.encode([n.text for n in nodes]))
            if self.matrix.shape[0] != len(nodes):
                raise ValueError("Embedder returned the wrong number of vectors")
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("wb") as stream:
                    np.savez_compressed(
                        stream,
                        vectors=self.matrix,
                        ids=np.array(self.ids),
                        fingerprint=signature,
                        embedder=embedder.identity,
                    )

    def scores(self, query, ids):
        q = normalize_vectors(self.embedder.encode([query]))
        values = self.matrix @ q[0]
        return np.array([values[self.positions[key]] for key in ids])


class BM25:
    def __init__(self, nodes, k1=1.5, b=0.75):
        self.ids = [n.id for n in nodes]
        self.k1, self.b = k1, b
        self.lengths = np.array([len(tokens(n.text)) for n in nodes], dtype=float)
        self.avg = max(float(self.lengths.mean()) if nodes else 0, 1)
        self.postings = defaultdict(list)
        for i, node in enumerate(nodes):
            for term, count in Counter(tokens(node.text)).items():
                self.postings[term].append((i, count))

    def scores(self, query):
        result = np.zeros(len(self.ids))
        for term in set(tokens(query)):
            postings = self.postings.get(term, [])
            idf = math.log(1 + (len(self.ids) - len(postings) + 0.5) / (len(postings) + 0.5))
            for i, count in postings:
                result[i] += (
                    idf
                    * count
                    * (self.k1 + 1)
                    / (count + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg))
                )
        return result


def calibrate(scores):
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return scores
    # Positive scores retain signal even when all candidates tie.
    low, high = min(0.0, float(scores.min())), float(scores.max())
    return (scores - low) / (high - low) if high > low else np.zeros_like(scores)


class HybridRetriever:
    def __init__(self, graph: GraphStore, dense: DenseSearch | None = None, weight=0.5):
        if not 0 <= weight <= 1:
            raise ValueError("Dense weight must be between 0 and 1")
        if weight > 0 and dense is None:
            raise ValueError("Hybrid retrieval needs a dense index; use weight=0 for BM25")
        self.graph, self.dense, self.weight = graph, dense, weight
        self.nodes = units(graph)
        self.bm25 = BM25(self.nodes)

    def seeds(self, query, k=5, document_ids=()):
        if k < 1:
            raise ValueError("k must be positive")
        sparse = calibrate(self.bm25.scores(query))
        dense = calibrate(self.dense.scores(query, self.bm25.ids)) if self.dense else 0
        scores = (1 - self.weight) * sparse + self.weight * dense
        # Explicit document IDs are ranking hints, not a hard filter on related regulations.
        for i, node in enumerate(self.nodes):
            if node.document_id in document_ids:
                scores[i] += 0.1
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], self.nodes[i].id))
        return [Evidence(self.nodes[i], float(scores[i])) for i in order[:k] if scores[i] > 0]


class GraphPropagator:
    def __init__(self, graph: GraphStore, max_depth=4, max_nodes=24, max_chars=16000):
        if min(max_depth, max_nodes, max_chars) < 1:
            raise ValueError("Propagation limits must be positive")
        self.graph = graph
        self.max_depth, self.max_nodes, self.max_chars = max_depth, max_nodes, max_chars

    def expand(self, seeds):
        chosen = {e.node.id: Evidence(e.node, e.score, e.origin, e.terminal) for e in seeds}
        edge_set, warnings = set(), set()
        # Bound discovery as well as final output; no unbounded high-degree graph walk.
        discovery_limit = max(self.max_nodes, len(chosen)) * 4
        queue = deque((e.node.id, 0, e.score) for e in seeds)
        seen = set()

        def add(key, score, origin):
            if key in chosen:
                if origin == "validity":
                    chosen[key].origin = "validity"
                return True
            if len(chosen) >= discovery_limit:
                warnings.add("graph_discovery_budget")
                return False
            node = self.graph.get(key)
            if node is None or node.placeholder:
                warnings.add("unresolved_reference:" + key)
                return False
            chosen[key] = Evidence(node, score, origin)
            return True

        while queue:
            key, depth, score = queue.popleft()
            if key in seen:
                continue
            seen.add(key)
            outgoing = self.graph.edges(key)
            changes = [e for e in outgoing if e.relation in VALIDITY]
            chosen[key].terminal = not changes
            for edge in changes:
                if depth >= self.max_depth:
                    warnings.add("validity_depth_limit:" + key)
                    continue
                edge_set.add(edge)
                if add(edge.target, score, "validity"):
                    queue.append((edge.target, depth + 1, score))
            # Reference edges are expanded once from seeds and validity-chain nodes.
            for edge in outgoing:
                if edge.relation == "REFERS_TO":
                    edge_set.add(edge)
                    add(edge.target, score * 0.8, "reference")
            # Upstream hierarchy also exposes document-level validity links.
            for edge in self.graph.edges(key, incoming=True):
                if edge.relation in STRUCTURAL:
                    edge_set.add(edge)
                    if add(edge.source, score * 0.7, "provenance"):
                        queue.append((edge.source, depth, score * 0.7))

        # Reference neighbors also need provenance, without expanding their legal edges.
        hierarchy = deque(chosen)
        traced = set()
        while hierarchy:
            key = hierarchy.popleft()
            if key in traced:
                continue
            traced.add(key)
            for edge in self.graph.edges(key, incoming=True):
                if edge.relation in STRUCTURAL:
                    edge_set.add(edge)
                    if add(edge.source, chosen[key].score * 0.7, "provenance"):
                        hierarchy.append(edge.source)

        # Detect actual cycles, not shared paths reached from multiple seeds.
        adjacency = defaultdict(set)
        degrees = {key: 0 for key in chosen}
        for edge in edge_set:
            if edge.relation in VALIDITY and edge.source in chosen and edge.target in chosen:
                if edge.target not in adjacency[edge.source]:
                    adjacency[edge.source].add(edge.target)
                    degrees[edge.target] += 1
        ready = deque(key for key, degree in degrees.items() if degree == 0)
        while ready:
            key = ready.popleft()
            degrees.pop(key)
            for target in adjacency[key]:
                degrees[target] -= 1
                if degrees[target] == 0:
                    ready.append(target)
        if degrees:
            warnings.add("validity_cycle:" + ",".join(sorted(degrees)))

        ordered = sorted(
            chosen.values(),
            key=lambda e: (
                e.node.kind == "Document",
                not (e.origin == "validity" and e.terminal),
                -e.score,
                e.node.id,
            ),
        )
        final, used = [], 0
        for item in ordered:
            # Provenance carries metadata only; avoid re-including the entire parent document.
            size = len(item.node.text) if item.node.kind != "Document" else 0
            if len(final) >= self.max_nodes or used + size > self.max_chars:
                warnings.add("context_budget")
                continue
            used += size
            final.append(item)
        return Retrieval(
            final,
            sorted(edge_set, key=lambda e: (e.source, e.target, e.relation)),
            sorted(warnings),
        )
