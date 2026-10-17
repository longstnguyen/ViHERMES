<p align="center"><img src="assets/guide.svg" width="100%" alt="ViHERMES research code and data"></p>

# Architecture

[↑ Overview](../README.md) · [Usage](usage.md) · [Dataset](../dataset/README.md)

ViHERMES separates domain objects and policies from persistence and model services. `cli.py` is the composition root: it selects adapters and injects them into the pipeline. The same retrieval and agent code runs against SQLite/NumPy or optional Neo4j/Milvus adapters.

## Contracts

| Layer | Responsibility | Implementation |
| :-- | :-- | :-- |
| Domain | Typed nodes, edges, claims, evidence, and answers | `models.py` |
| Ports | Graph, embedding, dense search, and structured LLM contracts | `ports.py` |
| Repositories | Stable node storage, adjacency, placeholder resolution | `graph.py`, `adapters/neo4j.py` |
| Retrieval strategies | BM25, dense scores, weighted fusion | `retrieval.py`, `adapters/milvus.py` |
| Propagation policy | Typed expansion and context budgeting | `GraphPropagator` |
| Agent orchestration | Interpret → retrieve → draft → audit | `agents.py` |
| Research workflows | Loading, evaluation, generation, and validation | `data.py`, `evaluation.py`, `construction.py` |

## Graph identity and direction

Raw documents use deterministic identifiers:

```text
01/2020/TT-BYT
01/2020/TT-BYT::article:3
01/2020/TT-BYT::article:3::clause:2
```

Imported context CSVs retain their released `context_id` values so retrieval results match gold evidence IDs. Their parent document IDs come from an explicit document number in the title, or a deterministic title hash. Article/clause parsing applies to raw document JSONL; imported context passages retain their supplied boundaries.

| Relation | Direction | Retrieval behavior |
| :-- | :-- | :-- |
| `HAS_ARTICLE` | Document → Article | Follow upstream for provenance |
| `HAS_CLAUSE` | Article/Document → Clause | Follow upstream for provenance |
| `AMENDS` | Affected unit → Updating unit | Recursively trace updates |
| `REPLACES` | Affected unit → Replacing unit | Recursively trace replacements |
| `SUPPLEMENTS` | Affected unit → Supplementing unit | Recursively collect supplements |
| `REFERS_TO` | Citing unit → Referenced unit | Expand one hop |

Validity edges point toward the update: “B amends A” becomes `A → B`. Terminal update nodes receive priority; originals remain available because an amendment may affect only part of a provision. The Auditor checks supplied effective dates, expiry dates, and legal status when assessing applicability.

References to missing units create placeholders. Ingesting the matching ID replaces the placeholder while preserving its edges. Resolved nodes enter the retrievable text corpus once their source text is available. Unresolved references are exposed in the query trace.

Ingestion updates nodes and edges by their stable IDs, so repeated imports are idempotent. Use a new graph database when replacing a corpus that removes or renumbers provisions.

## Retrieval and budgets

BM25 and cosine similarities are calibrated independently into a nonnegative range, then combined:

```text
score = dense_weight × dense_score + (1 − dense_weight) × bm25_score
```

A small ranking hint promotes explicitly named document IDs without filtering out related documents. The graph policy bounds seed count, validity depth, discovered nodes, final nodes, and text characters. References expand by one hop. Structural links supply parent-document metadata for both seeds and referenced evidence; parent documents are serialized as metadata.

NumPy indexes record the corpus fingerprint, ordered IDs, and embedding model identity. A changed corpus or model requires rebuilding the index. Milvus uses separate collection generations keyed by corpus/model identity; its exact fusion adapter supports up to 16,384 units per collection.

## Grounding and abstention

Generated answers consist of individual claims, source IDs, and exact supporting quotations. The Auditor first checks source membership, quoted spans, known status/dates, and replacement conflicts. In generative mode, an additional structured LLM call checks semantic entailment and amendment consistency.

Failed grounding returns an abstention. Offline extractive mode returns source quotations for local pipeline checks. Full generative runs can use a different verifier by injecting another `LanguageModel` into `Auditor`.

## Adding a backend

Implement a protocol in `ports.py`, then supply the instance to the domain service. Keep provider credentials in environment variables. Provider errors propagate to the caller for diagnosis. Evaluation retains completed predictions when a provider call fails.

```python
from vihermes.agents import Auditor, Conductor, Interpreter, Pathfinder
from vihermes.retrieval import GraphPropagator, HybridRetriever

# graph, dense, and model implement the protocols in ports.py.
retriever = HybridRetriever(graph, dense, weight=0.5)
pathfinder = Pathfinder(retriever, GraphPropagator(graph), k=5)
engine = Conductor(Interpreter(model), pathfinder, Auditor(model), model)
answer = engine.ask("Hồ sơ đăng ký gồm những gì?")
```
