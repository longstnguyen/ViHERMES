<p align="center"><img src="assets/guide.svg" width="100%" alt="ViHERMES research code and data"></p>

# Usage guide

[↑ Overview](../README.md) · [Architecture](architecture.md) · [Dataset](../dataset/README.md)

Run commands from the repository root. Every command exposes `--help`. Generated artifacts belong in `outputs/`, which is excluded from Git.

## Ingest documents

Use UTF-8 JSONL for complete documents. Supply the canonical document number as `id` and retain available provenance in `metadata`:

```json
{"id":"01/2020/TT-BYT","title":"Thông tư 01/2020/TT-BYT","text":"Điều 1. Hồ sơ\n1. Hồ sơ gồm bản điện tử.","metadata":{"source_url":"https://example.org/document","effective_date":"2020-07-01"}}
```

The snippet is a synthetic format example. Supported temporal fields are `effective_date` and `expiry_date` in `YYYY-MM-DD` form. Status may include `effective`, `amended`, `draft`, `expired`, or `repealed`. Unknown status/date values should remain absent; publication dates are not automatically treated as effective dates.

```bash
vihermes ingest --input your_documents.jsonl --db outputs/vihermes.db
```

For presegmented corpora, use a CSV with `context_id,title,text` columns:

```bash
vihermes extract-contexts --qa dataset/dataset.jsonl \
  --output outputs/contexts.csv
vihermes ingest --input outputs/contexts.csv --db outputs/vihermes.db
```

## Models and retrieval

### OpenAI-compatible models

```bash
pip install -e '.[openai]'
export OPENAI_API_KEY='your-key'
vihermes index --db outputs/vihermes.db \
  --embedding openai --embedding-model text-embedding-3-small
vihermes ask --db outputs/vihermes.db \
  --model gpt-4o-mini --dense-weight 0.5 \
  'Hồ sơ đăng ký khám chữa bệnh gồm những gì?'
```

`OPENAI_BASE_URL` selects another OpenAI-compatible endpoint. The endpoint must support structured JSON chat responses and the embedding model used for indexing. Keep model settings identical between `index` and subsequent retrieval commands; rebuild with `index --force` after changing the corpus or embedding model.

### Local sentence encoders

```bash
pip install -e '.[local-models]'
vihermes index --embedding sentence-transformers \
  --embedding-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
vihermes retrieve --embedding sentence-transformers \
  --embedding-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
  'Quy định về hồ sơ bệnh án điện tử'
```

### Lexical and extractive modes

`--dense-weight 0` selects BM25 and skips embeddings entirely. `--generator extractive` returns a source quotation without calling an LLM:

```bash
vihermes ask --dense-weight 0 --generator extractive \
  'Quy định về hồ sơ bệnh án điện tử'
```

`--embedding hash` selects a deterministic lexical test encoder for integration checks. Use OpenAI or a sentence encoder for neural semantic retrieval.

### Control context size

| Flag | Default | Meaning |
| :-- | --: | :-- |
| `-k` | 5 | Number of ranked seed units |
| `--dense-weight` | 0.5 | Dense contribution to the hybrid score |
| `--max-depth` | 4 | Maximum validity-chain depth |
| `--max-nodes` | 24 | Maximum nodes in assembled context |
| `--max-chars` | 16000 | Maximum legal-unit text characters |

## External stores

Install adapters with `pip install -e '.[stores]'`. Set `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `MILVUS_URI`, and optionally `MILVUS_TOKEN` in your shell; `.env.example` lists them. The CLI reads exported environment variables and does not auto-load `.env`.

```bash
vihermes ingest --graph neo4j --input your_documents.jsonl
vihermes index --graph neo4j --vector milvus --embedding openai
vihermes ask --graph neo4j --vector milvus --embedding openai \
  'Quy định về hồ sơ bệnh án điện tử'
```

The Neo4j adapter uses the `ViHERMES` node label and an ID uniqueness constraint. Milvus collection names include a corpus/model fingerprint, so separate index generations do not overwrite one another. Provision the services separately and manage storage retention according to your deployment needs.

## Evaluate retrieval and answers

```bash
vihermes evaluate --qa dataset/dataset.jsonl --dense-weight 0 \
  --output outputs/retrieval --ks 1,5,10
```

For generated answers and external judging:

```bash
vihermes evaluate --qa dataset/dataset.jsonl \
  --embedding openai --model gpt-4o-mini \
  --answers --judge --judge-model gpt-4o \
  --output outputs/qa-evaluation --ks 1,5,10
```

Use `--limit 20` for a short run. Omitting it evaluates every record. The output includes macro-averaged Recall@k, token-level F1 when answers are requested, and metrics grouped by hop level. Recall uses the unique gold context IDs and the first k retrieved legal units; provenance document nodes do not occupy the ranking. Queries without gold IDs have a null recall and an explicit denominator count. Failed provider calls stop the run, retaining completed predictions for inspection.

F1 uses Unicode-normalized, case-folded word tokens with multiset overlap; rendered citation markers are excluded. The optional judge scores correctness, completeness, and evidence consistency on 0–1 scales; their mean is the reported judge score. Its prompt is in `evaluation.py`. Use the same rubric, model, corpus, and settings when comparing systems.

## Construct a dataset

Ingest your corpus, then select an embedding backend and generator:

```bash
pip install -e '.[openai,dataset]'
vihermes construct --embedding openai --model gpt-4o-mini \
  --hops 1,2,3,4,5 --per-hop 10 \
  --clusters 20 --per-cluster 3 --overlap 0.5 \
  --test-fraction 0.2 --seed 42 --output outputs/candidates
```

The workflow normalizes embeddings, clusters them, and selects contexts nearest each centroid. It samples tuples with distinct document titles and bounded pairwise context overlap within each hop group. Generated records must use every selected context, include reasoning, and quote exact source spans.

`candidates.jsonl` contains structurally validated records with proposed hop-stratified splits and `review_status: pending`. `rejected.jsonl` records validation failures; `manifest.json` captures the sampling configuration. Annotators and a domain reviewer then verify that the reasoning genuinely requires every context and that the legal interpretation is sound. The proposed split is by QA record; a document-disjoint evaluation requires a separate corpus partition before generation.
