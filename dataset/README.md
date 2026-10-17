<p align="center"><img src="../docs/assets/guide.svg" width="100%" alt="ViHERMES research code and data"></p>

# ViHERMES dataset

[↑ Dataset overview](../README.md#dataset) · [Usage guide](../docs/usage.md) · [Citation](../README.md#citation)

This directory contains the question–answer records in [JSONL](dataset.jsonl) and [CSV](dataset.csv). Both formats contain questions, reference answers, reasoning, evidence, hop levels, and the source context blocks.

| Field | Meaning |
| :-- | :-- |
| `id` | Original generation-local question identifier |
| `question`, `answer` | Vietnamese question and reference answer |
| `hop_level` | Number of supporting contexts |
| `used_context_ids` | IDs of the supporting regulatory passages |
| `evidence` | Extractive supporting spans |
| `reasoning` | Explanation connecting the evidence |
| `context_block` | Context IDs, titles, and passage text |

The release contains **1,561 records**: 312 examples at each of hops 1–4 and 313 at hop 5. The study reports a 1,560-example test set. Evaluation outputs identify each released record by a stable row ID and retain its original question ID and row number.

```bash
vihermes audit-data --qa dataset/dataset.jsonl
vihermes extract-contexts --qa dataset/dataset.jsonl \
  --output outputs/contexts.csv
```

Context extraction yields **998 unique passages** and verifies that repeated context IDs have identical titles and text. The retrieval corpus is built from `context_block`; `answer` and `reasoning` serve as evaluation annotations.

If you use this dataset, please cite the [ViHERMES paper](../README.md#citation). Regulatory document titles and context IDs provide provenance for the supplied passages.
