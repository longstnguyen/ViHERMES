# ViHERMES dataset

[↑ Dataset overview](../README.md#dataset) · [Usage guide](../docs/usage.md) · [Citation](../README.md#citation)

This directory preserves the released question–answer records in [JSONL](dataset.jsonl) and [CSV](dataset.csv). Both formats contain questions, reference answers, reasoning, evidence, hop levels, and the source context blocks.

| Field | Meaning |
| :-- | :-- |
| `id` | Original generation-local question identifier |
| `question`, `answer` | Vietnamese question and reference answer |
| `hop_level` | Number of supporting contexts |
| `used_context_ids` | IDs of the supporting regulatory passages |
| `evidence` | Extractive supporting spans |
| `reasoning` | Explanation connecting the evidence |
| `context_block` | Context IDs, titles, and passage text |

The bundled records comprise 312 examples at each of hops 1–4 and 313 at hop 5: **1,561 total**. The paper's statistics describe a 1,560-example test set. The loader preserves every bundled row and exposes stable evaluation IDs plus the original ID and row number, so repeated generation-local IDs cannot overwrite predictions.

```bash
vihermes audit-data --qa dataset/dataset.jsonl
vihermes extract-contexts --qa dataset/dataset.jsonl \
  --output outputs/contexts.csv
```

Context extraction yields **998 unique passages** and verifies that repeated context IDs have identical titles and text. It reads only the source context blocks; reference answers and reasoning are not inserted into the retrieval corpus.

If you use this dataset, please cite the [ViHERMES paper](../README.md#citation). Regulatory document titles and context IDs provide provenance for the supplied passages.
