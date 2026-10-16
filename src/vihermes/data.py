"""Load released QA records without changing source IDs or silently dropping examples."""

import csv
import json
import re
from collections import Counter
from pathlib import Path

from .models import QAExample


def records(path):
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as f:
        if path.suffix == ".csv":
            return list(csv.DictReader(f))
        return [json.loads(line) for line in f if line.strip()]


def load_qa(path):
    rows = records(path)
    counts = Counter(str(r.get("id", "")) for r in rows)
    examples = []
    for i, row in enumerate(rows):
        original = str(row.get("id", ""))
        # Stable row keys preserve duplicate original IDs in metadata for traceability.
        key = original if original and counts[original] == 1 else f"row:{i:06d}:{original}"
        gold = row.get("gold_context_ids", row.get("used_context_ids", []))
        if isinstance(gold, str):
            gold = json.loads(gold)
        if not isinstance(gold, list) or not all(isinstance(x, str) for x in gold):
            raise ValueError(f"Invalid gold context IDs at row {i}")
        if not row.get("question", "").strip():
            raise ValueError(f"Empty question at row {i}")
        examples.append(
            QAExample(
                key,
                row["question"],
                row.get("answer", ""),
                list(dict.fromkeys(gold)),
                int(row.get("hop_level", 1)),
                {"original_id": original, "row": i},
            )
        )
    return examples


def extract_contexts(path, output):
    """Recover the released corpus from context blocks, checking repeated IDs agree."""
    contexts = {}
    pattern = re.compile(
        r"\[CONTEXT\s+\d+\]\s*\nID:\s*([^\n]+)\nTitle:\s*([^\n]+)\nBody:\s*\n(.*?)(?=\n\[CONTEXT\s+\d+\]|\Z)",
        re.S,
    )
    for i, row in enumerate(records(path)):
        matches = list(pattern.finditer(row.get("context_block", "")))
        if not matches:
            raise ValueError(f"No parseable contexts at row {i}")
        for match in matches:
            key, title, text = (v.strip() for v in match.groups())
            value = {"context_id": key, "title": title, "text": text}
            if key in contexts and contexts[key] != value:
                raise ValueError(f"Conflicting source text for context {key}")
            contexts[key] = value
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["context_id", "title", "text"])
        writer.writeheader()
        writer.writerows(contexts.values())
    return len(contexts)
