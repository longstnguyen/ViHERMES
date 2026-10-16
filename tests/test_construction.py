import json

from vihermes.adapters.models import HashEmbedder
from vihermes.construction import construct
from vihermes.models import Node


class GenerationModel:
    def json(self, system, payload):
        contexts = payload["contexts"]
        return {
            "question": "Kết hợp " + ",".join(n["id"] for n in contexts),
            "answer": "Đáp án",
            "hop_level": len(contexts),
            "reasoning": "Kết hợp các quy định.",
            "used_context_ids": [n["id"] for n in contexts],
            "evidence": {n["id"]: n["text"] for n in contexts},
        }


def test_cluster_generate_validate_and_stratified_split(tmp_path):
    nodes = [
        Node(str(i), str(i), title=f"Document {i}", text=f"Quy định hồ sơ {i}") for i in range(15)
    ]
    report = construct(
        nodes,
        HashEmbedder(),
        GenerationModel(),
        tmp_path,
        hops=(1, 2),
        per_hop=4,
        clusters=5,
        per_cluster=3,
        seed=9,
    )
    assert report["accepted"] == 8 and report["rejected"] == 0
    rows = [json.loads(line) for line in (tmp_path / "candidates.jsonl").read_text().splitlines()]
    for h in (1, 2):
        group = [r for r in rows if r["hop_level"] == h]
        assert {r["split"] for r in group} == {"train", "test"}
    assert all(r["review_status"] == "pending" for r in rows)
