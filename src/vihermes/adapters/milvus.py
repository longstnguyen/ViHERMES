"""Milvus vector ranking; sparse scores and graph policy stay in the core."""

import hashlib

import numpy as np

from ..retrieval import fingerprint, normalize_vectors


class MilvusDenseIndex:
    def __init__(self, nodes, embedder, uri, token="", collection="vihermes", batch_size=128):
        from pymilvus import MilvusClient

        self.client = MilvusClient(uri=uri, token=token)
        self.embedder = embedder
        self.ids = [n.id for n in nodes]
        if not nodes:
            raise ValueError("No legal units to index")
        signature = hashlib.sha256((fingerprint(nodes) + embedder.identity).encode()).hexdigest()[
            :16
        ]
        # Immutable corpus/model generations prevent stale vectors after graph edits.
        self.collection = collection + "_" + signature
        if not self.client.has_collection(self.collection):
            vectors = normalize_vectors(embedder.encode([n.text for n in nodes]))
            self.client.create_collection(
                collection_name=self.collection,
                dimension=vectors.shape[1],
                primary_field_name="id",
                id_type="string",
                max_length=1024,
                metric_type="COSINE",
                consistency_level="Strong",
            )
            for start in range(0, len(nodes), batch_size):
                self.client.upsert(
                    collection_name=self.collection,
                    data=[
                        {"id": nodes[i].id, "vector": vectors[i].tolist()}
                        for i in range(start, min(start + batch_size, len(nodes)))
                    ],
                )
        count = self.client.query(
            collection_name=self.collection, filter="", output_fields=["count(*)"]
        )
        if count[0]["count(*)"] != len(nodes):
            raise ValueError(
                "Incomplete Milvus index generation; rebuild this collection before querying"
            )

    def scores(self, query, ids):
        # Milvus limits a single search window; use its full supported candidate window.
        if len(self.ids) > 16384:
            raise ValueError(
                "Exact hybrid fusion supports at most 16384 Milvus units per collection"
            )
        vector = normalize_vectors(self.embedder.encode([query]))[0].tolist()
        hits = self.client.search(
            collection_name=self.collection,
            data=[vector],
            limit=len(self.ids),
            search_params={"metric_type": "COSINE"},
        )[0]
        scores = {str(hit["id"]): float(hit["distance"]) for hit in hits}
        return np.array([scores.get(key, 0.0) for key in ids])

    def close(self):
        self.client.close()
