"""OpenAI-compatible and Sentence Transformers adapters, plus an offline fixture encoder."""

import hashlib
import json

import numpy as np

from ..retrieval import tokens


class HashEmbedder:
    """Deterministic lexical hashing for smoke tests, not neural semantic retrieval."""

    identity = "fixture-hash-256-v1"

    def encode(self, texts):
        result = np.zeros((len(texts), 256), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in tokens(text):
                digest = hashlib.sha256(word.encode()).digest()
                result[row, int.from_bytes(digest[:4], "little") % 256] += 1
        return result


class OpenAIEmbedder:
    def __init__(self, model="text-embedding-3-small", batch_size=64):
        from openai import OpenAI

        self.client, self.model, self.batch_size = OpenAI(), model, batch_size
        self.identity = "openai:" + str(self.client.base_url) + ":" + model

    def encode(self, texts):
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            result = self.client.embeddings.create(
                model=self.model, input=texts[start : start + self.batch_size]
            )
            vectors.extend(item.embedding for item in sorted(result.data, key=lambda x: x.index))
        return np.asarray(vectors, dtype=np.float32)


class SentenceEmbedder:
    def __init__(self, model, batch_size=32):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model)
        self.identity, self.batch_size = "sentence-transformers:" + model, batch_size

    def encode(self, texts):
        return self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True)


class OpenAILanguageModel:
    def __init__(self, model="gpt-4o-mini", temperature=0):
        from openai import OpenAI

        self.client, self.model, self.temperature = OpenAI(), model, temperature

    def json(self, system, payload):
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
        result = json.loads(response.choices[0].message.content or "{}")
        if not isinstance(result, dict):
            raise ValueError("Expected a JSON object from the language model")
        return result
