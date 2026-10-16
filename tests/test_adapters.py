"""Optional adapter contract tests; service tests run only with explicit test endpoints."""

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from vihermes.models import Edge, Node


def test_openai_sdk_against_local_endpoint(monkeypatch):
    pytest.importorskip("openai")
    from vihermes.adapters.models import OpenAIEmbedder, OpenAILanguageModel

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path.endswith("/embeddings"):
                result = {
                    "object": "list",
                    "data": [
                        {"object": "embedding", "index": i, "embedding": [float(i), 1.0, 0.0]}
                        for i, _ in enumerate(data["input"])
                    ],
                    "model": data["model"],
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                }
            else:
                assert data["response_format"]["type"] == "json_object"
                result = {
                    "id": "test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": data["model"],
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": '{"supported": true}'},
                        }
                    ],
                }
            body = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("OPENAI_API_KEY", "local-test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1")
    try:
        assert OpenAILanguageModel().json("Return JSON", {"question": "test"}) == {
            "supported": True
        }
        embeddings = OpenAIEmbedder(batch_size=1).encode(["a", "b"])
        assert embeddings.shape == (2, 3)
        assert np.isfinite(embeddings).all()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_milvus_lite_roundtrip(tmp_path):
    pytest.importorskip("pymilvus")
    pytest.importorskip("milvus_lite")
    from vihermes.adapters.milvus import MilvusDenseIndex
    from vihermes.adapters.models import HashEmbedder

    nodes = [Node("a", "doc", text="hồ sơ bệnh án"), Node("b", "doc", text="học phí")]
    index = MilvusDenseIndex(nodes, HashEmbedder(), str(tmp_path / "milvus.db"))
    try:
        scores = index.scores("bệnh án", ["a", "b"])
        assert scores[0] > scores[1]
        assert len(index.ids) == 2
    finally:
        index.close()


@pytest.mark.skipif(
    not os.environ.get("VIHERMES_TEST_NEO4J_URI"),
    reason="isolated Neo4j test service not configured",
)
def test_neo4j_store_contract():
    from vihermes.adapters.neo4j import Neo4jGraph

    graph = Neo4jGraph(
        os.environ["VIHERMES_TEST_NEO4J_URI"], "neo4j", os.environ["VIHERMES_TEST_NEO4J_PASSWORD"]
    )
    try:
        graph.upsert_edge(Edge("test::old", "test::new", "REPLACES"))
        assert graph.get("test::old").placeholder
        graph.upsert_node(Node("test::old", "test", text="Old source"))
        graph.upsert_node(Node("test::old", "test", placeholder=True))
        assert not graph.get("test::old").placeholder
        assert graph.edges("test::new", incoming=True)[0].source == "test::old"
        assert any(n.id == "test::old" for n in graph.nodes())
    finally:
        graph._query(
            "MATCH (n:ViHERMES) WHERE n.id IN $ids DETACH DELETE n", ids=["test::old", "test::new"]
        )
        graph.close()
