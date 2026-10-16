"""Persistent local SRKG repository; all writes preserve stable node IDs."""

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

from .models import Edge, Node


class SQLiteGraph:
    def __init__(self, path: str | Path = ":memory:"):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS nodes (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS edges (
          source TEXT NOT NULL, target TEXT NOT NULL, relation TEXT NOT NULL,
          PRIMARY KEY(source, target, relation));
        CREATE INDEX IF NOT EXISTS edge_target ON edges(target);
        """)

    def upsert_node(self, node: Node):
        existing = self.get(node.id)
        if existing and not existing.placeholder and node.placeholder:
            return
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO nodes VALUES (?,?)",
                (node.id, json.dumps(asdict(node), ensure_ascii=False)),
            )

    def upsert_edge(self, edge: Edge):
        for key in (edge.source, edge.target):
            if self.get(key) is None:
                doc = key.split("::")[0]
                kind = "Document" if key == doc else ("Clause" if "::clause:" in key else "Article")
                self.upsert_node(Node(key, doc, kind=kind, placeholder=True))
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO edges VALUES (?,?,?)",
                (edge.source, edge.target, edge.relation),
            )

    def get(self, node_id):
        row = self.db.execute("SELECT payload FROM nodes WHERE id=?", (node_id,)).fetchone()
        return Node(**json.loads(row[0])) if row else None

    def nodes(self):
        return [
            Node(**json.loads(r[0]))
            for r in self.db.execute("SELECT payload FROM nodes ORDER BY id")
        ]

    def edges(self, node_id, incoming=False):
        column = "target" if incoming else "source"
        return [
            Edge(*r)
            for r in self.db.execute(
                f"SELECT source,target,relation FROM edges WHERE {column}=? ORDER BY relation,source,target",
                (node_id,),
            )
        ]

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
