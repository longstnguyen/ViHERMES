"""Neo4j GraphStore adapter with the same ID and placeholder semantics as SQLite."""

import json
from dataclasses import asdict

from ..models import Edge, Node


class Neo4jGraph:
    def __init__(self, uri, user, password, database="neo4j"):
        from neo4j import GraphDatabase

        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self._query(
            "CREATE CONSTRAINT vihermes_id IF NOT EXISTS FOR (n:ViHERMES) REQUIRE n.id IS UNIQUE"
        )

    def _query(self, query, **params):
        return self.driver.execute_query(query, parameters_=params, database_=self.database).records

    def upsert_node(self, node):
        self._query(
            """MERGE (n:ViHERMES {id:$id})
        WITH n WHERE n.placeholder IS NULL OR n.placeholder OR NOT $placeholder
        SET n.payload=$payload, n.placeholder=$placeholder""",
            id=node.id,
            placeholder=node.placeholder,
            payload=json.dumps(asdict(node), ensure_ascii=False),
        )

    def upsert_edge(self, edge):
        for key in (edge.source, edge.target):
            doc = key.split("::")[0]
            kind = "Document" if key == doc else ("Clause" if "::clause:" in key else "Article")
            self.upsert_node(Node(key, doc, kind=kind, placeholder=True))
        # Edge validates the relation against a closed vocabulary before interpolation.
        Edge(edge.source, edge.target, edge.relation)
        self._query(
            f"""MATCH (a:ViHERMES {{id:$source}}), (b:ViHERMES {{id:$target}})
        MERGE (a)-[:{edge.relation}]->(b)""",
            source=edge.source,
            target=edge.target,
        )

    def get(self, node_id):
        rows = self._query("MATCH (n:ViHERMES {id:$id}) RETURN n.payload AS payload", id=node_id)
        return Node(**json.loads(rows[0]["payload"])) if rows else None

    def nodes(self):
        return [
            Node(**json.loads(r["payload"]))
            for r in self._query("MATCH (n:ViHERMES) RETURN n.payload AS payload ORDER BY n.id")
        ]

    def edges(self, node_id, incoming=False):
        condition = "b.id=$id" if incoming else "a.id=$id"
        return [
            Edge(r["source"], r["target"], r["relation"])
            for r in self._query(
                f"""MATCH (a:ViHERMES)-[e]->(b:ViHERMES) WHERE {condition}
            RETURN a.id AS source, b.id AS target, type(e) AS relation
            ORDER BY relation, source, target""",
                id=node_id,
            )
        ]

    def close(self):
        self.driver.close()
