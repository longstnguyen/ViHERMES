import pytest

from vihermes.agents import Auditor, Conductor, Interpreter, Pathfinder
from vihermes.graph import SQLiteGraph
from vihermes.ingestion import parse_document
from vihermes.models import Node
from vihermes.retrieval import GraphPropagator, HybridRetriever


class ScriptedModel:
    def __init__(self, grounded=True):
        self.calls = []
        self.grounded = grounded

    def json(self, system, payload):
        self.calls.append(payload)
        if "Interpreter" in system:
            return {"regulatory": True, "use_graph": True, "document_ids": []}
        if "Conductor" in system:
            return {
                "abstain": False,
                "claims": [
                    {
                        "text": "Cần bản điện tử.",
                        "citations": ["c1"],
                        "quotes": {"c1": "Hồ sơ gồm bản điện tử."},
                    }
                ],
            }
        return {"supported": self.grounded, "reasons": []}


def test_four_role_pipeline_and_semantic_abstention():
    with SQLiteGraph() as graph:
        graph.upsert_node(Node("c1", "doc", title="Quy định", text="Hồ sơ gồm bản điện tử."))
        finder = Pathfinder(HybridRetriever(graph, weight=0), GraphPropagator(graph))
        for grounded in [True, False]:
            llm = ScriptedModel(grounded)
            engine = Conductor(Interpreter(llm), finder, Auditor(llm), llm, repair_attempts=0)
            answer, retrieval = engine.run("Hồ sơ cần gì?")
            assert len(llm.calls) == 3
            assert answer.abstained is not grounded
            assert retrieval.evidence[0].node.id == "c1"
            assert bool(answer.sources) is grounded


@pytest.mark.parametrize("invalid", ["citation", "quote", "structure"])
def test_one_repair_uses_same_sources_and_reaudits(invalid):
    class RepairModel(ScriptedModel):
        def __init__(self):
            super().__init__()
            self.drafts = 0

        def json(self, system, payload):
            result = super().json(system, payload)
            if "Conductor" in system:
                self.drafts += 1
                if self.drafts == 1:
                    if invalid == "structure":
                        return {"abstain": False, "claims": [None]}
                    claim = result["claims"][0]
                    if invalid == "citation":
                        claim.update(citations=["c"], quotes={"c": "Hồ sơ gồm bản điện tử."})
                    else:
                        claim["quotes"]["c1"] = "Hồ sơ bao gồm bản điện tử."
                else:
                    assert payload["validation_errors"]
                    assert payload["evidence"] == self.calls[1]["evidence"]
                    assert payload["previous_draft"]
            return result

    with SQLiteGraph() as graph:
        graph.upsert_node(Node("c1", "doc", text="Hồ sơ gồm bản điện tử."))
        finder = Pathfinder(HybridRetriever(graph, weight=0), GraphPropagator(graph))
        model = RepairModel()
        answer = Conductor(Interpreter(model), finder, Auditor(model), model).ask("Hồ sơ?")
        assert model.drafts == 2
        assert not answer.abstained
        assert answer.claims[0].citations == ["c1"]
        assert answer.sources[0]["text"] == "Hồ sơ gồm bản điện tử."
        assert any("repair 1" in step for step in answer.trace)
        assert model.calls[-1]["claims"][0]["citations"] == ["c1"]


def test_semantic_failure_still_abstains_after_bounded_repair():
    with SQLiteGraph() as graph:
        graph.upsert_node(Node("c1", "doc", text="Hồ sơ gồm bản điện tử."))
        finder = Pathfinder(HybridRetriever(graph, weight=0), GraphPropagator(graph))
        model = ScriptedModel(grounded=False)
        answer = Conductor(Interpreter(model), finder, Auditor(model), model).ask("Hồ sơ?")
        assert answer.abstained
        assert not answer.claims and not answer.sources
        assert "semantic_grounding_failed" in answer.reasons
        assert len(model.calls) == 5  # Route, draft, audit, repair, audit; then stop.


def test_graph_route_traces_amendment_and_preserves_new_evidence():
    class AmendmentModel:
        def json(self, system, payload):
            if "Interpreter" in system:
                return {"regulatory": True, "use_graph": True}
            if "Conductor" in system:
                current = next(e for e in payload["evidence"] if e["id"] == "new::body")
                assert current["terminal"]
                assert any(e["relation"] == "AMENDS" for e in payload["edges"])
                return {
                    "abstain": False,
                    "claims": [
                        {
                            "text": "Hồ sơ gồm bản điện tử.",
                            "citations": [current["id"]],
                            "quotes": {current["id"]: "Hồ sơ gồm bản điện tử."},
                        }
                    ],
                }
            return {"supported": True}

    with SQLiteGraph() as graph:
        for did, text in [
            ("01/2020/TT-BYT", "Điều 2. Hồ sơ gồm bản giấy."),
            ("new", "Sửa đổi Điều 2 của Thông tư 01/2020/TT-BYT. Hồ sơ gồm bản điện tử."),
        ]:
            nodes, edges = parse_document(did, did, text)
            for node in nodes:
                graph.upsert_node(node)
            for edge in edges:
                graph.upsert_edge(edge)
        finder = Pathfinder(HybridRetriever(graph, weight=0), GraphPropagator(graph), k=1)
        model = AmendmentModel()
        answer, retrieval = Conductor(Interpreter(model), finder, Auditor(model), model).run(
            "Hồ sơ bản giấy?"
        )
        assert not answer.abstained
        assert "graph=true" in answer.trace[0]
        assert any(e.relation == "AMENDS" for e in retrieval.edges)


@pytest.mark.parametrize("attempts", [-1, 3, True, 1.5])
def test_invalid_repair_budget(attempts):
    with pytest.raises(ValueError, match="repair_attempts"):
        Conductor(None, None, None, repair_attempts=attempts)
