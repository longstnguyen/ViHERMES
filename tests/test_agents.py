from vihermes.agents import Auditor, Conductor, Interpreter, Pathfinder
from vihermes.graph import SQLiteGraph
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
            engine = Conductor(Interpreter(llm), finder, Auditor(llm), llm)
            answer, retrieval = engine.run("Hồ sơ cần gì?")
            assert len(llm.calls) == 3
            assert answer.abstained is not grounded
            assert retrieval.evidence[0].node.id == "c1"
            assert bool(answer.sources) is grounded
