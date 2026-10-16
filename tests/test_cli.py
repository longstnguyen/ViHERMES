import json
import subprocess
import sys


def run(*args):
    return subprocess.run(
        [sys.executable, "-m", "vihermes", *map(str, args)], capture_output=True, text=True
    )


def test_demo_and_cli_workflow(tmp_path):
    demo = run("demo", "--output", tmp_path / "demo")
    assert demo.returncode == 0, demo.stderr
    assert not json.loads(demo.stdout)["abstained"]
    docs = tmp_path / "docs.jsonl"
    docs.write_text(
        json.dumps({"id": "D", "title": "Thông tư", "text": "Điều 1. Hồ sơ gồm bệnh án."})
    )
    db = tmp_path / "graph.db"
    assert run("ingest", "--input", docs, "--db", db).returncode == 0
    result = run(
        "ask", "--db", db, "--dense-weight", 0, "--generator", "extractive", "Hồ sơ gồm gì?"
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["sources"]
    bad = run("retrieve", "--db", db, "--dense-weight", 2, "question")
    assert bad.returncode != 0
