import json

from kgrag_eval.fixture import FIXTURE_DIR, run_fixture


def test_genesis_benchmark_matches_the_committed_baseline():
    # Reruns resolution, graph build and every scorer on the committed Genesis fixture.
    # A failure means a metric moved: if the change was intended, run `kgrag-eval fixture`.
    assert run_fixture() == json.loads(
        (FIXTURE_DIR / "expected_report.json").read_text(encoding="utf-8")
    )
