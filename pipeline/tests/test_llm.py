import json

import pytest
from pydantic import BaseModel

from kgrag.llm import (
    Budget,
    BudgetExceeded,
    Ledger,
    LLMCallError,
    LLMClient,
    ModelConfig,
    claude_code_command,
    claude_code_transport,
    default_transport,
    openai_compatible_transport,
)


class Answer(BaseModel):
    word: str


def fake_response(
    structured_output: dict | None, cost_usd: float = 0.01, is_error: bool = False
) -> dict:
    return {
        "is_error": is_error,
        "result": "usage limit reached" if is_error else "",
        "structured_output": structured_output,
        "total_cost_usd": cost_usd,
        "duration_ms": 1200,
        "usage": {
            "input_tokens": 5,
            "cache_creation_input_tokens": 100,
            "cache_read_input_tokens": 0,
            "output_tokens": 40,
            "output_tokens_details": {"thinking_tokens": 10},
        },
    }


def test_claude_code_command_isolates_user_settings_and_pins_the_model():
    command = claude_code_command(
        ModelConfig("claude-haiku-5-5", "low"), "Be brief.", {"type": "object"}, 0.1
    )

    assert command == [
        "claude",
        "-p",
        "--setting-sources",
        "",
        "--strict-mcp-config",
        "--tools",
        "",
        "--no-session-persistence",
        "--model",
        "claude-haiku-5-5",
        "--effort",
        "low",
        "--output-format",
        "json",
        "--json-schema",
        '{"type": "object"}',
        "--system-prompt",
        "Be brief.",
        "--max-budget-usd",
        "0.1000",
    ]


def test_budget_refuses_new_calls_once_the_run_limit_is_reached():
    budget = Budget(
        run_limit_usd=0.25,
        project_limit_usd=40,
        call_limit_usd=0.10,
        spent_before_run_usd=0,
    )

    first = budget.reserve()
    second = budget.reserve()
    with pytest.raises(BudgetExceeded):
        budget.reserve()
    budget.settle(first, 0.02)
    budget.settle(second, 0.03)

    assert budget.spent_this_run_usd == pytest.approx(0.05)
    assert budget.reserve() == 0.10


def test_budget_refuses_to_start_when_the_project_budget_is_used_up():
    with pytest.raises(
        BudgetExceeded, match=r"project budget used up: \$40.00 of \$40.00"
    ):
        Budget(
            run_limit_usd=2,
            project_limit_usd=40,
            call_limit_usd=0.10,
            spent_before_run_usd=40,
        )


def test_client_retries_invalid_output_caches_the_valid_one_and_records_every_paid_attempt(
    tmp_path,
):
    responses = [fake_response({"wrong": "shape"}), fake_response({"word": "OK"})]
    calls = []

    def transport(config, system_prompt, json_schema, max_call_usd, prompt):
        calls.append(prompt)
        return responses.pop(0)

    ledger = Ledger(tmp_path / "ledger.jsonl")
    client = LLMClient(
        ModelConfig("claude-haiku-5-5", "low"),
        budget=Budget(2, 40, 0.10, 0),
        ledger=ledger,
        run_id="run-1",
        purpose="extract",
        transport=transport,
        cache_dir=tmp_path / "cache",
    )

    first = client.call("Gen.4.p1", "Be brief.", "Say OK.", Answer)
    second = client.call("Gen.4.p1", "Be brief.", "Say OK.", Answer)

    assert (first.output, first.cached, second.output, second.cached) == (
        {"word": "OK"},
        False,
        {"word": "OK"},
        True,
    )
    assert calls == ["Say OK.", "Say OK."]
    assert [
        {
            key: row[key]
            for key in (
                "run_id",
                "item_id",
                "attempt",
                "is_error",
                "cost_usd",
                "thinking_tokens",
            )
        }
        for row in ledger.rows()
    ] == [
        {
            "run_id": "run-1",
            "item_id": "Gen.4.p1",
            "attempt": 1,
            "is_error": False,
            "cost_usd": 0.01,
            "thinking_tokens": 10,
        },
        {
            "run_id": "run-1",
            "item_id": "Gen.4.p1",
            "attempt": 2,
            "is_error": False,
            "cost_usd": 0.01,
            "thinking_tokens": 10,
        },
    ]
    assert ledger.total_cost() == pytest.approx(0.02)


def test_client_stops_without_retrying_when_claude_reports_an_error(tmp_path):
    calls = []

    def transport(config, system_prompt, json_schema, max_call_usd, prompt):
        calls.append(prompt)
        return fake_response(None, cost_usd=0.0, is_error=True)

    client = LLMClient(
        ModelConfig("claude-haiku-5-5", "low"),
        budget=Budget(2, 40, 0.10, 0),
        ledger=Ledger(tmp_path / "ledger.jsonl"),
        run_id="run-1",
        purpose="extract",
        transport=transport,
        cache_dir=tmp_path / "cache",
    )

    with pytest.raises(LLMCallError, match="Gen.4.p1: usage limit reached"):
        client.call("Gen.4.p1", "Be brief.", "Say OK.", Answer)
    assert len(calls) == 1
    assert json.loads((tmp_path / "ledger.jsonl").read_text())["is_error"] is True


class FakeHTTPResponse:
    def __init__(self, payload: dict) -> None:
        self.body = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, *args):
        return self.body

    def close(self):
        pass


def test_openai_compatible_transport_sends_the_schema_and_returns_a_claude_shaped_response(
    monkeypatch,
):
    sent = []

    def urlopen(request, timeout):
        sent.append((request.full_url, request.headers, json.loads(request.data)))
        return FakeHTTPResponse(
            {
                "choices": [{"message": {"content": '```json\n{"word": "OK"}\n```'}}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 200},
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1/")
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("KGRAG_PRICE_INPUT_PER_MTOK", "0.5")
    monkeypatch.setenv("KGRAG_PRICE_OUTPUT_PER_MTOK", "2")

    response = openai_compatible_transport(
        ModelConfig("qwen3.5:4b", "medium"),
        "Be brief.",
        {"type": "object"},
        0.1,
        "Say OK.",
    )

    assert sent == [
        (
            "http://localhost:11434/v1/chat/completions",
            {"Content-type": "application/json", "Authorization": "Bearer secret"},
            {
                "model": "qwen3.5:4b",
                "messages": [
                    {"role": "system", "content": "Be brief."},
                    {"role": "user", "content": "Say OK."},
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "output",
                        "schema": {"type": "object"},
                        "strict": False,
                    },
                },
            },
        )
    ]
    assert {
        key: response[key]
        for key in ("is_error", "structured_output", "usage", "total_cost_usd")
    } == {
        "is_error": False,
        "structured_output": {"word": "OK"},
        "usage": {"input_tokens": 1000, "output_tokens": 200},
        "total_cost_usd": pytest.approx(0.0009),
    }


def test_openai_compatible_transport_reports_http_errors_as_errors(monkeypatch):
    import urllib.error

    def urlopen(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url,
            401,
            "Unauthorized",
            {},
            FakeHTTPResponse({"error": "bad key"}),
        )

    monkeypatch.setattr("urllib.request.urlopen", urlopen)

    response = openai_compatible_transport(
        ModelConfig("gpt-4.1-mini", "medium"), "s", {}, 0.1, "p"
    )

    assert response == {"is_error": True, "result": 'HTTP 401: {"error": "bad key"}'}


def test_default_transport_follows_the_provider_setting(monkeypatch):
    monkeypatch.setenv("KGRAG_LLM_PROVIDER", "openai")
    assert default_transport() is openai_compatible_transport
    monkeypatch.setenv("KGRAG_LLM_PROVIDER", "claude-code")
    assert default_transport() is claude_code_transport
    monkeypatch.setenv("KGRAG_LLM_PROVIDER", "gemini")
    with pytest.raises(LLMCallError, match="use one of claude-code, openai"):
        default_transport()
