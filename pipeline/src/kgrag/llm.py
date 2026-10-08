import hashlib
import json
import statistics
import subprocess
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ValidationError

from kgrag.corpus import DATA_DIR

CACHE_DIR = DATA_DIR / "cache" / "llm"
LEDGER_PATH = DATA_DIR / "ledger" / "llm_calls.jsonl"

Runner = Callable[[list[str], str], dict]


class BudgetExceeded(Exception):
    pass


class LLMCallError(Exception):
    pass


@dataclass(frozen=True)
class ModelConfig:
    model: str
    effort: str


@dataclass(frozen=True)
class CallUsage:
    input_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    output_tokens: int
    thinking_tokens: int
    cost_usd: float
    duration_ms: int


@dataclass(frozen=True)
class CallResult:
    output: dict
    usage: CallUsage
    cached: bool


def claude_code_command(
    config: ModelConfig, system_prompt: str, json_schema: dict, max_call_usd: float
) -> list[str]:
    # Without the isolation flags every call loads the user's settings, plugins, hooks
    # and MCP servers (~28k extra tokens plus side effects). --bare would avoid that
    # but ignores the subscription login, so it cannot be used here.
    return [
        "claude",
        "-p",
        "--setting-sources",
        "",
        "--strict-mcp-config",
        "--tools",
        "",
        "--no-session-persistence",
        "--model",
        config.model,
        "--effort",
        config.effort,
        "--output-format",
        "json",
        "--json-schema",
        json.dumps(json_schema),
        "--system-prompt",
        system_prompt,
        "--max-budget-usd",
        f"{max_call_usd:.4f}",
    ]


def run_claude_code(command: list[str], prompt: str) -> dict:
    try:
        completed = subprocess.run(
            command, input=prompt, capture_output=True, text=True, timeout=600
        )
        return json.loads(completed.stdout)
    except subprocess.TimeoutExpired as error:
        raise LLMCallError("claude -p timed out after 600 s") from error
    except json.JSONDecodeError as error:
        raise LLMCallError(
            f"claude -p returned no JSON: {completed.stderr.strip()[:500]}"
        ) from error


def usage_from_response(response: dict) -> CallUsage:
    usage = response.get("usage") or {}
    return CallUsage(
        input_tokens=usage.get("input_tokens", 0),
        cache_creation_input_tokens=usage.get("cache_creation_input_tokens", 0),
        cache_read_input_tokens=usage.get("cache_read_input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        thinking_tokens=(usage.get("output_tokens_details") or {}).get(
            "thinking_tokens", 0
        ),
        cost_usd=response.get("total_cost_usd", 0.0),
        duration_ms=response.get("duration_ms", 0),
    )


class Ledger:
    def __init__(self, path: Path = LEDGER_PATH) -> None:
        self.path = path
        self._lock = threading.Lock()

    def append(self, row: dict) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(row) + "\n")

    def rows(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as file:
            return [json.loads(line) for line in file]

    def total_cost(self) -> float:
        return sum(row["cost_usd"] for row in self.rows())

    def average_call_cost(self, config: ModelConfig, purpose: str) -> float | None:
        costs = [
            row["cost_usd"]
            for row in self.rows()
            if row["model"] == config.model
            and row["effort"] == config.effort
            and row["purpose"] == purpose
            and not row["is_error"]
        ]
        return statistics.fmean(costs) if costs else None


def summarize_usage(
    rows: list[dict], since: datetime | None = None
) -> dict[tuple[str, str, str], dict]:
    totals: dict[tuple[str, str, str], dict] = {}
    for row in rows:
        if since is not None and datetime.fromisoformat(row["time"]) < since:
            continue
        group = totals.setdefault(
            (row["model"], row["effort"], row["purpose"]),
            {
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "thinking_tokens": 0,
                "cost_usd": 0.0,
            },
        )
        group["calls"] += 1
        group["input_tokens"] += (
            row["input_tokens"]
            + row["cache_creation_input_tokens"]
            + row["cache_read_input_tokens"]
        )
        group["output_tokens"] += row["output_tokens"]
        group["thinking_tokens"] += row["thinking_tokens"]
        group["cost_usd"] += row["cost_usd"]
    return totals


class Budget:
    """Spending caps in API-equivalent USD, as reported by claude -p.

    Each in-flight call reserves its full per-call cap, so parallel workers can never
    push the run or project total past its limit.
    """

    def __init__(
        self,
        run_limit_usd: float,
        project_limit_usd: float,
        call_limit_usd: float,
        spent_before_run_usd: float,
    ) -> None:
        if spent_before_run_usd >= project_limit_usd:
            raise BudgetExceeded(
                f"project budget used up: ${spent_before_run_usd:.2f} of ${project_limit_usd:.2f}"
            )
        self.run_limit_usd = run_limit_usd
        self.project_limit_usd = project_limit_usd
        self.call_limit_usd = call_limit_usd
        self.spent_before_run_usd = spent_before_run_usd
        self.spent_this_run_usd = 0.0
        self._reserved_usd = 0.0
        self._lock = threading.Lock()

    def remaining_usd(self) -> float:
        return min(
            self.run_limit_usd - self.spent_this_run_usd,
            self.project_limit_usd
            - self.spent_before_run_usd
            - self.spent_this_run_usd,
        )

    def reserve(self) -> float:
        with self._lock:
            available = self.remaining_usd() - self._reserved_usd
            if available < self.call_limit_usd:
                raise BudgetExceeded(
                    f"run budget reached: spent ${self.spent_this_run_usd:.4f} this run, "
                    f"${self.spent_before_run_usd + self.spent_this_run_usd:.4f} in total"
                )
            self._reserved_usd += self.call_limit_usd
            return self.call_limit_usd

    def settle(self, reserved_usd: float, actual_usd: float) -> None:
        with self._lock:
            self._reserved_usd -= reserved_usd
            self.spent_this_run_usd += actual_usd


class ClaudeCodeClient:
    def __init__(
        self,
        config: ModelConfig,
        budget: Budget | None,
        ledger: Ledger,
        run_id: str,
        purpose: str,
        runner: Runner = run_claude_code,
        cache_dir: Path = CACHE_DIR,
        max_attempts: int = 3,
    ) -> None:
        self.config = config
        self.budget = budget
        self.ledger = ledger
        self.run_id = run_id
        self.purpose = purpose
        self.runner = runner
        self.cache_dir = cache_dir
        self.max_attempts = max_attempts

    def cache_path(
        self, system_prompt: str, user_prompt: str, output_model: type[BaseModel]
    ) -> Path:
        key_material = json.dumps(
            [
                self.config.model,
                self.config.effort,
                system_prompt,
                output_model.model_json_schema(),
                user_prompt,
            ]
        )
        return (
            self.cache_dir / f"{hashlib.sha256(key_material.encode()).hexdigest()}.json"
        )

    def is_cached(
        self, system_prompt: str, user_prompt: str, output_model: type[BaseModel]
    ) -> bool:
        return self.cache_path(system_prompt, user_prompt, output_model).exists()

    def call(
        self,
        item_id: str,
        system_prompt: str,
        user_prompt: str,
        output_model: type[BaseModel],
    ) -> CallResult:
        cache_path = self.cache_path(system_prompt, user_prompt, output_model)
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            return CallResult(
                output=cached["output"], usage=CallUsage(**cached["usage"]), cached=True
            )
        if self.budget is None:
            raise BudgetExceeded("no budget configured for uncached calls")

        last_problem = ""
        for attempt in range(1, self.max_attempts + 1):
            reserved = self.budget.reserve()
            response: dict = {}
            try:
                response = self.runner(
                    claude_code_command(
                        self.config,
                        system_prompt,
                        output_model.model_json_schema(),
                        reserved,
                    ),
                    user_prompt,
                )
            finally:
                usage = usage_from_response(response)
                self.budget.settle(reserved, usage.cost_usd)
                if response:
                    self._record(item_id, attempt, usage, response)
            if response.get("is_error"):
                # Errors such as usage limits or the per-call cap would only repeat, so stop.
                raise LLMCallError(f"{item_id}: {str(response.get('result'))[:300]}")
            try:
                output = output_model.model_validate(
                    response.get("structured_output")
                ).model_dump()
            except ValidationError as error:
                last_problem = str(error)[:300]
                continue
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps({"output": output, "usage": asdict(usage)}), encoding="utf-8"
            )
            return CallResult(output=output, usage=usage, cached=False)
        raise LLMCallError(
            f"{item_id}: no valid output after {self.max_attempts} attempts: {last_problem}"
        )

    def _record(
        self, item_id: str, attempt: int, usage: CallUsage, response: dict
    ) -> None:
        self.ledger.append(
            {
                "time": datetime.now(UTC).isoformat(timespec="seconds"),
                "run_id": self.run_id,
                "purpose": self.purpose,
                "model": self.config.model,
                "effort": self.config.effort,
                "item_id": item_id,
                "attempt": attempt,
                "is_error": bool(response.get("is_error")),
                **asdict(usage),
            }
        )
