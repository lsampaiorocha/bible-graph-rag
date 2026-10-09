import json
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from pathlib import Path

from kgrag.corpus import DATA_DIR
from kgrag.llm import BudgetExceeded, LLMCallError, LLMClient, ModelConfig
from kgrag.schema import PassageExtraction
from kgrag.validate import validate_extraction

PROMPT_VERSION = "extract_v1"
PURPOSE = "extract"


@dataclass
class ExtractionSummary:
    extracted: int = 0
    from_cache: int = 0
    failed: list[str] = field(default_factory=list)
    stopped_by_budget: str | None = None
    not_attempted: int = 0


def load_system_prompt() -> str:
    return (Path(__file__).parent / "prompts" / f"{PROMPT_VERSION}.md").read_text(
        encoding="utf-8"
    )


def user_prompt(passage: dict) -> str:
    return f"Passage:\n{passage['text']}"


def extraction_dir(config: ModelConfig) -> Path:
    return (
        DATA_DIR / "extractions" / f"{PROMPT_VERSION}__{config.model}__{config.effort}"
    )


def extract_passage(
    client: LLMClient, system_prompt: str, passage: dict, out_dir: Path
) -> bool:
    result = client.call(
        passage["id"], system_prompt, user_prompt(passage), PassageExtraction
    )
    validated, drops = validate_extraction(
        PassageExtraction.model_validate(result.output), passage["text"]
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{passage['id']}.json").write_text(
        json.dumps(
            {
                "passage_id": passage["id"],
                "model": client.config.model,
                "effort": client.config.effort,
                "prompt_version": PROMPT_VERSION,
                "usage": asdict(result.usage),
                "raw": result.output,
                "validated": validated.model_dump(),
                "dropped": [asdict(drop) for drop in drops],
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    return result.cached


def extract_passages(
    client: LLMClient, passages: list[dict], out_dir: Path, workers: int
) -> ExtractionSummary:
    system_prompt = load_system_prompt()
    summary = ExtractionSummary()

    def record(future: Future) -> None:
        try:
            cached = future.result()
        except BudgetExceeded as error:
            summary.stopped_by_budget = summary.stopped_by_budget or str(error)
            return
        except LLMCallError as error:
            summary.failed.append(str(error))
            return
        if cached:
            summary.from_cache += 1
        else:
            summary.extracted += 1

    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {
            executor.submit(extract_passage, client, system_prompt, passage, out_dir)
            for passage in passages
        }
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                record(future)
            if summary.stopped_by_budget:
                still_running = {future for future in pending if not future.cancel()}
                wait(still_running)
                for future in still_running:
                    record(future)
                break
    summary.not_attempted = (
        len(passages) - summary.extracted - summary.from_cache - len(summary.failed)
    )
    return summary


def load_extractions(
    config: ModelConfig, passage_ids: list[str]
) -> tuple[list[dict], list[str]]:
    found, missing = [], []
    for passage_id in passage_ids:
        path = extraction_dir(config) / f"{passage_id}.json"
        if path.exists():
            found.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            missing.append(passage_id)
    return found, missing
