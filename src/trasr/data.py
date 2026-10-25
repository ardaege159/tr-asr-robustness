"""Paths and loaders for scenarios, answer keys and generated conversations."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
SCENARIOS = ROOT / "scenarios"
RESULTS = ROOT / "results"

CONDITIONS = ("clean", "asr", "selfcorr", "selfcorr_asr")


@cache
def scenario(company: str) -> dict:
    return yaml.safe_load((SCENARIOS / f"{company}.yaml").read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    """Rows of a JSONL file. A run killed mid-write can leave a cut-off last line; it is skipped."""
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def specs() -> list[dict]:
    return read_jsonl(DATA / "specs.jsonl")


def conversations(condition: str) -> list[dict]:
    """Rows of data/<condition>.jsonl: {"id", "condition", "turns": [{"role", "text"}, ...]}."""
    return read_jsonl(DATA / f"{condition}.jsonl")
