"""Play every conversation to every model, score the tool calls, and save raw results.

    python -m trasr.run --models gemini:gemini-3.1-flash-lite --conditions clean asr
    python -m trasr.run --plan --models groq:openai/gpt-oss-120b    # the experiment for one model, on data/subset.txt
    python -m trasr.run --summary

Results go to results/raw/<model>.jsonl, one row per (spec, condition, fix); reruns skip rows
that already exist, so an interrupted run picks up where it stopped.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict

from .data import CONDITIONS, DATA, RESULTS, append_jsonl, conversations, read_jsonl, specs
from .models import DailyQuota, run_assistant
from .score import score

RAW = RESULTS / "raw"

MODELS = ["gemini:gemini-3.1-flash-lite", "groq:openai/gpt-oss-120b", "groq:openai/gpt-oss-20b", "groq:qwen/qwen3.8-27b"]

# (fix, conditions). The fixes target speech errors, so they run only on the speech conditions.
BASELINE = [("none", ["clean", "asr", "selfcorr", "selfcorr_asr"])]
FIXES = [("confirm", ["asr", "selfcorr_asr"]), ("prompt", ["asr"])]

# Groq's free tier allows 200k tokens/day per model, about 80 conversations, which covers the
# baseline but not the fixes; those run on the model with a separate request quota.
FIX_MODELS = {"gemini:gemini-3.1-flash-lite"}

# The plan runs on a fixed subset of 50 answer keys so that every condition and fix can finish on
# free-tier quotas; rows already run on other keys are kept and reported separately.
SUBSET = DATA / "subset.txt"


def subset_ids() -> list[str]:
    return [line.strip() for line in SUBSET.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]


def plan(model_name: str) -> list[tuple[str, list[str]]]:
    return BASELINE + (FIXES if model_name in FIX_MODELS else [])


def slug(model_name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", model_name)


def run(model_name: str, conditions: list[str], ids: list[str] | None, fix: str = "none") -> None:
    from . import fixes

    path = RAW / f"{slug(model_name)}.jsonl"
    done = {(r["id"], r["condition"], r["fix"]) for r in read_jsonl(path) if not r.get("error")}
    by_id = {s["id"]: s for s in specs()}
    for cond in conditions:
        for conv in conversations(cond):
            if (conv["id"], cond, fix) in done or (ids and conv["id"] not in ids):
                continue
            spec = by_id[conv["id"]]
            turns, extra_system, responder = fixes.apply(fix, conv["turns"], spec, cond)
            res = run_assistant(model_name, spec, turns, extra_system, responder)
            row = {"id": conv["id"], "condition": cond, "fix": fix, "model": model_name, **res}
            if not res["error"]:
                s = score(spec, res["calls"], cond)
                outcome = s.outcome
                # Ending on a question instead of acting is caution, not the same failure as ignoring the request.
                if outcome == "missed" and "?" in res["reply"]:
                    outcome = "asked"
                row |= {"outcome": outcome, "detail": s.detail}
            append_jsonl(path, row)
            print(f"{model_name} {fix:7} {cond:13} {conv['id']} {row.get('outcome', 'ERROR ' + str(res['error'])[:80])}", flush=True)


def summary() -> None:
    rows = [r for p in sorted(RAW.glob("*.jsonl")) for r in read_jsonl(p)]
    latest = {}
    for r in rows:  # a successful rerun replaces an earlier error
        key = (r["model"], r["id"], r["condition"], r["fix"])
        if key not in latest or latest[key].get("error"):
            latest[key] = r
    table: dict[tuple, Counter] = defaultdict(Counter)
    lat: dict[tuple, list] = defaultdict(list)
    for r in latest.values():
        key = (r["model"], r["fix"], r["condition"])
        table[key][r.get("outcome", "error")] += 1
        if not r.get("error"):
            lat[key].append((r.get("latency_total_s") or 0, r.get("followups", 0)))
    print(f"{'model':40} {'fix':8} {'condition':13} {'n':>3} {'correct':>8} {'p50 s':>6} {'extra':>6}  outcomes")
    for key, c in sorted(table.items()):
        model, fix, cond = key
        n = sum(v for k, v in c.items() if k != "error")
        acc = c["correct"] / n if n else 0
        times = sorted(t for t, _ in lat[key])
        p50 = times[len(times) // 2] if times else 0
        extra = sum(f for _, f in lat[key]) / len(lat[key]) if lat[key] else 0
        print(f"{model:40} {fix:8} {cond:13} {n:3} {acc:8.0%} {p50:6.1f} {extra:6.2f}  {dict(c)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--conditions", nargs="*", default=list(CONDITIONS))
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--fix", default="none")
    ap.add_argument("--plan", action="store_true", help="run every step of plan(model)")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.plan and args.ids is None:
        args.ids = subset_ids()
    for m in args.models or (MODELS if args.plan else []):
        try:
            for fix, conds in plan(m) if args.plan else [(args.fix, args.conditions)]:
                run(m, conds, args.ids, fix)
        except DailyQuota as e:
            print(f"{m}: daily quota used up, rerun later to continue ({e})")
    if args.summary or not args.models:
        summary()


if __name__ == "__main__":
    main()
