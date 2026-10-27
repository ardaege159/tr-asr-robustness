"""OpenAI-compatible clients for free model endpoints, and the tool-calling loop under test."""

from __future__ import annotations

import json
import os
import re
import time
from functools import cache

from dotenv import load_dotenv
from openai import APIStatusError, APITimeoutError, OpenAI, RateLimitError

from .data import ROOT, scenario
from .score import READ_ONLY_TOOLS

load_dotenv(ROOT / ".env")

PROVIDERS = {
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
}

MAX_TOOL_ROUNDS = 4


@cache
def client(provider: str) -> OpenAI:
    base_url, env = PROVIDERS[provider]
    return OpenAI(base_url=base_url, api_key=os.environ[env], max_retries=0, timeout=180)


def parse_model(name: str) -> tuple[str, str]:
    """'openrouter:qwen/qwen3-32b:free' -> ('openrouter', 'qwen/qwen3-32b:free')."""
    provider, _, model = name.partition(":")
    return provider, model


class DailyQuota(Exception):
    """A per-day limit that will not free up soon enough to wait for."""


MAX_QUOTA_WAIT_S = 30 * 60


def _rate_limit_text(e: Exception) -> str:
    """Message plus response body: Gemini sends its 429 details as a JSON list, so str(e) is only 'Error code: 429'."""
    response = getattr(e, "response", None)
    try:
        body = response.text if response is not None else ""
    except Exception:
        body = ""
    return f"{e} {body}"


def _retry_after(message: str) -> float | None:
    """Seconds from 'try again in 3m12.2s' (Groq) or 'retry in 22h39m9.4s' (Gemini)."""
    if m := re.search(r"(?:try again|retry) in ([\d.]+)ms", message):
        return float(m.group(1)) / 1000
    m = re.search(r"(?:try again|retry) in (?:(\d+)h)?(?:(\d+)m)?([\d.]+)s", message)
    if not m:
        return None
    h, mins, s = m.groups()
    return int(h or 0) * 3600 + int(mins or 0) * 60 + float(s)


def _create(provider: str, attempts: int = 6, **kwargs):
    """chat.completions.create with backoff; free tiers rate-limit hard.

    Returns (response, seconds taken by the successful attempt), so waiting out a rate limit
    never counts as model latency. Groq's daily token limit is a rolling window that frees up
    within minutes, so it is waited out; a quota that resets hours away raises DailyQuota.
    """
    i = 0
    while True:
        try:
            t0 = time.perf_counter()
            resp = client(provider).chat.completions.create(**kwargs)
            return resp, time.perf_counter() - t0
        except RateLimitError as e:
            msg = _rate_limit_text(e)
            if "PerDay" in msg:
                # Gemini's per-day request quota resets at midnight Pacific; its retryDelay is not that.
                raise DailyQuota(msg[:300]) from e
            if re.search(r"per day|\(TPD\)|\(RPD\)", msg):
                wait = _retry_after(msg)
                if wait is None or wait > MAX_QUOTA_WAIT_S:
                    raise DailyQuota(msg[:300]) from e
                time.sleep(wait + 2)
                continue
            if i == attempts - 1:
                raise
            wait = 10 * 2**i
        except APITimeoutError:
            if i == attempts - 1:
                raise
            wait = 5
        except APIStatusError as e:
            if e.status_code < 500 or i == attempts - 1:
                raise
            wait = 5 * 2**i
        i += 1
        time.sleep(min(wait, 120))


def chat_json(model_name: str, prompt: str, temperature: float = 0.9, reasoning_effort: str | None = "low") -> dict:
    provider, model = parse_model(model_name)
    extra = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
    resp, _ = _create(
        provider,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=temperature,
        **extra,
    )
    text = resp.choices[0].message.content or ""
    # Some models wrap the JSON in prose or code fences; take the outermost object.
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"no JSON object in reply: {text[:200]!r}")
    return json.loads(text[start : end + 1])


def tool_defs(company: str) -> list[dict]:
    return [{"type": "function", "function": t} for t in scenario(company)["tools"]]


def system_prompt(company: str, extra: str = "") -> str:
    sc = scenario(company)
    return f"{sc['policy'].strip()}\nBugün: {sc['today']}.\n{extra}".strip()


def mock_result(spec: dict, tool: str) -> dict:
    """What the fake backend returns. Lookups can carry per-spec facts (spec["lookup"])."""
    if tool in READ_ONLY_TOOLS:
        return {"durum": "bulundu", **spec.get("lookup", {})}
    return {"sonuc": "basarili"}


def run_assistant(
    model_name: str, spec: dict, turns: list[dict], extra_system: str = "", responder=None, max_followups: int = 2
) -> dict:
    """Play the conversation to the model and let it act until it answers in text.

    responder: optional callable(dialogue, assistant_reply) -> customer's next utterance. When given,
    a text reply that comes before any action gets a customer answer and the model continues,
    up to max_followups times. Without it the model's first decision is final.

    Returns {"calls", "reply", "latency_s", "latency_total_s", "followups", "dialogue", "error"}.
    latency_s is the time to the first response, which is what a caller on the phone waits for;
    latency_total_s adds up every model response, including those after follow-ups.
    """
    provider, model = parse_model(model_name)
    messages = [{"role": "system", "content": system_prompt(spec["company"], extra_system)}]
    messages += [{"role": t["role"], "content": t["text"]} for t in turns]
    dialogue = list(turns)
    calls: list[dict] = []
    latencies: list[float] = []
    followups = 0

    def result(reply: str = "", error: str | None = None) -> dict:
        return {
            "calls": calls,
            "reply": reply,
            "latency_s": latencies[0] if latencies else None,
            "latency_total_s": sum(latencies),
            "followups": followups,
            "dialogue": dialogue[len(turns):],
            "error": error,
        }

    try:
        rounds = 0
        while rounds < MAX_TOOL_ROUNDS:
            resp, seconds = _create(provider, model=model, messages=messages, tools=tool_defs(spec["company"]), temperature=0)
            latencies.append(seconds)
            msg = resp.choices[0].message
            if not msg.tool_calls:
                reply = msg.content or ""
                acted = any(c["tool"] not in READ_ONLY_TOOLS for c in calls)
                if responder is None or acted or followups >= max_followups:
                    return result(reply)
                answer = responder(dialogue, reply)
                dialogue += [{"role": "assistant", "text": reply}, {"role": "user", "text": answer}]
                messages += [{"role": "assistant", "content": reply}, {"role": "user", "content": answer}]
                followups += 1
                rounds = 0
                continue
            rounds += 1
            messages.append(msg.model_dump(exclude_none=True))
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {"_raw": tc.function.arguments}
                calls.append({"tool": tc.function.name, "args": args})
                out = mock_result(spec, tc.function.name)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(out, ensure_ascii=False)})
        return result(error="max_tool_rounds")
    except DailyQuota:
        raise  # not the model's fault; the run stops and resumes later
    except Exception as e:  # recorded and excluded from scores, never silently dropped
        return result(error=f"{type(e).__name__}: {e}"[:500])
