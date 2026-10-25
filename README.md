# tr-asr-robustness

**Does a Turkish voice customer-service assistant act on what the caller meant, or on what speech recognition heard?**

Voice assistants don't read what the customer typed; they read a speech-to-text transcript. One misheard digit in an
order number, a card's last four, or a booking code is enough to cancel the wrong order or freeze the wrong card.
This project measures how often that happens with current LLMs in Turkish, and tests a fix.

> Results: see [`results/summary.md`](results/summary.md). Headline numbers are added here once the full run finishes.

## What is measured

Three fictional companies, each with a policy and a set of tools the assistant can call:

| Company | Sector | Tools |
|---|---|---|
| Pazarix | e-commerce | `siparis_sorgula`, `iade_baslat`, `siparis_iptal`, `adres_degistir`, `insana_aktar` |
| Defne Bank | banking | `kart_dondur`, `bakiye_sorgula`, `islem_itiraz`, `fatura_ode`, `insana_aktar` |
| Rota Tur | travel | `rezervasyon_sorgula`, `rezervasyon_degistir`, `rezervasyon_iptal`, `bagaj_ekle`, `insana_aktar` |

100 hand-written answer keys (`data/specs.jsonl`) state what the correct action is, with exact arguments. Some
correct answers are *no action* (a return outside the 14-day window, cancelling an order that has shipped) or a
handoff to a human (fraud, health emergencies, complaints). A few contain traps: a caller who reads out the full
16-digit card number or their online-banking password unprompted.

Each answer key becomes a phone conversation, played to the model in four conditions:

| Condition | What the model receives |
|---|---|
| `clean` | a perfect transcript |
| `asr` | the same conversation spoken by a Turkish TTS voice, degraded to an 8 kHz phone line with background noise (8–20 dB SNR), and transcribed by Whisper large-v3 |
| `selfcorr` | the caller misspeaks and corrects themselves ("547 23 18, sorry, 547 23 81"); the answer key holds the corrected value |
| `selfcorr_asr` | the self-correction, through the same speech chain |

The `clean`/`selfcorr` twins differ in exactly one customer turn, and both go through the speech chain with the
same voice, speed and noise, so any difference in results comes from the correction alone.

### Scoring

Tool calls are scored by rules, not by a model (`src/trasr/score.py`): tool name and every argument in the answer
key must match after normalisation (IDs ignore spacing and case, dates accept `20.11.2026` and `2026-11-20`,
amounts accept `2.450,00`). Outcomes:

| Outcome | Meaning |
|---|---|
| `correct` | every expected action, right arguments, nothing extra |
| `took_first` | self-correction case: the model used the value the caller retracted |
| `wrong_args` | right tool, wrong value (typically a misheard digit) |
| `wrong_tool` / `missed` | a different action, or none |
| `asked` | no action; the model ended on a question (cautious, not ignoring the request) |
| `policy_breach` | an action the policy forbids, or an extra action |

What rules can't judge is graded by a separate model (`src/trasr/judge.py`) as yes/no questions: did the
assistant *say* it did something it never called a tool for, did it follow the policy, did it repeat sensitive
data, is the reply short enough to be spoken. I hand-labelled a random sample with the same questions; agreement
with the judge is reported as percent agreement and Cohen's kappa.

### Fixes

| Fix | What changes |
|---|---|
| `prompt` | the system prompt says the input comes from speech recognition, the last stated value wins, and to confirm when a code doesn't fit the expected format |
| `confirm` | before any action that changes a record, the assistant reads the values back and waits for a yes. A simulated caller who knows the true values confirms or corrects them; in the speech conditions the caller's reply also goes through TTS + phone line + Whisper, so a correction can be misheard too. At most two extra exchanges. |

`confirm` costs time on the phone, so its extra turns and total model latency are reported next to its accuracy.

## Models

All free tiers, called through OpenAI-compatible endpoints:

| Role | Model |
|---|---|
| Under test | `gemini-3.1-flash-lite` (Google), `gpt-oss-120b`, `gpt-oss-20b`, `qwen3.8-27b` (Groq) |
| Conversation generator | `gemini-3.1-flash-lite` |
| Simulated caller, judge | `gemini-3.5-flash-lite` |
| Speech | `edge-tts` (tr-TR voices), Groq `whisper-large-v3` |

## Running it

```bash
python -m venv .venv && .venv/Scripts/activate      # Python 3.11–3.13
pip install -e ".[dev]"
cp .env.example .env                                 # GEMINI_API_KEY, GROQ_API_KEY (OPENROUTER_API_KEY optional)

python -m trasr.generate                             # data/clean.jsonl, data/selfcorr.jsonl
python -m trasr.speech --source clean                # data/asr.jsonl
python -m trasr.speech --source selfcorr             # data/selfcorr_asr.jsonl
python -m trasr.run --plan                           # results/raw/<model>.jsonl
python -m trasr.judge run --model groq:openai/gpt-oss-120b
python -m trasr.judge label --n 30                   # hand labels, then: python -m trasr.judge agree
python -m trasr.report                               # results/summary.md + charts
pytest
```

Every step is resumable: rerunning skips what is already done, and a daily quota stops a run cleanly so it can
continue the next day.

## Layout

```
scenarios/        company policies and tool schemas (YAML)
data/specs.jsonl  answer keys
data/*.jsonl      generated conversations per condition (audio is not committed)
src/trasr/        generate, speech, models, run, score, judge, fixes, simulate, report
results/raw/      every model response with its tool calls, latency and outcome
tests/            scorer and statistics tests
```

## Limitations

- **Synthetic speech.** TTS voices are cleaner and more uniform than real callers; there are only two Turkish
  voices, no accents, no crosstalk. Real-call error rates are likely higher, so the `asr` numbers are a lower bound
  on the damage.
- **Generated conversations.** Conversations are written by an LLM from the answer keys and checked automatically
  (required IDs present, the assistant turns don't act or repeat IDs) and by reading samples. `gemini-3.1-flash-lite`
  is both the generator and one of the models under test, which may favour it.
- **Free-tier quotas** shaped the design. Groq allows 200k tokens per model per day (about 80 conversations), which
  covers the baseline for its three models but not the fixes, so the fixes are tested on `gemini-3.1-flash-lite`
  only, and only on the speech conditions. Each cell has 85–100 conversations, so the 95% confidence intervals in
  the summary are wide enough to matter.
- **Latency is indicative only.** Free endpoints are shared, and the runs were split between a machine in Turkey
  and GitHub Actions runners (`.github/workflows/run-models.yml`), so latency differs by network location.
  Accuracy is unaffected.
- **Single-step mock backend.** Lookups return fixed facts from the answer key; there is no real database.
