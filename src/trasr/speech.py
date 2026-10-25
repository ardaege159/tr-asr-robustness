"""Run customer turns through a simulated phone call and real speech recognition.

text -> edge-tts (tr-TR) -> 8 kHz, 300-3400 Hz band, background noise -> Whisper (Groq) -> text

Assistant turns stay as text: they are the system's own speech, not something it has to hear.

    python -m trasr.speech --source clean --limit 5      # writes data/asr.jsonl
    python -m trasr.speech --source selfcorr             # writes data/selfcorr_asr.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import re
import time

import edge_tts
import jiwer
import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfilt

from .data import DATA, append_jsonl, conversations, specs
from .models import client
from .score import ID_ARGS, norm_id, tr_lower

VOICES = ["tr-TR-AhmetNeural", "tr-TR-EmelNeural"]
RATES = ["-10%", "+0%", "+10%", "+20%"]
SNR_DB = (8.0, 20.0)
PHONE_SR = 8000
WHISPER_MODEL = "whisper-large-v3"
AUDIO = DATA / "audio"

OUTPUT = {"clean": "asr", "selfcorr": "selfcorr_asr"}


def rng_for(key: str) -> np.random.Generator:
    """Same conversation, same voice and noise on every run."""
    return np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:8], 16))


async def _tts(text: str, voice: str, rate: str) -> bytes:
    buf = bytearray()
    async for chunk in edge_tts.Communicate(text, voice, rate=rate).stream():
        if chunk["type"] == "audio":
            buf += chunk["data"]
    return bytes(buf)


def tts(text: str, voice: str, rate: str) -> tuple[np.ndarray, int]:
    # The free TTS service drops requests now and then, or leaves the connection hanging;
    # they succeed on a later try.
    for attempt in range(8):
        try:
            data = asyncio.run(asyncio.wait_for(_tts(text, voice, rate), timeout=90))
            break
        except (edge_tts.exceptions.NoAudioReceived, TimeoutError):
            if attempt == 7:
                raise
            time.sleep(3 + 4 * attempt)
    audio, sr = sf.read(io.BytesIO(data), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def _pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    spectrum = np.fft.rfft(rng.standard_normal(n))
    spectrum /= np.sqrt(np.maximum(np.arange(len(spectrum)), 1))
    return np.fft.irfft(spectrum, n)


def phone_channel(audio: np.ndarray, sr: int, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Narrowband telephone audio with pink background noise at the given SNR."""
    x = resample_poly(audio, PHONE_SR, sr).astype(np.float64)
    x = sosfilt(butter(4, [300, 3400], btype="bandpass", fs=PHONE_SR, output="sos"), x)
    noise = sosfilt(butter(4, [300, 3400], btype="bandpass", fs=PHONE_SR, output="sos"), _pink_noise(len(x), rng))
    sig_p, noise_p = np.mean(x**2), np.mean(noise**2)
    x = x + noise * np.sqrt(sig_p / (noise_p * 10 ** (snr_db / 10)))
    x = np.clip(x / (np.max(np.abs(x)) + 1e-9) * 1.1, -1, 1)  # slight clipping, like a hot handset mic
    return x.astype(np.float32)


def transcribe(wav: np.ndarray, sr: int) -> str:
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
    for attempt in range(6):
        try:
            resp = client("groq").audio.transcriptions.create(
                model=WHISPER_MODEL, file=("turn.wav", buf.getvalue()), language="tr", temperature=0
            )
            return resp.text.strip()
        except Exception as e:
            if "rate" not in str(e).lower() or attempt == 5:
                raise
            time.sleep(15 * (attempt + 1))
    return ""


def norm_for_wer(s: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", tr_lower(s)).split())


def ids_kept(spec: dict, turns: list[dict]) -> dict[str, bool]:
    """For each identifier the customer gives, did it survive recognition intact?"""
    said = norm_id(" ".join(t["text"] for t in turns if t["role"] == "user"))
    vals = dict(spec.get("args_mentioned", {}))
    for e in spec["expected"]:
        vals.update(e["args"])
    return {k: norm_id(v) in said for k, v in vals.items() if k in ID_ARGS}


def caller(conv_id: str) -> tuple[str, str, float]:
    """Voice, speed and SNR of one simulated caller.

    They depend only on the conversation, and noise only on the turn's text, so a clean
    conversation and its self-correction twin differ only in the corrected turn.
    """
    rng = rng_for(conv_id)
    voice, rate = VOICES[rng.integers(len(VOICES))], RATES[rng.integers(len(RATES))]
    return voice, rate, float(rng.uniform(*SNR_DB))


def hear(conv_id: str, text: str, wav_name: str | None = None) -> str:
    """What the assistant receives when this caller says `text` over the phone."""
    voice, rate, snr = caller(conv_id)
    audio, sr = tts(text, voice, rate)
    wav = phone_channel(audio, sr, snr, rng_for(conv_id + text))
    if wav_name:
        AUDIO.mkdir(parents=True, exist_ok=True)
        sf.write(AUDIO / wav_name, wav, PHONE_SR)
    return transcribe(wav, PHONE_SR)


def process(conv: dict, spec: dict) -> dict:
    voice, rate, snr = caller(conv["id"])
    turns, refs, hyps = [], [], []
    for i, t in enumerate(conv["turns"]):
        if t["role"] != "user":
            turns.append(t)
            continue
        heard = hear(conv["id"], t["text"], f"{conv['id']}_{conv['condition']}_{i}.wav")
        turns.append({"role": "user", "text": heard, "spoken": t["text"]})
        refs.append(norm_for_wer(t["text"]))
        hyps.append(norm_for_wer(heard))
    return {
        "id": conv["id"],
        "condition": OUTPUT[conv["condition"]],
        "turns": turns,
        "wer": round(jiwer.wer(refs, hyps), 4),
        "ids_kept": ids_kept(spec, turns),
        "voice": voice,
        "rate": rate,
        "snr_db": round(snr, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=list(OUTPUT), default="clean")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--ids", nargs="*")
    args = ap.parse_args()

    out_name = OUTPUT[args.source]
    by_id = {s["id"]: s for s in specs()}
    done = {c["id"] for c in conversations(out_name)}
    todo = [c for c in conversations(args.source) if c["id"] not in done and (not args.ids or c["id"] in args.ids)]
    for conv in todo[: args.limit]:
        try:
            row = process(conv, by_id[conv["id"]])
        except Exception as e:
            print(f"FAIL {conv['id']}: {type(e).__name__}: {e}")
            continue
        append_jsonl(DATA / f"{out_name}.jsonl", row)
        print(f"ok {conv['id']} wer={row['wer']:.2f} snr={row['snr_db']} ids_kept={row['ids_kept']}")


if __name__ == "__main__":
    main()
