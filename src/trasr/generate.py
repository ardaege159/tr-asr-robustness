"""Turn answer keys into spoken-style phone conversations.

Each spec yields a clean conversation and, when it has a correction, a self-correction twin.
The twin is made by rewriting only the customer turn that carries the corrected value, so the
two differ in that one turn and nowhere else.

    python -m trasr.generate --limit 5
"""

from __future__ import annotations

import argparse
import re
from datetime import date

from .data import DATA, append_jsonl, conversations, scenario, specs
from .models import chat_json
from .score import ID_ARGS, norm_id, tr_lower

DEFAULT_MODEL = "gemini:gemini-3.1-flash-lite"

PROMPT = """Bir Türkçe sesli müşteri hizmetleri değerlendirme veri seti için telefon konuşması yazıyorsun.

Şirket: {company} ({sector}). Bugün: {today}.
Şirket politikası (asistanın talimatı):
{policy}

Müşterinin niyeti: {intent}
Müşteri tipi: {persona}
Arka plan: {notes}
Müşterinin konuşmada söylemesi gereken bilgiler (doğru değerler):
{values}

Kurallar:
- Konuşma asistanın selamlamasıyla başlar, müşterinin cümlesiyle biter. Toplam 4 ya da 6 tur, roller sırayla.
- Konuşma, asistanın karar vermesi gereken anda kesilir: son müşteri cümlesiyle gerekli bütün bilgiler verilmiş olur ve asistanın ne yapacağı henüz söylenmemiştir.
- Asistan turları yalnızca selamlar ve eksik bilgiyi sorar. Asistan hiçbir işlem yapmaz, sistemde bir şey kontrol ettiğini söylemez, sonuç ya da karar bildirmez (izin verildi, reddedildi, süre geçmiş gibi), politikada olmayan bir şart uydurmaz, müşterinin verdiği numara ve kodları tekrar etmez.
- Müşteri yalnızca arka planda yazanı ister; arka planda yazmıyorsa temsilci istemez, yeni bir talep eklemez.
- Müşteri konuşma dilinde, müşteri tipine uygun konuşur (dolgu sözcükleri, yarım cümleler olabilir). Markdown, emoji, parantez yok.
- Konuşma ses tanımadan geçmiş kusursuz bir döküm gibi yazılır: sipariş, kart, abone numaralarını rakamla ve insanların söylediği gibi gruplayarak yaz (ör. "547 23 81"). Rezervasyon kodlarını harf ve rakamları boşlukla ayırarak yaz (ör. "K 7 M 2 X A"). Tutarları rakamla yaz (ör. "2450 lira").
- Tarihleri müşteri doğal söyler ("bu perşembe", "ayın yirmisi", "dün"); bugünün tarihine göre doğru güne denk gelmeli.
- Araç adlarını ya da kodlu değerleri (ör. yanlis_urun, supheli_islem) asla aynen söyleme, doğal anlat.

Yalnızca şu JSON'u döndür:
{{"turns": [{{"role": "assistant", "text": "..."}}, {{"role": "user", "text": "..."}}, ...]}}"""

SELFCORR_PROMPT = """Aşağıda bir telefon konuşmasında müşterinin cümleleri numaralı olarak var.
Müşteri "{arg}" bilgisini söylüyor. {meaning}

{lines}

Bu bilginin söylendiği müşteri cümlesini seç ve yeniden yaz: müşteri önce yanlış değeri söylesin, hemen ardından
doğal bir şekilde düzeltip doğru değeri söylesin ("yok pardon", "hayır hayır", "pardon yanlış söyledim" gibi).
Doğru değer en son söylenen olmalı. Cümlenin geri kalanını olduğu gibi bırak, başka bir şey ekleme.
Numara ve kodları cümledeki biçimle yaz (rakamla, gruplayarak; kodlarda harfleri boşlukla ayırarak).

Yalnızca şu JSON'u döndür: {{"index": <cümle numarası>, "text": "..."}}"""

# How an argument value reads to a person, so the correction is said naturally, not as a code.
ENUM_MEANING = {
    "kayip": "kart kayboldu", "calinti": "kart çalındı", "supheli_islem": "şüpheli işlem",
    "hasarli": "ürün hasarlı geldi", "yanlis_urun": "yanlış ürün geldi", "begenmedim": "ürünü beğenmedi", "diger": "başka bir sebep",
    "vadesiz": "vadesiz hesap", "vadeli": "vadeli hesap", "kredi_karti": "kredi kartı",
    "elektrik": "elektrik faturası", "su": "su faturası", "dogalgaz": "doğalgaz faturası",
    "internet": "internet faturası", "telefon": "telefon faturası",
}


MONTHS = ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"]
WEEKDAYS = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]

# An assistant turn that reports doing or checking something takes the decision away from the model under test.
ASSISTANT_ACTS = re.compile(r"kontrol ett|paylaşıyorum|başlattım|oluşturdum|iptal ett|dondurdum|baktım|görüyorum|gerçekleştirdim|tamamladım")


def _values(spec: dict) -> dict:
    vals = dict(spec.get("args_mentioned", {}))
    for e in spec["expected"]:
        vals.update(e["args"])
    return vals


def _clean_notes(spec: dict) -> str:
    """Notes without the 'first says X, then Y' part, which belongs only to the self-correction twin."""
    notes = re.sub(r"Önce [^.]*\.", "", spec.get("notes", "")).strip()
    return notes or "-"


def _say(value) -> str:
    """'2026-11-05' -> '5 Kasım Perşembe'; enum codes -> their meaning; anything else unchanged."""
    if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        d = date.fromisoformat(value)
        return f"{d.day} {MONTHS[d.month - 1]} {WEEKDAYS[d.weekday()]}"
    return ENUM_MEANING.get(value, str(value))


def build_prompt(spec: dict) -> str:
    sc = scenario(spec["company"])
    return PROMPT.format(
        company=sc["company"],
        sector=sc["sector"],
        today=sc["today"],
        policy=sc["policy"].strip(),
        intent=spec["intent"],
        persona=spec["persona"],
        notes=_clean_notes(spec),
        values="\n".join(f"- {k}: {v}" for k, v in _values(spec).items()) or "- (yok)",
    )


def _customer_ids(turns: list[dict]) -> str:
    return norm_id(" ".join(t["text"] for t in turns if t["role"] == "user"))


def check(spec: dict, turns) -> list[str]:
    """Cheap sanity checks on a clean conversation; generation is retried when any fail."""
    if not isinstance(turns, list) or len(turns) < 2:
        return ["missing turns"]
    problems = []
    roles = [t.get("role") for t in turns]
    if roles[0] != "assistant" or roles[-1] != "user" or any(a == b for a, b in zip(roles, roles[1:])):
        problems.append(f"bad roles {roles}")
    said = _customer_ids(turns)
    assistant_said = norm_id(" ".join(t["text"] for t in turns if t["role"] == "assistant"))
    for k, val in _values(spec).items():
        if k in ID_ARGS and norm_id(val) not in said:
            problems.append(f"{k}={val} not said")
        if k in ID_ARGS and norm_id(val) in assistant_said:
            problems.append(f"assistant repeats {k}")
    for t in turns:
        if t["role"] == "assistant" and (m := ASSISTANT_ACTS.search(tr_lower(t["text"]))):
            problems.append(f"assistant acts: {m.group()!r}")
    return problems


def check_selfcorr(spec: dict, turns: list[dict]) -> list[str]:
    corr = spec["correction"]
    if corr["arg"] not in ID_ARGS:
        return []
    said = _customer_ids(turns)
    wrong, right = norm_id(corr["wrong"]), norm_id(_values(spec)[corr["arg"]])
    if wrong not in said or said.rfind(right) < said.find(wrong):
        return ["wrong value missing or not before the right one"]
    return []


def _normalise_roles(turns) -> list:
    for t in turns if isinstance(turns, list) else []:
        t["role"] = {"customer": "user", "musteri": "user", "asistan": "assistant"}.get(t.get("role"), t.get("role"))
    return turns


def _retry(make, validate, spec_id: str, tries: int = 3):
    problems: list[str] = []
    for _ in range(tries):
        try:
            out = make()
            problems = validate(out)
        except (ValueError, KeyError, TypeError, IndexError) as e:
            problems = [f"{type(e).__name__}: {e}"]
        if not problems:
            return out
        print(f"  {spec_id} retry: {problems}")
    raise ValueError(f"{spec_id}: {problems}")


def generate_clean(spec: dict, model: str) -> list[dict]:
    prompt = build_prompt(spec)
    return _retry(lambda: _normalise_roles(chat_json(model, prompt).get("turns")), lambda t: check(spec, t), spec["id"])


def make_selfcorr(spec: dict, clean: list[dict], model: str) -> list[dict]:
    """Rewrite only the customer turn that carries the corrected value; every other turn is copied."""
    corr = spec["correction"]
    right = _values(spec)[corr["arg"]]
    user_idx = [i for i, t in enumerate(clean) if t["role"] == "user"]
    meaning = f"Yanlış değer: {_say(corr['wrong'])}. Doğru değer: {_say(right)}. Tarihleri ve sebepleri doğal söyle, kod gibi değil."
    prompt = SELFCORR_PROMPT.format(
        arg=corr["arg"],
        meaning=meaning,
        lines="\n".join(f"{n}. {clean[i]['text']}" for n, i in enumerate(user_idx, 1)),
    )

    def make():
        out = chat_json(model, prompt, temperature=0.5)
        i = user_idx[int(out["index"]) - 1]
        turns = [dict(t) for t in clean]
        turns[i] = {"role": "user", "text": out["text"].strip()}
        return turns

    return _retry(make, lambda t: check_selfcorr(spec, t), spec["id"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--ids", nargs="*")
    args = ap.parse_args()

    done = {c["id"] for c in conversations("clean")}
    todo = [s for s in specs() if s["id"] not in done and (not args.ids or s["id"] in args.ids)]
    for spec in todo[: args.limit]:
        try:
            clean = generate_clean(spec, args.model)
            selfcorr = make_selfcorr(spec, clean, args.model) if spec.get("correction") else None
        except Exception as e:
            print(f"FAIL {spec['id']}: {e}")
            continue
        append_jsonl(DATA / "clean.jsonl", {"id": spec["id"], "condition": "clean", "turns": clean})
        if selfcorr:
            append_jsonl(DATA / "selfcorr.jsonl", {"id": spec["id"], "condition": "selfcorr", "turns": selfcorr})
        print(f"ok {spec['id']}")


if __name__ == "__main__":
    main()
