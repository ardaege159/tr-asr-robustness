"""Mitigations applied to a conversation before it reaches the assistant.

none     the conversation as heard; the model's first decision is final
prompt   tell the model its input comes from speech recognition (last value wins, confirm if unsure)
confirm  the model must read key values back and get a yes before acting; a simulated customer
         answers (see simulate.py), at most two extra exchanges
"""

from __future__ import annotations

from .simulate import make_responder

PROMPT_HINT = (
    "Müşterinin sözleri telefondan ses tanıma ile metne çevriliyor; harf, rakam ve kelime hataları olabilir. "
    "Müşteri bir bilgiyi düzeltirse en son söylediği değeri kullan. "
    "Bir numara ya da kod politikadaki biçime uymuyorsa ya da emin değilsen işlem yapmadan önce müşteriye teyit ettir."
)

CONFIRM_HINT = (
    "Müşterinin sözleri telefondan ses tanıma ile metne çevriliyor; harf, rakam ve kelime hataları olabilir. "
    "Müşteri bir bilgiyi düzeltirse en son söylediği değeri kullan. "
    "Kayıt değiştiren bir işlem yapmadan önce (dondurma, iade, iptal, değişiklik, ödeme, itiraz, bagaj) "
    "işlemi ve kullanacağın numara, kod, tarih, tutar gibi değerleri müşteriye tek cümlede okuyup onay iste. "
    "Numaraları ve kodları rakam rakam, harf harf oku. Müşteri onaylayınca işlemi yap; düzeltirse düzeltilmiş değerle tekrar teyit et. "
    "Kart numarasının tamamını, şifre ya da kodları asla okuma. Temsilciye aktarma ve sorgulama için onay gerekmez."
)

SPEECH_CONDITIONS = {"asr", "selfcorr_asr"}


def apply(fix: str, turns: list[dict], spec: dict, condition: str) -> tuple[list[dict], str, object]:
    """Return (turns to show the model, extra system prompt text, responder or None)."""
    if fix == "none":
        return turns, "", None
    if fix == "prompt":
        return turns, PROMPT_HINT, None
    if fix == "confirm":
        return turns, CONFIRM_HINT, make_responder(spec, spec["id"], condition in SPEECH_CONDITIONS)
    raise ValueError(f"unknown fix: {fix}")
