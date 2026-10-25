"""A simulated customer who answers the assistant's follow-up questions.

It knows the true values from the answer key, so it confirms a correct read-back and corrects a
wrong one, the way a real caller would. In the speech conditions its answer goes through the same
voice, phone channel and recogniser as the original turns, so a correction can be misheard again.
"""

from __future__ import annotations

from .data import scenario
from .models import chat_json

SIM_MODEL = "gemini:gemini-3.5-flash-lite"

PROMPT = """Telefonda {company} müşteri hizmetlerini arayan bir müşterisin. Müşteri tipin: {persona}.
Durumun: {notes}
Doğru bilgilerin (asistanın bunları duyması gerekiyor):
{values}

Konuşma şu ana kadar (asistan senin sözlerini ses tanımayla, hatalı duymuş olabilir):
{dialogue}

Asistanın son sözü: "{reply}"

Asistana tek bir kısa cevap ver:
- Asistan bilgileri okuyup onay istiyorsa ve hepsi doğruysa kısaca onayla.
- Okuduğu bir bilgi yanlışsa (tek harf ya da rakam bile) hayır de ve doğrusunu söyle.
- Bir bilgi istiyorsa doğru bilgiyi ver. Bilmediğin bir şey sorarsa makul bir cevap uydur ama yukarıdaki bilgilerle çelişme.
- Numaraları rakamla ve gruplayarak (ör. "547 23 81"), kodları harfleri boşlukla ayırarak (ör. "K 7 M 2 X A") söyle.
- Kart şifresi, CVV ya da SMS kodu istenirse verme.
Yalnızca şu JSON'u döndür: {{"text": "..."}}"""


def _values(spec: dict) -> dict:
    vals = dict(spec.get("args_mentioned", {}))
    for e in spec["expected"]:
        vals.update(e["args"])
    return vals


def make_responder(spec: dict, conv_id: str, through_speech: bool):
    sc = scenario(spec["company"])

    def respond(dialogue: list[dict], reply: str) -> str:
        lines = [
            f"{'Asistan' if t['role'] == 'assistant' else 'Sen'}: {t.get('spoken', t['text'])}" for t in dialogue
        ]
        prompt = PROMPT.format(
            company=sc["company"],
            persona=spec["persona"],
            notes=spec.get("notes", "-"),
            values="\n".join(f"- {k}: {v}" for k, v in _values(spec).items()) or "- (yok)",
            dialogue="\n".join(lines),
            reply=reply,
        )
        said = chat_json(SIM_MODEL, prompt, temperature=0.3)["text"].strip()
        if not through_speech:
            return said
        from .speech import hear

        return hear(conv_id, said)

    return respond
