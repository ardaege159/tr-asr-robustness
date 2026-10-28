"""Tables and charts from results/raw and results/judged.

    python -m trasr.report        # writes results/summary.md and results/*.png
"""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .data import CONDITIONS, RESULTS, conversations, read_jsonl  # noqa: E402
import re  # noqa: E402

from .judge import CHECKS, HUMAN, JUDGED, SOUNDS_DONE, cohen_kappa  # noqa: E402
from .run import RAW, subset_ids  # noqa: E402
from .score import READ_ONLY_TOOLS  # noqa: E402

CONDITION_LABELS = {"clean": "Temiz", "asr": "Ses tanıma", "selfcorr": "Kendini düzeltme", "selfcorr_asr": "Düzeltme + ses"}
FIX_LABELS = {"none": "İyileştirme yok", "prompt": "Talimat", "confirm": "Teyit"}

# Reference palette, light mode (categorical slots 1-4); text never wears series colors.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    v = sorted(values)
    return v[min(len(v) - 1, int(q * len(v)))]


def load() -> list[dict]:
    latest = {}
    for p in sorted(RAW.glob("*.jsonl")):
        for r in read_jsonl(p):
            key = (r["model"], r["id"], r["condition"], r["fix"])
            if key not in latest or latest[key].get("error"):
                latest[key] = r
    return [r for r in latest.values() if not r.get("error")]


def short(model: str) -> str:
    return model.split(":", 1)[1].split("/")[-1].removesuffix(":free")


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK_2)
    ax.tick_params(colors=INK_2, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))


def grouped_bars(path, groups: list[str], series: list[str], value, title: str, group_labels: dict, series_labels) -> None:
    """value(group, series) -> (k, n) or None."""
    fig, ax = plt.subplots(figsize=(9, 4.6), facecolor=SURFACE)
    _style(ax)
    width = 0.8 / len(series)
    for j, s in enumerate(series):
        xs, ys, lo, hi = [], [], [], []
        for i, g in enumerate(groups):
            kn = value(g, s)
            if not kn or not kn[1]:
                continue
            k, n = kn
            a, b = wilson(k, n)
            xs.append(i - 0.4 + width * (j + 0.5))
            ys.append(k / n)
            lo.append(k / n - a)
            hi.append(b - k / n)
        if not xs:
            continue  # color stays tied to the series index, so the others keep theirs
        # 2px surface gap between neighbouring bars
        ax.bar(xs, ys, width * 0.92, color=SERIES[j], label=series_labels(s), edgecolor=SURFACE, linewidth=1)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=INK_2, elinewidth=1, capsize=0)
    ax.set_xticks(range(len(groups)), [group_labels.get(g, g) for g in groups], color=INK)
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=30)
    ax.legend(frameon=False, ncol=len(series), loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def asr_stats() -> list[str]:
    lines = ["| Koşul | Konuşma | Ort. kelime hata oranı | Numara/kod bozulmadan ulaştı |", "|---|---:|---:|---:|"]
    for cond in ("asr", "selfcorr_asr"):
        rows = conversations(cond)
        if not rows:
            continue
        kept = [v for r in rows for v in r["ids_kept"].values()]
        kept_s = f"{sum(kept) / len(kept):.0%} ({sum(kept)}/{len(kept)})" if kept else "-"
        lines.append(f"| {CONDITION_LABELS[cond]} | {len(rows)} | {mean(r['wer'] for r in rows):.1%} | {kept_s} |")
    return lines


OUTCOME_LABELS = {
    "correct": "Doğru", "wrong_args": "Yanlış değerle işlem", "asked": "Sordu", "took_first": "İlk değeri aldı",
    "wrong_tool": "Yanlış işlem", "missed": "İşlem yok", "policy_breach": "Politika ihlali",
}


# The labelling sample's pattern, as whole words ("gerçekleşti" must not match "gerçekleştirebilirim",
# an offer), plus the present-tense "iptal ediyorum" it missed.
CLAIM = re.compile(rf"(?:{SOUNDS_DONE.pattern}|iptal ediyorum)(?!\w)", re.I)


def claimed_without_call(row: dict) -> bool:
    """The reply says something was done, but no call changed a record.

    The model judge missed both such cases in the hand-labelled sample, so this check is a rule.
    """
    return bool(CLAIM.search(row["reply"] or "")) and all(c["tool"] in READ_ONLY_TOOLS for c in row["calls"])


def heard_vs_meant(rows: list[dict]) -> list[str]:
    """Speech conditions split by whether every number and code reached the model intact."""
    intact = {(c["id"], cond): all(c["ids_kept"].values()) for cond in ("asr", "selfcorr_asr") for c in conversations(cond) if c["ids_kept"]}
    groups: dict[tuple, list[str]] = defaultdict(list)
    for r in rows:
        if (k := intact.get((r["id"], r["condition"]))) is not None:
            groups[(r["model"], r["fix"], k)].append(r["outcome"])
    shown = ["correct", "wrong_args", "asked"]
    lines = [
        "| Model | İyileştirme | Numara/kod | n | " + " | ".join(OUTCOME_LABELS[o] for o in shown) + " | Diğer |",
        "|---|---|---|---:|" + "---:|" * (len(shown) + 1),
    ]
    order = list(FIX_LABELS)
    for (model, fix, k), outs in sorted(groups.items(), key=lambda x: (x[0][0], order.index(x[0][1]), not x[0][2])):
        n = len(outs)
        cells = [f"{outs.count(o) / n:.0%}" for o in shown]
        other = n - sum(outs.count(o) for o in shown)
        lines.append(f"| {short(model)} | {FIX_LABELS[fix]} | {'sağlam' if k else '**bozuk**'} | {n} | " + " | ".join(cells) + f" | {other / n:.0%} |")
    return lines


def agreement() -> list[str]:
    """Judge and the claimed_without_call rule against the hand labels."""
    human = read_jsonl(HUMAN)
    judged = {(r["model"], r["id"], r["condition"], r["fix"]): r for p in JUDGED.glob("*.jsonl") for r in read_jsonl(p)}
    raw = {(r["model"], r["id"], r["condition"], r["fix"]): r for p in sorted(RAW.glob("*.jsonl")) for r in read_jsonl(p) if not r.get("error")}
    lines = ["| Kontrol | Puanlayan | n | Uyum | Kappa | İnsan evet | Puanlayan evet |", "|---|---|---:|---:|---:|---:|---:|"]

    def row(check, who, pairs):
        a, b = [x for x, _ in pairs], [y for _, y in pairs]
        if a:
            kappa = cohen_kappa(a, b)
            lines.append(f"| {check} | {who} | {len(a)} | {sum(x == y for x, y in pairs) / len(a):.0%} | {'-' if math.isnan(kappa) else f'{kappa:.2f}'} | {sum(a)} | {sum(b)} |")

    keyed = [((h["model"], h["id"], h["condition"], h["fix"]), h) for h in human]
    for check in CHECKS:
        row(check, "model", [(h[check], judged[k][check]) for k, h in keyed if k in judged])
        if check == "claimed_without_call":
            row(check, "kural", [(h[check], claimed_without_call(raw[k])) for k, h in keyed if k in raw])
    return lines


def main() -> None:
    every = load()
    subset = set(subset_ids())
    rows = [r for r in every if r["id"] in subset]
    RESULTS.mkdir(exist_ok=True)
    models = sorted({r["model"] for r in rows})
    cell: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        cell[(r["model"], r["fix"], r["condition"])].append(r)

    def kn(model, fix, cond):
        rs = cell.get((model, fix, cond), [])
        return sum(r["outcome"] == "correct" for r in rs), len(rs)

    md = ["# Sonuçlar", "", f"Ana tablolar {len(subset)} cevap anahtarlık sabit alt kümeden (`data/subset.txt`); her hücre aynı konuşmalar.", ""]
    md += ["## Ses tanıma zinciri", "", *asr_stats(), ""]
    md += [
        "## Duyduğuna mı, kastedilene mi göre davrandı?", "",
        "Ses koşulları, numara ve kodların ses tanımadan bozulmadan geçip geçmediğine göre ayrıldı. "
        "Bozuk satırda doğru cevap ancak model sorup doğrusunu öğrenirse mümkün; *yanlış değerle işlem* "
        "müşterinin söylemediği bir numarayla kayıt değiştirmek demek.", "",
        *heard_vs_meant(rows), "",
    ]
    md += ["## Doğru araç çağrısı oranı", "", "| Model | İyileştirme | Koşul | n | Doğru | %95 GA | İlk cevap p50 / p95 (s) | Ek tur | Sorup bıraktı | Düzeltilmişi değil ilkini aldı |", "|---|---|---|---:|---:|---|---|---:|---:|---:|"]
    for (model, fix, cond), rs in sorted(cell.items(), key=lambda x: (x[0][0], list(FIX_LABELS).index(x[0][1]) if x[0][1] in FIX_LABELS else 9, CONDITIONS.index(x[0][2]))):
        k, n = kn(model, fix, cond)
        a, b = wilson(k, n)
        lat = [r["latency_s"] for r in rs if r.get("latency_s") is not None]
        md.append(
            f"| {short(model)} | {FIX_LABELS.get(fix, fix)} | {CONDITION_LABELS[cond]} | {n} | {k / n:.0%} | {a:.0%}–{b:.0%} "
            f"| {pct(lat, 0.5):.1f} / {pct(lat, 0.95):.1f} | {mean(r.get('followups', 0) for r in rs):.2f} "
            f"| {sum(r['outcome'] == 'asked' for r in rs)} | {sum(r['outcome'] == 'took_first' for r in rs)} |"
        )

    md += ["", "## Yapmadığı işlemi yaptım dedi", "", "Kural: cevap bir işlemin yapıldığını söylüyor (`report.CLAIM`) ama kayıt değiştiren bir araç çağrısı yok. Kural, elle etiketlenen 40 örnekte görülen kalıplardan yazıldı; aşağıdaki uyum değeri bu yüzden iyimser.", ""]
    md += ["| Model | " + " | ".join(CONDITION_LABELS[c] for c in CONDITIONS) + " |", "|---|" + "---:|" * len(CONDITIONS)]
    for model in models:
        cells = []
        for cond in CONDITIONS:
            rs = cell.get((model, "none", cond), [])
            cells.append(f"{sum(map(claimed_without_call, rs))}/{len(rs)}" if rs else "-")
        md.append(f"| {short(model)} | " + " | ".join(cells) + " |")

    md += ["", "## Puanlayıcıların insan etiketleriyle uyumu", "", *agreement()]

    # Rows outside the subset were run before it was fixed; shown only as a check on the subset's numbers.
    full = defaultdict(list)
    for r in every:
        if r["fix"] == "none" and r["condition"] in ("clean", "asr"):
            full[(r["model"], r["condition"])].append(r["outcome"] == "correct")
    md += ["", "## Kontrol: tüm 100 cevap anahtarı (iyileştirme yok)", "", "| Model | Koşul | n | Doğru |", "|---|---|---:|---:|"]
    for (model, cond), oks in sorted(full.items()):
        md.append(f"| {short(model)} | {CONDITION_LABELS[cond]} | {len(oks)} | {sum(oks) / len(oks):.0%} |")

    grouped_bars(
        RESULTS / "accuracy_by_condition.png", list(CONDITIONS), models,
        lambda cond, m: kn(m, "none", cond), "Doğru araç çağrısı oranı (iyileştirme yok), %95 güven aralığı",
        CONDITION_LABELS, short,
    )
    fixes = [f for f in FIX_LABELS if any(k[1] == f for k in cell)]
    if len(fixes) > 1:
        for cond in ("asr", "selfcorr_asr"):
            grouped_bars(
                RESULTS / f"fix_effect_{cond}.png", models, fixes,
                lambda m, f, c=cond: kn(m, f, c), f"İyileştirmelerin etkisi: {CONDITION_LABELS[cond]}",
                {m: short(m) for m in models}, lambda f: FIX_LABELS[f],
            )
    md += ["", "![Koşullara göre doğruluk](accuracy_by_condition.png)"]
    if len(fixes) > 1:
        md += ["", "![İyileştirme etkisi, ses tanıma](fix_effect_asr.png)", "", "![İyileştirme etkisi, düzeltme + ses](fix_effect_selfcorr_asr.png)"]
    (RESULTS / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
