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
from .run import RAW  # noqa: E402

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


def main() -> None:
    rows = load()
    RESULTS.mkdir(exist_ok=True)
    models = sorted({r["model"] for r in rows})
    cell: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        cell[(r["model"], r["fix"], r["condition"])].append(r)

    def kn(model, fix, cond):
        rs = cell.get((model, fix, cond), [])
        return sum(r["outcome"] == "correct" for r in rs), len(rs)

    md = ["# Sonuçlar", "", "## Ses tanıma zinciri", "", *asr_stats(), ""]
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

    judged = [r for p in (RESULTS / "judged").glob("*.jsonl") for r in read_jsonl(p)]
    if judged:
        md += ["", "## Model değerlendirici (judge) bulguları", "", "| Model | İyileştirme | n | Yapmadan yaptım dedi | Politikaya uygun | Hassas bilgi tekrarı | Sese uygun |", "|---|---|---:|---:|---:|---:|---:|"]
        by: dict[tuple, list] = defaultdict(list)
        for r in judged:
            by[(r["model"], r["fix"])].append(r)
        for (model, fix), rs in sorted(by.items()):
            n = len(rs)
            md.append(
                f"| {short(model)} | {FIX_LABELS.get(fix, fix)} | {n} | "
                + " | ".join(f"{sum(r[k] for r in rs) / n:.0%}" for k in ("claimed_without_call", "policy_ok", "leaks_sensitive", "voice_ok"))
                + " |"
            )

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
