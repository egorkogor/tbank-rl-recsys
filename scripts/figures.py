"""Figures for the README from artifacts/runs/eval/val/*.json (static PNG, light surface).

1. track_length_vs_beta.png  - mean length of the top-10 against beta, by reward (two panels: probability
                               rewards, minute rewards), SFT as a reference line.
2. tradeoff.png              - gain of balanced minutes B (snapshot judges) against the NDCG@10 change, RL vs
                               the closed form pi*, coloured by reward class; the selection region shaded.
3. goodhart_lp.png           - lp reward at beta 0.02 over training steps: train RM, judges, NDCG@10.
Palette: validated categorical slots 1-5 of the dataviz reference palette (light mode).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401

from rlrec.config import ROOT, RUNS  # noqa: E402

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a63", "#e6e5e0", "#fcfcfb"
OUT = ROOT / "docs" / "figures"


def load() -> dict:
    out = {}
    for f in (RUNS / "eval" / "val").glob("*.json"):
        if f.stem.startswith(("compare", "selection", "decompose", "suboptimality", "multistep", "booster")):
            continue
        j = json.loads(f.read_text(encoding="utf-8"))
        out[j["spec"]] = j
    return out


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def rl_points(ev, reward):
    pts = []
    for spec, j in ev.items():
        if spec.startswith(f"rl:{reward}_b") and spec.endswith("_T1:400"):
            beta = float(spec.split("_b")[1].split("_")[0])
            pts.append((beta, j))
    return sorted(pts, key=lambda p: p[0])


def fig_length(ev):
    sft_len = ev["sft"]["behaviour"]["track_len_mean"]
    panels = [("Награды-вероятности", ["lp", "ratio", "fair", "mix", "quant"]),
              ("Награды в минутах", ["minutes", "wtg", "balanced", "add", "mlp"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    for ax, (title, rewards) in zip(axes, panels):
        style(ax)
        for k, r in enumerate(rewards):
            pts = [(b, j) for b, j in rl_points(ev, r) if b > 0]
            if not pts:
                continue
            xs = [b for b, _ in pts]
            ys = [j["behaviour"]["track_len_mean"] for _, j in pts]
            ax.plot(xs, ys, color=SLOTS[k], linewidth=2, marker="o", markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=r)
        ax.axhline(sft_len, color=MUTED, linewidth=1.2, linestyle="--")
        ax.set_xscale("log")
        betas = sorted({b for r in rewards for b, _ in rl_points(ev, r) if b > 0})
        ax.set_xticks(betas)
        ax.set_xticklabels([f"{b:g}".replace(".", ",") for b in betas])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        if "минут" in title:
            ax.set_yscale("log")
            ax.set_yticks([200, 300, 500, 800, 1300])
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
            ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.annotate(f"SFT {sft_len:.0f} с", (betas[0], sft_len), xytext=(0, 5), textcoords="offset points",
                    fontsize=8, color=MUTED)
        ax.set_xlabel("β (сила KL-поводка), лог. шкала", color=INK, fontsize=10)
        ax.set_ylabel("средняя длина трека в топ-10, с", color=INK, fontsize=10)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.legend(frameon=False, fontsize=8, loc="upper left" if "минут" in title else "lower right")
    fig.suptitle("Слабый поводок уводит в короткие треки (вероятности) или в длинные (минуты)", color=INK,
                 fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "track_length_vs_beta.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def reward_class(spec):
    r = spec.split(":")[1].split("_b")[0]
    if r in ("lp", "ratio", "fair", "mix", "quant"):
        return 0, "вероятности"
    if r in ("minutes", "wtg", "balanced", "add", "mlp"):
        return 1, "минуты"
    if r in ("cont", "disc", "spotify"):
        return 2, "долгосрочные"
    return 3, "плацебо"


def fig_tradeoff(ev):
    sft = ev["sft"]
    b0 = sft["r_judge_snap"]["balanced"]
    n0 = sft["relevance"]["ndcg@10"]["mean"]
    fig, ax = plt.subplots(figsize=(9, 5.4), facecolor=SURFACE)
    style(ax)
    ax.axhspan(-2, 6, color=GRID, alpha=0.5, linewidth=0)
    ax.annotate("допустимо по правилу: ΔNDCG ≥ −2%", (0.02, 4.6), xycoords=("axes fraction", "data"), fontsize=8,
                color=MUTED)
    seen = set()
    for spec, j in ev.items():
        if not (spec.startswith("rl:") or spec.startswith("pistar:")) or j.get("r_judge_snap") is None:
            continue
        if spec.startswith("rl:") and not spec.endswith(":400") and "spotify" not in spec:
            continue
        cls, name = reward_class(spec if spec.startswith("rl:") else "rl:" + spec.split(":")[1] + "_b")
        x = j["r_judge_snap"]["balanced"] - b0
        y = 100 * (j["relevance"]["ndcg@10"]["mean"] / n0 - 1)
        kind = "RL" if spec.startswith("rl:") else "π*"
        lab = f"{kind}, {name}" if (kind, cls) not in seen else None
        seen.add((kind, cls))
        if kind == "RL":
            ax.scatter(x, y, s=46, color=SLOTS[cls], edgecolor=SURFACE, linewidth=1.5, zorder=3, label=lab)
        else:
            ax.scatter(x, y, s=56, marker="D", facecolor="none", edgecolor=SLOTS[cls], linewidth=2, zorder=3,
                       label=lab)
    ch = json.loads((RUNS / "eval" / "val" / "selection.json").read_text(encoding="utf-8"))["choice"]
    if ch in ev:
        j = ev[ch]
        x, y = j["r_judge_snap"]["balanced"] - b0, 100 * (j["relevance"]["ndcg@10"]["mean"] / n0 - 1)
        ax.annotate(f"выбор правила: {ch}", (x, y), xytext=(18, -26), textcoords="offset points", fontsize=9,
                    color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=1))
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_ylim(-105, 6)
    ax.set_xlabel("прирост сбалансированных минут B у судей m4–m6, мин (symlog)", color=INK, fontsize=10)
    ax.set_ylabel("изменение NDCG@10 против SFT, %", color=INK, fontsize=10)
    ax.set_title("Компромисс: награда против релевантности (val)", color=INK, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower left", ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "tradeoff.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_goodhart(ev):
    steps, rows = [0], [ev["sft"]]
    for st in (100, 200, 300, 400):
        spec = f"rl:lp_b0.02_T1:{st}"
        if spec in ev:
            steps.append(st)
            rows.append(ev[spec])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 5.6), sharex=True, facecolor=SURFACE)
    for ax in (a1, a2):
        style(ax)
    series = [("обучающая RM m0", [j["r_train"]["lp"] for j in rows]),
              ("судьи m1–m3", [j["r_judge"]["lp"] for j in rows]),
              ("судьи m4–m6", [j["r_judge_snap"]["lp"] for j in rows])]
    for k, (name, ys) in enumerate(series):
        a1.plot(steps, ys, color=SLOTS[k], linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=name)
        a1.annotate(name, (steps[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", fontsize=8, color=INK,
                    va="center")
    a1.set_ylabel("P(Listen+) топ-10", color=INK, fontsize=10)
    a1.set_title("«Дослушает», β = 0,02: балл RM растёт, точность падает", color=INK, fontsize=11, loc="left")
    a1.legend(frameon=False, fontsize=8, loc="lower right")
    nd = [j["relevance"]["ndcg@10"]["mean"] for j in rows]
    a2.plot(steps, nd, color=INK, linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE,
            markeredgewidth=1.5)
    a2.set_ylabel("NDCG@10", color=INK, fontsize=10)
    a2.set_xlabel("шаг RL", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "goodhart_lp.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ev = load()
    fig_length(ev)
    fig_tradeoff(ev)
    fig_goodhart(ev)
    print("figures:", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
