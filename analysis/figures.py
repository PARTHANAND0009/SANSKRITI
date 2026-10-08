"""Regenerate every paper figure and table from results/ (and data/processed/).

Source selection, per model: the full GPU run (4 cyclic rotations, results/depth_{m}_cyc_*)
when present, otherwise the CPU pilot (original + permuted order, 2,000 questions). Stats
come from results/gpu_stats/ when present, otherwise results/prelim/. Every caption names its
source through the \\ResultsSource macro, and anything not yet produced (e.g. patching on the
real models) becomes a clearly marked placeholder, so the paper compiles at every stage.

Outputs: paper/figures/*.pdf, paper/generated/*.tex (tables, numbers.tex with in-text
macros, status.tex).

  python analysis/figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.depth import calibrate, letter_prior, probs_of, soft_layer  # noqa: E402
from crystal.io import load_prompts  # noqa: E402
from data.prep import option_order  # noqa: E402

RES, FIG, GEN = ROOT / "results", ROOT / "paper" / "figures", ROOT / "paper" / "generated"
MODELS = {"llama31_8b": "Llama-3.1-8B", "qwen25_7b": "Qwen2.5-7B", "gemma2_9b": "Gemma-2-9B"}
# dataviz reference palette, categorical slots 1-3 (validated for all pairs, light mode)
COLORS = {"llama31_8b": "#2a78d6", "qwen25_7b": "#eb6834", "gemma2_9b": "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEED = 1234
METRIC_LABELS = {"l_star": "$\\ell^*$", "l_star_cal": "$\\ell^*_{cal}$", "d_soft_cal": "$d_{soft,cal}$",
                 "l_star_mean": "$\\ell^*_{mean}$", "l_star_cal_mean": "$\\ell^*_{cal,mean}$",
                 "l_star_cyc": "$\\ell^*_{cyc}$", "d_margin": "$d_{margin}$", "d_soft": "$d_{soft}$"}

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "lines.linewidth": 1.4, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})


def tex_escape(s: str) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def fmt_int(n) -> str:
    return f"{int(n):,}".replace(",", "{,}")


# ------------------------------------------------------------------ sources

def per_variant_tables(m: str) -> tuple[str, dict]:
    """('full'|'pilot'|None, {variant: table}) for the logit lens."""
    cyc = {v: RES / f"depth_{m}_{v}_logit.parquet" for v in ("cyc0", "cyc1", "cyc2", "cyc3")}
    if all(p.exists() for p in cyc.values()):
        return "full", {v: pd.read_parquet(p) for v, p in cyc.items()}
    pil = {v: RES / f"depth_{m}_{v}_logit.parquet" for v in ("prompt", "prompt_permuted")}
    have = {v: pd.read_parquet(p) for v, p in pil.items() if p.exists()}
    return ("pilot", have) if have else (None, {})


def aggregate_table(m: str) -> tuple[str, pd.DataFrame | None]:
    p = RES / f"depth_{m}_cyc_logit_agg.parquet"
    if p.exists():
        return "full", pd.read_parquet(p)
    p = RES / f"depth_{m}_prompt+prompt_permuted_logit_agg.parquet"
    return ("pilot", pd.read_parquet(p)) if p.exists() else (None, None)


def stats_dir() -> tuple[str, Path | None]:
    for name, d in (("full", RES / "gpu_stats"), ("pilot", RES / "prelim")):
        if (d / "PRELIMINARY_regression.csv").exists():
            return name, d
    return None, None


SOURCE_TEXT = {"full": "full run (19{,}742 questions, four cyclic option orders)",
               "pilot": "CPU pilot (2{,}000 sampled questions, original and one permuted option order); "
                        "to be replaced by the full run",
               None: "pending"}


# ------------------------------------------------------------------ figures

def placeholder(path: Path, text: str, size=(3.2, 1.6)):
    fig, ax = plt.subplots(figsize=size)
    ax.axis("off")
    ax.text(0.5, 0.5, text, ha="center", va="center", color=INK2, fontsize=8, wrap=True)
    fig.savefig(path)
    plt.close(fig)


def fig_layer_curves(prompts: pd.DataFrame) -> str:
    """Share of questions with gold ranked first, by relative layer: raw vs letter-calibrated."""
    sources, panels = set(), []
    for m in MODELS:
        src, tabs = per_variant_tables(m)
        if not tabs:
            continue
        sources.add(src)
        qids = sorted(set.intersection(*[set(t.qid) for t in tabs.values()]))
        tabs = {v: t.set_index("qid").loc[qids].reset_index() for v, t in tabs.items()}
        probs = {v: probs_of(t) for v, t in tabs.items()}
        # letter prior pooled over the variants (balanced gold letters across rotations / orders)
        prior = letter_prior(*probs.values())
        raw, cal, modal = [], [], []
        for v, P in probs.items():
            top = P.argmax(-1)                                   # [n, L+1] letter ranked first
            modal.append(np.max([(top == k).mean(0) for k in range(4)], 0))
            g = np.array([option_order(v, int(gi), q, SEED).index(int(gi))
                          for q, gi in zip(qids, prompts.loc[qids, "gold_idx"])])
            raw.append((P.argmax(-1) == g[:, None]).mean(0))
            cal.append((calibrate(P, prior).argmax(-1) == g[:, None]).mean(0))
        panels.append((m, np.mean(raw, 0), np.mean(cal, 0), np.mean(modal, 0)))
    path = FIG / "layer_curves.pdf"
    if not panels:
        placeholder(path, "Layer curves: pending results")
        return "pending"
    fig, axes = plt.subplots(1, len(panels), figsize=(6.3, 1.9), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (m, raw, cal, modal) in zip(axes, panels):
        x = np.arange(len(raw)) / (len(raw) - 1)
        ax.plot(x, cal, color=COLORS[m], label="letter-calibrated")
        ax.plot(x, raw, color=COLORS[m], linestyle=(0, (3, 2)), alpha=0.8, label="raw")
        ax.plot(x, modal, color=INK2, linestyle=(0, (1, 1.5)), linewidth=1.1, label="most common letter ranked first")
        ax.axhline(0.25, color=INK2, linewidth=0.8, linestyle=":")
        ax.set_title(MODELS[m], color=INK)
        ax.set_xlabel("relative layer $\\ell/L$")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("share of questions")
    axes[0].text(0.02, 0.27, "chance", color=INK2, fontsize=6.5)
    from matplotlib.lines import Line2D
    proxies = [Line2D([], [], color=INK, label="gold first, letter-calibrated"),
               Line2D([], [], color=INK, linestyle=(0, (3, 2)), label="gold first, raw"),
               Line2D([], [], color=INK2, linestyle=(0, (1, 1.5)), label="most common letter first")]
    fig.legend(handles=proxies, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)
    fig.savefig(path)
    plt.close(fig)
    return "full" if sources == {"full"} else "pilot"


def fig_depth_distribution() -> str:
    data, srcs = [], set()
    for m in MODELS:
        src, agg = aggregate_table(m)
        if agg is None:
            continue
        srcs.add(src)
        c = agg[agg.correct_cyc_final]
        data.append((m, c.d_frac_d_soft.astype(float).dropna().to_numpy()))
    path = FIG / "depth_distribution.pdf"
    if not data:
        placeholder(path, "Depth distribution: pending results")
        return "pending"
    fig, ax = plt.subplots(figsize=(3.1, 1.9))
    bins = np.linspace(0, 1, 41)
    for m, d in data:
        ax.hist(d, bins=bins, histtype="step", color=COLORS[m], linewidth=1.4, density=True,
                label=f"{MODELS[m]} (n={len(d):,})")
    ax.set_xlabel("$d_{\\mathrm{soft}}$ (fraction of depth)")
    ax.set_ylabel("density")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(data), fontsize=6.5)
    fig.savefig(path)
    plt.close(fig)
    return "full" if srcs == {"full"} else "pilot"


def fig_frequency(freq: pd.DataFrame) -> str:
    rows, srcs = [], set()
    for m in MODELS:
        src, agg = aggregate_table(m)
        if agg is None:
            continue
        srcs.add(src)
        j = agg[agg.correct_cyc_final].merge(freq[["qid", "log_freq_lenadj"]], on="qid").dropna(subset=["log_freq_lenadj"])
        j["y"] = j.d_frac_d_soft.astype(float)
        j = j.dropna(subset=["y"])
        j["bin"] = pd.qcut(j.log_freq_lenadj, 10, labels=False, duplicates="drop")
        g = j.groupby("bin").agg(x=("log_freq_lenadj", "median"), y=("y", "mean"), sd=("y", "std"), n=("y", "size"))
        g["se"] = g.sd / np.sqrt(g.n)
        rows.append((m, g))
    path = FIG / "frequency.pdf"
    if not rows:
        placeholder(path, "Depth vs frequency: pending results")
        return "pending"
    fig, ax = plt.subplots(figsize=(3.1, 1.9))
    for m, g in rows:
        ax.errorbar(g.x, g.y, yerr=1.96 * g.se, color=COLORS[m], marker="o", markersize=3.5, linewidth=1.2,
                    capsize=0, elinewidth=0.8, label=MODELS[m])
    ax.set_xlabel("entity frequency, length-adjusted (decile medians)")
    ax.set_ylabel("mean $d_{\\mathrm{soft}}$")
    ax.legend(frameon=False, fontsize=6.5)
    fig.savefig(path)
    plt.close(fig)
    return "full" if srcs == {"full"} else "pilot"


def fig_reliability() -> str:
    path = FIG / "reliability.pdf"
    split = {m: RES / f"reliability_{m}_splithalf_logit.csv" for m in MODELS}
    if any(p.exists() for p in split.values()):
        src, frames = "full", {m: pd.read_csv(p) for m, p in split.items() if p.exists()}
        col, label = "spearman_brown_full", "split-half reliability (Spearman-Brown, 4 orders)"
        frames = {m: f[f.subset == "correct_both"] for m, f in frames.items()}
    else:
        pil = {m: RES / f"reliability_{m}_prompt_vs_prompt_permuted_logit.csv" for m in MODELS}
        frames = {m: pd.read_csv(p) for m, p in pil.items() if p.exists()}
        frames = {m: f[f.subset == "correct_both"] for m, f in frames.items()}
        src, col, label = "pilot", "spearman", "test-retest Spearman (original vs permuted)"
    if not frames:
        placeholder(path, "Reliability: pending results")
        return "pending"
    metrics = list(dict.fromkeys(sum([f.metric.tolist() for f in frames.values()], [])))
    fig, ax = plt.subplots(figsize=(3.1, 1.9))
    w = 0.8 / len(frames)
    for i, (m, f) in enumerate(frames.items()):
        f = f.set_index("metric").reindex(metrics)
        ax.bar(np.arange(len(metrics)) + (i - (len(frames) - 1) / 2) * w, f[col], width=w * 0.92,
               color=COLORS[m], label=MODELS[m], edgecolor="white", linewidth=0.6)
    ax.set_xticks(np.arange(len(metrics)), [METRIC_LABELS.get(x, x) for x in metrics], rotation=0)
    ax.set_ylabel(label, fontsize=6.5)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=6.5)
    fig.savefig(path)
    plt.close(fig)
    return src


def fig_patching() -> str:
    path = FIG / "patching.pdf"
    curves = []
    for m in MODELS:
        shards = sorted((RES / f"patch_{m}_prompt").glob("shard_*.parquet"))
        if not shards:
            continue
        df = pd.concat(pd.read_parquet(s) for s in shards)
        v = df[df.valid]
        if len(v):
            R = np.stack(v.recovery.to_numpy())
            curves.append((m, R, len(df)))
    if not curves:
        placeholder(path, "Activation patching: pending the GPU run")
        return "pending"
    fig, ax = plt.subplots(figsize=(3.1, 1.9))
    for m, R, n in curves:
        x = np.arange(R.shape[1]) / (R.shape[1] - 1)
        ax.plot(x, np.median(R, 0), color=COLORS[m], label=f"{MODELS[m]} ({len(R)} valid of {n})")
    ax.axhline(0.5, color=INK2, linewidth=0.8, linestyle=":")
    ax.set_xlabel("patched layer $\\ell/L$")
    ax.set_ylabel("median recovery")
    ax.legend(frameon=False, fontsize=6.5)
    fig.savefig(path)
    plt.close(fig)
    return "full"


# ------------------------------------------------------------------ tables

def tab_models(prompts) -> str:
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Model & $L$ & $n$ & Acc. & $d_{\mathrm{soft}}$ & $\ell^*_{\mathrm{cyc}}/L$ \\", r"\midrule"]
    src_all = set()
    for m, name in MODELS.items():
        src, agg = aggregate_table(m)
        if agg is None:
            lines.append(f"{name} & -- & -- & -- & -- & -- \\\\")
            continue
        src_all.add(src)
        c = agg[agg.correct_cyc_final]
        lines.append(f"{name} & {int(agg.L.iloc[0])} & {fmt_int(len(agg))} & {agg.correct_cyc_final.mean():.3f} & "
                     f"{c.d_frac_d_soft.astype(float).median():.2f} & {c.d_frac_l_star_cyc.astype(float).median():.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "tab_models.tex").write_text("\n".join(lines) + "\n")
    return "full" if src_all == {"full"} else ("pilot" if src_all else "pending")


def tab_effects() -> str:
    src, d = stats_dir()
    if d is None:
        body = [r"\multicolumn{6}{c}{\emph{pending results}} \\"]
    else:
        r = pd.read_csv(d / "PRELIMINARY_regression.csv")
        r = r[(r.estimator == "ols_cluster_state") & (r.freq_measure == "raw")]
        prim = "d_soft"
        if (RES / "primary_metric.json").exists():
            prim = json.loads((RES / "primary_metric.json").read_text())["primary_metric"]
        pm = pd.read_csv(d / "PRELIMINARY_state_permutation.csv")
        pm = pm[pm.freq_measure == "raw"] if "freq_measure" in pm else pm
        body = []
        for m, name in MODELS.items():
            e = r[(r.model == m) & (r.metric == prim)]
            if e.empty:
                body.append(f"{name} & \\multicolumn{{5}}{{c}}{{--}} \\\\")
                continue
            a = e[e.term == "z_freq"].iloc[0]
            s = e[e.term == "z_state_logfreq"].iloc[0]
            pp = pm[(pm.model == m) & (pm.metric == prim)]
            ppv = f"{pp.p_perm.iloc[0]:.2f}" if len(pp) else "--"
            body.append(f"{name} & {a.beta_layers:+.2f} [{a.ci_low_layers:+.2f}, {a.ci_high_layers:+.2f}] & "
                        f"{a.p_holm:.3f} & {s.beta_layers:+.2f} [{s.ci_low_layers:+.2f}, {s.ci_high_layers:+.2f}] & "
                        f"{ppv} & {int(a.n):,} \\\\".replace(",", "{,}"))
    lines = [r"\begin{tabular}{lccccr}", r"\toprule",
             r"Model & Entity freq. & Holm $p$ & State freq. & Perm.\ $p$ & $n$ \\", r"\midrule", *body,
             r"\bottomrule", r"\end{tabular}"]
    (GEN / "tab_effects.tex").write_text("\n".join(lines) + "\n")
    return src or "pending"


def tab_data(prompts, numbers) -> None:
    rows = [("Rows in the release", numbers["nRaw"]), ("Answer matches no option (dropped)", numbers["nDropped"]),
            ("Gold text in two options (excluded)", numbers["nAmbiguous"]),
            ("Stem reveals the answer (excluded)", numbers["nLeaks"]),
            ("Analysis set", numbers["nAnalysis"]),
            ("Duplicated distractor (kept, flagged)", numbers["nDupOnly"])]
    lines = [r"\begin{tabular}{lr}", r"\toprule", r"Rows & Count \\", r"\midrule"]
    lines += [f"{k} & {v} \\\\" for k, v in rows] + [r"\bottomrule", r"\end{tabular}"]
    (GEN / "tab_data.tex").write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------------ numbers

def numbers(prompts) -> dict:
    raw_n = 21853
    an = prompts[~prompts.ambiguous_gold & ~prompts.leaks_answer]
    gold = [o[i] for o, i in zip(prompts.options, prompts.gold_idx)]
    norm = lambda s: " ".join(str(s).replace("_", " ").lower().split())  # noqa: E731
    assoc = prompts[(prompts.question_type == "Association") & ~prompts.ambiguous_gold]
    agold = [o[i] for o, i in zip(assoc.options, assoc.gold_idx)]
    self_state = assoc[[norm(g) == norm(s) for g, s in zip(agold, assoc.state)]]
    ent = pd.read_parquet(ROOT / "data/processed/entities.parquet")
    ea = ent[ent.qid.isin(an.qid)]
    fmeta = json.loads((ROOT / "data/processed/freq_meta.json").read_text())
    fe = pd.read_parquet(ROOT / "data/processed/freq_entities.parquet")
    from scipy.stats import spearmanr
    b = fe.dropna(subset=["corpus_count", "wiki_pageviews_en"])
    rho = spearmanr(b.corpus_count, b.wiki_pageviews_en)[0]
    leaks = pd.read_csv(ROOT / "data/processed/leaks.csv")
    al = leaks[leaks.question_type == "Association"].leak_rule.value_counts()
    import re
    dq = (ROOT / "data/processed/data_quality.md").read_text()
    n_repeat = re.search(r"in (\d+) Association rows the gold option repeats", dq).group(1)
    n = {
        "nAssocGoldText": fmt_int(al.get("gold_text_in_stem", 0)),
        "nAssocGoldStateName": fmt_int(al.get("gold_state_name", 0)), "nAssocRepeat": n_repeat,
        "nRaw": fmt_int(raw_n), "nKept": fmt_int(len(prompts)), "nDropped": fmt_int(raw_n - len(prompts)),
        "nAmbiguous": fmt_int(prompts.ambiguous_gold.sum()), "nLeaks": fmt_int(prompts.leaks_answer.sum()),
        "nAnalysis": fmt_int(len(an)), "nDupOptions": fmt_int(prompts.duplicate_options.sum()),
        "nDupOnly": fmt_int((prompts.duplicate_options & ~prompts.ambiguous_gold).sum()),
        "nAssocState": fmt_int(len(self_state)), "nAssocStateLeak": fmt_int(self_state.leaks_answer.sum()),
        "nAssocLeak": fmt_int((assoc.leaks_answer).sum()),
        "pctEntity": f"{100 * ea.entity.notna().mean():.1f}", "pctSpan": f"{100 * ea.span_start.notna().mean():.1f}",
        "nEntities": fmt_int(fmeta["n_entities"]), "rhoPageviews": f"{rho:.2f}", "nPageviews": fmt_int(len(b)),
        "lenSlope": f"{fmeta['length_adjustment']['slope']:.2f}", "lenR": f"{fmeta['length_adjustment']['r']:.2f}",
        "nStates": fmt_int(prompts.state.nunique()),
    }
    return n


def write_numbers(n: dict, status: dict) -> None:
    lines = ["% generated by analysis/figures.py; do not edit"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in n.items()]
    lines += [f"\\newcommand{{\\Src{k}}}{{{SOURCE_TEXT.get(v, 'pending')}}}" for k, v in status.items()]
    overall = "full" if all(v == "full" for v in status.values()) else "pilot"
    lines.append(f"\\newcommand{{\\ResultsSource}}{{{SOURCE_TEXT[overall]}}}")
    (GEN / "numbers.tex").write_text("\n".join(lines) + "\n")
    (GEN / "status.json").write_text(json.dumps(status, indent=2))


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    GEN.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts()
    pr = prompts.set_index("qid")
    freq = pd.read_parquet(ROOT / "data/processed/freq.parquet")
    n = numbers(prompts)
    status = {
        "Curves": fig_layer_curves(pr), "Dist": fig_depth_distribution(), "Freq": fig_frequency(freq),
        "Rel": fig_reliability(), "Patch": fig_patching(), "Models": tab_models(prompts), "Effects": tab_effects(),
    }
    tab_data(prompts, n)
    write_numbers(n, status)
    for k, v in status.items():
        print(f"{k:8s} {v}")
    print(f"wrote {FIG} and {GEN}")


if __name__ == "__main__":
    main()
