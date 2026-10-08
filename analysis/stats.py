"""Stage 4b: does crystallization depth depend on documentation? PRELIMINARY.

Inputs: an across-variant depth table (analysis/depth.py aggregate) per model, joined
with freq.parquet (entity log_freq, state_log_freq) and prompts.parquet.

For each model and each depth metric (as d = layers / L):
  rows      questions correct under the content-averaged readout (correct_cyc_final)
            with a defined depth and an entity log_freq
  model     d ~ z(entity frequency) + z(state log_freq) + acc + z(stem words)
                + z(log entity tokens) + z(entity words) + C(attribute) + C(question_type)
            run twice, once per entity frequency measure (freq_measure):
              raw     log_freq = log1p(exact-string corpus count)
              lenadj  log_freq_lenadj = residual of log_freq on log entity token length
            Exact-string counts fall with entity length, so both models control for the
            entity's token length and word count.
            acc = mean correctness across the orders. Mixed model with a random
            intercept per state (statsmodels MixedLM, REML); if it fails to converge
            cleanly, OLS with state-clustered SEs is reported instead. OLS-clustered is
            always fitted as a robustness column.
  control   state-label permutation: state -> state_log_freq is shuffled across the
            states, the OLS is refitted, n_perm times; p = (1 + #|b_perm| >= |b_obs|) /
            (1 + n_perm) for the state_log_freq coefficient.
  per state median depth with percentile bootstrap CIs (resampling questions within state).
  Holm  p_holm: Holm-Bonferroni across the depth metrics, within (model, estimator,
        freq_measure, term).
  equivalence  beta_std = beta / SD(depth); equivalent_null when the 90% CI of beta_std
        lies within +/- EQUIV_BOUND (0.10), the pre-registered smallest effect of interest.

Outputs (every file says PRELIMINARY): results/prelim/PRELIMINARY_*.csv + README.md.

  python -m analysis.stats --models llama31_8b qwen25_7b --agg prompt+prompt_permuted
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from crystal.io import load_freq, load_prompts, load_run_config, model_config, resolve

LABEL = "PRELIMINARY"
METRICS = ("d_soft", "l_star_cyc", "d_margin", "l_star_cal_mean", "l_star_mean")
FORMULA = ("y ~ z_freq + z_state_logfreq + acc + z_stem_words + z_ent_tokens + z_ent_words"
           " + C(attribute) + C(question_type)")
FREQ_MEASURES = {"raw": "log_freq", "lenadj": "log_freq_lenadj"}
TERMS = ("z_freq", "z_state_logfreq", "acc", "z_stem_words", "z_ent_tokens", "z_ent_words")


def holm(p: pd.Series) -> pd.Series:
    """Holm-Bonferroni adjusted p-values (step-down, monotone, capped at 1)."""
    p = p.astype(float)
    order = p.sort_values().index
    m = p.notna().sum()
    adj, running = pd.Series(np.nan, index=p.index), 0.0
    for i, idx in enumerate(order):
        if np.isnan(p[idx]):
            continue
        running = max(running, min(1.0, (m - i) * p[idx]))
        adj[idx] = running
    return adj


def build_frame(model: str, agg_name: str, readout: str = "logit") -> pd.DataFrame:
    if model_config(model).get("proxy"):
        raise SystemExit(f"{model} is a proxy model; never used in analysis")
    agg = pd.read_parquet(resolve("results") / f"depth_{model}_{agg_name}_{readout}_agg.parquet")
    fr = load_freq()[["qid", "log_freq", "log_freq_lenadj", "entity_n_tokens", "entity_n_words", "state_log_freq"]]
    pr = load_prompts()[["qid", "stem"]]
    df = agg.merge(fr, on="qid", how="left").merge(pr, on="qid", how="left")
    df["stem_words"] = df.stem.str.split().str.len()
    df["log_ent_tokens"] = np.log(df.entity_n_tokens.astype(float))
    for c, z in (("log_freq", "z_freq_raw"), ("log_freq_lenadj", "z_freq_lenadj"),
                 ("state_log_freq", "z_state_logfreq"), ("stem_words", "z_stem_words"),
                 ("log_ent_tokens", "z_ent_tokens"), ("entity_n_words", "z_ent_words")):
        x = df[c].astype(float)
        df[z] = (x - x.mean()) / x.std()
    df["model"] = model
    return df


def metric_rows(df: pd.DataFrame, metric: str, freq_measure: str = "raw") -> pd.DataFrame:
    d = df[df.correct_cyc_final & df[FREQ_MEASURES[freq_measure]].notna() & df.z_ent_tokens.notna()].copy()
    d["y"] = d[f"d_frac_{metric}"].astype(float)
    d["z_freq"] = d[f"z_freq_{freq_measure}"]
    return d[d.y.notna()]


def formula_for(d: pd.DataFrame) -> str:
    """Drop a control with no variance in this subset (e.g. acc is always 1 for
    l_star_mean, which needs every order correct); otherwise it is collinear with
    the intercept and the fit is singular."""
    f = FORMULA
    if d["acc"].nunique() < 2:
        f = f.replace(" + acc", "")
    return f


def fit_ols(d: pd.DataFrame):
    return smf.ols(formula_for(d), d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.state)[0]})


def fit_mixed(d: pd.DataFrame):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            r = smf.mixedlm(formula_for(d), d, groups=d["state"]).fit(reml=True, method="lbfgs")
        except Exception as e:  # noqa: BLE001
            return None, f"failed: {type(e).__name__}"
    bad = [str(x.message)[:80] for x in w if "converge" in str(x.message).lower() or "singular" in str(x.message).lower()
           or "boundary" in str(x.message).lower()]
    if not r.converged or bad:
        return None, "not clean: " + ("; ".join(bad) or "converged=False")
    return r, "ok"


EQUIV_BOUND = 0.10   # pre-registered smallest effect of interest, |beta_std| (ANALYSIS_PLAN.md)


def coef_row(res, term, kind, y_sd=None):
    ci = res.conf_int().loc[term]
    ci90 = res.conf_int(alpha=0.10).loc[term]
    row = {"estimator": kind, "term": term, "beta": res.params[term], "ci_low": ci[0], "ci_high": ci[1],
           "p": res.pvalues[term]}
    if y_sd:
        # standardised: depth SDs per SD of the predictor (predictors are z-scored)
        row.update(beta_std=res.params[term] / y_sd, ci90_low_std=ci90[0] / y_sd, ci90_high_std=ci90[1] / y_sd)
        row["equivalent_null"] = bool(-EQUIV_BOUND < row["ci90_low_std"] and row["ci90_high_std"] < EQUIV_BOUND)
    return row


def permutation_state(d: pd.DataFrame, n_perm: int, rng: np.random.Generator) -> dict:
    f = formula_for(d)
    obs = smf.ols(f, d).fit().params["z_state_logfreq"]
    states = d.state.unique()
    level = d.groupby("state").z_state_logfreq.first()
    null = np.empty(n_perm)
    for i in range(n_perm):
        mapping = dict(zip(states, rng.permutation(level.loc[states].to_numpy())))
        dd = d.assign(z_state_logfreq=d.state.map(mapping))
        null[i] = smf.ols(f, dd).fit().params["z_state_logfreq"]
    return {"beta_obs": obs, "null_mean": null.mean(), "null_sd": null.std(),
            "null_q025": np.quantile(null, 0.025), "null_q975": np.quantile(null, 0.975),
            "p_perm": (1 + np.sum(np.abs(null) >= abs(obs))) / (1 + n_perm), "n_perm": n_perm}


def state_medians(d: pd.DataFrame, n_boot: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for s, g in d.groupby("state"):
        y = g.y.to_numpy()
        boots = np.median(rng.choice(y, (n_boot, len(y)), replace=True), axis=1)
        rows.append({"state": s, "n": len(y), "median_d": np.median(y), "ci_low": np.quantile(boots, 0.025),
                     "ci_high": np.quantile(boots, 0.975), "state_log_freq": g.state_log_freq.iloc[0]})
    return pd.DataFrame(rows).sort_values("median_d")


def main(argv=None):
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="+", default=["llama31_8b", "qwen25_7b"])
    ap.add_argument("--agg", default="prompt+prompt_permuted", help="aggregate name, e.g. cyc")
    ap.add_argument("--readout", default="logit")
    ap.add_argument("--metrics", nargs="+", default=list(METRICS))
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--freq-measures", nargs="+", default=list(FREQ_MEASURES))
    ap.add_argument("--out", default="results/prelim")
    args = ap.parse_args(argv)
    cfg = load_run_config()
    rng = np.random.default_rng(cfg["seed"])
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)

    reg, perm, med, desc = [], [], [], []
    for m in args.models:
        df = build_frame(m, args.agg, args.readout)
        L = int(df.L.iloc[0])
        for fm in args.freq_measures:
            for metric in args.metrics:
                d = metric_rows(df, metric, fm)
                desc.append({"model": m, "freq_measure": fm, "metric": metric, "n": len(d),
                             "n_states": d.state.nunique(), "L": L, "d_mean": d.y.mean(), "d_sd": d.y.std(),
                             "layers_sd": d.y.std() * L})
                ols = fit_ols(d)
                mixed, status = fit_mixed(d)
                for term in [x for x in TERMS if x in ols.params]:
                    base = {"label": LABEL, "model": m, "freq_measure": fm, "metric": metric, "n": len(d), "L": L,
                            "mixed_status": status}
                    if mixed is not None:
                        reg.append({**base, **coef_row(mixed, term, "mixedlm_state_re", d.y.std())})
                    reg.append({**base, **coef_row(ols, term, "ols_cluster_state", d.y.std())})
                perm.append({"label": LABEL, "model": m, "freq_measure": fm, "metric": metric, "n": len(d),
                             **permutation_state(d, args.n_perm, rng)})
                if fm == "raw":   # per-state medians do not depend on the frequency measure
                    sm = state_medians(d, args.n_boot, rng)
                    sm.insert(0, "metric", metric)
                    sm.insert(0, "model", m)
                    sm.insert(0, "label", LABEL)
                    med.append(sm)
                print(f"[{LABEL}] {m} {fm} {metric}: n={len(d)} mixed={status}")

    reg = pd.DataFrame(reg)
    reg["beta_layers"] = reg.beta * reg.L
    reg["ci_low_layers"] = reg.ci_low * reg.L
    reg["ci_high_layers"] = reg.ci_high * reg.L
    reg["p_holm"] = reg.groupby(["model", "estimator", "freq_measure", "term"]).p.transform(holm)
    reg.to_csv(out / "PRELIMINARY_regression.csv", index=False)
    pd.concat(med).to_csv(out / "PRELIMINARY_state_medians.csv", index=False)
    pd.DataFrame(desc).assign(label=LABEL).to_csv(out / "PRELIMINARY_descriptives.csv", index=False)
    print(f"wrote {out}/PRELIMINARY_*.csv")
    perm_df = pd.DataFrame(perm)
    perm_df["p_perm_holm"] = perm_df.groupby(["model", "freq_measure"]).p_perm.transform(holm)
    perm_df.to_csv(out / "PRELIMINARY_state_permutation.csv", index=False)
    focus = reg[reg.term.isin(["z_freq", "z_state_logfreq"])]
    print(focus[["model", "freq_measure", "metric", "estimator", "term", "n", "beta_layers", "ci_low_layers",
                 "ci_high_layers", "p", "p_holm"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
