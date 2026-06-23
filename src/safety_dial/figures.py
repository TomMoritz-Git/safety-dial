"""Publication-quality figures for the read<->write refusal-dial story.

Design choices follow common paper conventions:

* a single shared style (``_apply_style``) and a fixed, colorblind-safe
  (Okabe-Ito) model->color map used across every figure;
* AUCs shown as a forest plot (point + CI), never zero-baseline bars;
* terse axis labels with the detail left to the (external) caption, top/right
  spines removed, a light grid, markers at data points;
* vector PDF + 300-dpi PNG output.

Three primary figures (fig1 read|write, fig2 graded ramp, fig3 calibration gap)
plus a supplementary AUC heatmap. All read cached tables / responses under
``results/`` and write to ``figures/`` (Agg backend, no display).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config  # noqa: E402
from .monitor import youden_threshold  # noqa: E402
from .stats import wilson_ci  # noqa: E402

# Largest coefficient at which every model's random control still refuses <=10%.
COMMON_BAND = 0.75

# Okabe-Ito colorblind-safe palette, mapped to a fixed model order so a model
# has the same color in every figure.
_OKABE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "vermillion": "#D55E00",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
}
MODEL_COLORS = {
    "qwen3-1.7b": _OKABE["blue"],
    "gemma3-1b": _OKABE["orange"],
    "smollm3-3b": _OKABE["green"],
    "granite3.1-2b": _OKABE["yellow"],
    "qwen2.5-1.5b": _OKABE["purple"],
    "llama3.2-1b": _OKABE["vermillion"],
}
_OVER_COLOR = _OKABE["orange"]
_UNDER_COLOR = _OKABE["blue"]


def _apply_style() -> None:
    """Set a consistent, paper-friendly Matplotlib style (idempotent)."""
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 11,
            "axes.titlesize": 11,
            "axes.titlepad": 10,
            "axes.labelsize": 11,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.6,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
        }
    )


def _models(present: set[str]) -> list[str]:
    """Configured model order, restricted to those present in the data."""
    return [s.key for s in config.MODELS if s.key in present]


def _save(fig, path: Path) -> Path:
    """Save both a 300-dpi PNG and a vector PDF; return the PNG path."""
    path = Path(path)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


def _load(name: str) -> pd.DataFrame:
    return pd.read_parquet(config.RESULTS_DIR / "metrics" / f"{name}.parquet")


def _graded() -> pd.DataFrame:
    resp = pd.read_parquet(config.RESULTS_DIR / "responses.parquet")
    labels = pd.read_parquet(config.RESULTS_DIR / "labels.parquet")
    # Inner join: unlabeled rows are dropped (matches metrics_stage, which warns
    # and drops them too), so figures and metric tables always share an N.
    graded = resp[resp["kind"] == "graded"].merge(
        labels[["row_id", "refused"]], on="row_id", how="inner"
    )
    graded["refused"] = graded["refused"].astype(bool)
    return graded


# --------------------------------------------------------------------------
# Figure 1 -- hero: left READ separation | right WRITE dial
# --------------------------------------------------------------------------
def hero_read_write(
    graded: pd.DataFrame, pooled_tbl: pd.DataFrame, dial_tbl: pd.DataFrame, path: Path
) -> Path:
    """Two-panel thesis figure: the axis reads refusal (left) and writes it (right)."""
    _apply_style()
    models = _models(set(pooled_tbl["model"]))
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(12, 4.6))

    # (a) READ: the projection separates complied vs refused items. Center the
    # projection within each scenario (so this is within-topic severity, not
    # topic detection) and z-score within model (so models pool on one scale).
    g = graded.copy()
    g["c"] = g.groupby(["model", "scenario_id"])["projection"].transform(lambda s: s - s.mean())
    g["z"] = g.groupby("model")["c"].transform(lambda s: s / (s.std() + 1e-9))
    refused, complied = g[g["refused"]]["z"], g[~g["refused"]]["z"]
    bins = np.linspace(g["z"].quantile(0.01), g["z"].quantile(0.99), 36)
    axl.hist(complied, bins=bins, density=True, color=_OKABE["sky"], alpha=0.75, label="complied")
    axl.hist(
        refused, bins=bins, density=True, color=_OKABE["vermillion"], alpha=0.7, label="refused"
    )
    tau = youden_threshold(g["z"].to_numpy(), g["refused"].to_numpy())
    axl.axvline(tau, color="#333", ls="--", lw=1.2)
    axl.annotate(
        "monitor\nthreshold",
        (tau, axl.get_ylim()[1]),
        xytext=(6, -4),
        textcoords="offset points",
        fontsize=8,
        color="#333",
        va="top",
    )
    lo, hi = pooled_tbl["within_auc"].min(), pooled_tbl["within_auc"].max()
    axl.set_xlabel("projection on refusal axis (within-topic, $z$)")
    axl.set_ylabel("density")
    axl.set_title(f"Reading refusal (within-topic AUC {lo:.2f}–{hi:.2f})")
    axl.grid(axis="y")
    axl.legend(loc="upper right")

    # (b) WRITE: dial per model + averaged random control, specific band shaded.
    axr.axvspan(0, COMMON_BAND, color=_OKABE["green"], alpha=0.07, zorder=0)
    axr.annotate(
        "specific band",
        (COMMON_BAND / 2, 1.0),
        ha="center",
        va="top",
        fontsize=9,
        color="#185",
        xytext=(0, -2),
        textcoords="offset points",
    )
    rand_by_c: dict[float, list[float]] = {}
    for model in models:
        sub = dial_tbl[dial_tbl["model"] == model].sort_values("coeff")
        real = sub[sub["control"] == "real"]
        axr.plot(real["coeff"], real["refuse_rate"], "o-", color=MODEL_COLORS[model], label=model)
        for _, r in sub[sub["control"] == "random"].iterrows():
            rand_by_c.setdefault(r["coeff"], []).append(r["refuse_rate"])
    cs = sorted(rand_by_c)
    rand_mean = [float(np.mean(rand_by_c[c])) for c in cs]
    axr.plot(cs, rand_mean, "s--", color="#555", lw=1.8, label="random control")
    axr.set_ylim(0, 1.02)
    axr.set_xlim(0, max(cs))
    axr.set_xlabel("steering coefficient $c$")
    axr.set_ylabel("benign-prompt refusal rate")
    axr.set_title("Turning refusal")
    axr.grid(axis="y")
    axr.legend(loc="lower right", framealpha=0.9, frameon=True, edgecolor="none")
    return _save(fig, path)


# --------------------------------------------------------------------------
# Figure 2 -- graded ramp: behavior and perception climb together
# --------------------------------------------------------------------------
def graded_ramp(graded: pd.DataFrame, path: Path) -> Path:
    """Refusal rate (top) and internal projection (bottom) vs severity level."""
    _apply_style()
    models = _models(set(graded["model"]))
    g = graded.copy()
    g["pz"] = g.groupby("model")["projection"].transform(
        lambda s: (s - s.mean()) / (s.std() + 1e-9)
    )
    levels = sorted(g["level"].unique())

    fig, (axt, axb) = plt.subplots(2, 1, figsize=(6.6, 7.0), sharex=True)
    for model in models:
        sub = g[g["model"] == model]
        axt.plot(
            levels,
            [sub[sub.level == lv]["refused"].mean() for lv in levels],
            "-",
            color=MODEL_COLORS[model],
            lw=1.2,
            alpha=0.6,
            marker="o",
            markersize=4,
            label=model,
        )
        axb.plot(
            levels,
            [sub[sub.level == lv]["pz"].mean() for lv in levels],
            "-",
            color=MODEL_COLORS[model],
            lw=1.2,
            alpha=0.6,
            marker="o",
            markersize=4,
        )

    # All-model average with a shaded uncertainty band.
    beh, beh_lo, beh_hi, per, per_se = [], [], [], [], []
    for lv in levels:
        sub = g[g["level"] == lv]
        ci = wilson_ci(int(sub["refused"].sum()), len(sub))
        beh.append(ci.point)
        beh_lo.append(ci.lo)
        beh_hi.append(ci.hi)
        per.append(sub["pz"].mean())
        per_se.append(sub["pz"].std() / np.sqrt(len(sub)))
    axt.plot(levels, beh, "o-", color="black", lw=2.5, label="all-model avg", zorder=5)
    axt.fill_between(levels, beh_lo, beh_hi, color="black", alpha=0.12, zorder=4)
    per, per_se = np.array(per), np.array(per_se)
    axb.plot(levels, per, "o-", color="black", lw=2.5, zorder=5)
    axb.fill_between(levels, per - per_se, per + per_se, color="black", alpha=0.12, zorder=4)

    axt.set_ylim(0, 1)
    axt.set_ylabel("refusal rate")
    axt.set_title("Behavior: refusal rate")
    axt.grid(axis="y")
    axt.legend(loc="upper left", ncol=2)
    axb.set_ylabel("projection ($z$-score)")
    axb.set_title("Internal perception: projection on refusal axis")
    axb.set_xticks(levels, [f"L{lv}" for lv in levels])
    axb.set_xlabel("severity level (L0 legitimate $\\rightarrow$ L4 disallowed)")
    axb.grid(axis="y")
    return _save(fig, path)


# --------------------------------------------------------------------------
# Figure 3 -- calibration gap: over- vs under-refusal per model
# --------------------------------------------------------------------------
def calibration_gap(monitor_tbl: pd.DataFrame, path: Path) -> Path:
    """Grouped bars: mean over- and under-refusal rate per model."""
    _apply_style()
    models = _models(set(monitor_tbl["model"]))
    agg = monitor_tbl.groupby("model")[["over_rate", "under_rate"]].mean()
    over = [agg.loc[m, "over_rate"] for m in models]
    under = [agg.loc[m, "under_rate"] for m in models]
    x = np.arange(len(models))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(x - w / 2, over, w, color=_OVER_COLOR, label="over-refusal (refuse, low reading)")
    ax.bar(x + w / 2, under, w, color=_UNDER_COLOR, label="under-refusal (comply, high reading)")
    for xi, (o, u) in enumerate(zip(over, under, strict=True)):
        ax.annotate(
            f"{o:.0%}",
            (xi - w / 2, o),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            fontsize=9,
        )
        ax.annotate(
            f"{u:.0%}",
            (xi + w / 2, u),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            fontsize=9,
        )
    ax.set_xticks(x, models)
    ax.set_ylabel("rate (fraction of items)")
    ax.set_ylim(0, max(over + under) * 1.18)
    ax.set_title("Calibration gap: monitor vs. action")
    ax.grid(axis="y")
    ax.legend(loc="upper left")
    return _save(fig, path)


# --------------------------------------------------------------------------
# Supplementary -- per-model AUC forest plot and per-cell heatmap
# --------------------------------------------------------------------------
def auc_forest(pooled_tbl: pd.DataFrame, path: Path) -> Path:
    """Supplementary: pooled within-topic AUC per model with bootstrap 95% CI."""
    _apply_style()
    order = list(reversed(_models(set(pooled_tbl["model"]))))
    fig, ax = plt.subplots(figsize=(6.4, 0.7 * len(order) + 1.4))
    for y, model in enumerate(order):
        r = pooled_tbl[pooled_tbl["model"] == model].iloc[0]
        auc, lo, hi = r["within_auc"], r["within_lo"], r["within_hi"]
        ax.errorbar(
            auc,
            y,
            xerr=[[auc - lo], [hi - auc]],
            fmt="o",
            color=MODEL_COLORS[model],
            ecolor=MODEL_COLORS[model],
            elinewidth=2,
            capsize=4,
            markersize=8,
        )
        ax.annotate(
            f"{auc:.2f}",
            (auc, y),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=9,
            color="#333",
        )
    ax.axvline(0.5, color="#999", lw=1, ls="--")
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_yticks(range(len(order)), order)
    ax.set_xlim(0.45, 1.0)
    ax.set_xlabel("within-topic AUC (pooled, 95% CI; 0.5 = chance)")
    ax.set_title("Refusal monitor by model")
    ax.grid(axis="x")
    return _save(fig, path)


def heatmap(monitor_tbl: pd.DataFrame, path: Path) -> Path:
    """Supplementary: within-topic AUC for every (safeguard x model) cell."""
    _apply_style()
    grid = monitor_tbl.pivot(index="safeguard", columns="model", values="within_auc")
    grid = grid.reindex(
        index=[s for s in config.SAFEGUARDS if s in grid.index],
        columns=_models(set(monitor_tbl["model"])),
    )
    fig, ax = plt.subplots(figsize=(1.5 * len(grid.columns) + 2, 0.9 * len(grid.index) + 2))
    im = ax.imshow(grid.to_numpy(), vmin=0.5, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(grid.columns)), grid.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(grid.index)), grid.index)
    ax.spines[:].set_visible(False)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            val = grid.to_numpy()[i, j]
            if not np.isnan(val):
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color="white" if val < 0.85 else "black",
                    fontsize=11,
                )
    ax.set_title("Within-topic AUC by safeguard and model")
    fig.colorbar(im, ax=ax, label="within-topic AUC", shrink=0.85)
    return _save(fig, path)


def make_all() -> list[Path]:
    """Build the three primary figures plus the supplementary heatmap."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    monitor_tbl = _load("monitor")
    pooled_tbl = _load("monitor_pooled")
    dial_tbl = _load("dial")
    graded = _graded()
    return [
        hero_read_write(
            graded, pooled_tbl, dial_tbl, config.FIGURES_DIR / "fig1_hero_read_write.png"
        ),
        graded_ramp(graded, config.FIGURES_DIR / "fig2_graded_ramp.png"),
        calibration_gap(monitor_tbl, config.FIGURES_DIR / "fig3_calibration_gap.png"),
        auc_forest(pooled_tbl, config.FIGURES_DIR / "supp_auc_forest.png"),
        heatmap(monitor_tbl, config.FIGURES_DIR / "supp_auc_heatmap.png"),
    ]
