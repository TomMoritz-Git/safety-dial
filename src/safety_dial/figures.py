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
from matplotlib.lines import Line2D  # noqa: E402

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
# Display labels for legends (Title Case, matching the write-up).
MODEL_LABELS = {
    "qwen3-1.7b": "Qwen3-1.7B",
    "gemma3-1b": "Gemma-3-1B",
    "smollm3-3b": "SmolLM3-3B",
    "granite3.1-2b": "Granite-3.1-2B",
    "qwen2.5-1.5b": "Qwen2.5-1.5B",
    "llama3.2-1b": "Llama-3.2-1B",
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
        labels[["row_id", "refused", "label"]], on="row_id", how="inner"
    )
    graded["refused"] = graded["refused"].astype(bool)
    return graded


def _dial_real() -> pd.DataFrame:
    """Real-control dial rows (per benign prompt and coeff) with refusal labels."""
    resp = pd.read_parquet(config.RESULTS_DIR / "responses.parquet")
    labels = pd.read_parquet(config.RESULTS_DIR / "labels.parquet")
    dial = resp[resp["kind"] == "dial"].merge(
        labels[["row_id", "refused"]], on="row_id", how="inner"
    )
    real = dial[dial["control"] == "real"].copy()
    real["refused"] = real["refused"].astype(bool)
    return real[["model", "coeff", "prompt", "refused"]]


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
    axl.set_xlabel("projection on the refusal direction")
    axl.set_ylabel("density")
    axl.set_title("Reading refusal")
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
        axr.plot(
            real["coeff"],
            real["refuse_rate"],
            "o-",
            color=MODEL_COLORS[model],
            lw=1.3,
            markersize=4,
            alpha=0.85,
            label=model,
        )
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
# Figure 1 (hero) -- one direction reads and writes refusal, on one axis
# --------------------------------------------------------------------------
def hero_refusal_curve(
    graded: pd.DataFrame, benign_proj: pd.DataFrame, dial_real: pd.DataFrame, path: Path
) -> Path:
    """Refusal vs. position on the refusal direction, by two routes onto one axis.

    Natural graded prompts sit at their measured projection (the *read*: position
    predicts refusal). Steered benign prompts sit at ``u0 + coeff*||raw||`` -- their
    read-layer position once the dial pushes them along the direction (the *write*).
    Each model is standardised onto a shared axis (centred on its Youden threshold,
    scaled by its graded-projection SD), so x=0 is the monitor threshold for every
    model and the two routes can be pooled and compared.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(7.4, 5.4))

    nat_parts, st_parts = [], []
    for model, g in graded.groupby("model"):
        u = g["projection"].to_numpy(float)
        ref = g["refused"].to_numpy(bool)
        tau = youden_threshold(u, ref)
        sd = float(u.std()) + 1e-9
        nat_parts.append(pd.DataFrame({"x": (u - tau) / sd, "refused": ref}))
        bm = benign_proj[benign_proj["model"] == model][["prompt", "u0", "rawnorm"]]
        sub = dial_real[dial_real["model"] == model].merge(bm, on="prompt", how="inner")
        if len(sub):
            x = (sub["u0"] + sub["coeff"] * sub["rawnorm"] - tau) / sd
            st_parts.append(
                pd.DataFrame({"x": x.to_numpy(), "refused": sub["refused"].to_numpy(bool)})
            )
    nat = pd.concat(nat_parts, ignore_index=True)
    st = pd.concat(st_parts, ignore_index=True)

    edges = np.arange(-2.5, 3.5001, 0.5)

    def binned(df: pd.DataFrame, min_n: int = 8):
        mid, pt, lo, hi = [], [], [], []
        for i in range(len(edges) - 1):
            m = (df["x"] >= edges[i]) & (df["x"] < edges[i + 1])
            k = int(m.sum())
            if k < min_n:
                continue
            ci = wilson_ci(int(df.loc[m, "refused"].sum()), k)
            mid.append((edges[i] + edges[i + 1]) / 2)
            pt.append(ci.point)
            lo.append(ci.lo)
            hi.append(ci.hi)
        return np.array(mid), np.array(pt), np.array(lo), np.array(hi)

    ax.axvline(0, color="#333", ls="--", lw=1.2, zorder=1)
    ax.annotate(
        "monitor\nthreshold",
        (0, 1.0),
        xytext=(7, -2),
        textcoords="offset points",
        fontsize=8,
        color="#333",
        va="top",
    )

    mx, my, mlo, mhi = binned(nat)
    ax.fill_between(mx, mlo, mhi, color=_OKABE["sky"], alpha=0.18, zorder=2)
    ax.plot(
        mx,
        my,
        "o-",
        color=_OKABE["blue"],
        lw=2.0,
        markersize=6,
        label="natural prompts (read)",
        zorder=4,
    )
    sx, sy, _, _ = binned(st)
    ax.plot(
        sx,
        sy,
        "s--",
        color=_OKABE["vermillion"],
        lw=1.8,
        markersize=6,
        label="steered benign prompts (write)",
        zorder=3,
    )

    ax.set_ylim(-0.02, 1.04)
    ax.set_xlim(edges[0], edges[-1])
    ax.set_xlabel("position along the refusal direction (0 = monitor threshold)")
    ax.set_ylabel("probability of refusing")
    ax.set_title("One direction reads and writes refusal")
    ax.grid(axis="y")
    ax.legend(loc="lower right")
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
# Figure 4 -- the over-/under-refusal frontier (Act III: the use)
# --------------------------------------------------------------------------
def calibration_frontier(pooled_intent_tbl: pd.DataFrame, path: Path) -> Path:
    """Each model as a point on the over- vs under-refusal frontier.

    Over-refusal (x) is the refusal rate on L0 *legitimate* requests; under-
    refusal (y) is the comply rate on L4 *disallowed* requests. Both are scored
    against the ladder's ground-truth intent (not the monitor's own threshold),
    with Wilson 95% CIs. Models trade one error for the other: the dashed guide
    is the equal-error diagonal.
    """
    _apply_style()
    models = _models(set(pooled_intent_tbl["model"]))
    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    lim = 0.0
    for model in models:
        r = pooled_intent_tbl[pooled_intent_tbl["model"] == model].iloc[0]
        x, y = r["over_refusal"], r["under_refusal"]
        ax.errorbar(
            x,
            y,
            xerr=[[x - r["over_lo"]], [r["over_hi"] - x]],
            yerr=[[y - r["under_lo"]], [r["under_hi"] - y]],
            fmt="o",
            color=MODEL_COLORS[model],
            ecolor=MODEL_COLORS[model],
            elinewidth=1.6,
            capsize=3,
            markersize=9,
            label=model,
        )
        ax.annotate(
            model,
            (x, y),
            textcoords="offset points",
            xytext=(8, 5),
            fontsize=9,
            color="#333",
        )
        lim = max(lim, r["over_hi"], r["under_hi"])
    lim = min(1.0, lim * 1.15)
    ax.plot([0, lim], [0, lim], ls="--", color="#bbb", lw=1, zorder=0)
    ax.annotate(
        "equal error",
        (lim, lim),
        textcoords="offset points",
        xytext=(-4, -12),
        ha="right",
        fontsize=8,
        color="#999",
    )
    ax.set_xlim(-0.01, lim)
    ax.set_ylim(-0.01, lim)
    ax.set_xlabel("over-refusal: refuse rate on L0 legitimate requests")
    ax.set_ylabel("under-refusal: comply rate on L4 disallowed requests")
    ax.set_title("Two kinds of mistake")
    ax.grid(True)
    return _save(fig, path)


# --------------------------------------------------------------------------
# Figure 5 -- where the single-direction thesis breaks (Act II: the limit)
# --------------------------------------------------------------------------
def misinformation_breakdown(
    graded: pd.DataFrame, monitor_tbl: pd.DataFrame, path: Path, healthy: str = "privacy"
) -> Path:
    """Contrast a healthy safeguard with misinformation: ramp + per-cell AUC.

    Left: all-model-average refusal ramp (Wilson band) climbs monotonically for
    the healthy safeguard but is flat/non-monotone for misinformation. Right:
    within-topic AUC per (model, safeguard) — misinformation cells sit far below
    the rest, where they are reportable at all.
    """
    _apply_style()
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    levels = sorted(graded["level"].unique())
    sg_color = {healthy: _OKABE["green"], "misinformation": _OKABE["vermillion"]}

    for sg in (healthy, "misinformation"):
        pts, los, his = [], [], []
        for lv in levels:
            sub = graded[(graded["safeguard"] == sg) & (graded["level"] == lv)]
            ci = wilson_ci(int(sub["refused"].sum()), len(sub))
            pts.append(ci.point)
            los.append(ci.lo)
            his.append(ci.hi)
        axl.plot(levels, pts, "o-", color=sg_color[sg], label=sg)
        axl.fill_between(levels, los, his, color=sg_color[sg], alpha=0.12)
    axl.set_ylim(0, 1)
    axl.set_xticks(levels, [f"L{lv}" for lv in levels])
    axl.set_xlabel("intent level (L0 legitimate to L4 disallowed)")
    axl.set_ylabel("refusal rate (average over models)")
    axl.set_title("Refusal by intent level")
    axl.grid(axis="y")
    axl.legend(loc="upper left")

    # Right: within-topic AUC per safeguard, one marker per model.
    sgs = [s for s in config.SAFEGUARDS if s in set(monitor_tbl["safeguard"])]
    plotted: list[str] = []
    for xi, sg in enumerate(sgs):
        cells = monitor_tbl[monitor_tbl["safeguard"] == sg]
        for _, r in cells.iterrows():
            if np.isnan(r["within_auc"]):
                continue
            jitter = (hash(r["model"]) % 5 - 2) * 0.04
            axr.plot(
                xi + jitter,
                r["within_auc"],
                "o",
                color=MODEL_COLORS.get(r["model"], "#666"),
                markersize=7,
            )
            if r["model"] not in plotted:
                plotted.append(r["model"])
    axr.axhline(0.5, color="#999", lw=1, ls="--")
    axr.annotate(
        "chance",
        (len(sgs) - 1, 0.5),
        xytext=(0, 4),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#999",
    )
    axr.set_xticks(range(len(sgs)), [s.replace("_", " ") for s in sgs], rotation=20, ha="right")
    axr.set_ylim(0.45, 1.02)
    axr.set_ylabel("within-topic AUC")
    axr.set_title("Read quality by domain")
    axr.grid(axis="y")
    # One marker per model, so a legend is needed to read the colors.
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=7,
            color=MODEL_COLORS.get(m, "#666"),
            label=MODEL_LABELS.get(m, m),
        )
        for m in _models(set(plotted))
    ]
    axr.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 0.12),
        fontsize=8,
        handletextpad=0.2,
    )
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


def label_composition(graded: pd.DataFrame, path: Path) -> Path:
    """Supplementary: the three-way response mix (refuse / partial / comply) by level.

    The headline metrics binarize to refuse/comply; this keeps the ``partial``
    middle that binarization hides. Pooled over models and safeguards, stacked by
    severity level. The partial mass sits at the *legitimate* end (L0/L1) — these
    small models half-answer benign asks more than they refuse them — and gives
    way to refusal by L3/L4.
    """
    _apply_style()
    levels = sorted(graded["level"].unique())
    order = ["full_comply", "partial", "full_refuse"]
    colors = {
        "full_comply": _OKABE["sky"],
        "partial": _OKABE["yellow"],
        "full_refuse": _OKABE["vermillion"],
    }
    pretty = {"full_comply": "comply", "partial": "partial", "full_refuse": "refuse"}
    fracs = {
        lbl: [
            (graded[graded["level"] == lv]["label"] == lbl).mean()
            if (graded["level"] == lv).any()
            else 0.0
            for lv in levels
        ]
        for lbl in order
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    bottom = np.zeros(len(levels))
    for lbl in order:
        vals = np.array(fracs[lbl])
        ax.bar(levels, vals, bottom=bottom, color=colors[lbl], label=pretty[lbl], width=0.7)
        bottom += vals
    ax.set_xticks(levels, [f"L{lv}" for lv in levels])
    ax.set_ylim(0, 1)
    ax.set_xlabel("severity level (L0 legitimate $\\rightarrow$ L4 disallowed)")
    ax.set_ylabel("fraction of responses")
    ax.set_title("Response mix by level (partial is hidden by refuse/comply binarization)")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    return _save(fig, path)


def truncation_comparison(old_graded: pd.DataFrame, new_graded: pd.DataFrame, path: Path) -> Path:
    """Supplementary: the ``partial`` label is mostly a 64-token truncation artifact.

    Left: fraction labelled ``partial`` by level, original 64-token run vs. the
    re-judged 10240-token run — it collapses (L0 68%->6%), so the apparent
    "hedging on benign requests" was largely answers cut off mid-sentence. Right:
    refusal rate by level barely moves, i.e. the binary refuse/comply split that
    the headline metrics use is robust to the cap.
    """
    _apply_style()
    levels = sorted(set(old_graded["level"]) | set(new_graded["level"]))

    def by_level(df, fn):
        return [fn(df[df["level"] == lv]) for lv in levels]

    p_old = by_level(old_graded, lambda s: (s["label"] == "partial").mean())
    p_new = by_level(new_graded, lambda s: (s["label"] == "partial").mean())
    r_old = by_level(old_graded, lambda s: s["refused"].mean())
    r_new = by_level(new_graded, lambda s: s["refused"].mean())

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    x = np.arange(len(levels))
    w = 0.38
    axl.bar(x - w / 2, p_old, w, color="#bbb", label="64 tok (original)")
    axl.bar(x + w / 2, p_new, w, color=_OKABE["yellow"], label="10240 tok (un-truncated)")
    axl.set_title("`partial` label collapses without truncation")
    axl.set_ylabel("fraction labelled partial")
    for ax, lo in ((axl, "upper right"),):
        ax.set_xticks(x, [f"L{lv}" for lv in levels])
        ax.legend(loc=lo)
    axl.set_xlabel("severity level")

    axr.bar(x - w / 2, r_old, w, color="#bbb", label="64 tok")
    axr.bar(x + w / 2, r_new, w, color=_OKABE["vermillion"], label="10240 tok")
    axr.set_title("Refusal rate is stable (binary metric is robust)")
    axr.set_ylabel("refusal rate")
    axr.set_xticks(x, [f"L{lv}" for lv in levels])
    axr.set_xlabel("severity level")
    axr.set_ylim(0, 1)
    axr.legend(loc="upper left")
    return _save(fig, path)


def anchor_robustness(robust_tbl: pd.DataFrame, nsweep_tbl: pd.DataFrame, path: Path) -> Path:
    """Supplementary: the read does not hinge on the 8 deployed anchors.

    Left: within-topic AUC vs. anchors-per-class N (4..32); the deployed N=8 is
    marked, and every curve is flat — N=8 is on the plateau. Right: cosine
    similarity between each bootstrap-resampled pool direction and the deployed
    8-anchor direction (point = full-pool vs. deployed, bar = resample mean).
    """
    _apply_style()
    models = _models(set(robust_tbl["model"]))
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    for model in models:
        sub = nsweep_tbl[nsweep_tbl["model"] == model].sort_values("n_per_class")
        axl.plot(
            sub["n_per_class"],
            sub["mean_auc"],
            "o-",
            color=MODEL_COLORS[model],
            label=model,
            markersize=5,
        )
    axl.axvline(8, color="#333", ls="--", lw=1.1)
    axl.annotate(
        "deployed N=8",
        (8, axl.get_ylim()[0]),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=8,
        color="#333",
    )
    axl.set_xscale("log", base=2)
    axl.set_xticks([4, 8, 16, 32], ["4", "8", "16", "32"])
    axl.set_xlabel("anchors per class (N)")
    axl.set_ylabel("within-topic AUC (resample mean)")
    axl.set_title("Read is flat in anchor count")
    axl.grid(True)
    axl.legend(loc="lower right", ncol=1)

    y = np.arange(len(models))
    for yi, model in enumerate(models):
        r = robust_tbl[robust_tbl["model"] == model].iloc[0]
        axr.barh(yi, r["cos_resampled_mean"], color=MODEL_COLORS[model], alpha=0.5, height=0.6)
        axr.plot(r["cos_pool_deployed"], yi, "D", color=MODEL_COLORS[model], markersize=8)
    axr.axvline(1.0, color="#999", lw=1, ls="--")
    axr.set_yticks(y, models)
    axr.set_xlim(0.9, 1.005)
    axr.set_xlabel("cosine to deployed 8-anchor direction")
    axr.set_title("Direction is stable (bar = resample mean, ◆ = full pool)")
    axr.grid(axis="x")
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
    pooled_intent_tbl = _load("intent_calibration_pooled")
    dial_tbl = _load("dial")
    graded = _graded()
    bp_path = config.RESULTS_DIR / "metrics" / "benign_proj.parquet"
    fig1_path = config.FIGURES_DIR / "fig1_hero_read_write.png"
    if bp_path.exists():
        fig1 = hero_refusal_curve(graded, pd.read_parquet(bp_path), _dial_real(), fig1_path)
    else:  # fall back to the two-panel hero if benign projections weren't captured
        fig1 = hero_read_write(graded, pooled_tbl, dial_tbl, fig1_path)
    return [
        fig1,
        graded_ramp(graded, config.FIGURES_DIR / "fig2_graded_ramp.png"),
        calibration_gap(monitor_tbl, config.FIGURES_DIR / "fig3_calibration_gap.png"),
        calibration_frontier(
            pooled_intent_tbl, config.FIGURES_DIR / "fig4_calibration_frontier.png"
        ),
        misinformation_breakdown(
            graded, monitor_tbl, config.FIGURES_DIR / "fig5_misinfo_breakdown.png"
        ),
        auc_forest(pooled_tbl, config.FIGURES_DIR / "supp_auc_forest.png"),
        heatmap(monitor_tbl, config.FIGURES_DIR / "supp_auc_heatmap.png"),
        anchor_robustness(
            _load("robustness"),
            _load("robustness_nsweep"),
            config.FIGURES_DIR / "supp_anchor_robustness.png",
        ),
        label_composition(graded, config.FIGURES_DIR / "supp_label_composition.png"),
    ]
