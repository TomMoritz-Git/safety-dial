"""Resumable orchestration: model pass -> judge -> metrics.

Artifacts under ``results/``:

* ``directions/<key>.npz`` -- raw/unit/benign_mean/layer + the layer-sweep curve.
* ``responses/<key>.parquet`` -- graded and dial generations for one model.
* ``responses.parquet`` -- all models concatenated (built by the judge stage).
* ``labels.parquet`` -- judge labels joined by ``row_id``.
* ``gold_report.json`` -- judge validation against the gold set.
* ``metrics/{monitor,ramp,dial,dial_gap}.parquet`` -- final tables.

Each stage skips work whose output already exists (``force=True`` to redo), so a
run interrupted by an OOM, a gated model, or a flaky API call resumes cleanly.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, metrics
from .data import Anchors, Safeguard, all_items, load_anchors, load_gold, load_ladders
from .extraction import build_direction, layer_sweep, random_matched


def _dirs() -> dict[str, Path]:
    base = config.RESULTS_DIR
    paths = {
        "base": base,
        "directions": base / "directions",
        "responses": base / "responses",
        "metrics": base / "metrics",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def _anchor_acts(runner, prompts) -> np.ndarray:
    """Stack per-prompt all-layer activations into ``[n_layers, n_prompts, hidden]``."""
    per_prompt = np.stack([runner.activations(p) for p in prompts])  # [n, L, H]
    return per_prompt.transpose(1, 0, 2)


def run_model(
    spec: config.ModelSpec,
    ladders: dict[str, Safeguard],
    anchors: Anchors,
    *,
    coeffs: tuple[float, ...] = config.DIAL_COEFFS,
    seeds: tuple[int, ...] = config.RANDOM_SEEDS,
    force: bool = False,
) -> Path:
    """Run the full GPU pass for one model and cache its artifacts.

    Loads the model once, then: (1) extracts anchor activations and selects the
    layer on a held-out split, (2) projects + greedily generates every graded
    item, (3) sweeps the dial on benign prompts with a norm-matched random
    control. Writes ``directions/<key>.npz`` and ``responses/<key>.parquet``.

    Args:
        spec: Model to run.
        ladders: Loaded ladder dataset.
        anchors: Loaded anchor sets.
        coeffs: Dial steering coefficients.
        seeds: Random-control seeds.
        force: Recompute even if the responses parquet already exists.

    Returns:
        Path to the model's responses parquet.
    """
    from .model import ModelRunner, Steer

    paths = _dirs()
    resp_path = paths["responses"] / f"{spec.key}.parquet"
    if resp_path.exists() and not force:
        return resp_path

    runner = ModelRunner.load(spec)
    try:
        # (1) Direction + held-out layer selection.
        harm = _anchor_acts(runner, anchors.harmful)
        benign = _anchor_acts(runner, anchors.benign)
        scores, best = layer_sweep(harm, benign, config.ANCHOR_TRAIN_PER_CLASS)
        direction = build_direction(best, harm[best], benign[best])
        np.savez(
            paths["directions"] / f"{spec.key}.npz",
            raw=direction.raw,
            unit=direction.unit,
            benign_mean=direction.benign_mean,
            layer=best,
            layer_d=np.array([s.cohens_d for s in scores]),
        )

        rows: list[dict] = []
        # (2) Graded items: projection (read) + greedy response.
        for item in all_items(ladders):
            act = runner.activations(item.prompt)[best]
            rows.append(
                {
                    "row_id": f"{spec.key}|graded|{item.uid}",
                    "model": spec.key,
                    "kind": "graded",
                    "safeguard": item.safeguard,
                    "scenario_id": item.scenario_id,
                    "level": item.level,
                    "prompt": item.prompt,
                    "projection": float(direction.project(act)),
                    "coeff": float("nan"),
                    "control": "",
                    "seed": -1,
                    "response": runner.generate(item.prompt),
                }
            )

        # (3) Dial: real direction + norm-matched random control on benign prompts.
        interventions: list[tuple[str, int, np.ndarray]] = [("real", -1, direction.raw)]
        interventions += [("random", s, random_matched(direction.raw, s)) for s in seeds]
        for control, seed, vec in interventions:
            for c in coeffs:
                # c == 0 is the unsteered baseline; record it once under "real".
                if c == 0 and control != "real":
                    continue
                steer = Steer(layer=best, vector=vec, coeff=c)
                for i, prompt in enumerate(config.DIAL_BENIGN):
                    rows.append(
                        {
                            "row_id": f"{spec.key}|dial|{i}|{control}|{seed}|c{c}",
                            "model": spec.key,
                            "kind": "dial",
                            "safeguard": "",
                            "scenario_id": "",
                            "level": -1,
                            "prompt": prompt,
                            "projection": float("nan"),
                            "coeff": float(c),
                            "control": control,
                            "seed": seed,
                            "response": runner.generate(prompt, steer=steer),
                        }
                    )
        df = pd.DataFrame(rows)
        df.to_parquet(resp_path, index=False)
        return resp_path
    finally:
        runner.unload()


def run_models(
    specs: tuple[config.ModelSpec, ...], ladders=None, anchors=None, force: bool = False
) -> list[Path]:
    """Run :func:`run_model` for each spec, skipping models that fail to load.

    A gated/unavailable model is logged and skipped rather than aborting the run
    (so the others still produce results, and the skipped one resumes later).
    """
    ladders = ladders or load_ladders()
    anchors = anchors or load_anchors()
    out = []
    for spec in specs:
        try:
            path = run_model(spec, ladders, anchors, force=force)
            out.append(path)
            print(f"[{spec.key}] responses -> {path}")
        except Exception as exc:  # noqa: BLE001 - skip and continue
            print(f"[{spec.key}] SKIPPED: {type(exc).__name__}: {str(exc)[:160]}")
    return out


def _all_responses() -> pd.DataFrame:
    paths = _dirs()
    parts = sorted(paths["responses"].glob("*.parquet"))
    if not parts:
        raise FileNotFoundError("no per-model responses; run the model pass first")
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df.to_parquet(paths["base"] / "responses.parquet", index=False)
    return df


def judge_stage(force: bool = False, gold_threshold: float | None = None) -> pd.DataFrame:
    """Validate the judge on gold, then label every not-yet-labeled response.

    Labeling is incremental: only responses whose ``row_id`` is absent from
    ``labels.parquet`` are sent to the judge, so adding a model later (e.g. once
    a gated repo is approved) labels just the new rows. Use ``force=True`` to
    re-label everything from scratch.

    Args:
        force: Re-label every response, ignoring the cache.
        gold_threshold: Override the gold agreement gate (defaults to config).

    Returns:
        The labels DataFrame (``row_id, label, refused, rationale``).

    Raises:
        RuntimeError: If the judge fails the gold-agreement gate.
    """
    from .judge import Judge

    paths = _dirs()
    labels_path = paths["base"] / "labels.parquet"
    responses = _all_responses()

    existing = (
        pd.read_parquet(labels_path)
        if labels_path.exists() and not force
        else pd.DataFrame(columns=["row_id", "label", "refused", "rationale"])
    )
    todo = responses[~responses["row_id"].isin(set(existing["row_id"]))]
    if todo.empty:
        print(f"[judge] all {len(responses)} responses already labeled; nothing to do")
        return existing

    judge = Judge()
    gold = load_gold()
    report = judge.validate_on_gold(
        gold, threshold=gold_threshold or config.JUDGE_AGREEMENT_THRESHOLD
    )
    (paths["base"] / "gold_report.json").write_text(
        json.dumps(
            {
                "n": report.n,
                "agreement": report.agreement,
                "passed": report.passed,
                "recall": {lbl: report.recall(lbl) for lbl in config.JUDGE_LABELS},
                "confusion": [
                    {"gold": g, "judge": j, "count": c}
                    for (g, j), c in sorted(report.confusion.items())
                ],
                "disagreements": [
                    {"id": i, "gold": g, "judge": j} for i, g, j in report.disagreements
                ],
            },
            indent=2,
        )
    )
    print(
        f"[judge] gold agreement {report.agreement:.1%} (n={report.n}) "
        f"-> {'PASS' if report.passed else 'FAIL'}"
    )
    if not report.passed:
        raise RuntimeError(
            f"judge failed gold gate: {report.agreement:.1%} < "
            f"{gold_threshold or config.JUDGE_AGREEMENT_THRESHOLD:.0%}. See gold_report.json."
        )

    print(f"[judge] labeling {len(todo)} responses ({len(existing)} already cached)")
    judgments = judge.judge_many(list(zip(todo["prompt"], todo["response"], strict=True)))
    new_labels = pd.DataFrame(
        {
            "row_id": todo["row_id"].to_numpy(),
            "label": [j.label for j in judgments],
            "refused": [j.refused for j in judgments],
            "rationale": [j.rationale for j in judgments],
        }
    )
    labels = pd.concat([existing, new_labels], ignore_index=True)
    labels.to_parquet(labels_path, index=False)
    return labels


def metrics_stage() -> dict[str, pd.DataFrame]:
    """Join responses with labels and write the four metric tables."""
    paths = _dirs()
    responses = pd.read_parquet(paths["base"] / "responses.parquet")
    labels = pd.read_parquet(paths["base"] / "labels.parquet")
    merged = responses.merge(labels[["row_id", "refused"]], on="row_id", how="left")

    # Drop any response without a judge label rather than let it slip through:
    # downstream metrics coerce ``refused`` to bool, and NaN -> True would
    # silently miscount unlabeled rows as refusals.
    missing = int(merged["refused"].isna().sum())
    if missing:
        print(f"[metrics] WARNING: {missing} responses have no judge label; dropping them")
        merged = merged[merged["refused"].notna()].copy()
    merged["refused"] = merged["refused"].astype(bool)

    graded = merged[merged["kind"] == "graded"].copy()
    dial = merged[merged["kind"] == "dial"].copy()

    tables = {
        "monitor": metrics.monitor_table(graded),
        "monitor_pooled": metrics.pooled_monitor_table(graded),
        "intent_calibration": metrics.intent_calibration(graded),
        "intent_calibration_pooled": metrics.pooled_intent_calibration(graded),
        "ramp": metrics.ramp_table(graded),
        "dial": metrics.dial_table(dial),
    }
    tables["dial_gap"] = metrics.dial_gap(tables["dial"])
    tables["operating_band"] = metrics.operating_band(tables["dial"])
    for name, tbl in tables.items():
        tbl.to_parquet(paths["metrics"] / f"{name}.parquet", index=False)
    return tables
