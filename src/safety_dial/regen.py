"""Regenerate graded responses at a larger token budget, for a truncation check.

The original run capped generation at 64 new tokens, which truncated 73-100% of
graded answers and likely inflated the ``partial`` label at the legitimate levels
(a model's genuine help got cut off mid-sentence). This regenerates *only* the
graded responses at ``config.MAX_NEW_TOKENS`` (now 1024), leaving the original
64-token run and the dial responses untouched so the two can be compared.

Projections are reused from the original run -- they are activations of the
*prompt*, independent of how long the completion is -- so only the response text
(and hence its refusal label) changes. Output goes to ``results/responses_long/``;
the run is resumable per model and checkpoints within a model.
"""

from pathlib import Path

import pandas as pd

from . import config
from .data import Safeguard, all_items, load_ladders


def _long_dir() -> Path:
    d = config.RESULTS_DIR / "responses_long"
    d.mkdir(parents=True, exist_ok=True)
    return d


def regen_graded_model(
    spec: config.ModelSpec,
    ladders: dict[str, Safeguard],
    old_graded: pd.DataFrame,
    *,
    checkpoint_every: int = 25,
) -> Path:
    """Regenerate one model's graded responses at ``config.MAX_NEW_TOKENS``.

    Resumable: rows already present in the per-model output are skipped, and the
    output is rewritten every ``checkpoint_every`` new generations so an
    interrupted run loses at most that many.

    Args:
        spec: Model to run.
        ladders: Loaded ladder dataset.
        old_graded: The original graded rows (for this model) -- supplies the
            reused projection and the row metadata.
        checkpoint_every: Flush cadence.

    Returns:
        Path to ``results/responses_long/<key>.parquet``.
    """
    from .model import ModelRunner

    out = _long_dir() / f"{spec.key}.parquet"
    done = pd.read_parquet(out) if out.exists() else pd.DataFrame(columns=["row_id"])
    done_ids = set(done["row_id"])
    proj_by_uid = {
        rid.split("|graded|")[1]: pr
        for rid, pr in zip(old_graded["row_id"], old_graded["projection"], strict=True)
    }

    items = [it for it in all_items(ladders) if f"{spec.key}|graded|{it.uid}" not in done_ids]
    if not items:
        return out

    runner = ModelRunner.load(spec)
    rows = done.to_dict("records")
    try:
        for i, it in enumerate(items, 1):
            rows.append(
                {
                    "row_id": f"{spec.key}|graded|{it.uid}",
                    "model": spec.key,
                    "kind": "graded",
                    "safeguard": it.safeguard,
                    "scenario_id": it.scenario_id,
                    "level": it.level,
                    "prompt": it.prompt,
                    "projection": float(proj_by_uid[it.uid]),
                    "coeff": float("nan"),
                    "control": "",
                    "seed": -1,
                    "response": runner.generate(it.prompt),
                }
            )
            if i % checkpoint_every == 0 or i == len(items):
                pd.DataFrame(rows).to_parquet(out, index=False)
    finally:
        runner.unload()
    return out


def run(specs=None, force: bool = False) -> list[Path]:
    """Regenerate graded responses for each spec that has an original run.

    Skips models whose long output already covers all 400 graded items (unless
    ``force``). Models that fail to load are logged and skipped.
    """
    ladders = load_ladders()
    old = pd.read_parquet(config.RESULTS_DIR / "responses.parquet")
    old_graded = old[old["kind"] == "graded"]
    keys = [s.key for s in (specs or config.MODELS) if s.key in set(old_graded["model"])]
    out = []
    for key in keys:
        spec = config.MODELS_BY_KEY[key]
        sub = old_graded[old_graded["model"] == key]
        target = _long_dir() / f"{key}.parquet"
        if force and target.exists():
            target.unlink()
        try:
            path = regen_graded_model(spec, ladders, sub)
            n = len(pd.read_parquet(path))
            print(f"[{key}] graded regen -> {path} ({n} rows)", flush=True)
            out.append(path)
        except Exception as exc:  # noqa: BLE001 - skip and continue
            print(f"[{key}] SKIPPED: {type(exc).__name__}: {str(exc)[:160]}", flush=True)
    return out
