"""Command-line entry point: ``safety-dial <stage>``.

Stages mirror the pipeline and are individually resumable::

    safety-dial smoke      # verify each model loads, generates, refuses
    safety-dial models     # GPU pass: directions + graded/dial responses
    safety-dial judge      # gold gate + label every response
    safety-dial metrics    # AUC / ramps / dial / calibration tables
    safety-dial figures    # hero heatmap + supporting panels
    safety-dial all        # models -> judge -> metrics -> figures
"""

from __future__ import annotations

import argparse

from . import config


def _select(only: list[str] | None) -> tuple[config.ModelSpec, ...]:
    if not only:
        return config.MODELS
    chosen = []
    for key in only:
        if key not in config.MODELS_BY_KEY:
            raise SystemExit(f"unknown model key {key!r}; known: {list(config.MODELS_BY_KEY)}")
        chosen.append(config.MODELS_BY_KEY[key])
    return tuple(chosen)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the requested stage(s)."""
    parser = argparse.ArgumentParser(prog="safety-dial", description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)

    for name in ("smoke", "models", "judge", "metrics", "figures", "all"):
        sp = sub.add_parser(name)
        if name in ("smoke", "models", "all"):
            sp.add_argument("--only", nargs="+", help="restrict to these model keys")
        if name in ("models", "judge", "all"):
            sp.add_argument("--force", action="store_true", help="recompute cached outputs")

    args = parser.parse_args(argv)
    config.load_env()

    if args.stage == "smoke":
        from . import smoke

        results = smoke.run(_select(args.only))
        return 0 if all(r.ok for r in results) else 1

    from . import pipeline

    if args.stage in ("models", "all"):
        pipeline.run_models(_select(getattr(args, "only", None)), force=args.force)
    if args.stage in ("judge", "all"):
        pipeline.judge_stage(force=args.force)
    if args.stage in ("metrics", "all"):
        pipeline.metrics_stage()
    if args.stage in ("figures", "all"):
        from . import figures

        for path in figures.make_all():
            print(f"[figures] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
