# Reading and Turning the Safety Dial

One internal linear direction that both **predicts** and **controls** a small
instruct model's refusals — measured across several models and safeguard
domains.

> **Thesis.** In small instruct models, the decision to refuse is not a binary
> switch but a *graded threshold on a single linear direction* in the residual
> stream. The same direction **reads** the safeguard (its projection predicts
> refusal before generation — a monitor) and **writes** it (adding it slides the
> model across its refusal threshold — a dial). Where the monitor's reading and
> the model's action disagree, the safeguard is miscalibrated (over- /
> under-refusal).

See [`DESIGN.md`](DESIGN.md) for the full experiment design, hypotheses, and the
hard-won GTX-1070 (Pascal) environment notes.

## Layout

```
src/safety_dial/      # the package
  config.py           # models, paths, sweep grids, env loading
  data.py             # ladder / anchor / gold loaders + validation
  stats.py            # AUC, Cohen's d, Wilson & bootstrap CIs (pure NumPy)
  extraction.py       # diff-of-means direction + held-out layer sweep
  monitor.py          # read-side AUC, threshold, calibration gap
  model.py            # fp16 model runner (sdpa; eager for Gemma): acts + steered gen
  judge.py            # Anthropic structured-output refusal judge + gold gate
  metrics.py          # assemble per-(model x safeguard) results tables
  figures.py          # hero heatmap, ramps, read<->write exemplar
  pipeline.py         # resumable orchestration
  cli.py              # `safety-dial <stage>`
data/                 # ladders.json, anchors.json, gold.json
tests/                # pure-Python unit tests (no GPU/API)
```

## Setup

```bash
uv sync                       # CPU/GPU deps (torch 2.6.0+cu124 for Pascal)
cp .env.example .env          # then add ANTHROPIC_API_KEY and HF_TOKEN
```

The GTX 1070 is Pascal (sm_61): models load **fp16** (no bf16) with **SDPA
attention** by default — Gemma-3 uses **eager** for its attention soft-capping
(see `config.py`; fp16 + eager overflows to NaN on some other models). The torch
index is pinned to **cu124** (cu128/cu130 dropped Pascal kernels). Do not bump it.

## Run

```bash
uv run safety-dial smoke         # verify each model loads, generates, refuses
uv run safety-dial extract       # directions + held-out layer sweep
uv run safety-dial generate      # graded + dial generations (greedy)
uv run safety-dial judge         # validate on gold, then label everything
uv run safety-dial metrics       # AUC / ramps / dial / calibration tables
uv run safety-dial figures       # the hero heatmap and supporting panels
# or: uv run safety-dial all
```

Each stage caches to `results/` and is resumable.

## Test

```bash
uv run pytest                       # pure-Python core (fast)
uv run pytest -m "gpu"              # add the live model smoke checks
uv run ruff check . && uv run ruff format --check .
```
