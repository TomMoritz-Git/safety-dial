# safety-dial

A small interpretability study of refusals in open instruct models.

One linear direction in the residual stream both predicts a model's refusal
before it generates (a monitor) and, when added back during generation, causes
one (a dial). This holds for three safeguard domains and breaks down for a
fourth (misinformation).

Full write-up: [Reading and turning the safety
dial](https://tommoritz-git.github.io/blog/reading-and-turning-the-safety-dial/).

## How it works

- **Models.** Five small instruct models from four labs: Qwen3-1.7B, Gemma-3-1B,
  SmolLM3-3B, Granite-3.1-2B, Qwen2.5-1.5B. All run on one consumer GPU.
- **Data.** Four domains (privacy, cyber-access, fraud, misinformation), each an
  intent ladder: 20 scenarios, each written at five levels from a legitimate
  request (L0) to a clearly disallowed one (L4), with the topic held fixed. 400
  prompts in all. See `data/ladders.json`.
- **Direction.** Difference of mean harmful and benign anchor activations at the
  last prompt token. The layer is picked on a held-out split of the anchors.
- **Read.** Project a prompt's activation onto the direction. The score predicts
  refusal before generation. The main metric is within-topic AUC (same-scenario
  pairs only, so topic alone can't drive the score).
- **Write.** Add the direction during generation to push benign prompts into
  refusal, with a norm-matched random direction as a control.
- **Judge.** Refusals are labeled by Claude Haiku, checked against a 55-item
  hand-labeled gold set before it runs on everything.

## Results

The read separates intent at AUC 0.92–0.98 across models, pooled over domains.
The dial drives benign prompts to refusal well above the random control.
Misinformation is the exception: the read drops toward chance and refusals don't
track intent. Scored against the ladder's intent, the models spread along a
trade-off between over-refusal (refusing legitimate L0 requests) and
under-refusal (complying with disallowed L4 ones).

Tables and figures are written to `results/`.

## Setup

```bash
uv sync                       # torch 2.6.0+cu124 (Pascal-compatible)
cp .env.example .env          # then add ANTHROPIC_API_KEY and HF_TOKEN
```

GPU notes (GTX 1070, Pascal / sm_61): models load in fp16 with SDPA attention;
Gemma-3 uses eager attention for its soft-capping. The torch index is pinned to
cu124 because later builds dropped Pascal kernels, so don't bump it.

## Run

```bash
uv run safety-dial smoke         # check each model loads, generates, refuses
uv run safety-dial models        # build directions + graded and dial generations
uv run safety-dial judge         # validate on gold, then label everything
uv run safety-dial metrics       # AUC / ramps / dial / calibration tables
uv run safety-dial figures       # plots
# or run the whole thing: uv run safety-dial all
```

Each stage caches to `results/` and is resumable.

## Test

```bash
uv run pytest                    # pure-Python core, no GPU or API needed
uv run pytest -m gpu             # add the live model checks
uv run ruff check . && uv run ruff format --check .
```

## Layout

```
src/safety_dial/
  config.py       models, paths, sweep grids, env loading
  data.py         ladder / anchor / gold loaders and validation
  stats.py        AUC, Cohen's d, Wilson and bootstrap CIs (pure NumPy)
  extraction.py   diff-of-means direction + held-out layer sweep
  monitor.py      read-side AUC and threshold
  model.py        fp16 model runner: activations + steered generation
  judge.py        refusal judge + gold gate
  intent.py       blind request-intent rater (validates the ladder levels)
  metrics.py      result tables
  robustness.py   anchor-pool resampling checks
  regen.py        regenerate responses at a larger token cap
  figures.py      plots
  pipeline.py     resumable orchestration
  cli.py          `safety-dial <stage>`
data/             ladders.json, anchors.json, anchor_pool.json, gold.json
tests/            pure-Python unit tests
```
