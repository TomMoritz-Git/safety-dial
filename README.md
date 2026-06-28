# Reading and Turning the Safety Dial

One internal linear direction that both **predicts** and **controls** a small
instruct model's refusals — measured across several models and safeguard
domains, with the boundary where that picture breaks made explicit.

> **Thesis.** In small instruct models, the decision to refuse is — for the
> safeguards where post-training installed a coherent one — a *graded threshold
> on a single linear direction* in the residual stream. The same direction
> **reads** the safeguard (its projection predicts refusal before generation — a
> monitor) and **writes** it (adding it slides the model across its refusal
> threshold — a dial). The story comes in three acts:
>
> - **I — Structure (it works).** One vector reads refusal (within-topic AUC
>   0.92–0.98) and writes it (the dial drives benign prompts to refusal, beating
>   a norm-matched random control, with *coherent* refusals inside the operating
>   band).
> - **II — Limit (where it breaks).** The linear-threshold account is
>   *domain-specific*: it is crisp for privacy / cyber-access / fraud and *fails
>   for misinformation* (AUC near chance, a non-monotone severity ramp, degenerate
>   base rates). Refusal is not one mechanism — it is one mechanism *where the
>   model has a safeguard*, and several of these models simply don't have a
>   coherent one for misinformation.
> - **III — Use (what it's for).** Scored against the ladder's ground-truth
>   intent (not the monitor's own threshold), the models sit at different points
>   on an **over-refusal ↔ under-refusal frontier** — from maximally cautious
>   (qwen2.5: 26% over-refusal on legitimate L0 requests, 0% under-refusal on
>   disallowed L4) to permissive (smollm3: 1% / 31%). The dial moves a model along
>   that frontier and the monitor predicts where it sits before generation.

## Method

- **Models & data.** 5 small instruct models across 4 providers (Alibaba, Google,
  HuggingFace, IBM). 4 safeguard domains — privacy, cyber-access, fraud,
  misinformation — each an *intent ladder*: 20 scenarios escalated over 5 severity
  levels (L0 legitimate → L4 disallowed), 400 graded prompts in all. A
  topic-disjoint benign/harmful **anchor** set defines the direction; a 55-item
  human-labeled **gold** set validates the judge.
- **Direction.** Difference of mean harmful vs. benign anchor activations at the
  last prompt token; the layer is chosen on a held-out anchor split.
- **Monitor (read).** Projecting an item's activation onto the direction predicts
  refusal *before* generation. Primary metric: **within-topic AUC** (same-scenario
  pairs only, so it can't be a mere topic detector).
- **Dial (write).** Adding `c · direction` during generation slides benign prompts
  into refusal; a **norm-matched random direction** is the specificity control and
  a distinct-bigram check confirms the induced refusals are coherent, not breakage.
- **Calibration (intent-relative).** The headline calibration is scored against
  the ladder's **ground-truth intent**: over-refusal = refusing an L0 *legitimate*
  request; under-refusal = complying with an L4 *disallowed* one (Wilson CIs). The
  earlier monitor-vs-action "gap" is kept only as a *threshold-residual* diagnostic
  (how close refusal is to a pure 1-D threshold) — it is not a normative calibration
  measure, since its reference is the monitor's own operating point rather than intent.
- **Judge.** `claude-haiku-4-5` with forced structured output, gated at ≥95%
  agreement with the gold set before it labels the full run.

**Primary vs. exploratory.** To keep 5 models × 4 safeguards × several metrics
honest, the *primary* endpoints are: pooled within-topic AUC (read), the dial vs.
norm-matched-random gap within the operating band (write), and the intent-relative
over-/under-refusal frontier (calibration). Everything else — per-safeguard AUC
cells, the severity ramps, the threshold-residual gap — is exploratory and read
descriptively, not as a tested claim. The judge's three-way label is kept (not
just the refuse/comply binarization): in the original 64-token run the `partial`
mass (`label_mix`) sat at the *legitimate* L0/L1 levels and looked like the models
half-answering benign asks.

**Truncation control.** That `partial` mass turned out to be mostly an artifact of
the 64-token generation cap. Regenerating the graded responses at
`MAX_NEW_TOKENS=10240` (`regen.py`, GPU forward pass; dial untouched, since steering
induces refusal early) and re-judging collapses `partial` at L0 from **68% → 6%** —
the truncated answers were genuine, complete help cut off mid-sentence. Critically,
the **binary refuse/comply metrics are robust** to the cap: within-topic AUC moves
≤0.013 and per-level refusal rates move ≤5 points, so the read result and the intent
frontier stand (the un-truncated numbers show marginally *more* under-refusal, since
cut-off compliance with disallowed requests was previously mislabeled `partial`).
See `figures/supp_truncation_comparison.png` and
`results/metrics/truncation_comparison.parquet`.

**Robustness — the direction isn't an artifact of 8 prompts.** The deployed
direction is a diff-of-means over 8 anchors per class. To show the read doesn't
hinge on that handful, a separate 32/class anchor *pool* (`data/anchor_pool.json`)
is captured forward-only (no generation, no re-judge; the deployed direction and
judged labels are untouched) and resampled in NumPy (`robustness.py`):

- **Bootstrap.** Pool-resampled within-topic AUC reproduces the deployed AUC
  within ≤0.006 for every model, with tight CIs (e.g. qwen2.5 0.975 [0.972, 0.977]
  vs. deployed 0.977).
- **Anchor-count sweep.** AUC is *flat* from N=4 to N=32 (gemma 0.921→0.924,
  qwen2.5 0.974→0.975) — N=8 is on the plateau; more anchors don't change the read.
- **Cosine stability.** The full-pool direction sits at cosine 0.96–0.99 to the
  deployed 8-anchor one; bootstrap resamples stay >0.92. The axis is a property of
  the model, not of the anchor sample.

See `results/metrics/robustness*.parquet`, `results/robustness_report.json`, and
`figures/supp_anchor_robustness.png`.

**Stimulus & judge validation.** Two guardrails back the headline numbers. (1) An
independent blind rater (`intent.py`) scores each ladder prompt's intent given the
*request only* — L0/L1 read benign, L4 disallowed, monotone — so the intent-relative
calibration is not graded against the author's own labels. (2) The judge is gated
on a gold set stratified to the hard cells (refusing legitimate requests, complying
with disallowed); on it the refuse/comply split that every primary metric uses has
100% per-class recall (`results/gold_report.json`).

## Layout

```
src/safety_dial/      # the package
  config.py           # models, paths, sweep grids, env loading
  data.py             # ladder / anchor / gold loaders + validation
  stats.py            # AUC, Cohen's d, Wilson & bootstrap CIs (pure NumPy)
  extraction.py       # diff-of-means direction + held-out layer sweep
  monitor.py          # read-side AUC, threshold, threshold-residual gap
  model.py            # fp16 model runner (sdpa; eager for Gemma): acts + steered gen
  judge.py            # Anthropic structured-output refusal judge + gold gate + confusion
  intent.py           # blind request-intent rater (validates the ladder's levels)
  metrics.py          # results tables: monitor AUC, intent calibration, label mix, ramps, dial
  robustness.py       # anchor-pool capture + bootstrap / N-sweep / cosine stability
  regen.py            # regenerate graded responses at a larger token cap (truncation control)
  figures.py          # hero read<->write, ramps, calibration frontier, misinfo breakdown
  pipeline.py         # resumable orchestration
  cli.py              # `safety-dial <stage>`
data/                 # ladders.json, anchors.json, anchor_pool.json, gold.json
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
uv run safety-dial models        # directions (held-out layer sweep) + graded & dial gens
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
