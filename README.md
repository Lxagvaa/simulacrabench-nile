# SimulacraBench – Team Nile

Submission for [SimulacraBench](https://simulacrabench.org/) (NeurIPS 2026 competition track):
predict survey answers of held-out respondents for three instruments (UNHCR, UNICEF, World Bank)
from their `GIVEN` block.

## Usage

```bash
make setup       # uv env, starter kit, sandbox data, TabPFN checkpoint
make eval        # all three sandbox instruments, development budget
make eval-final  # final-phase conditions (TRAIN+DEV visible, 3300 s)
make test        # adversarial frames
make official    # the organisers' grader
make zip         # dist/submission.zip
```

The sandbox data are synthetic (one latent trait plus noise). They check that the pipeline runs within
time and memory and returns legal vectors; they say little about real performance.

## Task

`predict(frame, schema)` is called once per instrument. Visible respondents are complete; held-out
respondents only show `GIVEN`. For every blank cell it returns a distribution over the item's values,
plus a final `NA_GATED` slot for gated items. Score: log score as skill (uniform 0, perfect 1), mean
over the instruments. One H100, 16 GB RAM, 900 s (development) / 3600 s (final) for all three.

## Approach

**Gates.** For a gated item, `P(v) = P(asked) · P(v | asked)` and `P(NA_GATED) = 1 − P(asked)`.
`P(asked)` comes from the parent's predicted distribution and a table `P(asked | parent value, context)`
estimated from the data, with the schema rule as prior. Context columns (e.g. the survey round) matter
because the files do not always follow the routing. Items are composed in topological order.

**Components** (`simbench/models/`), all fitted on visible rows only, predicting from `GIVEN`:

| family | models |
|---|---|
| counts | marginal, hierarchical Dirichlet backoff, profile matching (same-round neighbours) |
| linear / nets | multi-task softmax regression (with pairwise interactions), MLP |
| latent structure | latent class regression, continuous factor model (VAE, IRT-style) |
| trees | per-item gradient boosting (three settings), multi-output random forest |
| other | proportional-odds ordinal model, TabPFN v2 (GPU) |

Most also exist in a "direct" variant that predicts `NA_GATED` as an ordinary answer.

**Blend** (`blend.py`). Out-of-fold predictions (3 folds, 5 in the final phase) → linear pool with EM
weights per item, shrunk toward per-instrument weights; then temperature calibration.

**Post-processing** (`postprocess.py`), each step kept only where it improves the out-of-fold score:
temperatures of gate parents chosen on parent + children, expected tick counts in multi-select
families, smoothing along numeric answer bands.

**Time and memory** (`pipeline.py`, `main.py`). The budget is split across instruments by expected cost.
Each component's first fold is timed and the component is dropped if all folds plus the refit would
not fit. If the refit on all rows would overrun, the average of the fold models is used instead.
Final-phase run on the sandboxes: about 2700 s and 5 GB peak memory.

## Results (development leaderboard)

| version | UNICEF | World Bank | UNHCR | mean |
|---|---|---|---|---|
| marginal baseline (sandbox) | 0.16 | 0.24 | 0.43 | 0.28 |
| v0.2 | 0.52 | 0.49 | 0.76 | 0.59 |
| v0.3 – gate context | 0.52 | 0.49 | 0.90 | 0.64 |
| v0.12 – gradient boosting on all items | 0.52 | 0.49 | 0.92 | 0.64 |

Scores are noised and rounded to 0.01. Experiment notes are in `notes/results.md`.

## Layout

```
submission/          contents of the zip
  main.py            entry point, budget split, fallback
  requirements.txt   tabpfn==9.1.0
  weights/           TabPFN checkpoint (fetched by scripts/fetch_weights.sh, not in git)
  simbench/
    schema_utils.py  option order, gates, topological order
    data.py          encoding, gate model
    models/          components
    blend.py         EM linear pool, temperature
    postprocess.py   gate parents, multi-select families, ordinal smoothing
    pipeline.py      scheduling, OOF, blend, refit, output
scripts/             setup, local evaluation, robustness test, model comparison, zip
```

## Third-party model

TabPFN v2 (Prior Labs, Prior Labs License: Apache 2.0 with attribution), checkpoint
`tabpfn-v2-classifier.ckpt` from `Prior-Labs/TabPFN-v2-clf` at commit
`f851f2a3c941544733b712d8c0f96dfae9b28862`.

## Licence

MIT, see `LICENSE`. TabPFN v2 keeps its own licence (see above).
