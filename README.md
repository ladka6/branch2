# branch2: what actually limits hierarchical speculative decoding?

First-round diagnostics for 3-level self-speculative decoding (L_d -> L_i -> L_f) on
LayerSkip early-exit checkpoints.

## Why this repo exists

The first HSD + HiSpec run (`legacy/hsd_hispec_experiments.py`, L4->L8->L16, gamma=6,
T=1.0, Ni=16) showed:

- L16 accepted on average 1.42 of 16 tentative tokens; 0 of 500 windows were fully accepted.
- L_d->L_i agreement did not predict final survival (Pearson -0.04).
- 52% of window tokens were "L_i overshoot", but that is expected by construction: for
  x ~ p_i, E[log p_i - log p_d] = KL(p_i||p_d) > 0 and E[log p_f - log p_i] = -KL(p_i||p_f) < 0.

The key observation behind this code: **under any lossless stage-1 verifier (tokenwise
or HSD), the tentative window handed to L_f is distributed exactly as autoregressive
sampling from L_i.** So L_f's acceptance behaviour does not depend on how stage 1 is
done. A better stage-1 verifier can only reduce the number of L_i calls; it cannot
change what L_f accepts. That lets us measure the final stage directly and cheaply,
without simulating the full pipeline.

## Questions the first experiments answer

| Section | Question | What would change our direction |
|---|---|---|
| A | Acceptance for every drafter->verifier depth pair | Baseline map of where the gaps are |
| B | Does the stage-1 outcome (L_i accepted L_d's draft vs L_i resampled) change L_f's acceptance? | If `a|res` << `a|acc`, L_i's corrections are unreliable and a target-aware intermediate verifier is well motivated. If equal, the stages are exactly decoupled. |
| C | Observed trajectory-class shares vs analytic null | Should match; confirms the overshoot result was an artifact |
| D | E[tokens accepted by L_f] vs window size Ni | Where the curve saturates tells you the useful Ni |
| E | Best 2-level vs best 3-level speedup, memory-bound cost model | If 3-level never wins, the hierarchy needs a new ingredient, not a better stage-1 verifier |
| F | Can signals available without L_f predict L_f's acceptance? | Strong predictors -> adaptive Ni / early L_f calls are viable |
| G | Does acceptance drift along the window? | Sanity check for the stationarity assumption in E |

Section B is the precise version of "does stage-1 quality predict final survival". For
each position it computes, analytically over the whole vocabulary,

- `a|acc = E[alpha_i->f | L_d's draft was accepted]` (token law proportional to min(p_d, p_i))
- `a|res = E[alpha_i->f | token came from L_i's residual]` (law proportional to (p_i - p_d)+)

These two always average to `1 - TV(p_i, p_f)`; the smoke test checks that identity.

The cost model (E) counts the LM head in layer units (`H`). For Llama-3.2-1B the tied
head is about 4 layers' worth of weights, so an L4 exit costs ~8.4 layer-units, not 4.
It reports two variants: `noreuse` (every call starts from layer 1) and `reuse`
(optimistic LayerSkip-style cache reuse, where a verifier at depth v after depth u pays
only v-u+H). The real number lies between them. It assumes a verification pass costs
the same regardless of how many tokens it scores (memory-bound decoding), and it treats
the stage-1 round count as Ni / block_efficiency (fractional).

## Layout

```
hsx/models.py       loading, early-exit truncation, cost units
hsx/data.py         fixed-length contexts from wikitext2 / humaneval / ultrachat / gsm8k
hsx/sampling.py     batched autoregressive sampling from an early exit
hsx/lens.py         all-depth logits from one forward pass (layer hooks) + per-token stats
experiments/collect.py   GPU: sample windows per source depth, store compact stats (.npz)
experiments/analyze.py   CPU: sections A-G, writes analysis/report.txt and CSVs
tests/smoke_test.py      offline test on a tiny random Llama (no downloads)
legacy/                  the original HSD + HiSpec harness
```

`hsx/lens.py` reads every depth from a single full-model pass with hooks and applies the
shared final norm + LM head. The smoke test checks that this matches truncated early-exit
models to 1e-4, and that the cached sampler agrees with the uncached lens.

## Running

Setup (the LayerSkip checkpoint is gated; accept the license on Hugging Face first):

```
pip install -r requirements.txt
huggingface-cli login
python tests/smoke_test.py
```

Main run (one A100 is plenty; each source depth is 512 windows x 32 sampled tokens):

```
python experiments/collect.py --out runs/wt2 --corpus wikitext2 \
    --sources 2 4 6 8 10 12 14 --depths 2 4 6 8 10 12 14 16 \
    --temperatures 0.6 1.0 --contexts 512 --window 32 --batch 8
python experiments/analyze.py --run runs/wt2
```

Then repeat `collect.py` with `--corpus humaneval`, `--corpus gsm8k` and `--corpus ultrachat`.
Code and math are usually much easier for early exits than open-ended text, and the
conclusions may differ by domain.

On Snellius, see `scripts/snellius_collect.sh`. Results land in `runs/<name>/analysis/`.

Memory note: `collect.py` keeps full-vocab probabilities for all depths of one batch
(about depths x batch x window x 128k x 4 bytes, ~1 GB at the defaults). Lower `--batch`
if needed.

## What the first run should settle

1. Is `a|res / a|acc` near 1 (decoupled) or clearly below 1 (L_i's corrections are
   what L_f rejects)? The second supports making the intermediate verifier target-aware
   (e.g. an adapter on the L_i exit distilled toward L_f).
2. Does any 3-level configuration beat the best 2-level one under either cost model?
3. Is there a cheap signal with AUROC well above 0.5 for L_f acceptance? AUROC below
   0.5 means the feature is inversely predictive, which is still usable (use 1 - AUROC).
