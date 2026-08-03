# What's new in Sentinel 2.0

Sentinel 1.0 gave you an index and a score. Getting a *good* score still meant
hand-building the index, guessing at hyperparameters, and re-encoding your corpus every
time you wanted to try something different.

Version 2.0 is about closing that loop: build an index in one call, resize it without
re-encoding, sweep the settings that matter, and see which text actually drove a score.

- [Build an index in one call](#build-an-index-in-one-call)
- [Resize an index without re-encoding](#resize-an-index-without-re-encoding)
- [Tune with the simulation harness](#tune-with-the-simulation-harness)
- [Sweep index size and ratio](#sweep-index-size-and-ratio)
- [Explanations survive a reload](#explanations-survive-a-reload)
- [Reproducible loading](#reproducible-loading)
- [Migrating from 1.0](#migrating-from-10)

## Build an index in one call

`SentinelLocalIndex.from_texts()` replaces an eight-step recipe, two steps of which fail
*silently* when you skip them: omit `normalize_embeddings=True` and the similarity maths
quietly returns wrong numbers, and omit the corpus and you lose explanations. Neither
raises. Neither warns.

```python
from sentinel import SentinelLocalIndex

index = SentinelLocalIndex.from_texts(
    positive_texts=["...examples of the rare class..."],
    negative_texts=["...examples of ordinary content..."],
    neg_to_pos_ratio=5.0,
    seed=42,
)
```

Surplus negatives are dropped *before* encoding rather than after. Encoding is per-text
and is the only expensive step, so at a 1:1 ratio against 1,000 positives, passing
100,000 negatives no longer means paying to encode 99,000 rows that never reach the index.

## Resize an index without re-encoding

Encoding a sentence produces the same numbers regardless of which index it ends up in, so
a small index is just a large one with rows removed. `subsample()` copies the rows you
want instead of re-running the model:

```python
smaller = index.subsample(n_positive=1000, neg_to_pos_ratio=5.0, seed=42)
```

It returns a new index and never mutates the original, which matters when you loop over
sizes: if each call shrank the receiver, run two would start from run one's leftovers and
every later result would be quietly wrong.

Each side draws from its own seeded generator. Sharing one would couple them, so a
configuration that kept every positive (and therefore drew nothing) would select
different negatives from one that subsampled - two cells meant to differ along one axis
would differ along two.

## Tune with the simulation harness

`sentinel.simulation` answers "which settings work best on *my* data?". It is numpy-only,
with no Ray, S3 or experiment trackers, and it separates two things that are easy to
confuse:

- The **summarize metric** (the aggregator) is how a group's many per-observation scores
  become one number. This is what you sweep.
- The **evaluation metric** is how good the resulting separation is. You do not choose
  one; every row reports three families, so you can tune for whatever matters.

Why the summarize metric matters so much: picture a hateful podcast episode with 97 dull
segments and 3 nasty ones, against a normal episode of 100 unremarkable segments.

| Episode | Segment scores | Average | Maximum |
|---|---|---|---|
| Hateful | 97 x 0.05, 3 x 0.90 | 0.0755 | 0.90 |
| Normal | 100 x 0.06 | 0.06 | 0.08 |

Averaging drowns the signal; the maximum finds it. But the maximum is fragile - one odd
segment flags an innocent episode - which is why the default is `skewness`, which looks
for a *pattern* of spikes and does not care how long the episode is.

```python
from sentinel.simulation import LabeledGroup, score_groups, compare_aggregators
import pandas as pd

groups = [
    LabeledGroup(name="source_a", label=1, observations=[...]),
    LabeledGroup(name="source_b", label=0, observations=[...]),
]

scored = score_groups(index, groups, top_k=5)     # expensive, once
pd.DataFrame(compare_aggregators(scored))          # cheap, all six aggregators
```

Every row carries all three families:

- Ranking: `roc_auc`, `recall_at_n`, `precision_at_n`, `rank_ratio`
- Threshold: `precision`, `recall`, `f1`, `false_positive_rate`
- Separation: `mean_separation`, `cohens_d`, `ks_statistic`

## Sweep index size and ratio

`run_grid_search` can now sweep the index itself, not just `top_k` and the threshold:

```python
pd.DataFrame(run_grid_search(
    index, groups,
    n_positive_values=[1000, 5000, 10000],
    neg_to_pos_ratios=[0.2, 1.0, 5.0],
    top_k_values=[3, 5, 10],
    index_seed=42,
))
```

The arguments are ordered to match the loop nesting, which is itself ordered by cost:

```
for n_positive in n_positive_values:      # cheap: subsample()
  for ratio in neg_to_pos_ratios:         # cheap: subsample()
    for top_k in top_k_values:            # re-scores
      for min_score in min_score_values:  # cheap
        for aggregator in aggregators:    # cheap
```

**Observations are encoded once for the whole sweep.** An observation's embedding depends
on the encoder, never on the index it is scored against, and a subsampled index shares its
parent's model - so re-encoding per pass was recomputing identical numbers. On a 2x2x3
sweep over 320 observations this took a real run from **2.99s to 0.52s, a 5.7x saving**,
producing byte-identical rows. The saving grows with the grid, because the one encoding
pass is amortised over more scoring passes.

Set `cache_observation_embeddings=False` if memory is tight; the cache holds one embedding
per observation.

Rows report both what you asked for and what you got, because a request is clipped when
the index is smaller than requested:

```
index_n_positive=10   -> index_n_positive_actual=10, index_n_negative_actual=20
index_n_positive=999  -> index_n_positive_actual=15, index_n_negative_actual=30
```

Without the actual counts, those two rows could look like a genuine size sweep when they
describe nearly the same index.

## Explanations survive a reload

In 1.0, `save()` wrote only embeddings and config. After any reload the corpus was gone,
so explanations reported the row number of a match instead of the matched sentence.

2.0 writes a `corpus.json` beside the embeddings. It is written **unconditionally**, with
nulls when there is no corpus, so it can never describe rows from an earlier save to the
same path. Saving without a corpus therefore clears any corpus already there - which is
deliberate, because keeping it is only correct if the new embeddings are the same rows in
the same order, and nothing enforces that.

Alignment is defended at three points: a mismatched corpus is refused at save time,
discarded with a warning at load time, and carried through row-for-row whenever rows are
dropped. Degrading to row numbers is recoverable; naming the wrong sentence is not.

Indices saved before this file existed simply lack it and continue to load normally.

## Reproducible loading

`load()` downsamples negatives to the requested ratio, and that choice is random. In 1.0
it was unseeded, so a saved index behaved like a slightly different model on every load -
on the shipped example index, the same text scored 0.028636 on one load and 0.011523 on
the next.

```python
index = SentinelLocalIndex.load(path="...", seed=42)
```

Seeding uses a private generator, so your own randomness is untouched.

## Migrating from 1.0

Most code needs no change. The breaking changes are deliberate and narrow.

**Arguments after the first are now keyword-only** on `calculate_rare_class_affinity`,
`from_texts` and `load`. This is what lets a new argument sit in a logical place instead
of being appended to an eleven-parameter list forever.

```python
# 1.0 - still works, but only because the first argument is positional
index.calculate_rare_class_affinity(texts)

# 1.0 positional style - no longer valid
index.calculate_rare_class_affinity(texts, 10)

# 2.0
index.calculate_rare_class_affinity(texts, top_k=10)
```

Every call site in this repository already used keywords, and the documentation always
taught that style, so in practice this is unlikely to affect you.

**Grid-search index columns are prefixed.** `evaluate_groups` returns an `n_positive`
meaning the number of positive *groups* in your evaluation set. The index size briefly
shared that name and silently overwrote it. If you read grid-search output, rename:

| Old | New |
|---|---|
| `n_positive` (index size) | `index_n_positive` |
| `neg_to_pos_ratio` | `index_neg_to_pos_ratio` |
| `n_positive_actual` | `index_n_positive_actual` |
| `n_negative_actual` | `index_n_negative_actual` |

`n_positive` now unambiguously means the positive-group count, as `evaluate_groups`
always documented. Writing a column twice raises instead of overwriting.

**Rows are still plain dicts**, so `pd.DataFrame(rows)` keeps working.

## Known limitations

- `top_k` still forces a re-score. Caching the neighbour search would make it as cheap as
  the index axes, but that is a deeper change.
- `RareClassAffinityResult.observation_scores` is keyed by observation text, so duplicate
  observations within one group collapse into a single entry.
