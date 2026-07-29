# Copyright 2025 Roblox Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the simulation / tuning harness (sentinel.simulation).

These tests are fast and require no model download: they either exercise the
pure metric logic directly on hand-crafted scores, or use a tiny stub index
that mimics ``calculate_rare_class_affinity``.
"""

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from sentinel.score_formulae import max_score
from sentinel.simulation import (
    DEFAULT_AGGREGATORS,
    GroupObservationScores,
    LabeledGroup,
    compare_aggregators,
    evaluate_groups,
    run_grid_search,
    score_groups,
)


def _make_group_scores(pairs):
    """Build GroupObservationScores from (label, affinity) pairs.

    Each group gets a single observation equal to the desired affinity, so that
    aggregating with ``max_score`` (and no threshold) reproduces that affinity.
    """
    return [
        GroupObservationScores(
            name=f"g{i}", label=label, observation_scores=np.array([affinity], dtype=float)
        )
        for i, (label, affinity) in enumerate(pairs)
    ]


# ---------------------------------------------------------------------------
# Ranking family
# ---------------------------------------------------------------------------


def test_roc_auc_separable():
    """Positives clearly above negatives -> AUC 1.0 and perfect top-N."""
    scores = _make_group_scores([(1, 0.9), (1, 0.8), (0, 0.2), (0, 0.1)])
    row = evaluate_groups(scores, max_score, min_score_to_consider=0.0)

    assert row["roc_auc"] == pytest.approx(1.0)
    assert row["recall_at_n"] == pytest.approx(1.0)
    assert row["precision_at_n"] == pytest.approx(1.0)
    # Positives should rank better (smaller rank number) than negatives.
    assert row["avg_rank_positive"] < row["avg_rank_negative"]
    assert row["rank_ratio"] < 1.0
    assert row["n_groups"] == 4
    assert row["n_positive"] == 2
    assert row["top_n"] == 2


def test_roc_auc_reversed():
    """Positives below negatives -> AUC 0.0."""
    scores = _make_group_scores([(1, 0.1), (1, 0.2), (0, 0.9), (0, 0.8)])
    row = evaluate_groups(scores, max_score, min_score_to_consider=0.0)
    assert row["roc_auc"] == pytest.approx(0.0)


def test_roc_auc_tied():
    """All affinities equal -> AUC 0.5 (ties are averaged)."""
    scores = _make_group_scores([(1, 0.5), (1, 0.5), (0, 0.5), (0, 0.5)])
    row = evaluate_groups(scores, max_score, min_score_to_consider=0.0)
    assert row["roc_auc"] == pytest.approx(0.5)


def test_ranking_auc_undefined_single_class():
    """AUC is nan when only one class is present."""
    scores = _make_group_scores([(1, 0.9), (1, 0.8)])
    row = evaluate_groups(scores, max_score, min_score_to_consider=0.0)
    assert np.isnan(row["roc_auc"])


# ---------------------------------------------------------------------------
# Threshold / classification family
# ---------------------------------------------------------------------------


def test_classification_fixed_threshold():
    """A cutoff between the two classes yields perfect precision/recall/F1."""
    scores = _make_group_scores([(1, 0.9), (1, 0.8), (0, 0.2), (0, 0.1)])
    row = evaluate_groups(scores, max_score, min_score_to_consider=0.0, decision_threshold=0.5)

    assert row["decision_threshold"] == pytest.approx(0.5)
    assert row["precision"] == pytest.approx(1.0)
    assert row["recall"] == pytest.approx(1.0)
    assert row["f1"] == pytest.approx(1.0)
    assert row["false_positive_rate"] == pytest.approx(0.0)
    assert (row["tp"], row["fp"], row["fn"], row["tn"]) == (2.0, 0.0, 0.0, 2.0)


def test_classification_flag_everything_threshold():
    """A very low cutoff flags all groups -> recall 1.0, precision = base rate."""
    scores = _make_group_scores([(1, 0.9), (0, 0.2), (0, 0.1)])
    row = evaluate_groups(scores, max_score, min_score_to_consider=0.0, decision_threshold=0.0)
    assert row["recall"] == pytest.approx(1.0)
    assert row["precision"] == pytest.approx(1.0 / 3.0)


def test_classification_best_f1_sweep():
    """With decision_threshold=None the best-F1 cutoff is found automatically."""
    scores = _make_group_scores([(1, 0.9), (1, 0.8), (0, 0.2), (0, 0.1)])
    row = evaluate_groups(scores, max_score, min_score_to_consider=0.0)
    # Perfectly separable -> a threshold achieving F1 == 1.0 exists.
    assert row["f1"] == pytest.approx(1.0)
    # The best cutoff is the lowest positive value (0.8), since predictions use >=.
    assert row["decision_threshold"] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Separation / distribution family
# ---------------------------------------------------------------------------


def test_separation_metrics_basic():
    scores = _make_group_scores([(1, 0.9), (1, 0.7), (0, 0.2), (0, 0.1)])
    row = evaluate_groups(scores, max_score, min_score_to_consider=0.0)
    assert row["mean_separation"] == pytest.approx(0.8 - 0.15)
    assert row["cohens_d"] > 0.0
    assert 0.0 <= row["ks_statistic"] <= 1.0
    assert row["ks_statistic"] == pytest.approx(1.0)  # fully separated


def test_cohens_d_nan_when_zero_variance():
    """Identical values within each class -> pooled std 0 -> Cohen's d is nan."""
    scores = _make_group_scores([(1, 0.5), (1, 0.5), (0, 0.2), (0, 0.2)])
    row = evaluate_groups(scores, max_score, min_score_to_consider=0.0)
    assert row["mean_separation"] == pytest.approx(0.3)
    assert np.isnan(row["cohens_d"])


def test_separation_nan_single_class():
    scores = _make_group_scores([(0, 0.2), (0, 0.1)])
    row = evaluate_groups(scores, max_score, min_score_to_consider=0.0)
    assert np.isnan(row["mean_separation"])
    assert np.isnan(row["ks_statistic"])


# ---------------------------------------------------------------------------
# compare_aggregators
# ---------------------------------------------------------------------------


def test_compare_aggregators_returns_row_per_aggregator():
    scores = _make_group_scores([(1, 0.9), (1, 0.8), (0, 0.2), (0, 0.1)])
    rows = compare_aggregators(scores, min_score_to_consider=0.0)

    assert len(rows) == len(DEFAULT_AGGREGATORS)
    assert {r["aggregator"] for r in rows} == set(DEFAULT_AGGREGATORS.keys())

    expected_keys = {
        # metadata
        "aggregator", "min_score_to_consider", "n_groups", "n_positive",
        # ranking
        "roc_auc", "recall_at_n", "precision_at_n", "top_n",
        "avg_rank_positive", "avg_rank_negative", "rank_ratio",
        # threshold / classification
        "decision_threshold", "tp", "fp", "fn", "tn",
        "precision", "recall", "f1", "false_positive_rate",
        # separation / distribution
        "mean_separation", "cohens_d", "ks_statistic",
    }
    for row in rows:
        assert expected_keys.issubset(row.keys())


# ---------------------------------------------------------------------------
# score_groups / run_grid_search plumbing (with a stub index)
# ---------------------------------------------------------------------------


class _StubIndex:
    """Minimal stand-in for SentinelLocalIndex.

    Interprets each observation string as its numeric score, so tests can
    control per-observation scores exactly with no model.
    """

    def __init__(self):
        self.calls = []

    def calculate_rare_class_affinity(self, text_samples, **kwargs):
        self.calls.append(kwargs)
        observation_scores = {text: float(text) for text in text_samples}
        return SimpleNamespace(observation_scores=observation_scores)


def test_score_groups_extracts_raw_scores():
    index = _StubIndex()
    groups = [
        LabeledGroup(name="pos", label=1, observations=["0.9", "0.1"]),
        LabeledGroup(name="neg", label=0, observations=["0.2", "0.05"]),
    ]

    scored = score_groups(index, groups, top_k=7)

    assert [s.name for s in scored] == ["pos", "neg"]
    assert [s.label for s in scored] == [1, 0]
    np.testing.assert_allclose(scored[0].observation_scores, [0.9, 0.1])
    np.testing.assert_allclose(scored[1].observation_scores, [0.2, 0.05])

    # score_groups must request raw scores (threshold 0) and disable extras.
    call = index.calls[0]
    assert call["top_k"] == 7
    assert call["min_score_to_consider"] == 0.0
    assert call["explain"] is False
    assert call["include_neighbors"] is False


def test_score_groups_handles_empty_group():
    index = _StubIndex()
    groups = [LabeledGroup(name="empty", label=1, observations=[])]
    scored = score_groups(index, groups)
    assert scored[0].observation_scores.size == 0
    # No scoring call should be made for an empty group.
    assert index.calls == []


def test_run_grid_search_rescoring_and_rows():
    index = _StubIndex()
    groups = [
        LabeledGroup(name="pos", label=1, observations=["0.9", "0.8"]),
        LabeledGroup(name="neg", label=0, observations=["0.2", "0.1"]),
    ]

    rows = run_grid_search(
        index,
        groups,
        top_k_values=[3, 5],
        min_score_values=[0.0, 0.2],
        top_n=1,
    )

    # 2 top_k x 2 thresholds x 6 aggregators.
    assert len(rows) == 2 * 2 * len(DEFAULT_AGGREGATORS)
    assert all("top_k" in row for row in rows)
    assert {row["top_k"] for row in rows} == {3, 5}
    # Re-scoring happens once per top_k value (each scores both groups), and NOT
    # again for every threshold/aggregator combination: 2 top_k x 2 groups = 4.
    assert len(index.calls) == 2 * len(groups)


class _StubSubsamplableIndex(_StubIndex):
    """Stub that also records subsample() calls and reports embedding row counts."""

    def __init__(self, n_positive=100, n_negative=500):
        super().__init__()
        self.subsample_calls = []
        self.positive_embeddings = np.zeros((n_positive, 4))
        self.negative_embeddings = np.zeros((n_negative, 4))

    def subsample(self, n_positive=None, neg_to_pos_ratio=None, seed=None):
        self.subsample_calls.append(
            {"n_positive": n_positive, "neg_to_pos_ratio": neg_to_pos_ratio, "seed": seed}
        )
        available_positive = self.positive_embeddings.shape[0]
        kept_positive = (
            min(n_positive, available_positive) if n_positive else available_positive
        )
        kept_negative = (
            min(int(kept_positive * neg_to_pos_ratio), self.negative_embeddings.shape[0])
            if neg_to_pos_ratio
            else self.negative_embeddings.shape[0]
        )
        smaller = _StubSubsamplableIndex(kept_positive, kept_negative)
        # Share the call log so assertions can see scoring done via the copy.
        smaller.calls = self.calls
        return smaller


def _two_groups():
    return [
        LabeledGroup(name="pos", label=1, observations=["0.9", "0.8"]),
        LabeledGroup(name="neg", label=0, observations=["0.2", "0.1"]),
    ]


def test_grid_search_without_index_axes_never_subsamples():
    """The default path must not touch the index at all.

    This is the backward-compatibility guarantee: callers passing any index-like
    object keep working, and behaviour is unchanged from before these axes existed.
    """
    index = _StubSubsamplableIndex()
    rows = run_grid_search(index, _two_groups(), top_k_values=[3], min_score_values=[0.1])

    assert index.subsample_calls == []
    assert len(rows) == len(DEFAULT_AGGREGATORS)
    # The new columns are still present, so the table shape is consistent.
    assert all(row["n_positive"] is None for row in rows)
    assert all(row["neg_to_pos_ratio"] is None for row in rows)


def test_grid_search_sweeps_index_size_and_ratio():
    """Both new axes multiply out, and each configuration is subsampled once."""
    index = _StubSubsamplableIndex()
    rows = run_grid_search(
        index,
        _two_groups(),
        top_k_values=[3, 5],
        min_score_values=[0.0],
        n_positive_values=[10, 20],
        neg_to_pos_ratios=[1.0, 2.0],
        index_seed=42,
    )

    # 2 sizes x 2 ratios x 2 top_k x 1 threshold x 6 aggregators.
    assert len(rows) == 2 * 2 * 2 * len(DEFAULT_AGGREGATORS)
    # subsample() is called once per (size, ratio) pair, NOT once per top_k: the
    # whole point is that reshaping the index is cheap and re-scoring is not.
    assert len(index.subsample_calls) == 4
    assert all(call["seed"] == 42 for call in index.subsample_calls)
    assert {(c["n_positive"], c["neg_to_pos_ratio"]) for c in index.subsample_calls} == {
        (10, 1.0), (10, 2.0), (20, 1.0), (20, 2.0)
    }
    # Scoring happens once per (size, ratio, top_k), each covering both groups.
    assert len(index.calls) == 4 * 2 * len(_two_groups())


def test_grid_search_reports_requested_and_actual_counts():
    """Actual counts are emitted, because a request is clipped on a small index.

    Without them, two rows can look identical while describing different indices.
    """
    index = _StubSubsamplableIndex(n_positive=15, n_negative=500)
    rows = run_grid_search(
        index,
        _two_groups(),
        top_k_values=[3],
        min_score_values=[0.0],
        n_positive_values=[10, 999],  # 999 exceeds the 15 available
        neg_to_pos_ratios=[2.0],
    )

    by_request = {row["n_positive"]: row for row in rows}
    assert by_request[10]["n_positive_actual"] == 10
    assert by_request[10]["n_negative_actual"] == 20
    # Clipped to what the index actually holds, and visible in the row.
    assert by_request[999]["n_positive_actual"] == 15
    assert by_request[999]["n_negative_actual"] == 30


def test_grid_search_one_axis_at_a_time():
    """Either axis can be swept on its own."""
    index = _StubSubsamplableIndex()
    size_only = run_grid_search(
        index, _two_groups(), top_k_values=[3], min_score_values=[0.0],
        n_positive_values=[10, 20],
    )
    assert {row["n_positive"] for row in size_only} == {10, 20}
    assert all(row["neg_to_pos_ratio"] is None for row in size_only)

    index2 = _StubSubsamplableIndex()
    ratio_only = run_grid_search(
        index2, _two_groups(), top_k_values=[3], min_score_values=[0.0],
        neg_to_pos_ratios=[0.5, 1.0],
    )
    assert {row["neg_to_pos_ratio"] for row in ratio_only} == {0.5, 1.0}
    assert all(row["n_positive"] is None for row in ratio_only)


def test_grid_search_logs_progress_per_configuration(caplog):
    """A long sweep logs progress so it cannot be mistaken for a hang."""
    caplog.set_level(logging.INFO)
    index = _StubSubsamplableIndex()
    run_grid_search(
        index, _two_groups(), top_k_values=[3], min_score_values=[0.0],
        n_positive_values=[10, 20], neg_to_pos_ratios=[1.0],
    )

    assert "index configuration 1/2" in caplog.text
    assert "index configuration 2/2" in caplog.text
