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

"""Tests for score_formulae module."""

import numpy as np
import pytest

from sentinel.score_formulae import (
    mean_of_positives,
    calculate_contrastive_score,
    skewness,
)


def test_calculate_contrastive_score():
    """Test calculate_contrastive_score function."""
    # Case 1: Text more similar to positive than negative examples
    positive_sims = [0.9, 0.8, 0.7]
    negative_sims = [0.5, 0.4, 0.3]
    score = calculate_contrastive_score(positive_sims, negative_sims)
    assert (
        score > 0
    ), "Score should be positive when text is more similar to positive examples"

    # Case 2: Text more similar to negative than positive examples
    # Based on the implementation, when negatives are more similar, the score is clipped to 0
    positive_sims = [0.5, 0.4, 0.3]
    negative_sims = [0.9, 0.8, 0.7]
    score = calculate_contrastive_score(positive_sims, negative_sims)
    assert (
        score == 0
    ), "Score should be 0 when text is more similar to negative examples"

    # Case 3: Text equally similar to both
    positive_sims = [0.7, 0.6, 0.5]
    negative_sims = [0.7, 0.6, 0.5]
    score = calculate_contrastive_score(positive_sims, negative_sims)
    assert (
        abs(score) < 1e-6
    ), "Score should be close to zero when equally similar to both"

    # Case 4: Test with different list lengths
    positive_sims = [0.9, 0.8]
    negative_sims = [0.5, 0.4, 0.3]
    score = calculate_contrastive_score(positive_sims, negative_sims)
    assert score > 0, "Score should handle different list lengths"


def test_mean_of_positives():
    """Test mean_of_positives function."""
    # Test with all positive scores
    scores = np.array([0.5, 0.3, 0.7])
    result = mean_of_positives(scores)
    assert result == 0.5, "Should return the mean of all positive scores"

    # Test with mixed scores (positive and negative)
    scores = np.array([0.5, -0.3, 0.7, -0.2])
    result = mean_of_positives(scores)
    assert result == 0.6, "Should ignore negative scores and return mean of positives"

    # Test with all negative scores - will raise a RuntimeWarning, but we're checking that
    # we handle the "mean of empty slice" case gracefully
    with pytest.warns(RuntimeWarning):
        scores = np.array([-0.5, -0.3, -0.7])
        result = mean_of_positives(scores)
        # When there's no positive scores, numpy returns NaN for an empty slice
        assert np.isnan(
            result
        ), "Should return NaN when there are no positive scores"  # Test with empty array - will raise a RuntimeWarning
        with pytest.warns(RuntimeWarning):
            scores = np.array([])
            result = mean_of_positives(scores)
            # When there's an empty array, numpy returns NaN
            assert np.isnan(result), "Should return NaN for empty array"


def test_skewness():
    """Test skewness function."""
    # Test with fewer scores than min_size_of_scores (which defaults to 10)
    # This should return 0.0 because we don't have enough data points
    scores = np.array([0.1, 0.2, 0.3, 0.9, 1.0])
    result = skewness(scores)
    assert np.isclose(
        result, 0.0
    ), "Should return 0.0 when fewer scores than min_size_of_scores"

    # Test with enough scores and explicitly set min_size_of_scores
    scores = np.array([0.1, 0.2, 0.3, 0.9, 1.0, 0.2, 0.3, 0.1, 0.2, 0.8, 0.7])
    result = skewness(scores, min_size_of_scores=5)
    assert result > 0, "Should return positive value for right-skewed distribution"

    # Test with negatively skewed distribution
    scores = np.array([0.0, 0.1, 0.7, 0.8, 0.9, 0.7, 0.8, 0.9, 0.8, 0.7])
    result = skewness(scores, min_size_of_scores=5)
    assert result < 0, "Should return negative value for left-skewed distribution"

    # Test with symmetric distribution
    symmetric_scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.1, 0.3, 0.5, 0.7, 0.9])
    result = skewness(symmetric_scores, min_size_of_scores=5)
    assert abs(result) < 0.01, "Should return close to zero for symmetric distribution"

    # Test with constant values
    constant_scores = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    result = skewness(constant_scores, min_size_of_scores=5)
    assert abs(result) < 1e-10, "Should return very close to 0.0 for constant values"

    # Test with insufficient scores
    small_scores = np.array([0.5])
    result = skewness(small_scores)
    assert np.isclose(result, 0.0), "Should return 0.0 for insufficient scores"

    # Test with empty array
    empty_scores = np.array([])
    result = skewness(empty_scores)
    assert np.isclose(result, 0.0), "Should return 0.0 for empty array"
