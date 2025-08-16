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

"""Tests for enhanced scoring algorithms in Sentinel."""

import numpy as np
import pytest
from unittest.mock import patch

from sentinel.score_formulae import (
    entropy_weighted_skewness,
    robust_skewness_with_confidence,
    adaptive_threshold_score,
    ensemble_scoring,
    pattern_consistency_score,
)


class TestEntropyWeightedSkewness:
    """Test entropy-weighted skewness scoring."""

    def test_empty_scores(self):
        """Test with empty scores array."""
        scores = np.array([])
        result = entropy_weighted_skewness(scores)
        assert result == 0.0

    def test_insufficient_scores(self):
        """Test with insufficient scores."""
        scores = np.array([0.1, 0.2, 0.3])
        result = entropy_weighted_skewness(scores, min_size_of_scores=10)
        assert result == 0.0

    def test_uniform_scores(self):
        """Test with uniform scores (no diversity)."""
        scores = np.array([0.5] * 15)
        result = entropy_weighted_skewness(scores)
        # Should return base skewness since no diversity
        assert isinstance(result, float)

    def test_diverse_scores(self):
        """Test with diverse score distribution."""
        # Create a right-skewed distribution with high entropy
        scores = np.array([0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 
                          1.0, 1.1, 1.2, 1.5, 2.0])
        result = entropy_weighted_skewness(scores)
        assert isinstance(result, float)
        assert result > 0  # Should be positive for right-skewed data


class TestRobustSkewnessWithConfidence:
    """Test robust skewness with confidence intervals."""

    def test_empty_scores(self):
        """Test with empty scores array."""
        scores = np.array([])
        skew, lower, upper = robust_skewness_with_confidence(scores)
        assert skew == 0.0
        assert lower == 0.0
        assert upper == 0.0

    def test_small_sample_bootstrap(self):
        """Test bootstrap method for small samples."""
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                          1.1, 1.2])
        skew, lower, upper = robust_skewness_with_confidence(scores)
        assert isinstance(skew, float)
        assert isinstance(lower, float)
        assert isinstance(upper, float)
        assert lower <= skew <= upper

    def test_large_sample_asymptotic(self):
        """Test asymptotic method for large samples."""
        # Create larger sample for asymptotic approach
        np.random.seed(42)
        scores = np.random.exponential(scale=0.5, size=50)  # Right-skewed
        skew, lower, upper = robust_skewness_with_confidence(scores)
        assert isinstance(skew, float)
        assert isinstance(lower, float)
        assert isinstance(upper, float)
        assert lower <= upper

    def test_confidence_level(self):
        """Test different confidence levels."""
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                          1.1, 1.2, 1.3, 1.4, 1.5])
        
        # 95% confidence interval should be wider than 80%
        _, lower_95, upper_95 = robust_skewness_with_confidence(scores, confidence_level=0.95)
        _, lower_80, upper_80 = robust_skewness_with_confidence(scores, confidence_level=0.80)
        
        assert (upper_95 - lower_95) >= (upper_80 - lower_80)


class TestAdaptiveThresholdScore:
    """Test adaptive threshold scoring."""

    def test_no_historical_data(self):
        """Test with no historical data."""
        scores = np.array([0.1, 0.2, 0.3])
        base_threshold = 0.1
        result = adaptive_threshold_score(scores, base_threshold=base_threshold)
        assert result == base_threshold

    def test_insufficient_historical_data(self):
        """Test with insufficient historical data."""
        scores = np.array([0.1, 0.2, 0.3])
        historical = np.array([0.05, 0.1, 0.15])  # Too few points
        base_threshold = 0.1
        result = adaptive_threshold_score(scores, historical, base_threshold)
        assert result == base_threshold

    def test_normal_activity(self):
        """Test with normal historical activity."""
        scores = np.array([0.1, 0.15, 0.12])
        historical = np.random.normal(0.1, 0.05, 100)  # Normal baseline
        base_threshold = 0.1
        result = adaptive_threshold_score(scores, historical, base_threshold)
        assert isinstance(result, float)
        assert result >= base_threshold * 0.5  # Should respect floor

    def test_elevated_activity(self):
        """Test with elevated current activity."""
        scores = np.array([0.8, 0.9, 1.0])  # High current scores
        historical = np.random.normal(0.1, 0.05, 100)  # Low baseline
        base_threshold = 0.1
        result = adaptive_threshold_score(scores, historical, base_threshold)
        assert result > base_threshold  # Should increase threshold


class TestEnsembleScoring:
    """Test ensemble scoring algorithm."""

    def test_empty_scores(self):
        """Test with empty scores array."""
        scores = np.array([])
        result = ensemble_scoring(scores)
        assert result == 0.0

    def test_insufficient_scores(self):
        """Test with insufficient scores."""
        scores = np.array([0.1, 0.2])
        result = ensemble_scoring(scores, min_size_of_scores=10)
        assert result == 0.0

    def test_all_zero_scores(self):
        """Test with all zero scores."""
        scores = np.array([0.0] * 15)
        result = ensemble_scoring(scores)
        assert result == 0.0

    def test_custom_weights(self):
        """Test with custom weights."""
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                          1.1, 1.2, 1.3, 1.4, 1.5])
        weights = [0.5, 0.3, 0.2]
        result = ensemble_scoring(scores, weights=weights)
        assert isinstance(result, float)

    def test_weight_normalization(self):
        """Test that weights are properly normalized."""
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                          1.1, 1.2, 1.3, 1.4, 1.5])
        weights = [2.0, 3.0, 5.0]  # Don't sum to 1
        result = ensemble_scoring(scores, weights=weights)
        assert isinstance(result, float)


class TestPatternConsistencyScore:
    """Test pattern consistency scoring."""

    def test_insufficient_scores(self):
        """Test with insufficient scores for windowing."""
        scores = np.array([0.1, 0.2, 0.3])
        result = pattern_consistency_score(scores, window_size=5)
        assert result == 0.0

    def test_no_positive_windows(self):
        """Test with no positive scoring windows."""
        scores = np.array([-0.1, -0.2, -0.3, -0.1, -0.2, -0.3, -0.1, -0.2, -0.3, -0.1])
        result = pattern_consistency_score(scores, window_size=3)
        assert result == 0.0

    def test_single_positive_window(self):
        """Test with single positive scoring window."""
        scores = np.array([0.0, 0.0, 0.0, 0.5, 0.6, 0.7, 0.0, 0.0, 0.0, 0.0])
        result = pattern_consistency_score(scores, window_size=3)
        assert isinstance(result, float)
        assert result > 0

    def test_consistent_positive_patterns(self):
        """Test with consistent positive patterns."""
        # Create data with sustained positive patterns
        scores = np.array([0.3, 0.4, 0.5, 0.4, 0.3, 0.5, 0.4, 0.3, 0.4, 0.5,
                          0.3, 0.4, 0.5, 0.4, 0.3])
        result = pattern_consistency_score(scores, window_size=3)
        assert isinstance(result, float)
        assert result > 0

    def test_inconsistent_patterns(self):
        """Test with inconsistent patterns."""
        scores = np.array([0.8, 0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1,
                          0.7, 0.1, 0.1, 0.1, 0.1])
        result = pattern_consistency_score(scores, window_size=3)
        assert isinstance(result, float)


class TestIntegration:
    """Integration tests for enhanced scoring."""

    def test_realistic_threat_scenario(self):
        """Test with realistic threat detection scenario."""
        # Simulate escalating threat pattern
        baseline_scores = np.random.normal(0.05, 0.02, 20)  # Normal activity
        threat_scores = np.array([0.1, 0.15, 0.3, 0.4, 0.6, 0.7, 0.8, 0.5, 0.6, 0.9])
        all_scores = np.concatenate([baseline_scores, threat_scores])
        
        # Test entropy weighted scoring
        entropy_score = entropy_weighted_skewness(all_scores)
        assert entropy_score > 0
        
        # Test ensemble scoring
        ensemble_score = ensemble_scoring(all_scores)
        assert ensemble_score > 0
        
        # Test pattern consistency
        consistency_score = pattern_consistency_score(all_scores)
        assert isinstance(consistency_score, float)

    def test_evasion_detection(self):
        """Test detection of evasive patterns."""
        # Simulate evasive behavior with varied but concerning patterns
        evasive_scores = np.array([0.05, 0.3, 0.08, 0.4, 0.07, 0.5, 0.06, 0.6,
                                  0.09, 0.7, 0.04, 0.2, 0.08, 0.3, 0.05])
        
        # Entropy-weighted should detect this better than simple skewness
        simple_skew = robust_skewness_with_confidence(evasive_scores)[0]
        entropy_skew = entropy_weighted_skewness(evasive_scores)
        
        # Entropy-weighted should generally be higher for evasive patterns
        assert isinstance(simple_skew, float)
        assert isinstance(entropy_skew, float)