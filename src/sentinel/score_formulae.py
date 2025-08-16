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

"""Score calculation functions for Sentinel index."""

import numpy as np
from typing import List, Callable, Tuple
from scipy import stats
import warnings


def mean_of_positives(scores: np.array) -> float:
    """Calculate the mean of positive contrastive scores across multiple observations.

    This function aggregates individual observation scores, focusing only on positive scores
    (observations that were more similar to rare class examples than common class examples).

    Unlike skewness, this aggregation method directly measures the average strength of rare class
    similarity but may be more sensitive to the number of observations. It's useful when you want
    to focus on the magnitude of similarity rather than the pattern across observations.

    Args:
        scores: Array of contrastive scores from multiple observations

    Returns:
        Mean of positive scores, indicating overall affinity to rare class content
    """
    return np.mean(scores[scores > 0])


def skewness(scores: np.array, min_size_of_scores: int = 10) -> float:
    """Calculate the skewness of contrastive scores to detect patterns of rare class content.

    Skewness measures the asymmetry in the distribution of contrastive scores across multiple observations.
    It is particularly effective for rare class detection because:

    1. It focuses on the pattern of scores rather than their quantity
    2. It's sensitive to occasional spikes in similarity that might indicate rare events
    3. It's robust to varying numbers of observations (e.g., recent chat volumes)
    4. It can work with a relatively small number of recent observations

    As a high-recall metric, a positive skewness suggests that while most observations are neutral/common,
    there are enough rare-class observations to create a right-skewed distribution.
    This makes it ideal for generating candidates for further investigation without being affected
    by the total volume of observations.

    Args:
        scores: Array of contrastive scores from multiple observations
        min_size_of_scores: Minimum number of scores required to calculate meaningful skewness

    Returns:
        Skewness value, where positive values suggest patterns of rare class content
    """
    if len(scores) < min_size_of_scores:
        return 0.0
    mean = np.mean(scores)
    median = np.median(scores)
    std = np.std(scores)
    if std == 0:
        return 0.0
    return (mean - median) / std


def calculate_contrastive_score(
    similarities_topk_pos: List[float],
    similarities_topk_neg: List[float],
    aggregation_fn: Callable[[np.array], float] = np.mean,
) -> float:
    """Calculate a contrastive score for a single observation.

    This function uses a contrastive learning approach to determine how similar an observation
    is to examples of the rare class compared to examples of the common class. It computes
    a ratio between similarities to positive (rare class) examples and similarities to
    negative (common class) examples.

    These individual observation scores are later aggregated across multiple observations
    (e.g., messages, posts) using functions like `skewness` to identify patterns indicative
    of the rare class, regardless of the total number of observations.

    Args:
        similarities_topk_pos: List of similarities between the observation and rare class examples
        similarities_topk_neg: List of similarities between the observation and common class examples
        aggregation_fn: Function to aggregate similarity values within each category

    Returns:
        A contrastive score where values > 0 indicate closer similarity to rare class content
    """
    if len(similarities_topk_pos) <= 0 or len(similarities_topk_neg) <= 0:
        raise ValueError(
            "The lists of similarities must have at least one element each."
        )

    similarities_topk_pos = np.array(similarities_topk_pos)
    similarities_topk_neg = np.array(similarities_topk_neg)

    positives_term = aggregation_fn(np.exp(similarities_topk_pos))
    negatives_term = aggregation_fn(np.exp(similarities_topk_neg))

    contrastive_score = positives_term / negatives_term

    if contrastive_score <= 1.0:
        return 0  # Clip to zero to avoid negative scores, since we accumulate this score for all chat lines of a user.
    return np.log(contrastive_score)


def entropy_weighted_skewness(scores: np.array, min_size_of_scores: int = 10) -> float:
    """Calculate entropy-weighted skewness to detect evasive patterns in score distributions.
    
    This enhanced scoring method combines traditional skewness with entropy analysis to identify
    sophisticated evasion attempts. High entropy (randomness) combined with positive skewness
    may indicate attempts to disguise harmful content through varied expression patterns.
    
    Args:
        scores: Array of contrastive scores from multiple observations
        min_size_of_scores: Minimum number of scores required for meaningful calculation
        
    Returns:
        Entropy-weighted skewness value for improved pattern detection
    """
    if len(scores) < min_size_of_scores:
        return 0.0
        
    # Calculate traditional skewness
    base_skewness = skewness(scores, min_size_of_scores)
    
    # Calculate entropy of score distribution
    if len(np.unique(scores)) < 2:
        return base_skewness  # No diversity to analyze
        
    # Discretize scores for entropy calculation
    hist, _ = np.histogram(scores, bins=min(10, len(scores)//2))
    hist = hist + 1e-10  # Avoid log(0)
    probabilities = hist / np.sum(hist)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    
    # Weight skewness by normalized entropy (higher entropy increases weight)
    max_entropy = np.log2(len(hist))
    entropy_weight = entropy / max_entropy if max_entropy > 0 else 0
    
    return base_skewness * (1 + entropy_weight)


def robust_skewness_with_confidence(scores: np.array, min_size_of_scores: int = 10, 
                                  confidence_level: float = 0.95) -> Tuple[float, float, float]:
    """Calculate robust skewness with confidence intervals for reliability assessment.
    
    This method provides not just the skewness value but also confidence bounds,
    allowing for more informed decision-making in high-stakes scenarios.
    
    Args:
        scores: Array of contrastive scores from multiple observations
        min_size_of_scores: Minimum number of scores required for calculation
        confidence_level: Confidence level for interval calculation (0.0-1.0)
        
    Returns:
        Tuple of (skewness_value, lower_confidence_bound, upper_confidence_bound)
    """
    if len(scores) < min_size_of_scores:
        return 0.0, 0.0, 0.0
        
    # Calculate base skewness
    base_skewness = skewness(scores, min_size_of_scores)
    
    if len(scores) < 30:  # Use bootstrap for small samples
        bootstrap_skewness = []
        n_bootstrap = 1000
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(scores, size=len(scores), replace=True)
            bootstrap_skewness.append(skewness(bootstrap_sample, min_size_of_scores))
        
        bootstrap_skewness = np.array(bootstrap_skewness)
        alpha = 1 - confidence_level
        lower_bound = np.percentile(bootstrap_skewness, 100 * alpha / 2)
        upper_bound = np.percentile(bootstrap_skewness, 100 * (1 - alpha / 2))
    else:
        # Use asymptotic distribution for larger samples
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            skew_stat = stats.skew(scores)
            n = len(scores)
            # Standard error approximation for skewness
            se_skew = np.sqrt(6 * n * (n - 1) / ((n - 2) * (n + 1) * (n + 3)))
            
        alpha = 1 - confidence_level
        z_critical = stats.norm.ppf(1 - alpha / 2)
        margin = z_critical * se_skew
        
        lower_bound = skew_stat - margin
        upper_bound = skew_stat + margin
    
    return base_skewness, lower_bound, upper_bound


def adaptive_threshold_score(scores: np.array, historical_scores: np.array = None,
                           base_threshold: float = 0.1) -> float:
    """Calculate adaptive threshold based on historical score distributions.
    
    This method adjusts the scoring threshold based on historical patterns,
    improving detection accuracy by accounting for baseline activity patterns.
    
    Args:
        scores: Current array of contrastive scores
        historical_scores: Historical scores for baseline calculation
        base_threshold: Base threshold when no historical data available
        
    Returns:
        Adaptive threshold value for the current context
    """
    if historical_scores is None or len(historical_scores) < 50:
        return base_threshold
        
    # Calculate historical distribution statistics
    hist_mean = np.mean(historical_scores)
    hist_std = np.std(historical_scores)
    hist_95th = np.percentile(historical_scores, 95)
    
    # Current scores statistics
    curr_mean = np.mean(scores) if len(scores) > 0 else 0
    
    # Adaptive threshold based on deviation from historical norm
    if hist_std > 0:
        z_score = (curr_mean - hist_mean) / hist_std
        # Increase threshold if current activity is unusually high
        threshold_multiplier = 1 + max(0, z_score * 0.5)
    else:
        threshold_multiplier = 1
        
    adaptive_threshold = min(base_threshold * threshold_multiplier, hist_95th)
    return max(adaptive_threshold, base_threshold * 0.5)  # Floor at 50% of base


def ensemble_scoring(scores: np.array, min_size_of_scores: int = 10,
                    weights: List[float] = None) -> float:
    """Ensemble scoring combining multiple detection strategies for robust assessment.
    
    This method combines multiple scoring approaches to provide a more comprehensive
    and reliable assessment, reducing the risk of evasion through single-method exploitation.
    
    Args:
        scores: Array of contrastive scores from multiple observations
        min_size_of_scores: Minimum number of scores required for calculation
        weights: Weights for different scoring methods [skewness, entropy_weighted, robust]
        
    Returns:
        Ensemble score combining multiple detection strategies
    """
    if len(scores) < min_size_of_scores:
        return 0.0
        
    if weights is None:
        weights = [0.4, 0.35, 0.25]  # Default weights
        
    # Ensure weights sum to 1
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    
    # Calculate individual scores
    skew_score = skewness(scores, min_size_of_scores)
    entropy_score = entropy_weighted_skewness(scores, min_size_of_scores)
    robust_score, _, _ = robust_skewness_with_confidence(scores, min_size_of_scores)
    
    # Normalize scores to similar ranges for fair combination
    scores_array = np.array([skew_score, entropy_score, robust_score])
    
    # Handle case where all scores are zero
    if np.all(scores_array == 0):
        return 0.0
        
    # Simple weighted average
    ensemble_score = np.average(scores_array, weights=weights)
    
    return ensemble_score


def pattern_consistency_score(scores: np.array, window_size: int = 5) -> float:
    """Analyze consistency of patterns across sliding windows for temporal detection.
    
    This method examines whether concerning patterns are sustained over time,
    helping distinguish persistent threats from isolated incidents.
    
    Args:
        scores: Array of contrastive scores ordered by time
        window_size: Size of sliding window for consistency analysis
        
    Returns:
        Pattern consistency score indicating sustained concerning behavior
    """
    if len(scores) < window_size * 2:
        return 0.0
        
    window_scores = []
    
    # Calculate skewness for each sliding window
    for i in range(len(scores) - window_size + 1):
        window = scores[i:i + window_size]
        if len(window) >= window_size:
            window_skew = skewness(window, min_size_of_scores=window_size)
            window_scores.append(window_skew)
    
    if not window_scores:
        return 0.0
        
    window_scores = np.array(window_scores)
    
    # Measure consistency: low variance in positive skewness indicates sustained patterns
    positive_windows = window_scores[window_scores > 0]
    
    if len(positive_windows) < 2:
        return 0.0
        
    # Consistency is high when we have multiple positive windows with low variance
    consistency = len(positive_windows) / len(window_scores)  # Frequency of positive windows
    if len(positive_windows) > 1:
        stability = 1 / (1 + np.var(positive_windows))  # Inverse variance for stability
        return consistency * stability * np.mean(positive_windows)
    else:
        return consistency * positive_windows[0]
