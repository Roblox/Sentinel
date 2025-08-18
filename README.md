# Sentinel


## Overview

Roblox Sentinel, part of the Roblox Safety Toolkit, is a Python library designed specifically for realtime detection of extremely rare classes of text by using contrastive learning principles. While traditional classifiers struggle with highly imbalanced datasets, Sentinel excels by:

1. Collecting recent observations from a single source (e.g., recent messages from a user)
2. Calculating individual observation scores using embedding similarity
3. Aggregating these scores using statistical measures like skewness to detect patterns

By prioritizing recall over precision, Sentinel serves as a high-recall candidate generator for more thorough investigation. This approach is particularly effective for applications where rare patterns are critical to identify. Rather than treating each message in isolation, Sentinel analyzes patterns across messages to identify concerning behavior.

## What’s New: Aggregation options and Explainability

Sentinel now includes multiple aggregation strategies and built‑in explainability to help you tune for your use case and understand why a score was assigned.

- Aggregators (in `sentinel.score_formulae`):
    - `skewness(scores, min_size_of_scores=10)`: default, pattern‑oriented and robust to message count
    - `top_k_mean(scores, k=3)`: focuses on the strongest signals
    - `percentile_score(scores, q=90.0)`: robust to outliers via a percentile over positives
    - `softmax_weighted_mean(scores, temperature=1.0)`: smoothly emphasizes higher scores
    - `max_score(scores)`: simplest, picks the highest positive score

- Explainability (in results):
    - Each call to `calculate_rare_class_affinity` returns a `RareClassAffinityResult` with:
        - `aggregation_name`, `aggregation_stats`: which aggregator was used and key params
        - `explanations`: per‑text details including top‑K positive/negative similarities, contrastive components, and neighbor snippets (when available)

## Terminology

In Sentinel's codebase:
- **Positive examples**: Examples of text that belong to the rare class of interest (e.g., harmful, unsafe, or critical content)
- **Negative examples**: Examples of text that belong to the common class (e.g., safe, neutral, or typical content)

## Installation

```bash
pip install .
```

By default `sentinel` doesn't pull in all transitive dependencies, specifically avoiding pulling in sentence transformers and its dependencies (torch).
To pull them in as well, use:

```bash
pip install '.[sbert]'
```

## Quick Start

```python
from sentinel.sentinel_local_index import SentinelLocalIndex

# Load a previously saved index from a local path
index = SentinelLocalIndex.load(path="path/to/local/index")

# Or load from S3
index = SentinelLocalIndex.load(
    path="s3://my-bucket/path/to/index",
    aws_access_key_id="YOUR_ACCESS_KEY_ID",  # Optional if using environment credentials
    aws_secret_access_key="YOUR_SECRET_ACCESS_KEY"  # Optional if using environment credentials
)

# Collect recent observations from a single source (e.g., recent messages from a user)
user_recent_messages = [
    "Hey how are you?",
    "What are you doing today?",
    "Do you have any pictures you can share?",
    "Where do you live?",
    "Are your parents home right now?"
]

# Calculate rare class affinity across all observations
result = index.calculate_rare_class_affinity(user_recent_messages)

# Get the overall score (uses skewness by default)
overall_score = result.rare_class_affinity_score
print(f"Overall rare class affinity score: {overall_score:.4f}")

# Examine individual observation scores
for message, score in result.observation_scores.items():
    risk_level = "High" if score > 0.5 else "Medium" if score > 0.1 else "Low"
    print(f"'{message}' - Score: {score:.4f} - Risk: {risk_level}")

# Inspect explainability
print("Aggregator:", result.aggregation_name)
print("Aggregation stats:", result.aggregation_stats)
for message, ex in result.explanations.items():
    print("--", message)
    print("   topk_positive:", ex["topk_positive"])  # scaled similarities
    print("   topk_negative:", ex["topk_negative"])  # scaled similarities
    print("   contrastive:", ex["contrastive"])      # positive_term, negative_term, log_ratio_unclipped
    print("   neighbors (sample):", ex["neighbors"][:2] if ex["neighbors"] else None)
```

## Creating a New Index

```python
import torch
from sentinel.sentinel_local_index import SentinelLocalIndex
from sentinel.embeddings.sbert import get_sentence_transformer_and_scaling_fn

# Initialize sentence model and get scaling function
model_name = "all-MiniLM-L6-v2"
model, scale_fn = get_sentence_transformer_and_scaling_fn(model_name)

# Prepare examples
positive_examples = ["positive message 1", "rare class example", "critical content example"]
negative_examples = ["neutral message 1", "common class example", "typical content"]

# Encode examples
positive_embeddings = model.encode(positive_examples, normalize_embeddings=True)
negative_embeddings = model.encode(negative_examples, normalize_embeddings=True)

# Create the index
index = SentinelLocalIndex(
    sentence_model=model,
    positive_embeddings=positive_embeddings,
    negative_embeddings=negative_embeddings,
    scale_fn=scale_fn,
    positive_corpus=positive_examples,
    negative_corpus=negative_examples,
)

# Save locally - provide the model name when saving
# You must provide the encoder model name as it can't be reliably extracted from a SentenceTransformer instance
# The save method returns the SavedIndexConfig for informational purposes, but it's already saved at the specified location
saved_config = index.save(path="path/to/local/index", encoder_model_name_or_path=model_name)
print(f"Saved index with encoder model: {saved_config.encoder_model_name_or_path}")

# Or save to S3
saved_config = index.save(
    path="s3://my-bucket/path/to/index",
    encoder_model_name_or_path=model_name,
    aws_access_key_id="YOUR_ACCESS_KEY_ID",  # Optional if using environment credentials
    aws_secret_access_key="YOUR_SECRET_ACCESS_KEY"  # Optional if using environment credentials
)

## Testing for optimal Thresholds and data ratio's

Usage of the 'examples\Example_Threshold_Script.py' script will allow for quick threshold checks for a variety of ratios, by default these are 10:1, 5:1 and 1:1 ratios. This has predefined example chat logs, and should, show optimal settings for the dataset being used based on an average score and average detection count.
You will be able to get detailed information output in the ratios with the -r or --review flags.

## Choosing an aggregation strategy

Different deployments optimize for different trade‑offs. You can swap in any aggregator using the `aggregation_function` argument:

```python
from sentinel.score_formulae import top_k_mean, percentile_score, softmax_weighted_mean, max_score

texts = ["msg a", "msg b", "msg c"]

# Focus on the strongest few signals
res1 = index.calculate_rare_class_affinity(texts, aggregation_function=lambda arr: top_k_mean(arr, k=3))

# Robust to outliers
res2 = index.calculate_rare_class_affinity(texts, aggregation_function=lambda arr: percentile_score(arr, q=90))

# Smoothly emphasize higher scores
res3 = index.calculate_rare_class_affinity(texts, aggregation_function=lambda arr: softmax_weighted_mean(arr, temperature=0.5))

# Simplest, picks the maximum
res4 = index.calculate_rare_class_affinity(texts, aggregation_function=max_score)
```

Notes:
- All aggregators operate over per‑observation scores where non‑confident observations are already clipped to 0.
- The default `skewness` remains a good choice when user activity volume varies widely.
```

## How It Works

Sentinel uses a two-step process to detect rare classes of text, focusing on high recall for realtime applications:

1. **Individual Observation Scoring**:
   - Each text observation (e.g., message, post) is compared against both rare class examples and common class examples
   - Using embedding similarity, we calculate how close the observation is to each class
   - The observation score is the ratio between rare class similarity and common class similarity
   - Scores > 0.1 indicate closer similarity to rare class examples

2. **Pattern Recognition via Skewness**:
   - Recent individual observation scores from the same source are collected
   - Skewness measures the asymmetry in the distribution of these scores
   - A positive skewness indicates a pattern where most content is common, but with enough rare-class similarities to create a right-skewed distribution
   - This method is resistant to variations in the number of observations, making it ideal for sources with different activity levels
   - By focusing on patterns rather than individual messages, it achieves higher recall for rare phenomena
   - The aggregated score reveals patterns that would be missed when analyzing messages individually

As a high-recall candidate generator, Sentinel identifies potential cases for further investigation, prioritizing not missing true positives even at the cost of some false positives.

## Motivating Use Case

Sentinel was developed to detect extremely rare classes of harmful content where traditional classification approaches fail due to the scarcity of examples. A prominent application was detecting child grooming attempts on Roblox:

1. **The Challenge**: Child grooming patterns are extremely rare in overall communications but devastating when they occur. Traditional classifiers struggle with such imbalanced classes.

2. **The Approach**:
   - Collect recent communications from a single source (e.g., a user's recent chat messages)
   - Score each message individually using contrastive learning to determine similarity to known harmful patterns
   - Aggregate these scores using skewness to detect overall patterns, regardless of message volume
   - Generate candidates for thorough investigation, prioritizing recall over precision

3. **Real-world Impact**: This approach led to over 1,000 NCMEC (National Center for Missing & Exploited Children) reports in just the first few months of deployment at Roblox, significantly improving platform safety.

The same methodology can be applied to any rare text classification problem where:
- Examples of the target class are extremely scarce
- Traditional classifiers would struggle with recall
- Realtime detection is required
- Context across multiple observations from the same source is meaningful
- High recall is prioritized over precision for initial screening

## Storage Options

Sentinel supports both local file storage and S3 storage:

- For local storage, use paths starting with `/` or a relative path
- For S3 storage, use URI format: `s3://bucket-name/path/to/index`

The storage is abstracted using `smart_open`, making it seamless to switch between storage backends.

## Examples
To run the notebook examples
```bash
# Install with examples dependencies
poetry install --with examples
poetry install --extras=sbert
poetry run jupyter notebook
```

## License

Apache License 2.0
