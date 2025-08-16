# Sentinel - Enhanced Multi-Platform Safety Detection

## ⚠️ IMPORTANT: DEFENSIVE SECURITY USE ONLY ⚠️

**This system is authorized EXCLUSIVELY for defensive security purposes:**
- ✅ Content safety and moderation systems
- ✅ Academic research on harmful content detection  
- ✅ Educational and training purposes
- ✅ Vulnerability assessment and security testing

**PROHIBITED USES:**
- ❌ Generating or amplifying harmful content
- ❌ Training offensive language models
- ❌ Creating attack tools or capabilities
- ❌ Surveillance without consent

## Overview

Sentinel is a comprehensive multi-platform safety detection system designed for real-time identification of harmful content across different platforms and violation types. The enhanced version provides:

### 🎯 **Multi-Category Detection**
- **Child Safety**: Grooming, exploitation, inappropriate contact
- **Harassment**: Cyberbullying, targeted abuse, doxxing threats
- **Hate Speech**: Discrimination, extremism, dehumanizing language
- **Gaming Toxicity**: Griefing, competitive harassment, toxic behavior
- **Platform Abuse**: Spam, scams, filter bypass attempts
- **Content Violations**: General harmful content patterns

### 🏢 **Platform-Specific Configurations**
- **Roblox**: Child-focused with strict safety thresholds
- **Discord**: Server-based with coordination abuse detection
- **Twitter/X**: Public discourse with hate speech focus
- **Reddit**: Forum-style with brigading detection
- **General**: Configurable for any platform

### 🛡️ **Advanced Safety Features**
- Comprehensive usage monitoring and audit trails
- Rate limiting and access control
- Automated threat assessment and alerting
- Compliance reporting and policy enforcement
- Validation framework for accuracy testing

Rather than treating each message in isolation, Sentinel analyzes patterns across multiple observations to identify concerning behavior trends, making it ideal for detecting sophisticated threats that evolve over time.

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

### Enhanced Multi-Category Detection

```python
from sentinel.multi_category_index import MultiCategorySentinelIndex
from sentinel.violation_taxonomy import Platform
from sentinel.safety_guardrails import initialize_global_guardrails, UsageContext

# Initialize safety guardrails (required)
guardrails = initialize_global_guardrails(enable_strict_mode=True)

# Initialize multi-category detection for a specific platform
index = MultiCategorySentinelIndex(
    platform=Platform.ROBLOX,  # or Platform.DISCORD, Platform.TWITTER, etc.
    model_base_path="./models"
)

# Auto-load all available trained models
loaded_models = index.auto_load_indices()
print(f"Loaded {loaded_models} detection models")

# Analyze content across multiple violation categories
user_messages = [
    "Hey, what's your real name and address?",
    "Want to play a private game just us two?",
    "Don't tell your parents about our conversations"
]

# Comprehensive violation detection
result = index.detect_violations(
    text_samples=user_messages,
    enable_keyword_detection=True,
    enable_pattern_matching=True
)

# Review results
print(f"Overall Risk Score: {result.overall_risk_score:.3f}")
print(f"Highest Risk Category: {result.highest_risk_category}")
print(f"High Risk Violations: {result.num_high_risk_violations}")
print(f"Platform: {result.platform.value}")

# Examine specific violation categories
for violation_type, violation_result in result.violation_results.items():
    print(f"\n{violation_type.value}:")
    print(f"  Affinity Score: {violation_result.affinity_score:.3f}")
    print(f"  Risk Level: {violation_result.risk_level}")
    print(f"  Confidence: {violation_result.confidence:.3f}")
    if violation_result.triggered_keywords:
        print(f"  Keywords: {violation_result.triggered_keywords}")
```

### Legacy Single-Category Detection

```python
from sentinel.sentinel_local_index import SentinelLocalIndex

# Load a previously saved index from a local path
index = SentinelLocalIndex.load(path="path/to/local/index")

# Calculate rare class affinity across all observations
result = index.calculate_rare_class_affinity(user_recent_messages)

# Get the overall score (uses skewness by default)
overall_score = result.rare_class_affinity_score
print(f"Overall rare class affinity score: {overall_score:.4f}")
```

## 🚀 Enhanced Features

### Comprehensive Training Data Builder

```python
from sentinel.training_data_builder import TrainingDataBuilder
from pathlib import Path

# Build detection models for all violation categories
builder = TrainingDataBuilder(
    training_data_dir=Path("./training_data"),
    model_output_dir=Path("./models")
)

# Validate training data
validation_results = builder.validate_training_data()
print(f"Valid training files: {len(validation_results['valid_files'])}")

# Build all models
results = builder.build_all_models()
for model_name, success in results.items():
    print(f"{model_name}: {'✓' if success else '✗'}")
```

### Validation Framework

```python
from sentinel.validation_framework import ValidationFramework

# Test model accuracy and performance
validator = ValidationFramework(
    test_data_dir=Path("./test_data"),
    models_dir=Path("./models"),
    results_dir=Path("./validation_results")
)

# Validate platform-specific performance
report = validator.validate_platform(Platform.ROBLOX)
print(f"Accuracy: {report.metrics.accuracy:.3f}")
print(f"F1 Score: {report.metrics.f1_score:.3f}")
print(f"Processing Time: {report.metrics.avg_processing_time:.3f}s")
```

### Safety Guardrails and Monitoring

```python
from sentinel.safety_guardrails import SafetyGuardrails, UsageContext

# Initialize comprehensive safety monitoring
guardrails = SafetyGuardrails(
    audit_log_dir=Path("./audit_logs"),
    enable_strict_mode=True
)

# Check access permissions
permitted, reason = guardrails.check_access_permission(
    user_id="researcher_123",
    usage_context=UsageContext.ACADEMIC_RESEARCH,
    component="multi_category_detection",
    request_details={"batch_size": 50, "purpose": "safety research"}
)

# Export compliance report
compliance_report = guardrails.export_compliance_report()
```

### Platform-Specific Configurations

```python
from sentinel.platform_config import get_recommended_config, create_custom_config

# Get optimized configuration for a platform
roblox_config = get_recommended_config(Platform.ROBLOX)
print(f"Child safety threshold: {roblox_config.get_violation_threshold(ViolationType.CHILD_SAFETY, 'high')}")

# Create custom configuration
custom_config = create_custom_config(
    base_platform=Platform.DISCORD,
    custom_name="Gaming Community",
    thresholds=DetectionThresholds(high_risk_threshold=0.4),
    enabled_violations={ViolationType.GAME_TOXICITY, ViolationType.HARASSMENT}
)
```

## 📊 Setup and Training

### Quick Setup

```bash
# Run the enhanced setup script
python setup_enhanced_sentinel.py

# This will create:
# - Training data directories and samples
# - Platform-specific configurations  
# - Safety policy templates
# - Validation test cases
```

### Building Detection Models

```python
# After adding your training data to ./training_data/
from sentinel.training_data_builder import TrainingDataBuilder

builder = TrainingDataBuilder(
    training_data_dir=Path("./training_data"),
    model_output_dir=Path("./models")
)

# Build models for specific violation types
builder.build_category_model(ViolationType.CHILD_SAFETY)
builder.build_category_model(ViolationType.HARASSMENT)

# Or build all available models
builder.build_all_models()
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
