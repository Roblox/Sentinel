#!/usr/bin/env python3
"""
Enhanced Sentinel Demo: Multi-Category Platform Safety Detection

This script demonstrates the enhanced Sentinel capabilities including:
- Multi-category violation detection
- Platform-specific configurations
- Comprehensive training data
- Safety guardrails and usage restrictions
- Validation framework

USAGE RESTRICTIONS:
This demo is for DEFENSIVE SECURITY PURPOSES ONLY. It demonstrates
how to detect and prevent harmful content, not how to create it.
"""

import sys
import json
from pathlib import Path
from typing import List, Dict

# Add the src directory to the path to import sentinel modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sentinel.multi_category_index import MultiCategorySentinelIndex
from sentinel.violation_taxonomy import ViolationType, Platform
from sentinel.platform_config import get_recommended_config
from sentinel.training_data_builder import TrainingDataBuilder
from sentinel.validation_framework import ValidationFramework
from sentinel.safety_guardrails import (
    SafetyGuardrails, UsageContext, require_authorization, initialize_global_guardrails
)


def setup_safety_system():
    """Initialize the safety guardrails system."""
    print("🔒 Initializing Safety Guardrails...")
    
    # Initialize global safety system
    guardrails = initialize_global_guardrails(
        audit_log_dir=Path("./audit_logs"),
        enable_strict_mode=True
    )
    
    print("✅ Safety system initialized with strict mode enabled")
    return guardrails


@require_authorization(
    usage_context=UsageContext.EDUCATIONAL_TRAINING,
    component="enhanced_demo"
)
def demonstrate_multi_category_detection(user_id: str = "demo_user"):
    """Demonstrate multi-category detection capabilities."""
    
    print("\n🔍 Multi-Category Detection Demo")
    print("=" * 50)
    
    # Initialize for different platforms
    platforms_to_test = [Platform.ROBLOX, Platform.DISCORD, Platform.GENERAL]
    
    for platform in platforms_to_test:
        print(f"\n📱 Testing {platform.value.upper()} Platform")
        print("-" * 30)
        
        # Get platform configuration
        config = get_recommended_config(platform)
        print(f"Primary age group: {config.characteristics.primary_age_group}")
        print(f"Moderation level: {config.characteristics.content_moderation_level}")
        print(f"Enabled violations: {len(config.enabled_violations)}")
        
        # Initialize multi-category index
        index = MultiCategorySentinelIndex(
            platform=platform,
            config=config,
            model_base_path=Path("./models")
        )
        
        # Try to load models (may not exist in demo)
        loaded_count = index.auto_load_indices()
        print(f"Loaded models: {loaded_count}")
        
        # Test with sample content
        test_messages = get_test_messages_for_platform(platform)
        
        if loaded_count > 0:
            print("\nAnalyzing test messages...")
            for message in test_messages[:3]:  # Limit to first 3 for demo
                result = index.detect_violations([message])
                print(f"  Message: '{message[:50]}...'")
                print(f"  Risk Score: {result.overall_risk_score:.3f}")
                print(f"  Risk Category: {result.highest_risk_category}")
                print(f"  High Risk Violations: {result.num_high_risk_violations}")
        else:
            print("⚠️  No trained models available for demonstration")
            print("   To see full functionality, run the training data builder first")


def get_test_messages_for_platform(platform: Platform) -> List[str]:
    """Get platform-appropriate test messages."""
    
    base_messages = [
        "Thanks for helping me with this game!",
        "That was a great match, well played everyone",
        "Can someone explain how this feature works?",
        "I love building in this game, it's so creative"
    ]
    
    if platform == Platform.ROBLOX:
        base_messages.extend([
            "What's your real name and age?",  # Child safety concern
            "Let's play a different game privately",  # Potential grooming
            "FREE ROBUX! Click this link now!"  # Scam attempt
        ])
    elif platform == Platform.DISCORD:
        base_messages.extend([
            "Everyone raid this server now!",  # Coordinated abuse
            "I found their address, let's dox them",  # Doxxing threat
            "Join my new crypto investment opportunity"  # Scam
        ])
    else:  # General
        base_messages.extend([
            "You're worthless and should delete yourself",  # Harassment
            "These people are inferior parasites",  # Hate speech
            "Click here for amazing weight loss results!"  # Spam
        ])
    
    return base_messages


def demonstrate_training_system():
    """Demonstrate the training data building system."""
    
    print("\n🏗️  Training System Demo")
    print("=" * 50)
    
    training_data_dir = Path("./training_data")
    models_dir = Path("./models")
    
    # Initialize training data builder
    builder = TrainingDataBuilder(
        training_data_dir=training_data_dir,
        model_output_dir=models_dir
    )
    
    # Validate training data
    print("Validating training data...")
    validation_results = builder.validate_training_data()
    
    print(f"Valid files: {len(validation_results['valid_files'])}")
    print(f"Invalid files: {len(validation_results['invalid_files'])}")
    print(f"Total positive examples: {validation_results['summary']['total_positive_examples']}")
    print(f"Total negative examples: {validation_results['summary']['total_negative_examples']}")
    
    if validation_results['warnings']:
        print("\nWarnings:")
        for warning in validation_results['warnings'][:5]:  # Show first 5 warnings
            print(f"  ⚠️  {warning}")
    
    # Show available violation categories
    print(f"\nAvailable violation categories:")
    for violation_type, stats in validation_results['statistics'].items():
        if isinstance(stats, dict) and 'positive_examples' in stats:
            print(f"  • {violation_type}: {stats['positive_examples']} examples (severity: {stats.get('severity', 'unknown')})")


def demonstrate_validation_framework():
    """Demonstrate the validation framework."""
    
    print("\n🧪 Validation Framework Demo")
    print("=" * 50)
    
    test_data_dir = Path("./test_data")
    models_dir = Path("./models")
    results_dir = Path("./validation_results")
    
    # Initialize validation framework
    validator = ValidationFramework(
        test_data_dir=test_data_dir,
        models_dir=models_dir,
        results_dir=results_dir
    )
    
    # Create sample test cases for demo
    for platform in [Platform.ROBLOX, Platform.DISCORD]:
        test_file = validator.create_sample_test_cases(platform)
        print(f"Created sample test cases: {test_file}")
    
    print("\nSample test case structure:")
    print("  • Positive interactions (expected: no violation)")
    print("  • Clear violations (expected: high risk)")
    print("  • Personal info requests (platform-specific handling)")
    print("  • Scam attempts (expected: medium risk)")
    
    # Show what validation would include
    print("\nValidation framework capabilities:")
    print("  • Accuracy metrics (precision, recall, F1-score)")
    print("  • Platform-specific thresholds")
    print("  • Category-specific performance analysis")
    print("  • Processing time benchmarks")
    print("  • Threshold optimization suggestions")
    print("  • Compliance reporting")


def demonstrate_safety_features():
    """Demonstrate safety guardrails and monitoring."""
    
    print("\n🛡️  Safety Features Demo")
    print("=" * 50)
    
    # Get global guardrails
    from sentinel.safety_guardrails import get_global_guardrails
    guardrails = get_global_guardrails()
    
    if not guardrails:
        print("⚠️  Safety guardrails not initialized")
        return
    
    # Show usage statistics
    stats = guardrails.get_usage_statistics()
    print("Current usage statistics:")
    for key, value in stats.items():
        print(f"  • {key}: {value}")
    
    # Show safety policies
    print(f"\nConfigured safety policies: {len(guardrails.policies)}")
    for policy_name, policy in guardrails.policies.items():
        print(f"  • {policy_name}:")
        print(f"    - Access level: {policy.access_level.value}")
        print(f"    - Max daily requests: {policy.max_daily_requests}")
        print(f"    - Audit required: {policy.audit_required}")
    
    # Export compliance report
    compliance_report = guardrails.export_compliance_report()
    print("\nCompliance status:")
    for key, value in compliance_report["safety_measures"].items():
        print(f"  • {key}: {value}")


def show_violation_taxonomy():
    """Show the comprehensive violation taxonomy."""
    
    print("\n📋 Violation Taxonomy")
    print("=" * 50)
    
    from sentinel.violation_taxonomy import VIOLATION_TAXONOMY, PLATFORM_VIOLATIONS
    
    print(f"Total violation types: {len(VIOLATION_TAXONOMY)}")
    print("\nViolation categories by severity:")
    
    # Group by severity
    by_severity = {}
    for violation_type, category in VIOLATION_TAXONOMY.items():
        severity = category.severity
        if severity not in by_severity:
            by_severity[severity] = []
        by_severity[severity].append((violation_type, category))
    
    for severity in sorted(by_severity.keys(), reverse=True):
        print(f"\n  Severity {severity}:")
        for violation_type, category in by_severity[severity]:
            print(f"    • {category.name}")
            print(f"      - Platforms: {[p.value for p in category.platforms]}")
            print(f"      - Detection patterns: {len(category.detection_patterns)}")
    
    print(f"\nPlatform-specific configurations:")
    for platform, violations in PLATFORM_VIOLATIONS.items():
        print(f"  • {platform.value}: {len(violations)} violation types enabled")


def main():
    """Main demonstration function."""
    
    print("🚀 Enhanced Sentinel Multi-Category Safety Detection Demo")
    print("=" * 60)
    print()
    print("This demo showcases Sentinel's enhanced capabilities for")
    print("comprehensive platform safety across multiple violation types.")
    print()
    
    try:
        # Initialize safety system
        guardrails = setup_safety_system()
        
        # Run demonstrations
        show_violation_taxonomy()
        demonstrate_training_system()
        demonstrate_validation_framework()
        demonstrate_safety_features()
        demonstrate_multi_category_detection()
        
        print("\n" + "=" * 60)
        print("🎉 Demo completed successfully!")
        print()
        print("Next steps:")
        print("1. Run the training data builder to create detection models")
        print("2. Use the validation framework to test model accuracy")
        print("3. Deploy with platform-specific configurations")
        print("4. Monitor usage through safety guardrails")
        print()
        print("Remember: This system is for DEFENSIVE SECURITY ONLY")
        
    except PermissionError as e:
        print(f"\n❌ Access denied: {e}")
        print("This demonstrates the safety guardrails in action.")
        
    except Exception as e:
        print(f"\n💥 Error during demo: {e}")
        print("This may be due to missing training data or models.")
        print("Run the setup scripts first to create the necessary files.")


if __name__ == "__main__":
    main()