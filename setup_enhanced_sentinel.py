#!/usr/bin/env python3
"""
Enhanced Sentinel Setup Script

This script sets up the enhanced Sentinel system with comprehensive
violation detection capabilities across multiple platforms.

IMPORTANT: This system is for DEFENSIVE SECURITY PURPOSES ONLY.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def setup_directories():
    """Create necessary directories for the enhanced system."""
    
    directories = [
        "models",
        "training_data", 
        "test_data",
        "validation_results",
        "audit_logs",
        "config"
    ]
    
    print("📁 Setting up directories...")
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"  ✓ {directory}/")


def create_training_config():
    """Create training configuration file."""
    
    config = {
        "description": "Enhanced Sentinel Training Configuration",
        "model_settings": {
            "encoder_model": "all-MiniLM-L6-v2",
            "batch_size": 32,
            "max_sequence_length": 512,
            "negative_to_positive_ratio": 10
        },
        "violation_categories": {
            "child_safety": {
                "priority": "critical",
                "platforms": ["roblox", "discord", "general"],
                "min_examples": 100
            },
            "harassment": {
                "priority": "high", 
                "platforms": ["twitter", "discord", "reddit", "general"],
                "min_examples": 200
            },
            "hate_speech": {
                "priority": "high",
                "platforms": ["twitter", "reddit", "general"],
                "min_examples": 300
            },
            "gaming_toxicity": {
                "priority": "medium",
                "platforms": ["discord", "roblox", "twitch"],
                "min_examples": 150
            },
            "filter_bypass": {
                "priority": "medium",
                "platforms": ["roblox", "discord", "general"],
                "min_examples": 100
            },
            "platform_abuse": {
                "priority": "medium",
                "platforms": ["discord", "twitter", "instagram"],
                "min_examples": 200
            }
        },
        "platform_priorities": {
            "roblox": ["child_safety", "grooming", "filter_bypass"],
            "discord": ["harassment", "doxxing", "coordinated_abuse"],
            "twitter": ["hate_speech", "harassment", "violence_threats"],
            "reddit": ["harassment", "coordinated_abuse", "hate_speech"]
        }
    }
    
    config_file = Path("config/training_config.json")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"📝 Created training configuration: {config_file}")


def create_platform_policies():
    """Create platform-specific safety policies."""
    
    policies = {
        "roblox_strict": {
            "description": "Strict policy for child-focused platform",
            "thresholds": {
                "child_safety": 0.3,
                "grooming": 0.25,
                "harassment": 0.4,
                "general": 0.5
            },
            "response_actions": {
                "high_risk": ["immediate_block", "admin_alert", "incident_report"],
                "medium_risk": ["temporary_restriction", "enhanced_monitoring"],
                "low_risk": ["flag_for_review", "pattern_tracking"]
            },
            "rate_limits": {
                "requests_per_minute": 1000,
                "requests_per_hour": 10000,
                "max_batch_size": 100
            }
        },
        "general_moderate": {
            "description": "Moderate policy for general platforms",
            "thresholds": {
                "harassment": 0.5,
                "hate_speech": 0.6,
                "spam": 0.7,
                "general": 0.5
            },
            "response_actions": {
                "high_risk": ["content_warning", "admin_review"],
                "medium_risk": ["user_warning", "content_flag"],
                "low_risk": ["log_only"]
            },
            "rate_limits": {
                "requests_per_minute": 500,
                "requests_per_hour": 5000,
                "max_batch_size": 500
            }
        }
    }
    
    policies_file = Path("config/platform_policies.json")
    with open(policies_file, 'w') as f:
        json.dump(policies, f, indent=2)
    
    print(f"🛡️  Created platform policies: {policies_file}")


def create_sample_training_data():
    """Create sample training data files if they don't exist."""
    
    # Check if training data already exists
    training_files = [
        "child_safety_violations.json",
        "harassment_violations.json", 
        "hate_speech_violations.json",
        "gaming_toxicity_violations.json",
        "filter_bypass_violations.json",
        "platform_abuse_violations.json",
        "safe_content_examples.json"
    ]
    
    existing_files = []
    missing_files = []
    
    for filename in training_files:
        filepath = Path("training_data") / filename
        if filepath.exists():
            existing_files.append(filename)
        else:
            missing_files.append(filename)
    
    print(f"📊 Training data status:")
    print(f"  ✓ Existing files: {len(existing_files)}")
    print(f"  ⚠️  Missing files: {len(missing_files)}")
    
    if existing_files:
        print("  Found existing training data files:")
        for filename in existing_files:
            print(f"    • {filename}")
    
    if missing_files:
        print("  Missing training data files:")
        for filename in missing_files:
            print(f"    • {filename}")
        print("  Note: Run the training data creation scripts to generate these files")


def validate_system_requirements():
    """Validate system requirements and dependencies."""
    
    print("🔍 Validating system requirements...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version >= (3, 8):
        print(f"  ✓ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print(f"  ❌ Python {python_version.major}.{python_version.minor} (requires 3.8+)")
        return False
    
    # Check required packages
    required_packages = [
        "torch",
        "numpy", 
        "sentence-transformers",
        "transformers",
        "scikit-learn"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"  ✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ❌ {package} (missing)")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install torch sentence-transformers transformers scikit-learn")
        return False
    
    return True


def show_next_steps():
    """Show next steps for using the enhanced system."""
    
    print("\n" + "="*60)
    print("🎉 Enhanced Sentinel Setup Complete!")
    print("="*60)
    
    print("\n📋 Next Steps:")
    print()
    print("1. 🏗️  BUILD TRAINING DATA:")
    print("   python -m examples.enhanced_sentinel_demo")
    print("   (This will show you what training data is available)")
    print()
    print("2. 🤖 TRAIN DETECTION MODELS:")
    print("   from sentinel.training_data_builder import TrainingDataBuilder")
    print("   builder = TrainingDataBuilder(Path('training_data'), Path('models'))")
    print("   builder.build_all_models()")
    print()
    print("3. 🧪 VALIDATE MODEL PERFORMANCE:")
    print("   from sentinel.validation_framework import ValidationFramework")
    print("   validator = ValidationFramework(Path('test_data'), Path('models'), Path('results'))")
    print("   report = validator.validate_platform(Platform.ROBLOX)")
    print()
    print("4. 🚀 DEPLOY WITH SAFETY GUARDRAILS:")
    print("   from sentinel.multi_category_index import MultiCategorySentinelIndex")
    print("   from sentinel.safety_guardrails import initialize_global_guardrails")
    print("   guardrails = initialize_global_guardrails()")
    print("   index = MultiCategorySentinelIndex(Platform.ROBLOX)")
    print()
    print("🛡️  SAFETY REMINDER:")
    print("This system is for DEFENSIVE SECURITY PURPOSES ONLY")
    print("• Content safety and moderation")
    print("• Academic research on harmful content detection") 
    print("• Educational and training purposes")
    print()
    print("❌ PROHIBITED USES:")
    print("• Generating or amplifying harmful content")
    print("• Training offensive language models")
    print("• Creating attack tools or capabilities")
    print()
    print("📖 For more information, see:")
    print("• training_data/README.md - Training data guidelines")
    print("• examples/enhanced_sentinel_demo.py - Comprehensive demo")
    print("• src/sentinel/safety_guardrails.py - Safety documentation")


def main():
    """Main setup function."""
    
    print("🚀 Enhanced Sentinel Setup")
    print("=" * 40)
    print()
    print("Setting up comprehensive multi-platform safety detection...")
    print()
    
    # Validate requirements
    if not validate_system_requirements():
        print("\n❌ Setup failed: Missing requirements")
        return False
    
    # Setup directories and configuration
    setup_directories()
    create_training_config()
    create_platform_policies()
    create_sample_training_data()
    
    # Show completion message
    show_next_steps()
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)