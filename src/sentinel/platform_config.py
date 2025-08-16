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

"""
Platform-specific configuration system for tailored content safety detection.

This module provides configuration classes and presets for different platforms,
allowing fine-tuned detection parameters based on platform characteristics,
user demographics, and content policies.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any
from enum import Enum
import json
from pathlib import Path

from .violation_taxonomy import Platform, ViolationType


@dataclass
class DetectionThresholds:
    """Threshold configurations for different severity levels."""
    
    # Individual observation score thresholds
    high_risk_threshold: float = 0.5      # Immediate action required
    medium_risk_threshold: float = 0.25   # Enhanced monitoring
    low_risk_threshold: float = 0.1       # Flagged for review
    
    # Aggregated affinity score thresholds
    episode_high_risk: float = 0.3        # Strong pattern detected
    episode_medium_risk: float = 0.15     # Moderate pattern
    episode_low_risk: float = 0.05        # Weak pattern
    
    # Confidence requirements
    min_confidence_high: float = 0.8      # High confidence required for high-risk actions
    min_confidence_medium: float = 0.6    # Medium confidence for medium-risk actions


@dataclass
class PlatformCharacteristics:
    """Platform-specific characteristics that influence detection behavior."""
    
    primary_age_group: str                # "children", "teens", "adults", "mixed"
    communication_style: str              # "public", "private", "group", "mixed"
    content_moderation_level: str         # "strict", "moderate", "permissive"
    real_time_requirements: bool          # True if real-time detection needed
    has_built_in_filters: bool           # True if platform has existing filters
    supports_multimedia: bool            # True if platform supports images/videos
    max_message_length: Optional[int]     # Maximum message length, None if unlimited


@dataclass 
class PlatformConfig:
    """Complete configuration for a specific platform."""
    
    platform: Platform
    platform_name: str
    characteristics: PlatformCharacteristics
    thresholds: DetectionThresholds
    enabled_violations: Set[ViolationType]
    disabled_violations: Set[ViolationType] = field(default_factory=set)
    
    # Scoring parameters
    top_k_neighbors: int = 5              # Number of neighbors for similarity matching
    min_observations: int = 3             # Minimum observations needed for pattern detection
    max_observations: int = 100           # Maximum observations to consider
    
    # Platform-specific adjustments
    scoring_adjustments: Dict[ViolationType, float] = field(default_factory=dict)
    keyword_boosters: Dict[str, float] = field(default_factory=dict)
    
    # Response configurations
    response_actions: Dict[str, Any] = field(default_factory=dict)
    escalation_rules: Dict[str, Any] = field(default_factory=dict)
    
    def get_violation_threshold(self, violation_type: ViolationType, risk_level: str) -> float:
        """Get threshold for specific violation type and risk level."""
        base_threshold = getattr(self.thresholds, f"{risk_level}_risk_threshold")
        adjustment = self.scoring_adjustments.get(violation_type, 1.0)
        return base_threshold * adjustment
    
    def is_violation_enabled(self, violation_type: ViolationType) -> bool:
        """Check if a violation type is enabled for this platform."""
        return (violation_type in self.enabled_violations and 
                violation_type not in self.disabled_violations)


# Pre-configured platform settings
PLATFORM_CONFIGS: Dict[Platform, PlatformConfig] = {
    
    Platform.ROBLOX: PlatformConfig(
        platform=Platform.ROBLOX,
        platform_name="Roblox",
        characteristics=PlatformCharacteristics(
            primary_age_group="children",
            communication_style="group", 
            content_moderation_level="strict",
            real_time_requirements=True,
            has_built_in_filters=True,
            supports_multimedia=False,
            max_message_length=200
        ),
        thresholds=DetectionThresholds(
            high_risk_threshold=0.3,      # Lower threshold for child safety
            medium_risk_threshold=0.15,
            low_risk_threshold=0.05,
            episode_high_risk=0.2,
            episode_medium_risk=0.1,
            episode_low_risk=0.03,
            min_confidence_high=0.9,      # Higher confidence required
            min_confidence_medium=0.7
        ),
        enabled_violations={
            ViolationType.CHILD_SAFETY, ViolationType.GROOMING, ViolationType.HARASSMENT,
            ViolationType.CYBERBULLYING, ViolationType.FILTER_BYPASS, ViolationType.GAME_TOXICITY,
            ViolationType.SEXUAL_CONTENT, ViolationType.SCAMS
        },
        scoring_adjustments={
            ViolationType.CHILD_SAFETY: 0.7,     # Even more sensitive
            ViolationType.GROOMING: 0.6,
            ViolationType.SEXUAL_CONTENT: 0.5,
            ViolationType.FILTER_BYPASS: 0.8
        },
        keyword_boosters={
            "age": 2.0, "parent": 2.0, "school": 1.5, "secret": 2.5,
            "private": 1.8, "real name": 2.0, "address": 2.5
        },
        response_actions={
            "high_risk": ["immediate_block", "admin_alert", "log_incident"],
            "medium_risk": ["enhanced_monitoring", "temporary_restriction"],
            "low_risk": ["flag_for_review", "log_pattern"]
        }
    ),
    
    Platform.DISCORD: PlatformConfig(
        platform=Platform.DISCORD,
        platform_name="Discord",
        characteristics=PlatformCharacteristics(
            primary_age_group="mixed",
            communication_style="mixed",
            content_moderation_level="moderate", 
            real_time_requirements=True,
            has_built_in_filters=True,
            supports_multimedia=True,
            max_message_length=2000
        ),
        thresholds=DetectionThresholds(
            high_risk_threshold=0.5,
            medium_risk_threshold=0.25,
            low_risk_threshold=0.1,
            episode_high_risk=0.3,
            episode_medium_risk=0.15,
            episode_low_risk=0.05
        ),
        enabled_violations={
            ViolationType.HARASSMENT, ViolationType.HATE_SPEECH, ViolationType.DOXXING,
            ViolationType.CHILD_SAFETY, ViolationType.GROOMING, ViolationType.SCAMS,
            ViolationType.COORDINATED_ABUSE, ViolationType.GAME_TOXICITY, ViolationType.FILTER_BYPASS
        },
        scoring_adjustments={
            ViolationType.DOXXING: 0.6,           # Very sensitive to doxxing
            ViolationType.COORDINATED_ABUSE: 0.7,
            ViolationType.HARASSMENT: 0.8
        },
        keyword_boosters={
            "raid": 2.0, "brigade": 2.0, "doxx": 2.5, "address": 2.0,
            "phone": 2.0, "swat": 3.0, "coordinate": 1.5
        }
    ),
    
    Platform.TWITTER: PlatformConfig(
        platform=Platform.TWITTER,
        platform_name="Twitter/X",
        characteristics=PlatformCharacteristics(
            primary_age_group="adults",
            communication_style="public",
            content_moderation_level="moderate",
            real_time_requirements=True,
            has_built_in_filters=True,
            supports_multimedia=True,
            max_message_length=280
        ),
        thresholds=DetectionThresholds(),  # Use defaults
        enabled_violations={
            ViolationType.HATE_SPEECH, ViolationType.HARASSMENT, ViolationType.DOXXING,
            ViolationType.COORDINATED_ABUSE, ViolationType.SCAMS, ViolationType.VIOLENCE_THREATS,
            ViolationType.IMPERSONATION, ViolationType.SPAM
        },
        scoring_adjustments={
            ViolationType.HATE_SPEECH: 0.8,
            ViolationType.VIOLENCE_THREATS: 0.6,
            ViolationType.IMPERSONATION: 0.7
        },
        keyword_boosters={
            "kill": 2.0, "bomb": 2.5, "shoot": 2.0, "attack": 1.8,
            "genocide": 2.5, "ethnic cleansing": 3.0
        }
    ),
    
    Platform.REDDIT: PlatformConfig(
        platform=Platform.REDDIT,
        platform_name="Reddit",
        characteristics=PlatformCharacteristics(
            primary_age_group="mixed",
            communication_style="public",
            content_moderation_level="moderate",
            real_time_requirements=False,
            has_built_in_filters=True,
            supports_multimedia=True,
            max_message_length=None
        ),
        thresholds=DetectionThresholds(
            high_risk_threshold=0.6,      # Slightly higher threshold for longer content
            medium_risk_threshold=0.3,
            low_risk_threshold=0.15
        ),
        enabled_violations={
            ViolationType.HARASSMENT, ViolationType.HATE_SPEECH, ViolationType.COORDINATED_ABUSE,
            ViolationType.DOXXING, ViolationType.SPAM, ViolationType.BAN_EVASION
        },
        scoring_adjustments={
            ViolationType.COORDINATED_ABUSE: 0.7,  # Reddit-specific brigading issues
            ViolationType.BAN_EVASION: 0.8
        },
        keyword_boosters={
            "brigade": 2.0, "raid": 2.0, "vote manipulation": 2.0,
            "ban evasion": 2.0, "alt account": 1.5
        }
    ),
    
    Platform.GENERAL: PlatformConfig(
        platform=Platform.GENERAL,
        platform_name="General Platform",
        characteristics=PlatformCharacteristics(
            primary_age_group="mixed",
            communication_style="mixed",
            content_moderation_level="moderate",
            real_time_requirements=True,
            has_built_in_filters=False,
            supports_multimedia=True,
            max_message_length=None
        ),
        thresholds=DetectionThresholds(),  # Use defaults
        enabled_violations=set(ViolationType),  # All violations enabled
        scoring_adjustments={},  # No adjustments
        keyword_boosters={}
    )
}


class ConfigManager:
    """Manages platform configurations and allows custom configurations."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize the configuration manager.
        
        Args:
            config_dir: Directory containing custom configuration files.
        """
        self.config_dir = config_dir
        self.custom_configs: Dict[str, PlatformConfig] = {}
        
        if config_dir and config_dir.exists():
            self._load_custom_configs()
    
    def get_config(self, platform: Platform) -> PlatformConfig:
        """Get configuration for a platform."""
        return PLATFORM_CONFIGS.get(platform, PLATFORM_CONFIGS[Platform.GENERAL])
    
    def get_custom_config(self, config_name: str) -> Optional[PlatformConfig]:
        """Get a custom configuration by name."""
        return self.custom_configs.get(config_name)
    
    def save_custom_config(self, config_name: str, config: PlatformConfig):
        """Save a custom configuration."""
        self.custom_configs[config_name] = config
        
        if self.config_dir:
            self.config_dir.mkdir(exist_ok=True)
            config_file = self.config_dir / f"{config_name}.json"
            
            # Convert config to JSON-serializable format
            config_dict = self._config_to_dict(config)
            
            with open(config_file, 'w') as f:
                json.dump(config_dict, f, indent=2)
    
    def _load_custom_configs(self):
        """Load custom configurations from files."""
        for config_file in self.config_dir.glob("*.json"):
            try:
                with open(config_file) as f:
                    config_dict = json.load(f)
                
                config = self._dict_to_config(config_dict)
                config_name = config_file.stem
                self.custom_configs[config_name] = config
                
            except Exception as e:
                print(f"Error loading config {config_file}: {e}")
    
    def _config_to_dict(self, config: PlatformConfig) -> Dict:
        """Convert PlatformConfig to dictionary for JSON serialization."""
        return {
            "platform": config.platform.value,
            "platform_name": config.platform_name,
            "characteristics": {
                "primary_age_group": config.characteristics.primary_age_group,
                "communication_style": config.characteristics.communication_style,
                "content_moderation_level": config.characteristics.content_moderation_level,
                "real_time_requirements": config.characteristics.real_time_requirements,
                "has_built_in_filters": config.characteristics.has_built_in_filters,
                "supports_multimedia": config.characteristics.supports_multimedia,
                "max_message_length": config.characteristics.max_message_length
            },
            "thresholds": {
                "high_risk_threshold": config.thresholds.high_risk_threshold,
                "medium_risk_threshold": config.thresholds.medium_risk_threshold,
                "low_risk_threshold": config.thresholds.low_risk_threshold,
                "episode_high_risk": config.thresholds.episode_high_risk,
                "episode_medium_risk": config.thresholds.episode_medium_risk,
                "episode_low_risk": config.thresholds.episode_low_risk,
                "min_confidence_high": config.thresholds.min_confidence_high,
                "min_confidence_medium": config.thresholds.min_confidence_medium
            },
            "enabled_violations": [v.value for v in config.enabled_violations],
            "disabled_violations": [v.value for v in config.disabled_violations],
            "top_k_neighbors": config.top_k_neighbors,
            "min_observations": config.min_observations,
            "max_observations": config.max_observations,
            "scoring_adjustments": {v.value: adj for v, adj in config.scoring_adjustments.items()},
            "keyword_boosters": config.keyword_boosters,
            "response_actions": config.response_actions,
            "escalation_rules": config.escalation_rules
        }
    
    def _dict_to_config(self, config_dict: Dict) -> PlatformConfig:
        """Convert dictionary back to PlatformConfig."""
        return PlatformConfig(
            platform=Platform(config_dict["platform"]),
            platform_name=config_dict["platform_name"],
            characteristics=PlatformCharacteristics(**config_dict["characteristics"]),
            thresholds=DetectionThresholds(**config_dict["thresholds"]),
            enabled_violations={ViolationType(v) for v in config_dict["enabled_violations"]},
            disabled_violations={ViolationType(v) for v in config_dict.get("disabled_violations", [])},
            top_k_neighbors=config_dict.get("top_k_neighbors", 5),
            min_observations=config_dict.get("min_observations", 3),
            max_observations=config_dict.get("max_observations", 100),
            scoring_adjustments={ViolationType(v): adj for v, adj in config_dict.get("scoring_adjustments", {}).items()},
            keyword_boosters=config_dict.get("keyword_boosters", {}),
            response_actions=config_dict.get("response_actions", {}),
            escalation_rules=config_dict.get("escalation_rules", {})
        )


def create_custom_config(
    base_platform: Platform,
    custom_name: str,
    **overrides
) -> PlatformConfig:
    """Create a custom configuration based on an existing platform config.
    
    Args:
        base_platform: Platform to use as base configuration.
        custom_name: Name for the custom configuration.
        **overrides: Configuration values to override.
    
    Returns:
        Custom PlatformConfig instance.
    """
    base_config = PLATFORM_CONFIGS[base_platform]
    
    # Create a copy and apply overrides
    custom_config = PlatformConfig(
        platform=base_config.platform,
        platform_name=custom_name,
        characteristics=base_config.characteristics,
        thresholds=base_config.thresholds,
        enabled_violations=base_config.enabled_violations.copy(),
        disabled_violations=base_config.disabled_violations.copy(),
        top_k_neighbors=base_config.top_k_neighbors,
        min_observations=base_config.min_observations,
        max_observations=base_config.max_observations,
        scoring_adjustments=base_config.scoring_adjustments.copy(),
        keyword_boosters=base_config.keyword_boosters.copy(),
        response_actions=base_config.response_actions.copy(),
        escalation_rules=base_config.escalation_rules.copy()
    )
    
    # Apply overrides
    for key, value in overrides.items():
        if hasattr(custom_config, key):
            setattr(custom_config, key, value)
    
    return custom_config


def get_recommended_config(platform: Platform) -> PlatformConfig:
    """Get the recommended configuration for a platform."""
    return PLATFORM_CONFIGS.get(platform, PLATFORM_CONFIGS[Platform.GENERAL])