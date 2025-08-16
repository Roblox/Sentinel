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
Multi-category detection system for comprehensive platform safety.

This module extends Sentinel's capabilities to detect multiple types of violations
simultaneously, with platform-specific configurations and enhanced scoring.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .sentinel_local_index import SentinelLocalIndex
from .violation_taxonomy import ViolationType, Platform, VIOLATION_TAXONOMY
from .platform_config import PlatformConfig, ConfigManager
from .score_types import RareClassAffinityResult
from .score_formulae import skewness, calculate_contrastive_score

LOG = logging.getLogger(__name__)


@dataclass
class ViolationDetectionResult:
    """Result for a specific violation type detection."""
    
    violation_type: ViolationType
    affinity_score: float
    confidence: float
    risk_level: str  # "high", "medium", "low", "none"
    individual_scores: Dict[str, float]
    triggered_keywords: List[str] = field(default_factory=list)
    detection_patterns: List[str] = field(default_factory=list)


@dataclass
class MultiCategoryDetectionResult:
    """Comprehensive detection result across all violation categories."""
    
    overall_risk_score: float
    highest_risk_category: Optional[ViolationType]
    platform: Platform
    violation_results: Dict[ViolationType, ViolationDetectionResult]
    
    # Summary statistics
    num_high_risk_violations: int = 0
    num_medium_risk_violations: int = 0
    num_low_risk_violations: int = 0
    
    # Metadata
    num_observations: int = 0
    analysis_timestamp: Optional[str] = None
    platform_config_used: Optional[str] = None


class MultiCategorySentinelIndex:
    """Multi-category detection system with platform-specific configurations."""
    
    def __init__(
        self,
        platform: Platform = Platform.GENERAL,
        config: Optional[PlatformConfig] = None,
        config_manager: Optional[ConfigManager] = None,
        model_base_path: Optional[Path] = None
    ):
        """Initialize the multi-category detection system.
        
        Args:
            platform: Target platform for detection.
            config: Optional custom platform configuration.
            config_manager: Configuration manager for custom configs.
            model_base_path: Base path for loading trained models.
        """
        self.platform = platform
        self.config_manager = config_manager or ConfigManager()
        self.config = config or self.config_manager.get_config(platform)
        self.model_base_path = model_base_path or Path("./models")
        
        # Individual category indices
        self.category_indices: Dict[ViolationType, SentinelLocalIndex] = {}
        self.loaded_categories: Set[ViolationType] = set()
        
        # Overall safety index (general violations)
        self.general_index: Optional[SentinelLocalIndex] = None
        
        LOG.info(f"Initialized MultiCategorySentinelIndex for {platform.value}")
    
    def load_category_index(self, violation_type: ViolationType, index_path: Path) -> bool:
        """Load a trained index for a specific violation category.
        
        Args:
            violation_type: Type of violation to load.
            index_path: Path to the saved index.
            
        Returns:
            True if successfully loaded, False otherwise.
        """
        try:
            if not self.config.is_violation_enabled(violation_type):
                LOG.info(f"Violation type {violation_type.value} is disabled for platform {self.platform.value}")
                return False
            
            index = SentinelLocalIndex.load(path=str(index_path))
            self.category_indices[violation_type] = index
            self.loaded_categories.add(violation_type)
            
            LOG.info(f"Loaded index for {violation_type.value} from {index_path}")
            return True
            
        except Exception as e:
            LOG.error(f"Failed to load index for {violation_type.value}: {e}")
            return False
    
    def load_general_index(self, index_path: Path) -> bool:
        """Load the general safety index for overall violation detection.
        
        Args:
            index_path: Path to the general safety index.
            
        Returns:
            True if successfully loaded, False otherwise.
        """
        try:
            self.general_index = SentinelLocalIndex.load(path=str(index_path))
            LOG.info(f"Loaded general safety index from {index_path}")
            return True
            
        except Exception as e:
            LOG.error(f"Failed to load general index: {e}")
            return False
    
    def auto_load_indices(self) -> int:
        """Automatically load all available indices for enabled violation types.
        
        Returns:
            Number of successfully loaded indices.
        """
        loaded_count = 0
        
        # Load general index if available
        general_path = self.model_base_path / "general_safety_index"
        if general_path.exists():
            if self.load_general_index(general_path):
                loaded_count += 1
        
        # Load category-specific indices
        for violation_type in self.config.enabled_violations:
            index_path = self.model_base_path / f"{violation_type.value}_index"
            if index_path.exists():
                if self.load_category_index(violation_type, index_path):
                    loaded_count += 1
        
        LOG.info(f"Auto-loaded {loaded_count} indices")
        return loaded_count
    
    def detect_violations(
        self,
        text_samples: List[str],
        enable_keyword_detection: bool = True,
        enable_pattern_matching: bool = True,
        min_confidence: float = 0.5
    ) -> MultiCategoryDetectionResult:
        """Detect violations across all enabled categories.
        
        Args:
            text_samples: List of text strings to analyze.
            enable_keyword_detection: Whether to use keyword boosting.
            enable_pattern_matching: Whether to apply pattern matching.
            min_confidence: Minimum confidence threshold for detections.
            
        Returns:
            Comprehensive detection results across all categories.
        """
        if not text_samples:
            return self._create_empty_result()
        
        # Limit observations based on platform config
        if len(text_samples) > self.config.max_observations:
            text_samples = text_samples[-self.config.max_observations:]
        
        violation_results = {}
        highest_risk_score = 0.0
        highest_risk_category = None
        
        # Analyze each enabled violation category
        for violation_type in self.config.enabled_violations:
            if violation_type not in self.loaded_categories:
                continue
            
            result = self._analyze_category(
                violation_type, 
                text_samples,
                enable_keyword_detection,
                enable_pattern_matching
            )
            
            violation_results[violation_type] = result
            
            # Track highest risk category
            if result.affinity_score > highest_risk_score:
                highest_risk_score = result.affinity_score
                highest_risk_category = violation_type
        
        # Use general index if no specific categories were analyzed
        if not violation_results and self.general_index:
            general_result = self._analyze_general_safety(text_samples)
            if general_result:
                violation_results[ViolationType.GENERAL_HATE] = general_result
                highest_risk_score = general_result.affinity_score
                highest_risk_category = ViolationType.GENERAL_HATE
        
        # Calculate summary statistics
        high_risk_count = sum(1 for r in violation_results.values() if r.risk_level == "high")
        medium_risk_count = sum(1 for r in violation_results.values() if r.risk_level == "medium")
        low_risk_count = sum(1 for r in violation_results.values() if r.risk_level == "low")
        
        # Calculate overall risk score (weighted combination)
        overall_risk = self._calculate_overall_risk(violation_results)
        
        return MultiCategoryDetectionResult(
            overall_risk_score=overall_risk,
            highest_risk_category=highest_risk_category,
            platform=self.platform,
            violation_results=violation_results,
            num_high_risk_violations=high_risk_count,
            num_medium_risk_violations=medium_risk_count,
            num_low_risk_violations=low_risk_count,
            num_observations=len(text_samples),
            platform_config_used=self.config.platform_name
        )
    
    def _analyze_category(
        self,
        violation_type: ViolationType,
        text_samples: List[str],
        enable_keyword_detection: bool,
        enable_pattern_matching: bool
    ) -> ViolationDetectionResult:
        """Analyze a specific violation category."""
        
        index = self.category_indices[violation_type]
        
        # Get base affinity result
        affinity_result = index.calculate_rare_class_affinity(
            text_samples=text_samples,
            top_k=self.config.top_k_neighbors,
            min_score_to_consider=self.config.get_violation_threshold(violation_type, "low")
        )
        
        # Apply scoring adjustments
        adjusted_score = self._apply_scoring_adjustments(
            affinity_result.rare_class_affinity_score,
            violation_type
        )
        
        # Keyword and pattern detection
        triggered_keywords = []
        detection_patterns = []
        
        if enable_keyword_detection:
            triggered_keywords = self._detect_keywords(text_samples, violation_type)
            if triggered_keywords:
                # Boost score based on keyword matches
                keyword_boost = self._calculate_keyword_boost(triggered_keywords, violation_type)
                adjusted_score = min(1.0, adjusted_score * (1.0 + keyword_boost))
        
        if enable_pattern_matching:
            detection_patterns = self._detect_patterns(text_samples, violation_type)
        
        # Determine risk level and confidence
        risk_level = self._determine_risk_level(adjusted_score, violation_type)
        confidence = self._calculate_confidence(affinity_result, triggered_keywords, detection_patterns)
        
        return ViolationDetectionResult(
            violation_type=violation_type,
            affinity_score=adjusted_score,
            confidence=confidence,
            risk_level=risk_level,
            individual_scores=affinity_result.observation_scores,
            triggered_keywords=triggered_keywords,
            detection_patterns=detection_patterns
        )
    
    def _analyze_general_safety(self, text_samples: List[str]) -> Optional[ViolationDetectionResult]:
        """Analyze using the general safety index."""
        if not self.general_index:
            return None
        
        try:
            affinity_result = self.general_index.calculate_rare_class_affinity(
                text_samples=text_samples,
                top_k=self.config.top_k_neighbors,
                min_score_to_consider=self.config.thresholds.low_risk_threshold
            )
            
            risk_level = self._determine_risk_level(affinity_result.rare_class_affinity_score, None)
            confidence = min(0.8, affinity_result.rare_class_affinity_score + 0.2)
            
            return ViolationDetectionResult(
                violation_type=ViolationType.GENERAL_HATE,
                affinity_score=affinity_result.rare_class_affinity_score,
                confidence=confidence,
                risk_level=risk_level,
                individual_scores=affinity_result.observation_scores
            )
            
        except Exception as e:
            LOG.error(f"Error in general safety analysis: {e}")
            return None
    
    def _apply_scoring_adjustments(self, base_score: float, violation_type: ViolationType) -> float:
        """Apply platform-specific scoring adjustments."""
        adjustment = self.config.scoring_adjustments.get(violation_type, 1.0)
        return min(1.0, base_score * adjustment)
    
    def _detect_keywords(self, text_samples: List[str], violation_type: ViolationType) -> List[str]:
        """Detect violation-specific keywords in text samples."""
        triggered_keywords = []
        
        # Get violation category info
        category_info = VIOLATION_TAXONOMY.get(violation_type)
        if not category_info:
            return triggered_keywords
        
        # Check for category keywords
        text_lower = " ".join(text_samples).lower()
        for keyword in category_info.keywords:
            if keyword.lower() in text_lower:
                triggered_keywords.append(keyword)
        
        # Check for platform-specific keyword boosters
        for keyword in self.config.keyword_boosters:
            if keyword.lower() in text_lower:
                triggered_keywords.append(keyword)
        
        return list(set(triggered_keywords))
    
    def _detect_patterns(self, text_samples: List[str], violation_type: ViolationType) -> List[str]:
        """Detect violation-specific patterns in text samples."""
        detected_patterns = []
        
        category_info = VIOLATION_TAXONOMY.get(violation_type)
        if not category_info:
            return detected_patterns
        
        text_combined = " ".join(text_samples).lower()
        
        # Check for known detection patterns
        for pattern in category_info.detection_patterns:
            # Simple pattern matching - could be enhanced with regex
            if pattern.lower().replace("_", " ") in text_combined:
                detected_patterns.append(pattern)
        
        return detected_patterns
    
    def _calculate_keyword_boost(self, triggered_keywords: List[str], violation_type: ViolationType) -> float:
        """Calculate keyword-based score boost."""
        boost = 0.0
        
        for keyword in triggered_keywords:
            # Platform-specific boosts
            platform_boost = self.config.keyword_boosters.get(keyword, 0.0)
            boost += platform_boost
        
        return min(0.5, boost)  # Cap boost at 50%
    
    def _determine_risk_level(self, score: float, violation_type: Optional[ViolationType]) -> str:
        """Determine risk level based on score and thresholds."""
        if violation_type:
            high_threshold = self.config.get_violation_threshold(violation_type, "high")
            medium_threshold = self.config.get_violation_threshold(violation_type, "medium")
            low_threshold = self.config.get_violation_threshold(violation_type, "low")
        else:
            high_threshold = self.config.thresholds.high_risk_threshold
            medium_threshold = self.config.thresholds.medium_risk_threshold
            low_threshold = self.config.thresholds.low_risk_threshold
        
        if score >= high_threshold:
            return "high"
        elif score >= medium_threshold:
            return "medium"
        elif score >= low_threshold:
            return "low"
        else:
            return "none"
    
    def _calculate_confidence(
        self,
        affinity_result: RareClassAffinityResult,
        keywords: List[str],
        patterns: List[str]
    ) -> float:
        """Calculate confidence based on multiple factors."""
        base_confidence = min(0.8, affinity_result.rare_class_affinity_score + 0.2)
        
        # Boost confidence based on supporting evidence
        keyword_boost = min(0.1, len(keywords) * 0.02)
        pattern_boost = min(0.1, len(patterns) * 0.03)
        
        # Factor in number of observations
        observation_boost = min(0.1, len(affinity_result.observation_scores) / 50.0)
        
        total_confidence = min(1.0, base_confidence + keyword_boost + pattern_boost + observation_boost)
        return total_confidence
    
    def _calculate_overall_risk(self, violation_results: Dict[ViolationType, ViolationDetectionResult]) -> float:
        """Calculate overall risk score from individual violation scores."""
        if not violation_results:
            return 0.0
        
        # Weight high-severity violations more heavily
        weighted_score = 0.0
        total_weight = 0.0
        
        for violation_type, result in violation_results.items():
            category_info = VIOLATION_TAXONOMY.get(violation_type)
            severity_weight = category_info.severity if category_info else 3
            
            weighted_score += result.affinity_score * result.confidence * severity_weight
            total_weight += severity_weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _create_empty_result(self) -> MultiCategoryDetectionResult:
        """Create an empty result for when no text is provided."""
        return MultiCategoryDetectionResult(
            overall_risk_score=0.0,
            highest_risk_category=None,
            platform=self.platform,
            violation_results={},
            num_observations=0,
            platform_config_used=self.config.platform_name
        )
    
    def get_platform_summary(self) -> Dict[str, Any]:
        """Get summary information about the current platform configuration."""
        return {
            "platform": self.platform.value,
            "platform_name": self.config.platform_name,
            "enabled_violations": [v.value for v in self.config.enabled_violations],
            "loaded_categories": [v.value for v in self.loaded_categories],
            "has_general_index": self.general_index is not None,
            "characteristics": {
                "primary_age_group": self.config.characteristics.primary_age_group,
                "moderation_level": self.config.characteristics.content_moderation_level,
                "real_time_requirements": self.config.characteristics.real_time_requirements
            },
            "thresholds": {
                "high_risk": self.config.thresholds.high_risk_threshold,
                "medium_risk": self.config.thresholds.medium_risk_threshold,
                "low_risk": self.config.thresholds.low_risk_threshold
            }
        }