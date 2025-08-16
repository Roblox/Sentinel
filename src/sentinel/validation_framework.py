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
Validation framework for testing Sentinel model accuracy and performance.

This module provides comprehensive testing tools to evaluate detection accuracy,
measure false positive/negative rates, and validate model performance across
different platforms and violation types.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
import numpy as np
from datetime import datetime
import random

from .multi_category_index import MultiCategorySentinelIndex, MultiCategoryDetectionResult
from .violation_taxonomy import ViolationType, Platform
from .platform_config import PlatformConfig

LOG = logging.getLogger(__name__)


@dataclass
class TestCase:
    """Individual test case for validation."""
    
    text: str
    expected_violation_type: Optional[ViolationType]
    expected_risk_level: str  # "high", "medium", "low", "none"
    platform: Platform
    description: str
    source: str = "manual"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationMetrics:
    """Metrics for a validation run."""
    
    # Basic accuracy metrics
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    
    # Risk level accuracy
    risk_level_accuracy: float = 0.0
    high_risk_precision: float = 0.0
    high_risk_recall: float = 0.0
    
    # Category-specific metrics
    category_accuracies: Dict[ViolationType, float] = field(default_factory=dict)
    
    # Performance metrics
    avg_processing_time: float = 0.0
    total_test_cases: int = 0
    
    @property
    def precision(self) -> float:
        """Calculate precision (true positives / (true positives + false positives))."""
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator > 0 else 0.0
    
    @property
    def recall(self) -> float:
        """Calculate recall (true positives / (true positives + false negatives))."""
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator > 0 else 0.0
    
    @property
    def f1_score(self) -> float:
        """Calculate F1 score (harmonic mean of precision and recall)."""
        p, r = self.precision, self.recall
        return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
    
    @property
    def accuracy(self) -> float:
        """Calculate overall accuracy."""
        total = self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
        return (self.true_positives + self.true_negatives) / total if total > 0 else 0.0


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    
    platform: Platform
    test_timestamp: str
    model_config: Dict[str, Any]
    metrics: ValidationMetrics
    
    # Detailed results
    failed_cases: List[Dict[str, Any]] = field(default_factory=list)
    edge_cases: List[Dict[str, Any]] = field(default_factory=list)
    performance_stats: Dict[str, Any] = field(default_factory=dict)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    threshold_suggestions: Dict[str, float] = field(default_factory=dict)


class ValidationFramework:
    """Framework for comprehensive model validation and testing."""
    
    def __init__(
        self,
        test_data_dir: Path,
        models_dir: Path,
        results_dir: Path
    ):
        """Initialize the validation framework.
        
        Args:
            test_data_dir: Directory containing test cases.
            models_dir: Directory containing trained models.
            results_dir: Directory to save validation results.
        """
        self.test_data_dir = test_data_dir
        self.models_dir = models_dir
        self.results_dir = results_dir
        
        # Ensure directories exist
        self.test_data_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        LOG.info("Initialized ValidationFramework")
    
    def load_test_cases(self, platform: Platform) -> List[TestCase]:
        """Load test cases for a specific platform.
        
        Args:
            platform: Platform to load test cases for.
            
        Returns:
            List of test cases.
        """
        test_cases = []
        
        # Load platform-specific test cases
        platform_test_file = self.test_data_dir / f"{platform.value}_test_cases.json"
        if platform_test_file.exists():
            test_cases.extend(self._load_test_file(platform_test_file, platform))
        
        # Load general test cases
        general_test_file = self.test_data_dir / "general_test_cases.json"
        if general_test_file.exists():
            test_cases.extend(self._load_test_file(general_test_file, platform))
        
        # Load edge cases
        edge_cases_file = self.test_data_dir / "edge_cases.json"
        if edge_cases_file.exists():
            test_cases.extend(self._load_test_file(edge_cases_file, platform))
        
        LOG.info(f"Loaded {len(test_cases)} test cases for {platform.value}")
        return test_cases
    
    def _load_test_file(self, file_path: Path, platform: Platform) -> List[TestCase]:
        """Load test cases from a JSON file."""
        test_cases = []
        
        try:
            with open(file_path) as f:
                data = json.load(f)
            
            for case_data in data.get("test_cases", []):
                # Parse violation type
                violation_type = None
                if case_data.get("expected_violation_type"):
                    try:
                        violation_type = ViolationType(case_data["expected_violation_type"])
                    except ValueError:
                        LOG.warning(f"Unknown violation type: {case_data['expected_violation_type']}")
                
                test_case = TestCase(
                    text=case_data["text"],
                    expected_violation_type=violation_type,
                    expected_risk_level=case_data.get("expected_risk_level", "none"),
                    platform=platform,
                    description=case_data.get("description", ""),
                    source=case_data.get("source", str(file_path)),
                    metadata=case_data.get("metadata", {})
                )
                test_cases.append(test_case)
                
        except Exception as e:
            LOG.error(f"Error loading test file {file_path}: {e}")
        
        return test_cases
    
    def validate_platform(self, platform: Platform, config: Optional[PlatformConfig] = None) -> ValidationReport:
        """Validate model performance for a specific platform.
        
        Args:
            platform: Platform to validate.
            config: Optional custom platform configuration.
            
        Returns:
            Comprehensive validation report.
        """
        LOG.info(f"Starting validation for platform: {platform.value}")
        
        # Initialize multi-category index
        index = MultiCategorySentinelIndex(
            platform=platform,
            config=config,
            model_base_path=self.models_dir
        )
        
        # Auto-load available models
        loaded_count = index.auto_load_indices()
        if loaded_count == 0:
            LOG.warning(f"No models loaded for platform {platform.value}")
        
        # Load test cases
        test_cases = self.load_test_cases(platform)
        if not test_cases:
            LOG.warning(f"No test cases found for platform {platform.value}")
            return self._create_empty_report(platform, index)
        
        # Run validation
        metrics = ValidationMetrics()
        failed_cases = []
        edge_cases = []
        processing_times = []
        
        for test_case in test_cases:
            start_time = datetime.now()
            
            # Run detection
            result = index.detect_violations([test_case.text])
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            processing_times.append(processing_time)
            
            # Evaluate result
            evaluation = self._evaluate_test_case(test_case, result)
            self._update_metrics(metrics, evaluation)
            
            # Track failed cases
            if not evaluation["correct"]:
                failed_cases.append({
                    "test_case": test_case,
                    "result": result,
                    "evaluation": evaluation,
                    "processing_time": processing_time
                })
            
            # Track edge cases (borderline results)
            if evaluation.get("is_edge_case", False):
                edge_cases.append({
                    "test_case": test_case,
                    "result": result,
                    "evaluation": evaluation
                })
        
        # Calculate summary metrics
        metrics.total_test_cases = len(test_cases)
        metrics.avg_processing_time = np.mean(processing_times) if processing_times else 0.0
        
        # Calculate category-specific accuracies
        metrics.category_accuracies = self._calculate_category_accuracies(test_cases, failed_cases)
        
        # Generate report
        report = ValidationReport(
            platform=platform,
            test_timestamp=datetime.now().isoformat(),
            model_config=index.get_platform_summary(),
            metrics=metrics,
            failed_cases=failed_cases[:50],  # Limit to first 50 failures
            edge_cases=edge_cases[:20],      # Limit to first 20 edge cases
            performance_stats={
                "avg_processing_time": metrics.avg_processing_time,
                "min_processing_time": min(processing_times) if processing_times else 0.0,
                "max_processing_time": max(processing_times) if processing_times else 0.0,
                "total_processing_time": sum(processing_times)
            }
        )
        
        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)
        report.threshold_suggestions = self._suggest_threshold_adjustments(failed_cases)
        
        # Save report
        self._save_report(report)
        
        LOG.info(f"Validation complete for {platform.value}. Accuracy: {metrics.accuracy:.3f}, F1: {metrics.f1_score:.3f}")
        return report
    
    def _evaluate_test_case(self, test_case: TestCase, result: MultiCategoryDetectionResult) -> Dict[str, Any]:
        """Evaluate a single test case result."""
        evaluation = {
            "correct": False,
            "predicted_violation": result.highest_risk_category,
            "predicted_risk_level": self._determine_overall_risk_level(result),
            "is_edge_case": False
        }
        
        # Check if violation type prediction is correct
        violation_correct = (
            test_case.expected_violation_type == result.highest_risk_category or
            (test_case.expected_violation_type is None and result.highest_risk_category is None)
        )
        
        # Check if risk level prediction is correct
        risk_level_correct = test_case.expected_risk_level == evaluation["predicted_risk_level"]
        
        # Overall correctness
        evaluation["correct"] = violation_correct and risk_level_correct
        
        # Check for edge cases (close to threshold boundaries)
        if result.overall_risk_score > 0.4 and result.overall_risk_score < 0.6:
            evaluation["is_edge_case"] = True
        
        evaluation["violation_correct"] = violation_correct
        evaluation["risk_level_correct"] = risk_level_correct
        evaluation["confidence"] = result.overall_risk_score
        
        return evaluation
    
    def _determine_overall_risk_level(self, result: MultiCategoryDetectionResult) -> str:
        """Determine overall risk level from detection result."""
        if result.num_high_risk_violations > 0 or result.overall_risk_score >= 0.5:
            return "high"
        elif result.num_medium_risk_violations > 0 or result.overall_risk_score >= 0.25:
            return "medium"
        elif result.num_low_risk_violations > 0 or result.overall_risk_score >= 0.1:
            return "low"
        else:
            return "none"
    
    def _update_metrics(self, metrics: ValidationMetrics, evaluation: Dict[str, Any]):
        """Update validation metrics based on evaluation."""
        if evaluation["correct"]:
            if evaluation["predicted_violation"] is not None:
                metrics.true_positives += 1
            else:
                metrics.true_negatives += 1
        else:
            if evaluation["predicted_violation"] is not None:
                metrics.false_positives += 1
            else:
                metrics.false_negatives += 1
        
        # Update risk level accuracy
        if evaluation["risk_level_correct"]:
            metrics.risk_level_accuracy += 1
    
    def _calculate_category_accuracies(
        self, 
        test_cases: List[TestCase], 
        failed_cases: List[Dict[str, Any]]
    ) -> Dict[ViolationType, float]:
        """Calculate accuracy for each violation category."""
        category_accuracies = {}
        
        # Count test cases by category
        category_counts = {}
        for test_case in test_cases:
            if test_case.expected_violation_type:
                category_counts[test_case.expected_violation_type] = category_counts.get(
                    test_case.expected_violation_type, 0) + 1
        
        # Count failures by category
        category_failures = {}
        for failed_case in failed_cases:
            test_case = failed_case["test_case"]
            if test_case.expected_violation_type:
                category_failures[test_case.expected_violation_type] = category_failures.get(
                    test_case.expected_violation_type, 0) + 1
        
        # Calculate accuracies
        for violation_type, total_count in category_counts.items():
            failure_count = category_failures.get(violation_type, 0)
            accuracy = (total_count - failure_count) / total_count
            category_accuracies[violation_type] = accuracy
        
        return category_accuracies
    
    def _generate_recommendations(self, report: ValidationReport) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        metrics = report.metrics
        
        # Accuracy recommendations
        if metrics.accuracy < 0.8:
            recommendations.append("Overall accuracy is below 80%. Consider retraining with more diverse data.")
        
        if metrics.precision < 0.7:
            recommendations.append("High false positive rate detected. Consider raising detection thresholds.")
        
        if metrics.recall < 0.7:
            recommendations.append("High false negative rate detected. Consider lowering detection thresholds.")
        
        # Category-specific recommendations
        for violation_type, accuracy in metrics.category_accuracies.items():
            if accuracy < 0.6:
                recommendations.append(
                    f"Poor accuracy for {violation_type.value} ({accuracy:.2f}). "
                    f"Consider adding more training data for this category."
                )
        
        # Performance recommendations
        if metrics.avg_processing_time > 1.0:
            recommendations.append("Processing time is high. Consider model optimization for real-time use.")
        
        # Platform-specific recommendations
        if report.platform == Platform.ROBLOX and len(report.failed_cases) > 0:
            child_safety_failures = [
                case for case in report.failed_cases 
                if case["test_case"].expected_violation_type in [ViolationType.CHILD_SAFETY, ViolationType.GROOMING]
            ]
            if child_safety_failures:
                recommendations.append("Child safety detection failures detected. This is critical for Roblox platform.")
        
        return recommendations
    
    def _suggest_threshold_adjustments(self, failed_cases: List[Dict[str, Any]]) -> Dict[str, float]:
        """Suggest threshold adjustments based on failed cases."""
        suggestions = {}
        
        # Analyze false positives (predicted violation when none expected)
        false_positives = [
            case for case in failed_cases
            if case["test_case"].expected_violation_type is None and
               case["result"].highest_risk_category is not None
        ]
        
        if false_positives:
            fp_scores = [case["result"].overall_risk_score for case in false_positives]
            suggested_threshold = np.percentile(fp_scores, 90)  # 90th percentile
            suggestions["reduce_false_positives"] = min(0.7, suggested_threshold + 0.1)
        
        # Analyze false negatives (missed violations)
        false_negatives = [
            case for case in failed_cases
            if case["test_case"].expected_violation_type is not None and
               case["result"].highest_risk_category is None
        ]
        
        if false_negatives:
            fn_scores = [case["result"].overall_risk_score for case in false_negatives]
            suggested_threshold = np.percentile(fn_scores, 10)  # 10th percentile
            suggestions["reduce_false_negatives"] = max(0.05, suggested_threshold - 0.05)
        
        return suggestions
    
    def _create_empty_report(self, platform: Platform, index: MultiCategorySentinelIndex) -> ValidationReport:
        """Create an empty validation report."""
        return ValidationReport(
            platform=platform,
            test_timestamp=datetime.now().isoformat(),
            model_config=index.get_platform_summary(),
            metrics=ValidationMetrics(),
            recommendations=["No test cases or models available for validation."]
        )
    
    def _save_report(self, report: ValidationReport):
        """Save validation report to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{report.platform.value}_validation_{timestamp}.json"
        filepath = self.results_dir / filename
        
        # Convert report to JSON-serializable format
        report_dict = {
            "platform": report.platform.value,
            "test_timestamp": report.test_timestamp,
            "model_config": report.model_config,
            "metrics": {
                "accuracy": report.metrics.accuracy,
                "precision": report.metrics.precision,
                "recall": report.metrics.recall,
                "f1_score": report.metrics.f1_score,
                "true_positives": report.metrics.true_positives,
                "true_negatives": report.metrics.true_negatives,
                "false_positives": report.metrics.false_positives,
                "false_negatives": report.metrics.false_negatives,
                "risk_level_accuracy": report.metrics.risk_level_accuracy,
                "avg_processing_time": report.metrics.avg_processing_time,
                "total_test_cases": report.metrics.total_test_cases,
                "category_accuracies": {k.value: v for k, v in report.metrics.category_accuracies.items()}
            },
            "performance_stats": report.performance_stats,
            "recommendations": report.recommendations,
            "threshold_suggestions": report.threshold_suggestions,
            "summary": {
                "failed_cases_count": len(report.failed_cases),
                "edge_cases_count": len(report.edge_cases),
                "overall_grade": self._calculate_overall_grade(report.metrics)
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        LOG.info(f"Validation report saved to {filepath}")
    
    def _calculate_overall_grade(self, metrics: ValidationMetrics) -> str:
        """Calculate an overall grade for the validation."""
        score = (metrics.accuracy + metrics.f1_score) / 2
        
        if score >= 0.9:
            return "A"
        elif score >= 0.8:
            return "B"
        elif score >= 0.7:
            return "C"
        elif score >= 0.6:
            return "D"
        else:
            return "F"
    
    def create_sample_test_cases(self, platform: Platform) -> Path:
        """Create sample test cases for a platform.
        
        Args:
            platform: Platform to create test cases for.
            
        Returns:
            Path to the created test file.
        """
        sample_cases = {
            "description": f"Sample test cases for {platform.value} platform",
            "platform": platform.value,
            "test_cases": [
                {
                    "text": "Thanks for helping me with this problem",
                    "expected_violation_type": None,
                    "expected_risk_level": "none",
                    "description": "Positive, helpful interaction"
                },
                {
                    "text": "You're worthless trash and should kill yourself",
                    "expected_violation_type": "harassment",
                    "expected_risk_level": "high",
                    "description": "Clear harassment with self-harm encouragement"
                },
                {
                    "text": "What's your real name and where do you live?",
                    "expected_violation_type": "child_safety" if platform == Platform.ROBLOX else "doxxing",
                    "expected_risk_level": "high",
                    "description": "Personal information solicitation"
                },
                {
                    "text": "You played really well in that game",
                    "expected_violation_type": None,
                    "expected_risk_level": "none",
                    "description": "Positive gaming interaction"
                },
                {
                    "text": "FREE ROBUX! Click this link now!",
                    "expected_violation_type": "scams" if platform == Platform.ROBLOX else "spam",
                    "expected_risk_level": "medium",
                    "description": "Scam attempt"
                }
            ]
        }
        
        test_file = self.test_data_dir / f"{platform.value}_test_cases.json"
        with open(test_file, 'w') as f:
            json.dump(sample_cases, f, indent=2)
        
        LOG.info(f"Created sample test cases at {test_file}")
        return test_file