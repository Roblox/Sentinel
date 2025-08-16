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
Training data builder for creating comprehensive violation detection models.

This module provides tools to load, process, and combine training data from multiple
sources to create robust detection models for each violation category.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import random

import torch
from sentence_transformers import SentenceTransformer

from .violation_taxonomy import ViolationType, Platform, VIOLATION_TAXONOMY
from .platform_config import Platform
from .sentinel_local_index import SentinelLocalIndex
from .embeddings.sbert import get_sentence_transformer_and_scaling_fn

LOG = logging.getLogger(__name__)


@dataclass
class TrainingDataset:
    """Container for training data."""
    
    positive_examples: List[str]
    negative_examples: List[str]
    violation_type: ViolationType
    source_files: List[str]
    metadata: Dict[str, any]


class TrainingDataBuilder:
    """Builder for creating comprehensive training datasets and models."""
    
    def __init__(
        self,
        training_data_dir: Path,
        model_output_dir: Path,
        encoder_model_name: str = "all-MiniLM-L6-v2"
    ):
        """Initialize the training data builder.
        
        Args:
            training_data_dir: Directory containing training data files.
            model_output_dir: Directory to save trained models.
            encoder_model_name: Name of the sentence transformer model to use.
        """
        self.training_data_dir = training_data_dir
        self.model_output_dir = model_output_dir
        self.encoder_model_name = encoder_model_name
        
        # Ensure directories exist
        self.training_data_dir.mkdir(exist_ok=True)
        self.model_output_dir.mkdir(exist_ok=True)
        
        # Load model and scaling function
        self.model, self.scale_fn = get_sentence_transformer_and_scaling_fn(encoder_model_name)
        
        LOG.info(f"Initialized TrainingDataBuilder with model {encoder_model_name}")
    
    def load_violation_training_data(self, violation_type: ViolationType) -> Optional[TrainingDataset]:
        """Load training data for a specific violation type.
        
        Args:
            violation_type: Type of violation to load data for.
            
        Returns:
            TrainingDataset if data exists, None otherwise.
        """
        # Look for violation-specific data file
        data_file = self.training_data_dir / f"{violation_type.value}_violations.json"
        
        if not data_file.exists():
            LOG.warning(f"Training data file not found: {data_file}")
            return None
        
        try:
            with open(data_file) as f:
                data = json.load(f)
            
            positive_examples = [example["text"] for example in data.get("examples", [])]
            
            if not positive_examples:
                LOG.warning(f"No positive examples found in {data_file}")
                return None
            
            # Load negative examples from safe content
            negative_examples = self._load_negative_examples(len(positive_examples) * 10)
            
            dataset = TrainingDataset(
                positive_examples=positive_examples,
                negative_examples=negative_examples,
                violation_type=violation_type,
                source_files=[str(data_file)],
                metadata={
                    "category": data.get("category", ""),
                    "description": data.get("description", ""),
                    "severity": data.get("severity", 3),
                    "platforms": data.get("platforms", []),
                    "usage_restriction": data.get("usage_restriction", "")
                }
            )
            
            LOG.info(f"Loaded {len(positive_examples)} positive and {len(negative_examples)} negative examples for {violation_type.value}")
            return dataset
            
        except Exception as e:
            LOG.error(f"Error loading training data for {violation_type.value}: {e}")
            return None
    
    def _load_negative_examples(self, target_count: int) -> List[str]:
        """Load negative examples from safe content files.
        
        Args:
            target_count: Number of negative examples to load.
            
        Returns:
            List of safe content examples.
        """
        negative_examples = []
        
        # Load from safe content file
        safe_content_file = self.training_data_dir / "safe_content_examples.json"
        if safe_content_file.exists():
            try:
                with open(safe_content_file) as f:
                    data = json.load(f)
                
                safe_examples = [example["text"] for example in data.get("examples", [])]
                negative_examples.extend(safe_examples)
                
            except Exception as e:
                LOG.error(f"Error loading safe content: {e}")
        
        # If we need more examples, create synthetic safe content
        if len(negative_examples) < target_count:
            additional_needed = target_count - len(negative_examples)
            synthetic_examples = self._generate_synthetic_negative_examples(additional_needed)
            negative_examples.extend(synthetic_examples)
        
        # Randomly sample to get the target count
        if len(negative_examples) > target_count:
            negative_examples = random.sample(negative_examples, target_count)
        
        return negative_examples
    
    def _generate_synthetic_negative_examples(self, count: int) -> List[str]:
        """Generate synthetic negative examples.
        
        Args:
            count: Number of examples to generate.
            
        Returns:
            List of synthetic safe content examples.
        """
        # Simple synthetic safe content templates
        templates = [
            "Thanks for the help with {}",
            "I really enjoyed {}",
            "Looking forward to {}",
            "Great job on {}",
            "Can someone help me understand {}",
            "I'm learning about {}",
            "This is interesting information about {}",
            "I appreciate your explanation of {}",
            "Does anyone have experience with {}",
            "Welcome to our community discussion about {}"
        ]
        
        topics = [
            "the game", "this project", "the tutorial", "the event", "coding",
            "mathematics", "science", "art", "music", "literature", "sports",
            "technology", "history", "geography", "languages", "cooking",
            "gardening", "photography", "movies", "books"
        ]
        
        synthetic_examples = []
        for i in range(count):
            template = random.choice(templates)
            topic = random.choice(topics)
            example = template.format(topic)
            synthetic_examples.append(example)
        
        return synthetic_examples
    
    def build_category_model(self, violation_type: ViolationType) -> bool:
        """Build and save a model for a specific violation category.
        
        Args:
            violation_type: Type of violation to build model for.
            
        Returns:
            True if model was successfully built and saved, False otherwise.
        """
        # Load training data
        dataset = self.load_violation_training_data(violation_type)
        if not dataset:
            LOG.error(f"Could not load training data for {violation_type.value}")
            return False
        
        try:
            # Encode examples
            LOG.info(f"Encoding positive examples for {violation_type.value}...")
            positive_embeddings = self.model.encode(
                dataset.positive_examples, 
                normalize_embeddings=True,
                show_progress_bar=True
            )
            
            LOG.info(f"Encoding negative examples for {violation_type.value}...")
            negative_embeddings = self.model.encode(
                dataset.negative_examples,
                normalize_embeddings=True, 
                show_progress_bar=True
            )
            
            # Create Sentinel index
            index = SentinelLocalIndex(
                sentence_model=self.model,
                positive_embeddings=torch.tensor(positive_embeddings),
                negative_embeddings=torch.tensor(negative_embeddings),
                scale_fn=self.scale_fn,
                positive_corpus=dataset.positive_examples,
                negative_corpus=dataset.negative_examples,
                model_card={
                    "violation_type": violation_type.value,
                    "description": dataset.metadata.get("description", ""),
                    "severity": dataset.metadata.get("severity", 3),
                    "training_data_size": {
                        "positive": len(dataset.positive_examples),
                        "negative": len(dataset.negative_examples)
                    },
                    "source_files": dataset.source_files,
                    "encoder_model": self.encoder_model_name,
                    "usage_restriction": dataset.metadata.get("usage_restriction", "")
                }
            )
            
            # Save the model
            output_path = self.model_output_dir / f"{violation_type.value}_index"
            saved_config = index.save(
                path=str(output_path),
                encoder_model_name_or_path=self.encoder_model_name
            )
            
            LOG.info(f"Successfully built and saved model for {violation_type.value} at {output_path}")
            return True
            
        except Exception as e:
            LOG.error(f"Error building model for {violation_type.value}: {e}")
            return False
    
    def build_general_safety_model(self) -> bool:
        """Build a general safety model combining multiple violation types.
        
        Returns:
            True if model was successfully built and saved, False otherwise.
        """
        all_positive_examples = []
        all_negative_examples = []
        source_files = []
        
        # Combine data from all available violation types
        for violation_type in ViolationType:
            if violation_type == ViolationType.GENERAL_HATE:
                continue  # Skip to avoid recursion
            
            dataset = self.load_violation_training_data(violation_type)
            if dataset:
                all_positive_examples.extend(dataset.positive_examples)
                source_files.extend(dataset.source_files)
        
        if not all_positive_examples:
            LOG.error("No positive examples found for general safety model")
            return False
        
        # Load negative examples (10:1 ratio)
        all_negative_examples = self._load_negative_examples(len(all_positive_examples) * 10)
        
        try:
            # Encode examples
            LOG.info("Encoding positive examples for general safety model...")
            positive_embeddings = self.model.encode(
                all_positive_examples,
                normalize_embeddings=True,
                show_progress_bar=True
            )
            
            LOG.info("Encoding negative examples for general safety model...")
            negative_embeddings = self.model.encode(
                all_negative_examples,
                normalize_embeddings=True,
                show_progress_bar=True
            )
            
            # Create general safety index
            index = SentinelLocalIndex(
                sentence_model=self.model,
                positive_embeddings=torch.tensor(positive_embeddings),
                negative_embeddings=torch.tensor(negative_embeddings),
                scale_fn=self.scale_fn,
                positive_corpus=all_positive_examples,
                negative_corpus=all_negative_examples,
                model_card={
                    "violation_type": "general_safety",
                    "description": "General safety model combining multiple violation types",
                    "severity": 4,
                    "training_data_size": {
                        "positive": len(all_positive_examples),
                        "negative": len(all_negative_examples)
                    },
                    "source_files": list(set(source_files)),
                    "encoder_model": self.encoder_model_name,
                    "usage_restriction": "DEFENSIVE SECURITY ONLY - For general content safety detection"
                }
            )
            
            # Save the general model
            output_path = self.model_output_dir / "general_safety_index"
            saved_config = index.save(
                path=str(output_path),
                encoder_model_name_or_path=self.encoder_model_name
            )
            
            LOG.info(f"Successfully built and saved general safety model at {output_path}")
            return True
            
        except Exception as e:
            LOG.error(f"Error building general safety model: {e}")
            return False
    
    def build_all_models(self) -> Dict[str, bool]:
        """Build models for all available violation types and general safety.
        
        Returns:
            Dictionary mapping model names to build success status.
        """
        results = {}
        
        # Build category-specific models
        for violation_type in ViolationType:
            if violation_type == ViolationType.GENERAL_HATE:
                continue  # Will build this separately as general safety
            
            model_name = f"{violation_type.value}_model"
            success = self.build_category_model(violation_type)
            results[model_name] = success
        
        # Build general safety model
        results["general_safety_model"] = self.build_general_safety_model()
        
        # Summary
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        LOG.info(f"Model building complete: {successful}/{total} models built successfully")
        
        return results
    
    def validate_training_data(self) -> Dict[str, any]:
        """Validate all training data files and return summary statistics.
        
        Returns:
            Dictionary with validation results and statistics.
        """
        validation_results = {
            "valid_files": [],
            "invalid_files": [],
            "statistics": {},
            "warnings": []
        }
        
        total_positive = 0
        total_negative = 0
        
        # Check each violation type
        for violation_type in ViolationType:
            if violation_type == ViolationType.GENERAL_HATE:
                continue
            
            data_file = self.training_data_dir / f"{violation_type.value}_violations.json"
            
            if not data_file.exists():
                validation_results["warnings"].append(f"Missing training data file: {data_file}")
                continue
            
            try:
                with open(data_file) as f:
                    data = json.load(f)
                
                examples = data.get("examples", [])
                if not examples:
                    validation_results["invalid_files"].append(str(data_file))
                    validation_results["warnings"].append(f"No examples in {data_file}")
                    continue
                
                positive_count = len(examples)
                total_positive += positive_count
                
                validation_results["valid_files"].append(str(data_file))
                validation_results["statistics"][violation_type.value] = {
                    "positive_examples": positive_count,
                    "severity": data.get("severity", "unknown"),
                    "platforms": data.get("platforms", [])
                }
                
            except Exception as e:
                validation_results["invalid_files"].append(str(data_file))
                validation_results["warnings"].append(f"Error reading {data_file}: {e}")
        
        # Check safe content
        safe_content_file = self.training_data_dir / "safe_content_examples.json"
        if safe_content_file.exists():
            try:
                with open(safe_content_file) as f:
                    data = json.load(f)
                
                safe_examples = len(data.get("examples", []))
                total_negative = safe_examples
                validation_results["statistics"]["safe_content"] = {
                    "negative_examples": safe_examples
                }
                
            except Exception as e:
                validation_results["warnings"].append(f"Error reading safe content file: {e}")
        
        validation_results["summary"] = {
            "total_positive_examples": total_positive,
            "total_negative_examples": total_negative,
            "valid_violation_types": len(validation_results["valid_files"]),
            "invalid_files_count": len(validation_results["invalid_files"])
        }
        
        return validation_results