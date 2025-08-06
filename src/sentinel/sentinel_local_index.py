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
Module for local Sentinel index implementation.

This module provides the implementation of the SentinelLocalIndex class for local semantic scoring.
"""

import logging
from typing import Optional, List, Mapping, Any, Callable

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search

from sentinel.score_formulae import calculate_contrastive_score, skewness
from sentinel.io.saved_index_config import SavedIndexConfig
from sentinel.io.index_io import save_index, load_index, create_s3_transport_params
from sentinel.embeddings.sbert import get_sentence_transformer_and_scaling_fn
from sentinel.score_types import RareClassAffinityResult

LOG = logging.getLogger(__name__)


class SentinelLocalIndex:
    """Calculate scores for detecting extremely rare classes of text using contrastive learning.

    This class implements a realtime approach specifically designed for detecting rare text patterns
    where traditional classifiers would fail due to extreme class imbalance. The core workflow is:

    1. Collect multiple observations from a single source (e.g., recent messages from a user)
    2. Calculate individual observation scores using contrastive learning
    3. Aggregate these scores using skewness to detect patterns, independent of observation count
    4. Apply optional threshold filtering for decision-making

    As a high-recall candidate generator, this approach prioritizes identifying potential cases for
    further investigation, emphasizing not missing true positives even at the cost of some false positives.

    The contrastive learning approach compares each observation against both rare class examples
    and common class examples, calculating a ratio of similarities. This ratio indicates whether
    the observation is more similar to the rare class than to the common class.

    By default, skewness is used as the aggregation method since it captures the prevalence of
    rare patterns without being affected by the total number of observations, making it ideal
    for scenarios with varying observation counts.

    For optimal results with English text, we recommend using the MiniLM-L6-v2 model with
    approximately 5-20k examples of the rare class.
    """

    def __init__(
        self,
        sentence_model: Optional[SentenceTransformer] = None,
        positive_embeddings: Optional[torch.Tensor] = None,
        negative_embeddings: Optional[torch.Tensor] = None,
        scale_fn: Optional[Callable[[float], float]] = None,
        encoding_additional_kwargs: Mapping[
            str, Any
        ] = {},  # Particularly of interest are prompt (or prompt_name) and precision
        positive_corpus: Optional[List[str]] = None,
        negative_corpus: Optional[List[str]] = None,
        model_card: Optional[
            Mapping[str, Any]
        ] = None,  # The description of where this model positive and negative examples came from, etc.
    ):
        """Initialize the SentinelLocalIndex.

        Args:
            sentence_model: A SentenceTransformer model instance.
            positive_embeddings: Tensor of embeddings for positive (rare class) examples.
            negative_embeddings: Tensor of embeddings for negative (common class) examples.
            scale_fn: Optional callable to scale similarity scores (needed for some models like E5).
            encoding_additional_kwargs: Additional keyword arguments for encoding.
            positive_corpus: List of original positive example texts (for debugging).
            negative_corpus: List of original negative example texts (for debugging).
            model_card: Dictionary with metadata about the model.

        Note:
            For direct initialization, you should get a model and scale_fn by calling:
            `model, scale_fn = get_sentence_transformer_and_scaling_fn(encoder_model_name_or_path)`

            When saving the index, you must provide the exact encoder_model_name_or_path
            as SentenceTransformer doesn't store the original model name.

        Use the class method `load` to load an index from S3 or local storage.
        """
        self.sentence_model: SentenceTransformer = sentence_model
        self.scale_fn: Optional[Callable[[float], float]] = scale_fn

        self.positive_embeddings: torch.Tensor = None
        if positive_embeddings is not None:
            if isinstance(positive_embeddings, torch.Tensor):
                self.positive_embeddings = positive_embeddings
            else:
                self.positive_embeddings = torch.tensor(positive_embeddings)

        self.negative_embeddings: torch.Tensor = None
        if negative_embeddings is not None:
            if isinstance(negative_embeddings, torch.Tensor):
                self.negative_embeddings = negative_embeddings
            else:
                self.negative_embeddings = torch.tensor(negative_embeddings)

        self.encoding_kwargs = {
            "normalize_embeddings": True,
        }
        self.encoding_kwargs.update(encoding_additional_kwargs)
        self.positive_corpus = positive_corpus
        self.negative_corpus = negative_corpus
        self.model_card = model_card

    def save(
        self,
        path: str,
        encoder_model_name_or_path: str,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ) -> SavedIndexConfig:
        """
        Save the index to a file or S3 path.

        Args:
            path: Path to save the index to (local directory or S3 URI).
            encoder_model_name_or_path: Name or path of the sentence transformer encoder model used.
                This must be the exact name used to create the SentenceTransformer as it cannot be
                reliably extracted from the model instance.
            aws_access_key_id: Optional AWS access key ID for S3 access.
            aws_secret_access_key: Optional AWS secret access key for S3 access.

        Returns:
            The SavedIndexConfig object that was saved. This is returned for informational purposes only,
            as the config has already been written to the specified location and will be automatically
            read by the load method.
        """
        # Create config
        config = SavedIndexConfig(
            encoder_model_name_or_path=encoder_model_name_or_path,
            encoding_kwargs=self.encoding_kwargs,
            model_card=self.model_card,
        )

        # Create transport parameters for S3 if needed
        transport_params = create_s3_transport_params(
            aws_access_key_id, aws_secret_access_key
        )

        # Save the index
        save_index(
            path=path,
            config=config,
            positive_embeddings=self.positive_embeddings,
            negative_embeddings=self.negative_embeddings,
            transport_params=transport_params,
        )

        # Return the config for informational purposes
        return config

    @classmethod
    def load(
        cls,
        path: str,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        negative_to_positive_ratio: float = 1.0,
    ) -> "SentinelLocalIndex":
        """
        Load the index from a path and returns a new SentinelLocalIndex instance.

        Args:
            path: Path to load the index from (local directory or S3 URI).
            aws_access_key_id: Optional AWS access key ID for S3 access.
            aws_secret_access_key: Optional AWS secret access key for S3 access.
            negative_to_positive_ratio: Ratio of negative examples to keep relative to positive examples.

        Returns:
            A new SentinelLocalIndex instance with the loaded model and embeddings.
        """
        # Create transport parameters for S3 if needed
        transport_params = create_s3_transport_params(
            aws_access_key_id, aws_secret_access_key
        )

        # Load the index
        config, positive_embeddings, negative_embeddings = load_index(
            path=path, transport_params=transport_params
        )

        # Create the sentence model and get the scaling function
        model_name = config.encoder_model_name_or_path

        sentence_model, scale_fn = get_sentence_transformer_and_scaling_fn(model_name)

        # Create a new instance with the loaded model and data
        instance = cls(
            sentence_model=sentence_model,
            scale_fn=scale_fn,
            positive_embeddings=positive_embeddings,
            negative_embeddings=negative_embeddings,
            encoding_additional_kwargs=config.encoding_kwargs,
            model_card=config.model_card,
        )

        # Apply negative ratio if needed
        instance._apply_negative_ratio(negative_to_positive_ratio)

        return instance

    def _apply_negative_ratio(self, negative_to_positive_ratio: float):
        """
        Apply the negative_to_positive_ratio to reduce the number of negative (common class) examples.

        Args:
            negative_to_positive_ratio: The ratio of negative samples to keep relative to positive samples.
        """
        # Calculate the number of negative samples to keep
        num_negative_to_keep = int(
            self.positive_embeddings.shape[0] * negative_to_positive_ratio
        )

        if self.negative_embeddings.shape[0] > num_negative_to_keep:
            LOG.info(
                "Keeping %d negative examples out of %d",
                num_negative_to_keep,
                self.negative_embeddings.shape[0],
            )
            # Randomly select a subset of the negative examples
            indices = torch.randperm(self.negative_embeddings.shape[0])[
                :num_negative_to_keep
            ]
            self.negative_embeddings = self.negative_embeddings[indices]
        else:
            LOG.info(
                "User requested %d negative examples but the model loaded only has %d",
                num_negative_to_keep,
                self.negative_embeddings.shape[0],
            )

    def calculate_rare_class_affinity(
        self,
        text_samples: List[str],
        top_k: int = 5,
        similarity_formula: Callable[
            [List[float], List[float]], float
        ] = calculate_contrastive_score,
        # Function to aggregate individual scores into an overall affinity score
        aggregation_function: Callable[[np.array], float] = skewness,
        # Margin to ignore when text is only slightly more similar to positive than negative.
        min_score_to_consider: float = 0.1,
        # Use when simulating by sampling texts from the same data indexed.
        prevent_exact_match: bool = False,
        encoding_additional_kwargs: Mapping[str, Any] = {},
        show_progress_bar: bool = False,
    ) -> RareClassAffinityResult:
        """Calculate rare class affinity for the given text samples in realtime.

        This method serves as a high-recall candidate generator for identifying potential rare class instances
        that warrant further investigation. It encodes recent observations from a single source and compares
        them to rare class and common class examples, prioritizing not missing true positives.

        For each observation, it calculates an individual score based on similarity to the rare class versus
        the common class. It then aggregates these scores, using an aggregation function like skewness,
        to detect patterns across multiple observations, independent of their total count.

        Args:
            text_samples: List of text strings to evaluate for rare class affinity.
            top_k: Number of closest neighbors to consider when calculating the score.
            similarity_formula: Function to calculate individual similarity scores.
            aggregation_function: Function to aggregate individual scores into an overall score.
            min_score_to_consider: Threshold below which scores are set to 0.
            prevent_exact_match: Whether to skip exact matches when scoring.
            encoding_additional_kwargs: Additional keyword arguments for encoding.
            show_progress_bar: Whether to display a progress bar during encoding.

        Returns:
            RareClassAffinityResult containing both the overall affinity score and
            individual observation scores for each text sample.
        """
        # Merge the default encoding kwargs with any additional ones provided
        effective_encoding_kwargs = self.encoding_kwargs.copy()
        effective_encoding_kwargs["show_progress_bar"] = show_progress_bar
        effective_encoding_kwargs.update(encoding_additional_kwargs)

        # Encode the input samples to get their embeddings
        # We currently don't support multi-process encoding in this method, because it is meant for online scoring.
        # We can add it if needed, probably by just allowing the caller to pass sample embeddings instead of text.
        sample_embeddings = self.sentence_model.encode(
            text_samples,
            **effective_encoding_kwargs,
        )

        # If we need to prevent exact matches (e.g., when scoring examples that are in the index),
        # request an additional neighbor so we can skip the exact match later
        additional_neighbors = 1 if prevent_exact_match else 0

        # Perform semantic search to find the most similar positive (rare class) examples
        positive_matches = semantic_search(
            sample_embeddings,
            self.positive_embeddings,
            top_k=top_k + additional_neighbors,
        )

        # Perform semantic search to find the most similar negative (common class) examples
        negative_matches = semantic_search(
            sample_embeddings,
            self.negative_embeddings,
            top_k=top_k + additional_neighbors,
        )

        observation_scores = {}

        for i, q in enumerate(text_samples):
            LOG.debug("Query: %s", q)

            # Combine and sort both positive and negative matches by similarity score
            # The "+" sign marks positive examples, "-" sign marks negative examples
            matches = sorted(
                [(hit["score"], hit["corpus_id"], "+") for hit in positive_matches[i]]
                + [
                    (hit["score"], hit["corpus_id"], "-") for hit in negative_matches[i]
                ],
                key=lambda x: x[0],
                reverse=True,
            )

            # Initialize lists to collect similarity scores for the top matches
            similarities_topk_positive = []
            similarities_topk_negative = []
            max_h = top_k  # Number of examples to consider

            # Process each match in order of similarity (highest first)
            for h, (score, corpus_id, sign) in enumerate(matches):
                # Stop once we've collected enough examples
                if h == max_h:
                    break

                # Skip exact matches if requested (when score is almost exactly 1.0)
                if prevent_exact_match and h == 0 and abs(score - 1.0) < 1e-3:
                    max_h += 1  # Compensate for skipping this match
                    continue

                # Apply scaling to the similarity score if a scaling function is available
                if self.scale_fn:
                    scaled_score = self.scale_fn(score)
                else:
                    scaled_score = score

                # Add the score to the appropriate list based on whether it's positive or negative
                if sign == "+":
                    similarities_topk_positive.append(scaled_score)
                    neighbor = (
                        self.positive_corpus[corpus_id]
                        if self.positive_corpus
                        else corpus_id
                    )
                else:
                    if sign != "-":
                        raise AssertionError(f"Unexpected sign: {sign}")
                    similarities_topk_negative.append(scaled_score)
                    neighbor = (
                        self.negative_corpus[corpus_id]
                        if self.negative_corpus
                        else corpus_id
                    )

                if LOG.level <= logging.DEBUG:
                    LOG.debug(
                        f"[{sign}] {neighbor} (Score: {score:.4f}, Scaled Score: {scaled_score:.4f})"
                    )

            # Ensure we have at least one similarity value for each category
            # If we didn't find any of a particular category in the top matches,
            # use the first match from the original search
            if not similarities_topk_positive:
                similarities_topk_positive = [positive_matches[i][0]["score"]]
            if not similarities_topk_negative:
                similarities_topk_negative = [negative_matches[i][0]["score"]]

            if LOG.level <= logging.DEBUG:
                LOG.debug(
                    f"Top {top_k} similarities for '{q}': "
                    f"[+] {similarities_topk_positive}, [-] {similarities_topk_negative}"
                )

            # Calculate the final score using the provided formula (default: calculate_contrastive_score)
            # This compares how close the observation is to positive examples vs. negative examples
            score = similarity_formula(
                similarities_topk_pos=similarities_topk_positive,
                similarities_topk_neg=similarities_topk_negative,
            )

            # Apply threshold to filter out borderline cases
            # If the score is below the minimum threshold, set it to zero
            if score < min_score_to_consider:
                observation_scores[q] = 0.0
            else:
                observation_scores[q] = score

        # Calculate the overall rare class affinity score by aggregating individual scores
        # If there are no scores, default to 0.0
        if not observation_scores:
            rare_class_score = 0.0
        else:
            rare_class_score = aggregation_function(
                np.array(list(observation_scores.values()))
            )

        return RareClassAffinityResult(
            rare_class_affinity_score=rare_class_score,
            observation_scores=observation_scores,
        )
