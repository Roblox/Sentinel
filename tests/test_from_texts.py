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

"""Tests for SentinelLocalIndex.from_texts()."""

import tempfile
import pytest
import torch

from sentinel.sentinel_local_index import SentinelLocalIndex
from sentinel.score_types import RareClassAffinityResult


class TestFromTexts:
    """Building an index in one call."""

    POSITIVE = ["unsafe content detected", "harmful behavior observed", "dangerous activity"]
    NEGATIVE = [
        "normal behavior detected",
        "regular activity observed",
        "safe content identified",
        "standard procedure followed",
        "ordinary events occurred",
        "the meeting went well",
    ]

    @pytest.mark.integration
    def test_builds_a_usable_index(self):
        """One call produces an index that scores text correctly."""
        index = SentinelLocalIndex.from_texts(
            positive_texts=self.POSITIVE,
            negative_texts=self.NEGATIVE,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )

        assert index.positive_embeddings.shape[0] == len(self.POSITIVE)
        assert index.negative_embeddings.shape[0] == len(self.NEGATIVE)
        assert index.sentence_model is not None

        result = index.calculate_rare_class_affinity(
            ["harmful unsafe behavior", "normal regular activity"]
        )
        assert isinstance(result, RareClassAffinityResult)

    @pytest.mark.integration
    def test_corpus_is_always_kept(self):
        """The corpus comes along automatically - half the point of the method.

        Forgetting it in the manual recipe costs you explanations, silently.
        """
        index = SentinelLocalIndex.from_texts(
            positive_texts=self.POSITIVE,
            negative_texts=self.NEGATIVE,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )

        assert index.positive_corpus == self.POSITIVE
        assert index.negative_corpus == self.NEGATIVE

    @pytest.mark.integration
    def test_normalization_is_applied_by_default(self):
        """Embeddings come out unit-length without the caller asking.

        Omitting normalize_embeddings by hand does not error; it just makes the
        similarity maths wrong. Asserting the norms catches a silent regression.
        """
        index = SentinelLocalIndex.from_texts(
            positive_texts=self.POSITIVE,
            negative_texts=self.NEGATIVE,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )

        norms = index.positive_embeddings.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
        assert index.encoding_kwargs["normalize_embeddings"] is True

    @pytest.mark.integration
    def test_ratio_downsamples_and_keeps_alignment(self):
        """The ratio is applied, and surviving negatives keep their own text."""
        index = SentinelLocalIndex.from_texts(
            positive_texts=self.POSITIVE,
            negative_texts=self.NEGATIVE,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            neg_to_pos_ratio=1.0,
            seed=42,
        )

        assert index.negative_embeddings.shape[0] == 3  # 3 positives * 1.0
        assert len(index.negative_corpus) == 3
        assert set(index.negative_corpus) <= set(self.NEGATIVE)

    @pytest.mark.integration
    def test_seeded_ratio_is_reproducible(self):
        """Same seed, same index."""
        kwargs = dict(
            positive_texts=self.POSITIVE,
            negative_texts=self.NEGATIVE,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            neg_to_pos_ratio=1.0,
        )
        a = SentinelLocalIndex.from_texts(seed=7, **kwargs)
        b = SentinelLocalIndex.from_texts(seed=7, **kwargs)

        assert torch.equal(a.negative_embeddings, b.negative_embeddings)
        assert a.negative_corpus == b.negative_corpus

    @pytest.mark.integration
    def test_round_trip_through_save_and_load(self):
        """An index built this way saves and reloads with explanations intact."""
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        index = SentinelLocalIndex.from_texts(
            positive_texts=self.POSITIVE,
            negative_texts=self.NEGATIVE,
            model_name=model_name,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            index.save(path=temp_dir, encoder_model_name_or_path=model_name)
            reloaded = SentinelLocalIndex.load(
                path=temp_dir, negative_to_positive_ratio=None, seed=1
            )

        assert reloaded.positive_corpus == self.POSITIVE
        assert reloaded.negative_corpus == self.NEGATIVE

    @pytest.mark.parametrize(
        "kwargs,message",
        [
            ({"positive_texts": "a bare string"}, "positive_texts must be a list"),
            ({"negative_texts": "a bare string"}, "negative_texts must be a list"),
            ({"positive_texts": []}, "positive_texts must not be empty"),
            ({"negative_texts": []}, "negative_texts must not be empty"),
            ({"neg_to_pos_ratio": 0}, "neg_to_pos_ratio must be positive"),
            ({"neg_to_pos_ratio": -1.0}, "neg_to_pos_ratio must be positive"),
        ],
    )
    def test_input_validation(self, kwargs, message):
        """Bad input is rejected up front, before any expensive encoding happens.

        A bare string is iterable, so without this check it would be encoded one
        character at a time - confusing, slow, and entirely silent.
        """
        call = {"positive_texts": self.POSITIVE, "negative_texts": self.NEGATIVE}
        call.update(kwargs)
        with pytest.raises(ValueError, match=message):
            SentinelLocalIndex.from_texts(**call)
