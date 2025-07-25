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
Sentinel - Semantic Ratio for Improved Recognition And Classification of Hurtful or Antagonistic content.

This library provides tools for semantic scoring of text based on contrastive learning principles.
"""

from sentinel.sentinel_local_index import SentinelLocalIndex
from sentinel.score_formulae import calculate_contrastive_score

__all__ = ["SentinelLocalIndex", "calculate_contrastive_score"]
