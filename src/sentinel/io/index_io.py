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
Module for abstracting reading and writing of SentinelLocalIndex.

This module provides functions for saving and loading SentinelLocalIndex instances
using smart_open, which allows working with both local and S3 paths.
"""

import json
import logging
import tempfile
import os
from typing import Dict, Any, Optional, Tuple

import torch
import safetensors
from safetensors.torch import save_file
import smart_open

from sentinel.io.saved_index_config import SavedIndexConfig

LOG = logging.getLogger(__name__)

# Constants for file names
CONFIG_FILE_NAME = "sentinel_local_index_config.json"
EMBEDDINGS_FILE_NAME = "embeddings.safetensors"
# Storage keys - kept as positive/negative for backward compatibility
POSITIVE_EMBEDDINGS_KEY = "positive_embeddings"  # Corresponds to rare class examples
NEGATIVE_EMBEDDINGS_KEY = "negative_embeddings"  # Corresponds to common class examples


def create_s3_transport_params(
    aws_access_key_id: Optional[str] = None, aws_secret_access_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create transport parameters for S3 access.

    Args:
        aws_access_key_id: AWS access key ID.
        aws_secret_access_key: AWS secret access key.

    Returns:
        Dictionary with transport parameters for smart_open.
    """
    if aws_access_key_id and aws_secret_access_key:
        return {
            "s3": {
                "client_kwargs": {
                    "aws_access_key_id": aws_access_key_id,
                    "aws_secret_access_key": aws_secret_access_key,
                }
            }
        }
    return {}


def _join_path(base_path: str, filename: str) -> str:
    """
    Join a base path and filename, handling both local and S3 paths.

    Args:
        base_path: Base directory path.
        filename: File name to append.

    Returns:
        Full path to the file.
    """
    if base_path.startswith("s3://"):
        if base_path.endswith("/"):
            return f"{base_path}{filename}"
        else:
            return f"{base_path}/{filename}"
    else:
        return os.path.join(base_path, filename)


def save_index(
    path: str,
    config: SavedIndexConfig,
    positive_embeddings: torch.Tensor,  # Represents rare class examples
    negative_embeddings: torch.Tensor,  # Represents common class examples
    transport_params: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save a SentinelLocalIndex to a path.

    Args:
        path: Path where to save the index.
        config: SavedIndexConfig instance containing configuration.
        positive_embeddings: Tensor of positive (rare class) example embeddings.
        negative_embeddings: Tensor of negative (common class) example embeddings.
        transport_params: Optional transport parameters for smart_open.
    """
    # Ensure directory exists for local paths
    if not path.startswith("s3://") and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    config_path = _join_path(path, CONFIG_FILE_NAME)
    embeddings_path = _join_path(path, EMBEDDINGS_FILE_NAME)

    # Save config file
    LOG.info("Saving index configuration to %s", config_path)
    with smart_open.open(config_path, "w", transport_params=transport_params) as f:
        json.dump(config.to_dict(), f)

    # For local paths, we can save tensors directly
    if not path.startswith("s3://"):
        LOG.info("Saving embeddings to %s", embeddings_path)
        save_file(
            {
                POSITIVE_EMBEDDINGS_KEY: positive_embeddings,  # Store as "positive_embeddings" (rare class examples)
                NEGATIVE_EMBEDDINGS_KEY: negative_embeddings,  # Store as "negative_embeddings" (common class examples)
            },
            embeddings_path,
        )
    else:
        # For S3, save to a temporary file first
        with tempfile.NamedTemporaryFile() as temp_file:
            LOG.info("Saving embeddings to temporary file before uploading to S3")
            save_file(
                {
                    POSITIVE_EMBEDDINGS_KEY: positive_embeddings,  # Store as "positive_embeddings" (rare class)
                    NEGATIVE_EMBEDDINGS_KEY: negative_embeddings,  # Store as "negative_embeddings" (common examples)
                },
                temp_file.name,
            )

            # Upload the temporary file to S3
            LOG.info("Uploading embeddings to %s", embeddings_path)
            with open(temp_file.name, "rb") as f_in:
                with smart_open.open(
                    embeddings_path, "wb", transport_params=transport_params
                ) as f_out:
                    f_out.write(f_in.read())

    LOG.info("Successfully saved index to %s", path)


def load_embeddings(
    file_path: str, transport_params: Optional[Dict[str, Any]] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load embeddings from a safetensors file.

    Args:
        file_path: Path to the embeddings file.
        transport_params: Optional transport parameters for smart_open.

    Returns:
        Tuple of (positive_embeddings, negative_embeddings) representing (rare class, common class)
    """
    if file_path.startswith("s3://"):
        # For S3, download to a temporary file first
        with tempfile.NamedTemporaryFile() as temp_file:
            LOG.info("Downloading embeddings from S3 to temporary file")
            with smart_open.open(
                file_path, "rb", transport_params=transport_params
            ) as f_in:
                with open(temp_file.name, "wb") as f_out:
                    f_out.write(f_in.read())

            # Load from the temporary file
            return _load_embeddings_from_file(temp_file.name)
    else:
        # For local paths, load directly
        return _load_embeddings_from_file(file_path)


def _load_embeddings_from_file(file_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load embeddings from a local file.

    Args:
        file_path: Path to the local embeddings file.

    Returns:
        Tuple of (positive_embeddings, negative_embeddings) representing (rare class, common class)
    """
    LOG.info("Loading embeddings from %s", file_path)
    with safetensors.safe_open(file_path, framework="pt", device="cpu") as f:
        # Get tensor slices for positive and negative embeddings
        positive_slice = f.get_slice(
            POSITIVE_EMBEDDINGS_KEY
        )  # Load from "positive_embeddings" key
        negative_slice = f.get_slice(
            NEGATIVE_EMBEDDINGS_KEY
        )  # Load from "negative_embeddings" key

        # Extract the data
        _, positive_dim = positive_slice.get_shape()
        _, negative_dim = negative_slice.get_shape()

        positive_embeddings = positive_slice[:, :positive_dim]
        negative_embeddings = negative_slice[:, :negative_dim]

    # Return in the legacy naming order for backward compatibility
    return positive_embeddings, negative_embeddings


def load_index(
    path: str, transport_params: Optional[Dict[str, Any]] = None
) -> Tuple[SavedIndexConfig, torch.Tensor, torch.Tensor]:
    """
    Load a SentinelLocalIndex from a path.

    Args:
        path: Path where the index is stored.
        transport_params: Optional transport parameters for smart_open.

    Returns:
        Tuple of (config, positive_embeddings, negative_embeddings) representing (config, rare class, common class)
    """
    config_path = _join_path(path, CONFIG_FILE_NAME)
    embeddings_path = _join_path(path, EMBEDDINGS_FILE_NAME)

    # Load config file
    LOG.info("Loading index configuration from %s", config_path)
    with smart_open.open(config_path, "r", transport_params=transport_params) as f:
        config_dict = json.load(f)

    config = SavedIndexConfig.from_dict(config_dict)

    # Load embeddings
    positive_embeddings, negative_embeddings = load_embeddings(
        embeddings_path, transport_params
    )

    LOG.info("Successfully loaded index from %s", path)

    return config, positive_embeddings, negative_embeddings
