# Copyright (c) 2023 Sophie Katz
#
# This file is part of Language Model.
#
# Language Model is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
#
# Language Model is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Language
# Model. If not, see <https://www.gnu.org/licenses/>.

"""Unit tests."""

import pytest
import torch as T

from language_model.models.transformer_from_scratch.token_embedding import (
    TokenEmbedding,
)


def test_token_embedding_simple() -> None:
    """Test the shape of token embeddings."""
    batch_size = 2
    vocabulary_size = 3
    sentence_length = 4
    feature_count = 8

    token_embedding_layer = TokenEmbedding(vocabulary_size, feature_count)

    token_indices = T.randint(
        vocabulary_size,
        (
            batch_size,
            sentence_length,
        ),
    )

    token_embeddings = token_embedding_layer(token_indices)

    assert token_embeddings.shape == (batch_size, sentence_length, feature_count)


@pytest.mark.skipif(not T.cuda.is_available(), reason="CUDA not available")
def test_token_embedding_cuda() -> None:
    """Test the shape of token embeddings."""
    batch_size = 2
    vocabulary_size = 3
    sentence_length = 4
    feature_count = 8

    token_embedding_layer = TokenEmbedding(vocabulary_size, feature_count).cuda()

    token_indices = T.randint(
        vocabulary_size,
        (
            batch_size,
            sentence_length,
        ),
    ).cuda()

    token_embeddings = token_embedding_layer(token_indices)

    assert token_embeddings.device.type == "cuda"
