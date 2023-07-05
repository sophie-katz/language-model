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

from language_model.models.transformer_from_scratch.word_embedding import WordEmbedding


def test_word_embedding_simple() -> None:
    """Test the shape of word embeddings."""
    batch_size = 2
    vocabulary_size = 3
    sentence_length = 4
    feature_count = 8

    word_embedding_layer = WordEmbedding(vocabulary_size, feature_count)

    word_indices = T.randint(
        vocabulary_size,
        (
            batch_size,
            sentence_length,
        ),
    )

    word_embeddings = word_embedding_layer(word_indices)

    assert word_embeddings.shape == (batch_size, sentence_length, feature_count)


@pytest.mark.skipif(not T.cuda.is_available(), reason="CUDA not available")
def test_word_embedding_cuda() -> None:
    """Test the shape of word embeddings."""
    batch_size = 2
    vocabulary_size = 3
    sentence_length = 4
    feature_count = 8

    word_embedding_layer = WordEmbedding(vocabulary_size, feature_count).cuda()

    word_indices = T.randint(
        vocabulary_size,
        (
            batch_size,
            sentence_length,
        ),
    ).cuda()

    word_embeddings = word_embedding_layer(word_indices)

    assert word_embeddings.device.type == "cuda"
