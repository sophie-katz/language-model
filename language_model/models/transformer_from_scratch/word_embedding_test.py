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

import torch as T

from language_model.models.transformer_from_scratch.word_embedding import WordEmbedding


def test_word_embedding() -> None:
    """Test the shape of word embeddings.

    Vocabulary size: 3
    Sentence length: 4
    Embedding size: 8
    """
    word_embedding_layer = WordEmbedding(3, 8)

    word_indices = T.randint(3, (4,))

    word_embeddings = word_embedding_layer(word_indices)

    assert word_embeddings.shape == (4, 8)
