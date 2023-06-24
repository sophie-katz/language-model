# Copyright (c) 2023 Sophie Katz
#
# This file is part of Language Model.
#
# Language Model is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later version.
#
# Language Model is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Language
# Model. If not, see <https://www.gnu.org/licenses/>.

# This code is based off notebooks/word_embedding_from_scratch.ipynb.

import torch as T
import torch.nn as nn
from typing import cast


class WordEmbedding(nn.Module):
    """
    A simple word embedding model that takes a word as input and returns its embedding.

    This module expects as input a tensor of word indices within the vocabulary of shape
    `(sentence_length,)`. It returns a tensor of word embeddings of shape
    `(sentence_length, embedding_size)`.

    Args:
        vocabulary_size: int
            The number of different words we expect to find in our input.
        embedding_size: int
            The size of the embedding vector for a given word.
    """

    def __init__(self, vocabulary_size: int, embedding_size: int) -> None:
        super().__init__()

        self.vocab_size = vocabulary_size
        self.embedding_size = embedding_size

        # We use Pytorch's built in embedding layer
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)

    def forward(self, sentence: T.Tensor) -> T.Tensor:
        # We expect sentence to be of shape (sentence_length,) and to be a tensor of
        # word indices within the vocabulary.

        assert sentence.ndim == 1

        result = self.embedding(sentence)

        # We expect result to be of shape (sentence_length, embedding_size)
        assert result.ndim == 2
        assert result.size(0) == sentence.size(0)
        assert result.size(1) == self.embedding_size

        return cast(T.Tensor, result)
