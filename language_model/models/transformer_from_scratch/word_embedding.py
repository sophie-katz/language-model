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

"""Simple word embedding implementation for a transformer.

This code is based off notebooks/word_embedding_from_scratch.ipynb.

https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch was
used to help with its implementation.
"""

import dataclasses

import torch as T
from torch import nn


@dataclasses.dataclass(unsafe_hash=True)
class WordEmbedding(nn.Module):
    """A simple word embedding model.

    Takes a word as input and returns its embedding

    This module expects as input a tensor of word indices within the vocabulary of shape
    `(sentence_length,)`. It returns a tensor of word embeddings of shape
    `(sentence_length, feature_count)`.

    Attributes
    ----------
    vocabulary_size : int
        The number of different words we expect to find in our input.
    feature_count : int
        The size of the embedding vector for a given word.
    embedding : nn.Embedding
        The embedding layer.
    """

    vocabulary_size: int
    feature_count: int

    embedding: nn.Embedding = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """Postinitialization for Pytorch module."""
        super().__init__()

        # We use Pytorch's built in embedding layer
        self.embedding = nn.Embedding(self.vocabulary_size, self.feature_count)

    def forward(self, sentence: T.Tensor) -> T.Tensor:
        """Forward function for network.

        Parameters
        ----------
        sentence : T.Tensor
            Tensor of shape (sentence_length,) and to be a tensor of word indices within
            the vocabulary.

        Returns
        -------
        T.Tensor
            A single tensor of shape `(sentence_length, feature_count)`
            that contains the embedding.
        """
        # pylint: disable=magic-value-comparison

        assert (
            sentence.ndim == 2
        ), f"input sentence should be a batch of vectors of word indices, not {sentence.shape}"

        result: T.Tensor = self.embedding(sentence)

        assert result.shape == (
            sentence.size(0),
            sentence.size(1),
            self.feature_count,
        ), "embedding should be a matrix of shape \
            (batch_size, sentence_length, feature_count)"

        return result
