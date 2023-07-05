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

"""A single pass of a transformer.

Can be used as a base class for either an encoder or a decoder.

This is heavily inspired by
https://medium.com/the-dl/transformers-from-scratch-in-pytorch-8777e346ca51.

https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch
was used to help with its implementation.
"""

import abc

import torch as T
from torch import nn

from language_model.models.transformer_from_scratch.positional_encoding import (
    get_positional_encoding,
)
from language_model.models.transformer_from_scratch.word_embedding import WordEmbedding


class TransformerPass(abc.ABC, nn.Module):
    """A single pass of a transformer.

    Can be used as a base class for either an encoder or a decoder.

    This is heavily inspired by
    https://medium.com/the-dl/transformers-from-scratch-in-pytorch-8777e346ca51.

    https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch
    was used to help with its implementation.

    Attributes
    ----------
    layer_count : int
        The number of decoder block layers to use.
    input_size : int
        The size of the input tensor.
    head_count : int
        The number of attention heads to use.
    feed_forward_hidden_size : int
        The size of the hidden layer in the feed forward layer.
    dropout_rate : float
        The dropout rate for the residual layers.
    positional_encoding_base : float
        The exponentiation base to use for generating the positional encoding matrix.
    vocabulary_size : int
        The number of different words we expect to find in our input.
    word_embedding_feature_count : int
        The size of the embedding vector for a given word.
    max_sequence_length : int
        The maximum length of a sequence.
    word_embedding : WordEmbedding
        The word embedding layer.
    layers : nn.ModuleList
        The decoder block layers.
    """

    def __init__(
        self,
        layer_count: int,
        word_embedding_vocabulary_size: int,
        word_embedding_feature_count: int,
        positional_encoding_max_sequence_length: int,
        positional_encoding_base: float,
    ) -> None:
        super().__init__()

        self.layer_count = layer_count
        self.word_embedding_vocabulary_size = word_embedding_vocabulary_size
        self.word_embedding_feature_count = word_embedding_feature_count
        self.positional_encoding_max_sequence_length = (
            positional_encoding_max_sequence_length
        )
        self.positional_encoding_base = positional_encoding_base

        self.word_embedding = WordEmbedding(
            word_embedding_vocabulary_size, word_embedding_feature_count
        )

        self.positional_encoding = get_positional_encoding(
            positional_encoding_max_sequence_length,
            word_embedding_feature_count,
            positional_encoding_base,
        )

    def forward(self, tensor: T.Tensor) -> T.Tensor:
        tensor = self.word_embedding(tensor)

        # TODO: Possibly scale up embedding -
        # https://www.notion.so/Confirm-if-embedding-should-be-scaled-up-55f74b736e724bf0b40788873a9235ed?pvs=4
        # tensor *= self.input_size ** 0.5

        tensor += self.positional_encoding[..., : tensor.size(1), :].to(tensor.device)

        return tensor
