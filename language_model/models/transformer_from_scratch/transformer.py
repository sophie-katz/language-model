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

"""Transformer model.

This is heavily inspired by
https://medium.com/the-dl/transformers-from-scratch-in-pytorch-8777e346ca51.

https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch was
used to help with its implementation.
"""

import dataclasses

import torch as T
from torch import nn

from language_model.models.transformer_from_scratch.decoder import Decoder
from language_model.models.transformer_from_scratch.encoder import Encoder


@dataclasses.dataclass
class Transformer(nn.Module):
    """Transformer model.

    This is heavily inspired by
    https://medium.com/the-dl/transformers-from-scratch-in-pytorch-8777e346ca51.

    https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch was
    used to help with its implementation.

    Attributes
    ----------
    encoder_layer_count : int
        The number of encoder block layers to use.
    decoder_layer_count : int
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
    embedding_size : int
        The size of the embedding vector for a given word.
    max_sequence_length : int
        The maximum length of a sequence.
    encoder : Encoder
        The encoder part of the model.
    decoder : Decoder
        The decoder part of the model.
    """

    encoder_layer_count: int
    decoder_layer_count: int
    input_size: int
    head_count: int
    feed_forward_hidden_size: int
    dropout_rate: float
    positional_encoding_base: float
    vocabulary_size: int
    embedding_size: int
    max_sequence_length: int

    encoder: Encoder = dataclasses.field(init=False)
    decoder: Decoder = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """Postinitialization for Pytorch module."""
        super().__init__()

        self.encoder = Encoder(
            layer_count=self.encoder_layer_count,
            input_size=self.input_size,
            head_count=self.head_count,
            feed_forward_hidden_size=self.feed_forward_hidden_size,
            dropout_rate=self.dropout_rate,
            positional_encoding_base=self.positional_encoding_base,
            vocabulary_size=self.vocabulary_size,
            embedding_size=self.embedding_size,
            max_sequence_length=self.max_sequence_length,
        )

        self.decoder = Decoder(
            layer_count=self.decoder_layer_count,
            input_size=self.input_size,
            head_count=self.head_count,
            feed_forward_hidden_size=self.feed_forward_hidden_size,
            dropout_rate=self.dropout_rate,
            positional_encoding_base=self.positional_encoding_base,
            vocabulary_size=self.vocabulary_size,
            embedding_size=self.embedding_size,
            max_sequence_length=self.max_sequence_length,
        )

    def forward(self, source: T.Tensor, target: T.Tensor) -> T.Tensor:
        """Forward function for network.

        Parameters
        ----------
        source : T.Tensor
            The source tensor.
        target : T.Tensor
            The target tensor.

        Returns
        -------
        T.Tensor
            A single tensor. TODO: Find the size of this.
        """
        result: T.Tensor = self.encoder(source)
        result = self.decoder(target, result)
        return result
