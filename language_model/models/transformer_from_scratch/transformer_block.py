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

"""A block from a transformer.

This could be either an encoder block or a decoder block. It is a base class to both
blocks.

This is heavily inspired by
https://medium.com/the-dl/transformers-from-scratch-in-pytorch-8777e346ca51.

https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch
was used to help with its implementation.
"""

import abc
import dataclasses

import torch as T
from torch import nn

from language_model.models.transformer_from_scratch.feed_forward import FeedForward
from language_model.models.transformer_from_scratch.multi_head_attention import (
    MultiHeadAttention,
)
from language_model.models.transformer_from_scratch.residual import Residual


@dataclasses.dataclass
class TransformerBlock(nn.Module, abc.ABC):
    """A block from a transformer.

    This could be either an encoder block or a decoder block. It is a base class to both
    blocks.

    This is heavily inspired by
    https://medium.com/the-dl/transformers-from-scratch-in-pytorch-8777e346ca51.

    https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch
    was used to help with its implementation.

    Attributes
    ----------
    input_size : int
        The size of the input tensor.
    head_count : int
        The number of attention heads to use.
    feed_forward_hidden_size : int
        The size of the hidden layer in the feed forward layer.
    dropout_rate : float
        The dropout rate for the residual layers.
    query_size : int
        The size of the query tensor.
    key_size : int
        The size of the key tensor.
    value_size : int
        The size of the value tensor.
    self_attention : Residual[MultiHeadAttention]
        The self attention layer.
    attention : Residual[MultiHeadAttention]
        The attention layer.
    feed_forward : Residual[FeedForward]
        The feed forward layer.
    """

    input_size: int
    head_count: int
    feed_forward_hidden_size: int
    dropout_rate: float

    query_size: int = dataclasses.field(init=False)
    key_size: int = dataclasses.field(init=False)
    value_size: int = dataclasses.field(init=False)

    attention: Residual[MultiHeadAttention] = dataclasses.field(init=False)
    feed_forward: Residual[FeedForward] = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """Postinitialization for Pytorch module."""
        super().__init__()

        self.query_size = max(self.input_size // self.head_count, 1)
        self.key_size = max(self.input_size // self.head_count, 1)
        self.value_size = max(self.input_size // self.head_count, 1)

        self.attention = Residual(
            internal_layer=MultiHeadAttention(
                head_count=self.head_count,
                input_size=self.input_size,
                query_size=self.query_size,
                key_size=self.key_size,
                value_size=self.value_size,
            ),
            input_size=self.input_size,
            dropout_rate=self.dropout_rate,
        )

        self.feed_forward = Residual(
            internal_layer=FeedForward(
                input_size=self.input_size,
                feed_forward_hidden_size=self.feed_forward_hidden_size,
            ),
            input_size=self.input_size,
            dropout_rate=self.dropout_rate,
        )
