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

# pylint: disable=abstract-method

import dataclasses

from torch import nn

from language_model.models.transformer_from_scratch.feed_forward import FeedForward
from language_model.models.transformer_from_scratch.multi_head_attention import (
    MultiHeadAttention,
)
from language_model.models.transformer_from_scratch.residual import Residual


@dataclasses.dataclass(unsafe_hash=True)
class TransformerBlock(nn.Module):
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
    attention : Residual[MultiHeadAttention]
        The attention layer.
    feed_forward : Residual[FeedForward]
        The feed forward layer.
    """

    input_feature_count: int
    head_count: int
    feed_forward_hidden_feature_count: int
    residual_dropout_rate: float

    qkv_feature_count: int = dataclasses.field(init=False)

    attention: Residual[MultiHeadAttention] = dataclasses.field(init=False)
    feed_forward: Residual[FeedForward] = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """Postinitialization for Pytorch module."""
        # pylint: disable=unidiomatic-typecheck

        super().__init__()

        if type(self) == TransformerBlock:
            raise TypeError(
                "TransformerBlock is an abstract class and cannot be instantiated."
            )

        self.qkv_feature_count = max(self.input_feature_count // self.head_count, 1)

        self.attention = Residual(
            internal_layer=MultiHeadAttention(
                head_count=self.head_count,
                input_feature_count=self.input_feature_count,
                qkv_feature_count=self.qkv_feature_count,
            ),
            input_feature_count=self.input_feature_count,
            dropout_rate=self.residual_dropout_rate,
        )

        feed_forward_internal = FeedForward(
            input_feature_count=self.input_feature_count,
            feed_forward_hidden_feature_count=self.feed_forward_hidden_feature_count,
        )

        self.feed_forward = Residual(
            internal_layer=feed_forward_internal,
            input_feature_count=self.input_feature_count,
            dropout_rate=self.residual_dropout_rate,
        )
