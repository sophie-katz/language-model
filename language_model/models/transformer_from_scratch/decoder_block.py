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

# This is heavily inspired by
# https://medium.com/the-dl/transformers-from-scratch-in-pytorch-8777e346ca51.
#
# https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch was
# used to help with its implementation.

import torch as T
import torch.nn as nn
from typing import cast
from language_model.models.transformer_from_scratch.feed_forward import FeedForward
from language_model.models.transformer_from_scratch.multi_head_attention import (
    MultiHeadAttention,
)
from language_model.models.transformer_from_scratch.residual import Residual
import dataclasses


@dataclasses.dataclass
class DecoderBlock(nn.Module):
    input_size: int
    head_count: int
    feed_forward_hidden_size: int
    dropout_rate: float

    query_size: int = dataclasses.field(init=False)
    key_size: int = dataclasses.field(init=False)
    value_size: int = dataclasses.field(init=False)

    self_attention: Residual[MultiHeadAttention] = dataclasses.field(init=False)
    attention: Residual[MultiHeadAttention] = dataclasses.field(init=False)
    feed_forward: Residual[FeedForward] = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        super().__init__()

        self.query_size = max(self.input_size // self.head_count, 1)
        self.key_size = max(self.input_size // self.head_count, 1)
        self.value_size = max(self.input_size // self.head_count, 1)

        self.self_attention = Residual(
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
                hidden_size=self.feed_forward_hidden_size,
            ),
            input_size=self.input_size,
            dropout_rate=self.dropout_rate,
        )

    def forward(self, target: T.Tensor, memory: T.Tensor) -> T.Tensor:
        result: T.Tensor = self.self_attention(target, target, target)
        result = self.attention(result, memory, memory)
        result = self.feed_forward(result)
        return result
