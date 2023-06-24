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
from language_model.models.transformer_from_scratch.positional_encoding import (
    get_positional_encoding,
)
from language_model.models.transformer_from_scratch.encoder_block import EncoderBlock


class Encoder(nn.Module):
    def __init__(
        self,
        layer_count: int,
        input_size: int,
        head_count: int,
        feed_forward_hidden_size: int,
        dropout_rate: float,
        positional_encoding_base: float,
    ) -> None:
        super().__init__()

        self.layer_count = layer_count
        self.input_size = input_size
        self.head_count = head_count
        self.feed_forward_hidden_size = feed_forward_hidden_size
        self.dropout_rate = dropout_rate
        self.positional_encoding_base = positional_encoding_base

        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    input_size=self.input_size,
                    head_count=self.head_count,
                    feed_forward_hidden_size=self.feed_forward_hidden_size,
                    dropout_rate=self.dropout_rate,
                )
                for _ in range(self.layer_count)
            ]
        )

    def forward(self, source: T.Tensor) -> T.Tensor:
        sequence_length = source.size(1)
        input_size = source.size(2)

        # TODO: Possibly scale up embedding:
        # source *= self.input_size ** 0.5

        # TODO: Pull this out of the forward function:
        source += get_positional_encoding(
            sequence_length, input_size, self.positional_encoding_base
        )

        for layer in self.layers:
            source = layer(source)

        return source
