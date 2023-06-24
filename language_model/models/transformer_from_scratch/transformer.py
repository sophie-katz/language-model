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
from language_model.models.transformer_from_scratch.decoder import (
    DECODER_LAYER_COUNT_DEFAULT,
    Decoder,
)
from language_model.models.transformer_from_scratch.encoder import (
    ENCODER_LAYER_COUNT_DEFAULT,
    Encoder,
)

TRANSFORMER_INPUT_SIZE_DEFAULT = 512
TRANSFORMER_HEAD_COUNT_DEFAULT = 6
TRANSFORMER_FEED_FORWARD_HIDDEN_SIZE_DEFAULT = 2048
TRANSFORMER_DROPOUT_RATE_DEFAULT = 0.1


class Transformer(nn.Module):
    def __init__(
        self,
        encoder_layer_count: int = ENCODER_LAYER_COUNT_DEFAULT,
        decoder_layer_count: int = DECODER_LAYER_COUNT_DEFAULT,
        input_size: int = TRANSFORMER_INPUT_SIZE_DEFAULT,
        head_count: int = TRANSFORMER_HEAD_COUNT_DEFAULT,
        feed_forward_hidden_size: int = TRANSFORMER_FEED_FORWARD_HIDDEN_SIZE_DEFAULT,
        dropout_rate: float = TRANSFORMER_DROPOUT_RATE_DEFAULT,
        activation: nn.Module = nn.ReLU(),  # TODO: Is this needed?
    ) -> None:
        super().__init__()

        self.encoder_layer_count = encoder_layer_count
        self.decoder_layer_count = decoder_layer_count
        self.input_size = input_size
        self.head_count = head_count
        self.feed_forward_hidden_size = feed_forward_hidden_size
        self.dropout_rate = dropout_rate
        self.activation = activation

        self.encoder = Encoder(
            layer_count=self.encoder_layer_count,
            input_size=self.input_size,
            head_count=self.head_count,
            feed_forward_hidden_size=self.feed_forward_hidden_size,
            dropout_rate=self.dropout_rate,
        )

        self.decoder = Decoder(
            layer_count=self.decoder_layer_count,
            input_size=self.input_size,
            head_count=self.head_count,
            feed_forward_hidden_size=self.feed_forward_hidden_size,
            dropout_rate=self.dropout_rate,
        )

    def forward(self, source: T.Tensor, target: T.Tensor) -> T.Tensor:
        return cast(T.Tensor, self.decoder(target, self.encoder(source)))
