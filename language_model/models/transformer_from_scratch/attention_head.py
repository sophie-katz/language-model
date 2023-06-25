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

import torch as T
import torch.nn as nn
from language_model.models.transformer_from_scratch.qkv import QKV
from .attention import attention
from .shapes import (
    has_sequence_shape,
    get_sequence_batch_size,
    get_sequence_feature_count,
)
import dataclasses


@dataclasses.dataclass
class AttentionHead(nn.Module):
    input_size: int
    query_size: int
    key_size: int
    value_size: int

    query_linear: nn.Linear = dataclasses.field(init=False)
    key_linear: nn.Linear = dataclasses.field(init=False)
    value_linear: nn.Linear = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        super().__init__()

        self.query_linear = nn.Linear(self.input_size, self.query_size)
        self.key_linear = nn.Linear(self.input_size, self.key_size)
        self.value_linear = nn.Linear(self.input_size, self.value_size)

    def forward(self, qkv: QKV) -> T.Tensor:
        assert (
            self.input_size == qkv.feature_count
        ), "QKV tensors must have the feature count equal to input size"

        return attention(
            QKV(
                self.query_linear(qkv.query),
                self.key_linear(qkv.key),
                self.value_linear(qkv.value),
            )
        )
