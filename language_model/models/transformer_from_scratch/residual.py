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


class Residual(nn.Module):
    def __init__(
        self,
        internal_layer: nn.Module,
        input_size: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()

        self.internal_layer = internal_layer
        self.input_size = input_size
        self.dropout_rate = dropout_rate

        self.normalization = nn.LayerNorm(input_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, *tensors: T.Tensor) -> T.Tensor:
        assert len(tensors) > 0
        assert tensors[0].ndim == 2
        assert tensors[0].size(-1) == self.input_size
        assert tensors[1].ndim == 2
        assert tensors[2].ndim == 2

        return cast(
            T.Tensor,
            self.normalization(
                tensors[0] + self.dropout(self.internal_layer(*tensors))
            ),
        )
