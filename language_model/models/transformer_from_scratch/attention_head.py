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

# This is heavily inspired by https://medium.com/the-dl/transformers-from-scratch-in-pytorch-8777e346ca51

import torch as T
import torch.nn as nn
from .attention import attention


class AttentionHead(nn.Module):
    def __init__(
        self, input_size: int, query_size: int, key_size: int, value_size: int
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.query_size = query_size
        self.key_size = key_size
        self.value_size = value_size

        self.query = nn.Linear(input_size, query_size)
        self.key = nn.Linear(input_size, key_size)
        self.value = nn.Linear(input_size, value_size)

    def forward(
        self, query_input: T.Tensor, key_input: T.Tensor, value_input: T.Tensor
    ) -> T.Tensor:
        return attention(
            self.query(query_input), self.key(key_input), self.value(value_input)
        )
