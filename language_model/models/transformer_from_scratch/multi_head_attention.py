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
from .attention_head import AttentionHead
from .qkv import QKV


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        head_count: int,
        input_size: int,
        query_size: int,
        key_size: int,
        value_size: int,
    ) -> None:
        super().__init__()

        self.head_count = head_count
        self.input_size = input_size
        self.query_size = query_size
        self.key_size = key_size
        self.value_size = value_size

        # TODO: Should each head have a separate key, query, and value matrix? - https://www.notion.so/Confirm-if-each-head-should-have-separate-trainable-weights-a3a189025e544dd58904ca23007a902d?pvs=4
        self.heads = nn.ModuleList(
            [
                AttentionHead(input_size, query_size, key_size, value_size)
                for _ in range(head_count)
            ]
        )

        assert (
            key_size == value_size
        ), "TODO: implement different key and value sizes"  # https://www.notion.so/Implement-different-key-and-value-sizes-3b4412dff5e9422caaa4cda49a0c4587?pvs=4

        self.linear = nn.Linear(head_count * key_size, input_size)

    def forward(self, qkv: QKV) -> T.Tensor:
        return cast(
            T.Tensor,
            self.linear(
                T.cat(
                    [head(qkv.query, qkv.key, qkv.value) for head in self.heads],
                    dim=-1,
                )
            ),
        )
