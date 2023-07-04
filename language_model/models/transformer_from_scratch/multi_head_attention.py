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

"""A multi-head attention module from a transformer.

This is heavily inspired by
https://medium.com/the-dl/transformers-from-scratch-in-pytorch-8777e346ca51.

https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch
was used to help with its implementation.
"""

import dataclasses
from typing import Optional

import torch as T
from torch import nn

from language_model.models.transformer_from_scratch.attention_head import AttentionHead
from language_model.models.transformer_from_scratch.qkv import QKV


@dataclasses.dataclass(unsafe_hash=True)
class MultiHeadAttention(nn.Module):
    """A multi-head attention module from a transformer.

    This is heavily inspired by
    https://medium.com/the-dl/transformers-from-scratch-in-pytorch-8777e346ca51.

    https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch
    was used to help with its implementation.

    Attributes
    ----------
    head_count : int
        The number of attention heads.
    input_size : int
        The size of the input tensor.
    query_size : int
        The size of the query tensor.
    key_size : int
        The size of the key tensor.
    value_size : int
        The size of the value tensor.
    heads: nn.ModuleList
        The list of attention heads.
    linear : nn.Linear
        The linear layer for the output tensor.
    """

    head_count: int
    input_feature_count: int
    qkv_feature_count: int

    heads: nn.ModuleList = dataclasses.field(init=False)
    linear: nn.Linear = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """Postinitialization for Pytorch module."""
        super().__init__()

        # TODO: Should each head have a separate key, query, and value matrix? -
        # https://www.notion.so/Confirm-if-each-head-should-have-separate-trainable-weights-a3a189025e544dd58904ca23007a902d?pvs=4
        self.heads = nn.ModuleList(
            [
                AttentionHead(
                    input_feature_count=self.input_feature_count,
                    qkv_feature_count=self.qkv_feature_count,
                )
                for _ in range(self.head_count)
            ]
        )

        self.linear = nn.Linear(
            self.head_count * self.qkv_feature_count, self.input_feature_count
        )

    def forward(self, qkv: QKV, mask: Optional[T.Tensor] = None) -> T.Tensor:
        """Forward function for network.

        Parameters
        ----------
        qkv : QKV
            The query, key, and value matrices to use as input to attention. Each of
            them should be of shape `(batch_size, sequence_length, feature_count)`. Note
            that each of the three matrices may have different feature counts, although
            they will all have the same batch size and sequence length.

        Returns
        -------
        T.Tensor
            A single tensor. TODO: Find the size of this.
        """
        head_results = [head(qkv, mask=mask) for head in self.heads]

        result: T.Tensor = T.cat(
            head_results,
            dim=-1,
        )

        result = self.linear(result)

        return result
