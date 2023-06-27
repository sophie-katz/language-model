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

"""A single attention head for use in a transformer model.

This is heavily inspired by
https://medium.com/the-dl/transformers-from-scratch-in-pytorch-8777e346ca51.
"""

import dataclasses

import torch as T
from torch import nn

from language_model.models.transformer_from_scratch.qkv import QKV

from .attention import attention


@dataclasses.dataclass
class AttentionHead(nn.Module):
    """A single attention head for use in a transformer model.

    This is heavily inspired by
    https://medium.com/the-dl/transformers-from-scratch-in-pytorch-8777e346ca51.

    Attributes
    ----------
    input_size : int
        The size of the input tensor.
    query_size : int
        The size of the query tensor.
    key_size : int
        The size of the key tensor.
    value_size : int
        The size of the value tensor.
    query_linear : nn.Linear
        The linear layer for the query tensor.
    key_linear : nn.Linear
        The linear layer for the key tensor.
    value_linear : nn.Linear
        The linear layer for the value tensor.
    """

    input_size: int
    query_size: int
    key_size: int
    value_size: int

    query_linear: nn.Linear = dataclasses.field(init=False)
    key_linear: nn.Linear = dataclasses.field(init=False)
    value_linear: nn.Linear = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """Postinitialization for Pytorch module."""
        super().__init__()

        self.query_linear = nn.Linear(self.input_size, self.query_size)
        self.key_linear = nn.Linear(self.input_size, self.key_size)
        self.value_linear = nn.Linear(self.input_size, self.value_size)

    def forward(self, qkv: QKV) -> T.Tensor:
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
