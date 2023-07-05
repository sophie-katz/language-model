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

from typing import Optional

import torch as T
from torch import nn

from language_model.models.transformer_from_scratch.qkv import QKV
from language_model.models.transformer_from_scratch.attention import attention
from language_model.models.transformer_from_scratch.shapes import (
    get_sequence_length,
)


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

    def __init__(self, input_feature_count: int, qkv_feature_count: int) -> None:
        super().__init__()

        self.input_feature_count = input_feature_count
        self.qkv_feature_count = qkv_feature_count

        self.query_linear = nn.Linear(self.input_feature_count, self.qkv_feature_count)
        self.key_linear = nn.Linear(self.input_feature_count, self.qkv_feature_count)
        self.value_linear = nn.Linear(self.input_feature_count, self.qkv_feature_count)

    def forward(self, qkv: QKV, mask: Optional[T.Tensor] = None) -> T.Tensor:
        """Forward function for network.

        Parameters
        ----------
        qkv : QKV
            The query, key, and value matrices to use as input to attention. Each of
            them should be of shape
            `(batch_size, sequence_length, input_feature_count)`. Note that each of the
            three matrices may have different feature counts, although they will all
            have the same batch size and sequence length.

        Returns
        -------
        T.Tensor
            A tensor containing the result of the attention calculation. The tensor
            should be of shape
            `(batch_size, query_sequence_length, query_feature_count)`.
        """
        assert (
            self.input_feature_count == qkv.feature_count
        ), "QKV tensors must have the feature count equal to input feature count"

        result = attention(
            QKV(
                self.query_linear(qkv.query),
                self.key_linear(qkv.key),
                self.value_linear(qkv.value),
            ),
            mask=mask,
        )

        assert result.shape == (
            qkv.batch_size,
            get_sequence_length(qkv.query),
            self.qkv_feature_count,
        )

        return result
