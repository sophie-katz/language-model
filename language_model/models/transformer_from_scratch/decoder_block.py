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

"""The decoder block from a transformer.

This is heavily inspired by
https://medium.com/the-dl/transformers-from-scratch-in-pytorch-8777e346ca51.

https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch
was used to help with its implementation.
"""

import dataclasses

import torch as T

from language_model.models.transformer_from_scratch.multi_head_attention import (
    MultiHeadAttention,
)
from language_model.models.transformer_from_scratch.qkv import QKV
from language_model.models.transformer_from_scratch.residual import Residual
from language_model.models.transformer_from_scratch.transformer_block import (
    TransformerBlock,
)


@dataclasses.dataclass
class DecoderBlock(TransformerBlock):
    """The decoder block from a transformer.

    This is heavily inspired by
    https://medium.com/the-dl/transformers-from-scratch-in-pytorch-8777e346ca51.

    https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch
    was used to help with its implementation.

    Attributes
    ----------
    self_attention : Residual[MultiHeadAttention]
        The self attention layer.
    """

    self_attention: Residual[MultiHeadAttention] = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """Postinitialization for Pytorch module."""
        super().__post_init__()

        self.self_attention = Residual(
            internal_layer=MultiHeadAttention(
                head_count=self.head_count,
                input_feature_count=self.input_feature_count,
                qkv_feature_count=self.qkv_feature_count,
            ),
            input_feature_count=self.input_feature_count,
            dropout_rate=self.residual_dropout_rate,
        )

    def forward(self, target: T.Tensor, memory: T.Tensor) -> T.Tensor:
        """Forward function for network.

        Parameters
        ----------
        target : T.Tensor
            The target tensor to be decoded into input-like data.
        memory : T.Tensor
            The memory tensor of the original input data.

        Returns
        -------
        T.Tensor
            A single tensor. TODO: Find the size of this.
        """
        result: T.Tensor = self.self_attention(QKV(target, target, target))
        result = self.attention(QKV(result, memory, memory))
        result = self.feed_forward(result)
        return result
