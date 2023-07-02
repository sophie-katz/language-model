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

"""A residual module from a transformer.

This is heavily inspired by
https://medium.com/the-dl/transformers-from-scratch-in-pytorch-8777e346ca51.

https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch was
used to help with its implementation.
"""

import dataclasses
from typing import Generic, TypeVar, Union

import torch as T
from torch import nn

from language_model.models.transformer_from_scratch.qkv import QKV

InternalLayer = TypeVar("InternalLayer", bound=nn.Module)


@dataclasses.dataclass
class Residual(nn.Module, Generic[InternalLayer]):
    """A residual module from a transformer.

    This is heavily inspired by
    https://medium.com/the-dl/transformers-from-scratch-in-pytorch-8777e346ca51.

    https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch was
    used to help with its implementation.

    Attributes
    ----------
    internal_layer : InternalLayer
        The internal layer of the residual module.
    input_feature_count : int
        The size of the input tensor.
    dropout_rate : float
        The dropout rate for the residual layers.
    """

    internal_layer: dataclasses.InitVar[InternalLayer]
    input_feature_count: int
    dropout_rate: float

    def __post_init__(self, internal_layer: InternalLayer) -> None:
        """Postinitialization for Pytorch module."""
        super().__init__()

        self.internal_layer = internal_layer
        self.normalization = nn.LayerNorm(self.input_feature_count)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, *tensors: Union[T.Tensor, QKV]) -> T.Tensor:
        """Forward function for network.

        Parameters
        ----------
        *tensors : Union[T.Tensor, QKV]
            The list of tensors to pass as input to the residual.

        Returns
        -------
        T.Tensor
            A result tensor that is the same shape as the first input tensor.
        """
        # pylint: disable=magic-value-comparison

        assert isinstance(
            self.internal_layer, nn.Module
        ), "internal layer must be a Pytorch module"
        assert len(tensors) > 0, "residual module needs at least one tensor as input"

        result: T.Tensor

        if isinstance(tensors[0], QKV):
            assert len(tensors) == 1, "residual module needs exactly one QKV as input"

            qkv = tensors[0]
            assert (
                qkv.feature_count == self.input_feature_count
            ), f"query, key, and value tensors must all be of the expected input size:\
                    {self.input_feature_count}"

            result = self.internal_layer(qkv)

            assert (
                result.shape == qkv.query.shape
            ), "output of internal layer must be same shape as query"

            result = self.dropout(result)

            assert (
                result.shape == qkv.query.shape
            ), "output of dropout must be same shape as query"

            result += qkv.query
            result = self.normalization(result)

            assert (
                result.shape == qkv.query.shape
            ), "output of normalization must be same shape as query"
        else:
            assert tensors[0].ndim > 1, "all tensors in residual module must be batched"
            assert (
                tensors[0].size(-1) == self.input_feature_count
            ), f"first input tensor must be of expected input size: \
                {self.input_feature_count}"

            batch_size = tensors[0].size(0)

            assert all(
                isinstance(tensor, T.Tensor)
                and tensor.ndim > 1
                and tensor.size(0) == batch_size
                for tensor in tensors[1:]
            ), "all tensors must have the same batch size"

            result = self.internal_layer(*tensors)

            assert (
                result.shape == tensors[0].shape
            ), "output of internal layer must be same shape as input"

            result = self.dropout(result)

            assert (
                result.shape == tensors[0].shape
            ), "output of dropout must be same shape as input"

            result += tensors[0]
            result = self.normalization(result)

            assert (
                result.shape == tensors[0].shape
            ), "output of normalization must be same shape as input"

        return result
