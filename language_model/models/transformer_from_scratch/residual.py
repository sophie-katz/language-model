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
from typing import Generic, TypeVar

import torch as T
from torch import nn

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
    input_size : int
        The size of the input tensor.
    dropout_rate : float
        The dropout rate for the residual layers.
    """

    internal_layer: InternalLayer
    input_size: int
    dropout_rate: float

    def __post_init__(self) -> None:
        """Postinitialization for Pytorch module."""
        super().__init__()

        self.normalization = nn.LayerNorm(self.input_size)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, *tensors: T.Tensor) -> T.Tensor:
        """Forward function for network.

        Parameters
        ----------
        *tensors : T.Tensor
            The list of tensors to pass as input to the residual.

        Returns
        -------
        T.Tensor
            A single tensor. TODO: Find the size of this.
        """
        # assert len(tensors) > 0
        # assert tensors[0].ndim == 2
        # assert tensors[0].size(-1) == self.input_size
        # assert tensors[1].ndim == 2
        # assert tensors[2].ndim == 2

        result: T.Tensor = self.internal_layer(*tensors)
        result = self.dropout(result)
        result += tensors[0]
        result = self.normalization(result)
        return result
