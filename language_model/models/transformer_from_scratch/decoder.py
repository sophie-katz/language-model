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

"""The decoder from a transformer.

This is heavily inspired by
https://medium.com/the-dl/transformers-from-scratch-in-pytorch-8777e346ca51.

https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch
was used to help with its implementation.
"""


import dataclasses

import torch as T
from torch import nn

from language_model.models.transformer_from_scratch.decoder_block import DecoderBlock
from language_model.models.transformer_from_scratch.positional_encoding import (
    get_positional_encoding,
)
from language_model.models.transformer_from_scratch.transformer_pass import (
    TransformerPass,
)


@dataclasses.dataclass
class Decoder(TransformerPass):
    """The decoder from a transformer.

    This is heavily inspired by
    https://medium.com/the-dl/transformers-from-scratch-in-pytorch-8777e346ca51.

    https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch
    was used to help with its implementation.

    Attributes
    ----------
    linear : nn.Linear
        The linear layer to increase dimensionality.
    """

    linear: nn.Linear = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """Postinitialization for Pytorch module."""
        self.linear = nn.Linear(self.input_size, self.input_size)

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
        sequence_length = target.size(1)
        input_size = target.size(2)

        # TODO: Possibly scale up embedding -
        # https://www.notion.so/Confirm-if-embedding-should-be-scaled-up-55f74b736e724bf0b40788873a9235ed?pvs=4
        # target *= self.input_size ** 0.5

        # TODO: Pull this out of the forward function -
        # https://www.notion.so/Pull-positional-encoding-out-of-the-forward-function-d034d6414f054f2aa163535c707182a3?pvs=4
        target += get_positional_encoding(
            sequence_length, input_size, self.positional_encoding_base
        )

        for layer in self.layers:
            target = layer(target, memory)

        return T.softmax(self.linear(target), dim=-1)
