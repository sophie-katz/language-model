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

"""The encoder block from a transformer.

This is heavily inspired by
https://medium.com/the-dl/transformers-from-scratch-in-pytorch-8777e346ca51.

https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch
was used to help with its implementation.
"""

import dataclasses

import torch as T

from language_model.models.transformer_from_scratch.qkv import QKV
from language_model.models.transformer_from_scratch.transformer_block import (
    TransformerBlock,
)


@dataclasses.dataclass(unsafe_hash=True)
class EncoderBlock(TransformerBlock):
    """The encoder block from a transformer.

    This is heavily inspired by
    https://medium.com/the-dl/transformers-from-scratch-in-pytorch-8777e346ca51.

    https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch
    was used to help with its implementation.
    """

    def forward(self, source: T.Tensor) -> T.Tensor:
        """Forward function for network.

        Parameters
        ----------
        source : T.Tensor
            The input source tensor to be encoded.

        Returns
        -------
        T.Tensor
            A single tensor. TODO: Find the size of this.
        """
        result: T.Tensor = self.attention(QKV(source, source, source))
        result = self.feed_forward(result)
        return result
