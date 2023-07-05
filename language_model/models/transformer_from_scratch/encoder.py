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

"""The encoder from a transformer.

This is heavily inspired by
https://medium.com/the-dl/transformers-from-scratch-in-pytorch-8777e346ca51.

https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch
was used to help with its implementation.
"""

import dataclasses

import torch as T
from torch import nn

from language_model.models.transformer_from_scratch.encoder_block import EncoderBlock
from language_model.models.transformer_from_scratch.transformer_pass import (
    TransformerPass,
)


@dataclasses.dataclass(unsafe_hash=True)
class Encoder(TransformerPass):
    """The encoder from a transformer.

    This is heavily inspired by
    https://medium.com/the-dl/transformers-from-scratch-in-pytorch-8777e346ca51.

    https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch
    was used to help with its implementation.
    """

    encoder_block_head_count: int
    encoder_block_feed_forward_hidden_feature_count: int
    encoder_block_residual_dropout_rate: float

    layers: nn.ModuleList = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """Postinitialization for Pytorch module."""
        super().__post_init__()

        # fmt: off
        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    input_feature_count=self.word_embedding_feature_count,
                    head_count=self.encoder_block_head_count,
                    feed_forward_hidden_feature_count=(
                        self.encoder_block_feed_forward_hidden_feature_count
                    ),
                    residual_dropout_rate=self.encoder_block_residual_dropout_rate,
                )
                for _ in range(self.layer_count)
            ]
        )
        # fmt: on

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
        source = self.word_embedding(source)

        # TODO: Possibly scale up embedding -
        # https://www.notion.so/Confirm-if-embedding-should-be-scaled-up-55f74b736e724bf0b40788873a9235ed?pvs=4
        # source *= self.input_size ** 0.5

        source += self.positional_encoding[..., : source.size(1), :].to(source.device)

        for layer in self.layers:
            source = layer(source)

        return source
