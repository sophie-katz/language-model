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


import torch as T
from torch import nn

from language_model.models.transformer_from_scratch.decoder_block import DecoderBlock
from language_model.models.transformer_from_scratch.shapes import (
    get_sequence_batch_size,
    get_sequence_length,
)
from language_model.models.transformer_from_scratch.transformer_pass import (
    TransformerPass,
)


class Decoder(nn.Module):
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

    def __init__(
        self,
        layer_count: int,
        token_embedding_vocabulary_size: int,
        token_embedding_feature_count: int,
        positional_encoding_max_sequence_length: int,
        positional_encoding_base: float,
        decoder_block_head_count: int,
        decoder_block_feed_forward_hidden_feature_count: int,
        decoder_block_residual_dropout_rate: float,
    ) -> None:
        # fmt: off
        super().__init__(
        )
        # fmt: on

        self.layer_count = layer_count
        self.token_embedding_vocabulary_size = token_embedding_vocabulary_size
        self.token_embedding_feature_count = token_embedding_feature_count
        self.positional_encoding_max_sequence_length = (
            positional_encoding_max_sequence_length
        )
        self.positional_encoding_base = positional_encoding_base
        self.token_embedding_vocabulary_size = token_embedding_vocabulary_size
        self.decoder_block_head_count = decoder_block_head_count
        self.decoder_block_feed_forward_hidden_feature_count = (
            decoder_block_feed_forward_hidden_feature_count
        )
        self.decoder_block_residual_dropout_rate = decoder_block_residual_dropout_rate

        self.transformer_pass = TransformerPass(
            layer_count=layer_count,
            token_embedding_vocabulary_size=token_embedding_vocabulary_size,
            token_embedding_feature_count=token_embedding_feature_count,
            positional_encoding_max_sequence_length=positional_encoding_max_sequence_length,
            positional_encoding_base=positional_encoding_base,
        )

        # fmt: off
        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    input_feature_count=token_embedding_feature_count,
                    head_count=decoder_block_head_count,
                    feed_forward_hidden_feature_count=(
                        decoder_block_feed_forward_hidden_feature_count
                    ),
                    residual_dropout_rate=decoder_block_residual_dropout_rate,
                )
                for _ in range(layer_count)
            ]
        )
        # fmt: on

        self.linear = nn.Linear(
            token_embedding_feature_count, token_embedding_vocabulary_size
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
        # target = self.token_embedding(target)

        # # TODO: Possibly scale up embedding -
        # # https://www.notion.so/Confirm-if-embedding-should-be-scaled-up-55f74b736e724bf0b40788873a9235ed?pvs=4
        # # target *= self.input_size ** 0.5

        # target += self.positional_encoding[..., : target.size(1), :].to(target.device)

        target = self.transformer_pass(target)

        mask = T.tril(
            T.ones(get_sequence_length(target), get_sequence_length(target))
        ).to(target.device)

        for layer in self.layers:
            target = layer(target, memory, mask=mask)

        result: T.Tensor = self.linear(target)

        assert result.shape == (
            get_sequence_batch_size(target),
            get_sequence_length(target),
            self.token_embedding_vocabulary_size,
        )

        result = T.softmax(result, dim=-1)

        assert result.shape == (
            get_sequence_batch_size(target),
            get_sequence_length(target),
            self.token_embedding_vocabulary_size,
        )

        return result
