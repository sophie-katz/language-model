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

"""Transformer model.

This is heavily inspired by
https://medium.com/the-dl/transformers-from-scratch-in-pytorch-8777e346ca51.

https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch was
used to help with its implementation.
"""

from collections.abc import Iterable

import torch as T
from torch import nn
import torch.nn.functional as F

from language_model.models.transformer_from_scratch.decoder import Decoder
from language_model.models.transformer_from_scratch.encoder import Encoder


class Transformer(nn.Module):
    """Transformer model.

    This is heavily inspired by
    https://medium.com/the-dl/transformers-from-scratch-in-pytorch-8777e346ca51.

    https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch was
    used to help with its implementation.

    Attributes
    ----------
    encoder_layer_count : int
        The number of encoder block layers to use.
    decoder_layer_count : int
        The number of decoder block layers to use.
    input_size : int
        The size of the input tensor.
    head_count : int
        The number of attention heads to use.
    feed_forward_hidden_size : int
        The size of the hidden layer in the feed forward layer.
    dropout_rate : float
        The dropout rate for the residual layers.
    positional_encoding_base : float
        The exponentiation base to use for generating the positional encoding matrix.
    vocabulary_size : int
        The number of different words we expect to find in our input.
    embedding_size : int
        The size of the embedding vector for a given word.
    max_sequence_length : int
        The maximum length of a sequence.
    encoder : Encoder
        The encoder part of the model.
    decoder : Decoder
        The decoder part of the model.
    """

    def __init__(
        self,
        encoder_layer_count: int,
        decoder_layer_count: int,
        word_embedding_vocabulary_size: int,
        word_embedding_feature_count: int,
        positional_encoding_max_sequence_length: int,
        positional_encoding_base: float,
        encoder_block_head_count: int,
        encoder_block_feed_forward_hidden_feature_count: int,
        encoder_block_residual_dropout_rate: float,
        decoder_block_head_count: int,
        decoder_block_feed_forward_hidden_feature_count: int,
        decoder_block_residual_dropout_rate: float,
    ) -> None:
        super().__init__()

        self.encoder_layer_count = encoder_layer_count
        self.decoder_layer_count = decoder_layer_count
        self.word_embedding_vocabulary_size = word_embedding_vocabulary_size
        self.word_embedding_feature_count = word_embedding_feature_count
        self.positional_encoding_max_sequence_length = (
            positional_encoding_max_sequence_length
        )
        self.positional_encoding_base = positional_encoding_base
        self.encoder_block_head_count = encoder_block_head_count
        self.encoder_block_feed_forward_hidden_feature_count = (
            encoder_block_feed_forward_hidden_feature_count
        )
        self.encoder_block_residual_dropout_rate = encoder_block_residual_dropout_rate
        self.decoder_block_head_count = decoder_block_head_count
        self.decoder_block_feed_forward_hidden_feature_count = (
            decoder_block_feed_forward_hidden_feature_count
        )
        self.decoder_block_residual_dropout_rate = decoder_block_residual_dropout_rate

        # fmt: off
        self.encoder = Encoder(
            layer_count=self.encoder_layer_count,
            word_embedding_vocabulary_size=self.word_embedding_vocabulary_size,
            word_embedding_feature_count=self.word_embedding_feature_count,
            positional_encoding_max_sequence_length=(
                self.positional_encoding_max_sequence_length
            ),
            positional_encoding_base=self.positional_encoding_base,
            encoder_block_head_count=self.encoder_block_head_count,
            encoder_block_feed_forward_hidden_feature_count=(
                self.encoder_block_feed_forward_hidden_feature_count
            ),
            encoder_block_residual_dropout_rate=(
                self.encoder_block_residual_dropout_rate
            ),
        )
        # fmt: on

        self.decoder = Decoder(
            layer_count=self.decoder_layer_count,
            word_embedding_vocabulary_size=self.word_embedding_vocabulary_size,
            word_embedding_feature_count=self.word_embedding_feature_count,
            positional_encoding_max_sequence_length=(
                self.positional_encoding_max_sequence_length
            ),
            positional_encoding_base=self.positional_encoding_base,
            decoder_block_head_count=self.decoder_block_head_count,
            decoder_block_feed_forward_hidden_feature_count=(
                self.decoder_block_feed_forward_hidden_feature_count
            ),
            decoder_block_residual_dropout_rate=(
                self.decoder_block_residual_dropout_rate
            ),
        )

    def forward(self, source: T.Tensor, target: T.Tensor) -> T.Tensor:
        """Forward function for network.

        Parameters
        ----------
        source : T.Tensor
            The source tensor.
        target : T.Tensor
            The target tensor.

        Returns
        -------
        T.Tensor
            A single tensor. TODO: Find the size of this.
        """
        memory: T.Tensor = self.encoder(source)
        # `target` is of shape (batch_size, sequence_length), dtype long, and is a batch
        # of word index vectors.
        #
        # `memory` is of shape (batch_size, sequence_length,
        # word_embedding_feature_count), dtype float32, and is a batch of word embedding
        # matrices.
        result: T.Tensor = self.decoder(target, memory)
        return result

    def infer(self, prompt: T.Tensor, max_length: int) -> Iterable[T.Tensor]:
        training_original = self.training
        self.eval()

        with T.no_grad():
            encoded = self.encoder(prompt.unsqueeze(dim=0))
            output = T.zeros(max_length, dtype=T.long)

            for word_index in range(1, max_length):
                decoded = self.decoder(output[:word_index].unsqueeze(dim=0), encoded)
                output_word = decoded[:, :, -1].squeeze(dim=(0, 1)).argmax(dim=-1)
                assert output_word.ndim == 0
                yield output_word.item()
                output[word_index] = output_word

        if training_original:
            self.train()
