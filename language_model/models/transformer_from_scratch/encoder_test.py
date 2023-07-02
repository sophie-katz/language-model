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

"""Unit tests."""

import torch as T

from language_model.models.transformer_from_scratch.encoder import Encoder


def test_encoder() -> None:
    """Test initialization and shape of encoder."""
    batch_size = 2
    input_sequence_length = 5
    layer_count = 3
    word_embedding_vocabulary_size = 13
    word_embedding_feature_count = 512
    positional_encoding_max_sequence_length = 4096
    positional_encoding_base = 1e4
    encoder_block_head_count = 6
    encoder_block_feed_forward_hidden_feature_count = 8
    encoder_block_residual_dropout_rate = 0.1

    # fmt: off
    encoder = Encoder(
        layer_count=layer_count,
        word_embedding_vocabulary_size=word_embedding_vocabulary_size,
        word_embedding_feature_count=word_embedding_feature_count,
        positional_encoding_max_sequence_length=positional_encoding_max_sequence_length,
        positional_encoding_base=positional_encoding_base,
        encoder_block_head_count=encoder_block_head_count,
        encoder_block_feed_forward_hidden_feature_count=(
            encoder_block_feed_forward_hidden_feature_count
        ),
        encoder_block_residual_dropout_rate=(
            encoder_block_residual_dropout_rate
        ),
    )
    # fmt: on

    source = T.randint(
        word_embedding_vocabulary_size, (batch_size, input_sequence_length)
    )

    result = encoder(source)

    assert result.shape == (
        batch_size,
        input_sequence_length,
        word_embedding_feature_count,
    )
