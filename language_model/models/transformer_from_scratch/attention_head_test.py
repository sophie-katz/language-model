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

from language_model.models.transformer_from_scratch.attention_head import AttentionHead
from language_model.models.transformer_from_scratch.qkv import QKV


def test_attention_head_simple() -> None:
    """Test initialization and shapes for attention head in a simple case."""
    batch_size = 2
    input_sequence_length = 4
    output_sequence_length = 5
    input_feature_count = 3
    qkv_feature_count = 6

    query = T.rand(batch_size, input_sequence_length, input_feature_count)
    key = T.rand(batch_size, output_sequence_length, input_feature_count)
    value = T.rand(batch_size, output_sequence_length, input_feature_count)

    attention_head = AttentionHead(
        input_feature_count=input_feature_count,
        qkv_feature_count=qkv_feature_count,
    )

    result = attention_head(QKV(query=query, key=key, value=value))

    assert result.shape == (batch_size, input_sequence_length, qkv_feature_count)
