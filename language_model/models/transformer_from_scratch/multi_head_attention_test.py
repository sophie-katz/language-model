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

import pytest
import torch as T

from language_model.models.transformer_from_scratch.multi_head_attention import (
    MultiHeadAttention,
)
from language_model.models.transformer_from_scratch.qkv import QKV


def test_multi_head_attention_simple() -> None:
    """Test initialization and shapes for multi-head attention in a simple case."""
    batch_size = 2
    input_sequence_length = 4
    input_feature_count = 3
    head_count = 6
    qkv_feature_count = 7

    query = T.rand(batch_size, input_sequence_length, input_feature_count)
    key = T.rand(batch_size, input_sequence_length, input_feature_count)
    value = T.rand(batch_size, input_sequence_length, input_feature_count)

    multi_head_attention = MultiHeadAttention(
        head_count=head_count,
        input_feature_count=input_feature_count,
        qkv_feature_count=qkv_feature_count,
    )

    result = multi_head_attention(QKV(query=query, key=key, value=value))

    assert result.shape == (batch_size, input_sequence_length, input_feature_count)


def test_multi_head_attention_parameters() -> None:
    """Test initialization and shapes for multi-head attention in a simple case."""
    input_feature_count = 3
    head_count = 6
    qkv_feature_count = 7

    multi_head_attention = MultiHeadAttention(
        head_count=head_count,
        input_feature_count=input_feature_count,
        qkv_feature_count=qkv_feature_count,
    )

    parameters = list(multi_head_attention.parameters())
    assert len(parameters) == 38


@pytest.mark.skipif(not T.cuda.is_available(), reason="CUDA not available")
def test_multi_head_attention_cuda() -> None:
    """Test initialization and shapes for multi-head attention in a simple case."""
    batch_size = 2
    input_sequence_length = 4
    input_feature_count = 3
    head_count = 6
    qkv_feature_count = 7

    query = T.rand(batch_size, input_sequence_length, input_feature_count).cuda()
    key = T.rand(batch_size, input_sequence_length, input_feature_count).cuda()
    value = T.rand(batch_size, input_sequence_length, input_feature_count).cuda()

    multi_head_attention = MultiHeadAttention(
        head_count=head_count,
        input_feature_count=input_feature_count,
        qkv_feature_count=qkv_feature_count,
    ).cuda()

    result = multi_head_attention(QKV(query=query, key=key, value=value))

    assert result.device.type == "cuda"
