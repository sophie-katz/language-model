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

from language_model.models.transformer_from_scratch.attention import attention
from language_model.models.transformer_from_scratch.qkv import QKV


def test_attention_simple() -> None:
    """Test attention in a simple case."""
    batch_size = 2
    input_sequence_length = 4
    output_sequence_length = 5
    feature_count = 3

    query = T.rand(batch_size, input_sequence_length, feature_count)
    key = T.rand(batch_size, output_sequence_length, feature_count)
    value = T.rand(batch_size, output_sequence_length, feature_count)

    result = attention(QKV(query, key, value))

    assert result.shape == (batch_size, input_sequence_length, feature_count)


def test_attention_shapes() -> None:
    """Test attention shapes in a more complicated case."""
    batch_size = 64
    query_sequence_length = 4
    key_and_value_sequence_length = 5
    feature_count = 512

    query = T.rand(batch_size, query_sequence_length, feature_count)
    key = T.rand(batch_size, key_and_value_sequence_length, feature_count)
    value = T.rand(batch_size, key_and_value_sequence_length, feature_count)

    result = attention(QKV(query, key, value))

    assert result.shape == (batch_size, query_sequence_length, feature_count)


def test_attention_mask_tril() -> None:
    """Test masking with attention."""
    batch_size = 64
    query_sequence_length = 4
    key_and_value_sequence_length = 5
    feature_count = 512

    query = T.rand(batch_size, query_sequence_length, feature_count)
    key = T.rand(batch_size, key_and_value_sequence_length, feature_count)
    value = T.rand(batch_size, key_and_value_sequence_length, feature_count)

    mask = T.tril(T.ones((query_sequence_length, key_and_value_sequence_length)))

    result_without_mask = attention(QKV(query, key, value))
    result_with_mask = attention(QKV(query, key, value), mask=mask)

    assert result_without_mask.shape == (
        batch_size,
        query_sequence_length,
        feature_count,
    )

    assert (result_with_mask != result_without_mask).any()


def test_attention_mask_ones() -> None:
    """Test masking with attention."""
    batch_size = 64
    query_sequence_length = 4
    key_and_value_sequence_length = 5
    feature_count = 512

    query = T.rand(batch_size, query_sequence_length, feature_count)
    key = T.rand(batch_size, key_and_value_sequence_length, feature_count)
    value = T.rand(batch_size, key_and_value_sequence_length, feature_count)

    mask = T.ones((query_sequence_length, key_and_value_sequence_length))

    result_without_mask = attention(QKV(query, key, value))
    result_with_mask = attention(QKV(query, key, value), mask=mask)

    assert result_without_mask.shape == (
        batch_size,
        query_sequence_length,
        feature_count,
    )

    assert not (result_with_mask != result_without_mask).any()
