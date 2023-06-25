# Copyright (c) 2023 Sophie Katz
#
# This file is part of Language Model.
#
# Language Model is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later version.
#
# Language Model is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Language
# Model. If not, see <https://www.gnu.org/licenses/>.

# This code is based off notebooks/attention_from_scratch.ipynb and
# https://medium.com/the-dl/transformers-from-scratch-in-pytorch-8777e346ca51.
#
# https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch was
# used to help with its implementation.

import torch as T
import torch.nn.functional as F
from language_model.models.transformer_from_scratch.shapes import (
    get_sequence_batch_size,
    get_sequence_feature_count,
    get_sequence_length,
    has_sequence_shape,
)


def attention(query: T.Tensor, key: T.Tensor, value: T.Tensor) -> T.Tensor:
    """
    Compute attention for a single head.

    Arguments:
        query: T.Tensor
            The query tensor of shape `(batch_size, query_sequence_length, feature_count)`.
        key: T.Tensor
            The key tensor of shape `(batch_size, key_sequence_length, feature_count)`.
        value: T.Tensor
            The value tensor of shape `(batch_size, value_sequence_length, feature_count)`.

    Returns:
        T.Tensor

        A tensor containing the result of the attention calculation.
    """

    assert has_sequence_shape(
        query
    ), "query tensor must be of shape (batch_size, sequence_length, feature_count)"
    assert has_sequence_shape(
        key
    ), "key tensor must be of shape (batch_size, sequence_length, feature_count)"
    assert has_sequence_shape(
        value
    ), "value tensor must be of shape (batch_size, sequence_length, feature_count)"

    batch_size = get_sequence_batch_size(query)
    feature_count = get_sequence_feature_count(query)
    query_sequence_length = get_sequence_length(query)
    key_sequence_length = get_sequence_length(key)

    assert (
        get_sequence_batch_size(query)
        == get_sequence_batch_size(key)
        == get_sequence_batch_size(value)
    ), "all tensors must have the same batch size"

    assert (
        get_sequence_feature_count(query)
        == get_sequence_feature_count(key)
        == get_sequence_feature_count(value)
    ), "all tensors must have the same feature count equal to input size"

    score = query.bmm(key.transpose(1, 2))

    assert score.shape == (batch_size, query_sequence_length, key_sequence_length)

    # TODO: Apply mask - https://www.notion.so/Apply-mask-a0a22426e0a94a3aa7d49abc21075fb9?pvs=4

    weight = F.softmax(score / feature_count**0.5)

    assert weight.shape == score.shape

    result = weight.bmm(value)

    assert result.shape == (batch_size, query_sequence_length, feature_count)

    return result
