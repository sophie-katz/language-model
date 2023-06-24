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

# This code is based off notebooks/attention_from_scratch.ipynb.
#
# https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch was
# used to help with its implementation.

import torch as T
import torch.nn.functional as F


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

    # This code is based on the code from
    # https://medium.com/the-dl/transformers-from-scratch-in-pytorch-8777e346ca51

    assert (
        query.ndim == 3
    ), "query tensor must be of shape (batch_size, query_sequence_length, feature_count)"
    assert (
        key.ndim == 3
    ), "key tensor must be of shape (batch_size, query_sequence_length, feature_count)"
    assert (
        value.ndim == 3
    ), "value tensor must be of shape (batch_size, query_sequence_length, feature_count)"

    batch_size = query.size(0)
    feature_count = query.size(2)
    query_sequence_length = query.size(1)
    key_sequence_length = key.size(1)
    value_sequence_length = value.size(1)

    assert (
        batch_size == key.size(0) == value.size(0)
    ), "all tensors must have the same batch size"
    assert (
        feature_count == key.size(2) == value.size(2)
    ), "all tensors must have the same feature count"

    score = query.bmm(key.transpose(1, 2))

    assert score.shape == (batch_size, query_sequence_length, key_sequence_length)

    # TODO: Apply mask

    weight = F.softmax(score / feature_count**0.5)

    assert weight.shape == score.shape

    result = weight.bmm(value)

    assert result.shape == (batch_size, query_sequence_length, feature_count)

    return result
