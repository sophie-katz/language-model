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

"""Implementation of attention.

This code is based off notebooks/attention_from_scratch.ipynb and
https://medium.com/the-dl/transformers-from-scratch-in-pytorch-8777e346ca51.

https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch
was used to help with its implementation.
"""

from typing import Optional

import torch as T
import torch.nn.functional as F

from language_model.models.transformer_from_scratch.qkv import QKV
from language_model.models.transformer_from_scratch.shapes import (
    get_sequence_length,
)

# Doesn't need to be a hyperparameter because it's basically just -inf but finite
MASK_FILL_VALUE = -1e20


def attention(qkv: QKV, mask: Optional[T.Tensor] = None) -> T.Tensor:
    """Compute attention for a single head.

    Parameters
    ----------
    qkv : QKV
        A `QKV` object containing the query, key, and value tensors. These are what is
        used to calculate the attention matrix that is returned.

    Returns
    -------
    T.Tensor
        A tensor containing the result of the attention calculation. The tensor should
        be of shape `(batch_size, query_sequence_length, feature_count)`.
    """
    assert get_sequence_length(qkv.key) == get_sequence_length(
        qkv.value
    ), "key and value sequences must have the same length"

    score = qkv.query.bmm(qkv.key.transpose(1, 2))

    assert score.shape == (
        qkv.batch_size,
        get_sequence_length(qkv.query),
        get_sequence_length(qkv.key),
    ), "score shape is unexpected, it should be (batch_size, sequence_length, \
        sequence_length)"

    # TODO: Apply mask
    # - https://www.notion.so/Apply-mask-a0a22426e0a94a3aa7d49abc21075fb9?pvs=4

    if mask is not None:
        assert mask.shape == (
            get_sequence_length(qkv.query),
            get_sequence_length(qkv.key),
        ), "mask must have the same shape as the score matrix, except for the batch \
            dimension"

        score = score.masked_fill(mask == 0, MASK_FILL_VALUE)

    weight = F.softmax(score / qkv.feature_count**0.5)

    assert (
        weight.shape == score.shape
    ), "softmax should not change the shape of the \
        tensor"

    result = weight.bmm(qkv.value)

    assert result.shape == (
        qkv.batch_size,
        get_sequence_length(qkv.query),
        qkv.feature_count,
    ), "result shape is unexpected, it should be (batch_size, sequence_length, \
        feature_count)"

    return result
