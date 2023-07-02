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

# pylint: disable=magic-value-comparison

"""Utility functions.

Utility functions For dealing with the shapes of sequence tensors that you expect to
find in a transformer.
"""

import torch as T


def has_sequence_shape(tensor: T.Tensor) -> bool:
    """Return whether or not the tensor has the shape of a sequence.

    The shape of a sequence tensor is `(batch_size, sequence_length, feature_count)`.

    Parameters
    ----------
    tensor : T.Tensor
        The tensor to check.

    Returns
    -------
    bool
        Whether or not the tensor has the shape of a sequence.
    """
    return tensor.ndim == 3 and tensor.size(2) > 0


def get_sequence_batch_size(tensor: T.Tensor) -> int:
    """Get the batch size of a sequence tensor.

    The shape of a sequence tensor is `(batch_size, sequence_length, feature_count)`.

    Parameters
    ----------
    tensor : T.Tensor
        The tensor to check.

    Returns
    -------
    int
        The batch size of the sequence tensor.
    """
    assert has_sequence_shape(
        tensor
    ), "this function is only applicable to tensors with a sequence shape"

    return tensor.size(0)


def get_sequence_length(tensor: T.Tensor) -> int:
    """Get the sequence length of a sequence tensor.

    The shape of a sequence tensor is `(batch_size, sequence_length, feature_count)`.

    Parameters
    ----------
    tensor : T.Tensor
        The tensor to check.

    Returns
    -------
    int
        The length of the sequence tensor.
    """
    assert has_sequence_shape(
        tensor
    ), "this function is only applicable to tensors with a sequence shape"

    return tensor.size(1)


def get_sequence_feature_count(tensor: T.Tensor) -> int:
    """Get the feature count of a sequence tensor.

    The shape of a sequence tensor is `(batch_size, sequence_length, feature_count)`.

    Parameters
    ----------
    tensor : T.Tensor
        The tensor to check.

    Returns
    -------
    int
        The feature count of the sequence tensor.
    """
    assert has_sequence_shape(
        tensor
    ), "this function is only applicable to tensors with a sequence shape"

    return tensor.size(2)
