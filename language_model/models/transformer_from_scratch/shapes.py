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

import torch as T


def has_sequence_shape(tensor: T.Tensor) -> bool:
    return tensor.ndim == 3 and tensor.size(2) > 1


def get_sequence_batch_size(tensor: T.Tensor) -> int:
    assert has_sequence_shape(
        tensor
    ), "this function is only applicable to tensors with a sequence shape"

    return tensor.size(0)


def get_sequence_length(tensor: T.Tensor) -> int:
    assert has_sequence_shape(
        tensor
    ), "this function is only applicable to tensors with a sequence shape"

    return tensor.size(1)


def get_sequence_feature_count(tensor: T.Tensor) -> int:
    assert has_sequence_shape(
        tensor
    ), "this function is only applicable to tensors with a sequence shape"

    return tensor.size(2)
