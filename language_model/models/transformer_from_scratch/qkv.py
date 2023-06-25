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
import dataclasses
from .shapes import (
    has_sequence_shape,
    get_sequence_batch_size,
    get_sequence_feature_count,
)


@dataclasses.dataclass(frozen=True)
class QKV(object):
    query: T.Tensor
    key: T.Tensor
    value: T.Tensor

    def __post_init__(self) -> None:
        assert has_sequence_shape(
            self.query
        ), "query must be of shape (batch_size, sequence_length, feature_count)"

        assert has_sequence_shape(
            self.key
        ), "query must be of shape (batch_size, sequence_length, feature_count)"

        assert has_sequence_shape(
            self.value
        ), "query must be of shape (batch_size, sequence_length, feature_count)"

        assert (
            get_sequence_batch_size(self.query)
            == get_sequence_batch_size(self.key)
            == get_sequence_batch_size(self.value)
        ), "query, key, and value must have the same batch size"

        assert (
            get_sequence_feature_count(self.query)
            == get_sequence_feature_count(self.key)
            == get_sequence_feature_count(self.value)
        ), "query, key, and value must have the same feature_count"

    @property
    def batch_size(self) -> int:
        return get_sequence_batch_size(self.query)

    @property
    def feature_count(self) -> int:
        return get_sequence_feature_count(self.query)
