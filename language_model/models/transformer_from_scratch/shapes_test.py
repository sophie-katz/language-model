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

"""Unit tests."""

import torch as T

from language_model.models.transformer_from_scratch.shapes import (
    get_sequence_batch_size,
    get_sequence_length,
    get_sequence_feature_count,
    has_sequence_shape,
)


def test_has_sequence_shape() -> None:
    """Test function."""
    assert has_sequence_shape(T.zeros(1, 1, 2))
    assert not has_sequence_shape(T.zeros(1, 2))
    assert not has_sequence_shape(T.zeros(1, 1, 1, 2))
    assert not has_sequence_shape(T.zeros(1, 1, 1))


def test_get_sequence_batch_size() -> None:
    """Test function."""
    assert get_sequence_batch_size(T.zeros(1, 2, 3)) == 1


def test_get_sequence_length() -> None:
    """Test function."""
    assert get_sequence_length(T.zeros(1, 2, 3)) == 2


def test_get_sequence_feature_count() -> None:
    """Test function."""
    assert get_sequence_feature_count(T.zeros(1, 2, 3)) == 3
