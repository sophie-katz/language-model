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

import torchdata.datapipes as dp

from language_model.data.data_pipelines.simple_space_split import SimpleSpaceSplit


def test_simple() -> None:
    """Test the simplest case."""
    datapipe = dp.iter.IterableWrapper(["a", "ab", "a b c"])
    datapipe = SimpleSpaceSplit(datapipe)

    assert list(datapipe) == [["a"], ["ab"], ["a", "b", "c"]]


def test_double_space() -> None:
    """Test the simplest case."""
    datapipe = dp.iter.IterableWrapper(["a", "ab", "a b  c"])
    datapipe = SimpleSpaceSplit(datapipe)

    assert list(datapipe) == [["a"], ["ab"], ["a", "b", "c"]]


def test_empty() -> None:
    """Test the simplest case."""
    datapipe = dp.iter.IterableWrapper(["a", "ab", " "])
    datapipe = SimpleSpaceSplit(datapipe)

    assert list(datapipe) == [["a"], ["ab"]]
