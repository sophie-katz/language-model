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

# pylint: disable=magic-value-comparison

from language_model.data.utils.wiki2_transformer import get_wiki2_transformer_datapipe


def test_simple() -> None:
    """Test the simplest case."""
    _, datapipe = get_wiki2_transformer_datapipe(["a b .", "c d . ef ."])

    assert [list(example) for example in datapipe] == [[[2, 3]], [[4, 5], [6]]]
