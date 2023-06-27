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

# import torch as T
import math

from language_model.models.transformer_from_scratch.positional_encoding import (
    get_positional_encoding,
)


EPSILON = 1e-5


def test_positional_encoding_against_simpler_implementation() -> None:
    """Compare positional encoding algorithm used against a non-optimized version."""
    encoding = get_positional_encoding(1000, 512, 10000)

    assert encoding.shape == (1000, 512)

    for sequence_position in range(100):
        for embedding_index in range(100):
            exponent = (embedding_index - (embedding_index % 2)) / 512

            phase = sequence_position / (10000**exponent)

            encoding_calculated = (
                math.sin(phase) if embedding_index % 2 == 0 else math.cos(phase)
            )

            assert (
                abs(
                    encoding[sequence_position, embedding_index].item()
                    - encoding_calculated
                )
                < EPSILON
            )


def test_positional_encoding_against_expected_values() -> None:
    """Compare positional encoding algorithm used against expected values.

    Expected values taken from
    https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/
    """
    encoding = get_positional_encoding(4, 4, 100)

    assert encoding.shape == (4, 4)

    assert abs(encoding[0, 0].item() - 0.0) < EPSILON
    assert abs(encoding[0, 1].item() - 1.0) < EPSILON
    assert abs(encoding[0, 2].item() - 0.0) < EPSILON
    assert abs(encoding[0, 3].item() - 1.0) < EPSILON
    assert abs(encoding[1, 0].item() - 0.84147098) < EPSILON
    assert abs(encoding[1, 1].item() - 0.54030231) < EPSILON
    assert abs(encoding[1, 2].item() - 0.09983342) < EPSILON
    assert abs(encoding[1, 3].item() - 0.99500417) < EPSILON
    assert abs(encoding[2, 0].item() - 0.90929743) < EPSILON
    assert abs(encoding[2, 1].item() - -0.41614684) < EPSILON
    assert abs(encoding[2, 2].item() - 0.19866933) < EPSILON
    assert abs(encoding[2, 3].item() - 0.98006658) < EPSILON
    assert abs(encoding[3, 0].item() - 0.14112001) < EPSILON
    assert abs(encoding[3, 1].item() - -0.9899925) < EPSILON
    assert abs(encoding[3, 2].item() - 0.29552021) < EPSILON
    assert abs(encoding[3, 3].item() - 0.95533649) < EPSILON
