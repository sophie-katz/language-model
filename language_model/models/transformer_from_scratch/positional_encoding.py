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

"""Positional encoding implementation for a transformer.

This code is based off notebooks/positional_encoding_from_scratch.ipynb.

https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch was
used to help with its implementation.
"""

import torch as T


def get_positional_encoding(
    sequence_length: int,
    embedding_size: int,
    base: float,
) -> T.Tensor:
    """Calculate positional encoding per Vaswani et. al. for use in transformers.

    Parameters
    ----------
    sequence_length : int
        The length of the sequence to calculate the positional encoding for.
    embedding_size : int
        The size of the embedding to calculate the positional encoding for.
    base : float
        The base of the exponent to use in the positional encoding calculation.

    Returns
    -------
    T.Tensor
        A tensor of shape (sequence_length, embedding_size) containing the positional
        encoding matrix for the given sequence length and embedding size.
    """
    assert sequence_length > 0, "sequences must be non-empty"
    assert embedding_size > 0, "embeddings must have at least 1 feature"

    exponent = (
        T.repeat_interleave(
            2 * T.arange(0, embedding_size // 2), 2, output_size=embedding_size
        )
        / embedding_size
    )

    assert exponent.shape == (
        embedding_size,
    ), "exponent shape is unexpected, it should be a vector of the same size as the \
        embedding"

    phase = T.arange(sequence_length).unsqueeze(1) / base**exponent

    assert phase.shape == (
        sequence_length,
        embedding_size,
    ), "phase shape is unexpected, it should be a matrix of shape (sequence_length, \
        embedding_size)"

    encoding = T.where(T.arange(embedding_size) % 2 == 0, T.sin(phase), T.cos(phase))

    assert encoding.shape == (
        sequence_length,
        embedding_size,
    ), "encoding shape is unexpected, it should be a matrix of shape (sequence_length, \
        embedding_size)"

    return encoding
