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

import itertools

import torch as T
import torch.utils.data
import torchtext.vocab
from torchtext.datasets import WikiText2

from language_model.data.utils.wiki2_transformer import get_wiki2_transformer_datapipe


def test_simple() -> None:
    """Test the simplest case."""
    _, datapipe = get_wiki2_transformer_datapipe(["a b .", "c d . ef ."])

    assert list(datapipe) == [[2, 3], [4, 5], [6]]


def test_with_vocabulary() -> None:
    """Test the simplest case."""
    vocabulary = torchtext.vocab.build_vocab_from_iterator(
        [["a", "b", ".", "c", "ef"]], specials=["<unk>"]
    )

    vocabulary.set_default_index(vocabulary["<unk>"])

    _, datapipe = get_wiki2_transformer_datapipe(
        ["a b .", "c d . ef ."], vocabulary=vocabulary
    )

    assert list(datapipe) == [[2, 3], [4, 0], [5]]


def test_integration_without_dataloader() -> None:
    """Test the simplest case."""
    train = WikiText2(root=".data", split="train")

    vocabulary, train_datapipe = get_wiki2_transformer_datapipe(train)
    vocabulary.set_default_index(vocabulary["<unk>"])

    for batch_index, batch in enumerate(itertools.islice(train_datapipe, 5)):
        assert isinstance(batch, T.Tensor)

        if batch_index == 0:
            assert batch.shape == (5,)
        elif batch_index == 1:
            assert batch.shape == (13,)


def test_integration_with_dataloader() -> None:
    """Test the simplest case."""
    train = WikiText2(root=".data", split="train")

    vocabulary, train_datapipe = get_wiki2_transformer_datapipe(train)
    vocabulary.set_default_index(vocabulary["<unk>"])

    train_dataloader = torch.utils.data.DataLoader(train_datapipe)  # type: ignore

    for batch_index, batch in enumerate(itertools.islice(train_dataloader, 5)):
        assert isinstance(batch, T.Tensor)

        if batch_index == 0:
            assert batch.shape == (1, 5)
        elif batch_index == 1:
            assert batch.shape == (1, 13)
