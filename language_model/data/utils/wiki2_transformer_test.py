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

    data = list(datapipe)
    assert len(data) == 3
    assert T.equal(data[0], T.tensor([3, 4, 1]))
    assert T.equal(data[1], T.tensor([5, 6, 1]))
    assert T.equal(data[2], T.tensor([7, 1]))


def test_with_vocabulary() -> None:
    """Test the simplest case."""
    vocabulary = torchtext.vocab.build_vocab_from_iterator(
        [["a", "b", ".", "c", "ef"]], specials=["<unk>", "<eos>"]
    )

    vocabulary.set_default_index(vocabulary["<unk>"])

    _, datapipe = get_wiki2_transformer_datapipe(
        ["a b .", "c d . ef ."], vocabulary=vocabulary
    )

    data = list(datapipe)
    assert len(data) == 3
    assert T.equal(data[0], T.tensor([3, 4, 1]))
    assert T.equal(data[1], T.tensor([5, 0, 1]))
    assert T.equal(data[2], T.tensor([6, 1]))


def test_integration_without_dataloader() -> None:
    """Test the simplest case."""
    train = WikiText2(root=".data", split="train")

    vocabulary, train_datapipe = get_wiki2_transformer_datapipe(train)
    vocabulary.set_default_index(vocabulary["<unk>"])

    for batch_index, batch in enumerate(itertools.islice(train_datapipe, 5)):
        assert isinstance(batch, T.Tensor)
        assert batch[-1].item() == vocabulary.get_stoi()["<eos>"]

        if batch_index == 0:
            assert batch.shape == (6,)
        elif batch_index == 1:
            assert batch.shape == (14,)


def test_integration_with_dataloader() -> None:
    """Test the simplest case."""
    train = WikiText2(root=".data", split="train")

    vocabulary, train_datapipe = get_wiki2_transformer_datapipe(train)
    vocabulary.set_default_index(vocabulary["<unk>"])

    train_dataloader = torch.utils.data.DataLoader(train_datapipe)  # type: ignore

    for batch_index, batch in enumerate(itertools.islice(train_dataloader, 5)):
        assert isinstance(batch, T.Tensor)

        if batch_index == 0:
            assert batch.shape == (1, 6)
        elif batch_index == 1:
            assert batch.shape == (1, 14)


def test_performance_with_dataloader() -> None:
    """Test the simplest case."""
    train = WikiText2(root=".data", split="train")

    vocabulary, train_datapipe = get_wiki2_transformer_datapipe(train)
    vocabulary.set_default_index(vocabulary["<unk>"])

    train_dataloader = torch.utils.data.DataLoader(train_datapipe)  # type: ignore

    assert len(list(itertools.islice(train_dataloader, 100))) == 100
