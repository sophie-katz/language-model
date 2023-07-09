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

import lightning as L
import torch as T
import torch.utils.data
from torchtext.datasets import WikiText2

from language_model.data.utils.wiki2_transformer import get_wiki2_transformer_datapipe
from language_model.models.transformer_from_scratch.transformer_module import (
    TransformerModule,
)


def test_transformer_module_optimizer_smoke() -> None:
    """Simple."""
    transformer_module = TransformerModule(13, 12)

    transformer_module.configure_optimizers()


def test_transformer_module_training_step() -> None:
    """Simple."""
    transformer_module = TransformerModule(13, 12)

    loss = transformer_module.training_step(T.tensor([[1, 4, 8]]), 0)

    assert isinstance(loss, T.Tensor)
    assert loss.ndim == 0
    assert loss.item() > 0


def test_integration() -> None:
    """Simple."""
    train = WikiText2(root=".data", split="train")

    vocabulary, train_datapipe = get_wiki2_transformer_datapipe(train)
    vocabulary.set_default_index(vocabulary["<unk>"])

    train_dataloader = torch.utils.data.DataLoader(train_datapipe)  # type: ignore

    transformer_module = TransformerModule(
        len(vocabulary), vocabulary.get_stoi()["<eos>"]
    )

    trainer = L.Trainer(max_steps=10)

    trainer.fit(transformer_module, train_dataloader)
