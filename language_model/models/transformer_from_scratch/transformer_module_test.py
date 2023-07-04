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

import torch as T

from language_model.models.transformer_from_scratch.transformer_module import (
    TransformerModule,
)


def test_transformer_module_optimizer_smoke() -> None:
    """Simple."""
    transformer_module = TransformerModule(13)

    transformer_module.configure_optimizers()


def test_transformer_module_training_step() -> None:
    """Simple."""
    transformer_module = TransformerModule(13)

    loss = transformer_module.training_step([[1, 4, 8]], 0)

    assert isinstance(loss, T.Tensor)
    assert loss.shape == (1,)
    assert loss.item() > 0
