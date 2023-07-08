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

import pytest
import torch as T
from torch import nn

from language_model.models.transformer_from_scratch.residual import Residual


def test_residual_simple() -> None:
    """Test initialization and shape of feed forward layer of transformer."""
    batch_size = 2
    input_feature_count = 4
    dropout_rate = 0.1

    residual: Residual = Residual(
        internal_layer=nn.Linear(input_feature_count, input_feature_count),
        input_feature_count=input_feature_count,
        dropout_rate=dropout_rate,
    )

    input_tensor = T.rand(batch_size, input_feature_count)

    result = residual(input_tensor)

    assert result.shape == (batch_size, input_feature_count)


def test_residual_parameters() -> None:
    """Test initialization and shape of feed forward layer of transformer."""
    input_feature_count = 4
    output_feature_count = 3
    dropout_rate = 0.1

    residual: Residual = Residual(
        internal_layer=nn.Linear(input_feature_count, output_feature_count),
        input_feature_count=input_feature_count,
        dropout_rate=dropout_rate,
    )

    # 1 weight and 1 bias for the linear layer, then 1 weight and 1 bias for the layer
    # normalization
    parameters = list(residual.parameters())
    assert len(parameters) == 4
    assert parameters[0].shape == (output_feature_count, input_feature_count)
    assert parameters[1].shape == (output_feature_count,)
    assert parameters[2].shape == (input_feature_count,)
    assert parameters[3].shape == (input_feature_count,)


@pytest.mark.skipif(not T.cuda.is_available(), reason="CUDA not available")
def test_residual_cuda() -> None:
    """Test initialization and shape of feed forward layer of transformer."""
    batch_size = 2
    input_feature_count = 4
    dropout_rate = 0.1

    residual: Residual = Residual(
        internal_layer=nn.Linear(input_feature_count, input_feature_count),
        input_feature_count=input_feature_count,
        dropout_rate=dropout_rate,
    ).cuda()

    input_tensor = T.rand(batch_size, input_feature_count).cuda()

    result = residual(input_tensor)

    assert result.device.type == "cuda"
