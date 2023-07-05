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

from language_model.models.transformer_from_scratch.feed_forward import FeedForward


def test_feed_forward_simple() -> None:
    """Test initialization and shape of feed forward layer of transformer."""
    batch_size = 2
    input_feature_count = 4
    feed_forward_hidden_feature_count = 6

    feed_forward = FeedForward(
        input_feature_count=input_feature_count,
        feed_forward_hidden_feature_count=feed_forward_hidden_feature_count,
    )

    input_tensor = T.rand(batch_size, input_feature_count)

    result = feed_forward(input_tensor)

    assert result.shape == (batch_size, input_feature_count)


def test_feed_forward_parameters() -> None:
    """Test initialization and shape of feed forward layer of transformer."""
    input_feature_count = 4
    feed_forward_hidden_feature_count = 6

    feed_forward = FeedForward(
        input_feature_count=input_feature_count,
        feed_forward_hidden_feature_count=feed_forward_hidden_feature_count,
    )

    # 2 weights and 2 biaes for the 2 linear layers
    parameters = list(feed_forward.parameters())
    assert len(parameters) == 4
    assert parameters[0].shape == (
        feed_forward_hidden_feature_count,
        input_feature_count,
    )
    assert parameters[1].shape == (feed_forward_hidden_feature_count,)
    assert parameters[2].shape == (
        input_feature_count,
        feed_forward_hidden_feature_count,
    )
    assert parameters[3].shape == (input_feature_count,)


@pytest.mark.skipif(not T.cuda.is_available(), reason="CUDA not available")
def test_feed_forward_cuda() -> None:
    """CUDA."""
    batch_size = 2
    input_feature_count = 4
    feed_forward_hidden_feature_count = 6

    feed_forward = FeedForward(
        input_feature_count=input_feature_count,
        feed_forward_hidden_feature_count=feed_forward_hidden_feature_count,
    ).cuda()

    input_tensor = T.rand(batch_size, input_feature_count).cuda()

    result = feed_forward(input_tensor)

    assert result.device.type == "cuda"
