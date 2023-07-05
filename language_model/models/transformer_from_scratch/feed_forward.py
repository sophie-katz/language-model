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

"""Feed forward network module for use in a transformer.

This is heavily inspired by
https://medium.com/the-dl/transformers-from-scratch-in-pytorch-8777e346ca51.
"""

from torch import nn


class FeedForward(nn.Sequential):
    """Feed forward network module for use in a transformer.

    This is heavily inspired by
    https://medium.com/the-dl/transformers-from-scratch-in-pytorch-8777e346ca51.

    Attributes
    ----------

    input_size : int
        The size of the input tensor.
    feed_forward_hidden_size : int
        The size of the hidden layer.
    """

    def __init__(
        self, input_feature_count: int, feed_forward_hidden_feature_count: int
    ) -> None:
        super().__init__(
            nn.Linear(input_feature_count, feed_forward_hidden_feature_count),
            nn.ReLU(),
            nn.Linear(feed_forward_hidden_feature_count, input_feature_count),
        )

        self.input_feature_count = input_feature_count
        self.feed_forward_hidden_feature_count = feed_forward_hidden_feature_count
