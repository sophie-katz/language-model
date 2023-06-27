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


from language_model.models.transformer_from_scratch.attention_head import AttentionHead


def test_attention_head() -> None:
    """Test initialization and shapes for attention head.

    Batch size: 2
    Input sequence length: 4
    Output sequence length: 5
    """
    #     query = T.rand(2, 5, 3)
    #     key = T.rand(2, 4, 3)
    #     value = T.rand(2, 4, 3)

    AttentionHead(2, 2, 2, 2)


#     result = attention(query, key, value)

#     assert result.shape == (2, 5, 3)
