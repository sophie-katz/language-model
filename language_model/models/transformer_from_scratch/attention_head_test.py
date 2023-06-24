# Copyright (c) 2023 Sophie Katz
#
# This file is part of Language Model.
#
# Language Model is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later version.
#
# Language Model is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Language
# Model. If not, see <https://www.gnu.org/licenses/>.

import torch as T
from .attention_head import AttentionHead


# def test_attention_head() -> None:
#     # Batch size: 2
#     #
#     # Input sequence length: 4
#     # Output sequence length: 5

#     query = T.rand(2, 5, 3)
#     key = T.rand(2, 4, 3)
#     value = T.rand(2, 4, 3)

#     attention_head = AttentionHead()

#     result = attention(query, key, value)

#     assert result.shape == (2, 5, 3)
