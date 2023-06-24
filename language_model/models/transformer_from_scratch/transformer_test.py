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

# This is heavily inspired by
# https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch.

import torch as T
from .word_embedding import WordEmbedding
from .transformer import Transformer


def test_transformer_simple_forward() -> None:
    vocabulary_size = 11
    embedding_size = 8
    source_length = 13
    target_length = 17

    # TODO: Move embedding into transformer
    # TODO: Should there be one embedding for encoding the source and one for decoding
    #       the target?
    word_embedding = WordEmbedding(vocabulary_size, embedding_size)

    transformer = Transformer(input_size=embedding_size)

    source_indices = T.randint(vocabulary_size, (source_length,))

    target_indices = T.randint(vocabulary_size, (target_length,))

    source_embedded = word_embedding(source_indices)

    target_embedded = word_embedding(target_indices)

    output = transformer(source_embedded, target_embedded)

    print(output.shape)
