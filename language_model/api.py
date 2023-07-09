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

from collections.abc import Callable, Iterable
from typing import Optional

import torch as T
import torchtext.datasets
import torchtext.vocab
import torchtext.data.utils

from language_model.data.utils.wiki2_transformer import get_wiki2_transformer_datapipe
from language_model.models.transformer_from_scratch.transformer import Transformer


class LanguageModelAPIState(object):
    def __init__(
        self,
        vocabulary: torchtext.vocab.Vocab,
        transformer: Transformer,
        tokenizer: Callable[[str], Iterable[str]],
    ) -> None:
        self.vocabulary = vocabulary
        self.transformer = transformer
        self.tokenizer = tokenizer


class LanguageModelAPI(object):
    def __init__(self, inferred_max_length: int) -> None:
        assert inferred_max_length > 0

        self.inferred_max_length = inferred_max_length

        self._state: Optional[LanguageModelAPIState] = None

    def load(self) -> None:
        vocabulary = self._load_vocabulary()
        self._state = LanguageModelAPIState(
            vocabulary=vocabulary,
            transformer=self._load_transformer(len(vocabulary)),
            tokenizer=torchtext.data.utils.get_tokenizer("basic_english"),
        )

    def finish_sentence(self, prompt_text: str) -> Iterable[str]:
        if self._state is None:
            raise Exception("load() must be called before any other API calls are made")

        prompt_tokens = self._state.tokenizer(prompt_text)

        prompt_token_indices = T.tensor(
            [self._state.vocabulary.get_stoi()[token] for token in prompt_tokens],
            dtype=T.long,
        )

        for inferred_token_index in self._state.transformer.infer(
            prompt_token_indices, self.inferred_max_length
        ):
            yield self._state.vocabulary.get_itos()[inferred_token_index]

    def _load_vocabulary(self) -> torchtext.vocab.Vocab:
        train = torchtext.datasets.WikiText2(root=".data", split="train")
        vocabulary, _ = get_wiki2_transformer_datapipe(train)
        return vocabulary

    def _load_transformer(self, vocabulary_size: int) -> Transformer:
        # pylint: disable=too-many-locals
        encoder_layer_count = 3
        decoder_layer_count = 3
        token_embedding_feature_count = 512
        positional_encoding_max_sequence_length = 4096
        positional_encoding_base = 1e4
        encoder_block_head_count = 6
        encoder_block_feed_forward_hidden_feature_count = 8
        encoder_block_residual_dropout_rate = 0.1
        decoder_block_head_count = 6
        decoder_block_feed_forward_hidden_feature_count = 8
        decoder_block_residual_dropout_rate = 0.1

        return Transformer(
            encoder_layer_count=encoder_layer_count,
            decoder_layer_count=decoder_layer_count,
            token_embedding_vocabulary_size=vocabulary_size,
            token_embedding_feature_count=token_embedding_feature_count,
            positional_encoding_max_sequence_length=positional_encoding_max_sequence_length,
            positional_encoding_base=positional_encoding_base,
            encoder_block_head_count=encoder_block_head_count,
            encoder_block_feed_forward_hidden_feature_count=(
                encoder_block_feed_forward_hidden_feature_count
            ),
            encoder_block_residual_dropout_rate=encoder_block_residual_dropout_rate,
            decoder_block_head_count=decoder_block_head_count,
            decoder_block_feed_forward_hidden_feature_count=(
                decoder_block_feed_forward_hidden_feature_count
            ),
            decoder_block_residual_dropout_rate=decoder_block_residual_dropout_rate,
        )
