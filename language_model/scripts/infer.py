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

from collections.abc import Iterable

# import torch as T
# from torchtext.datasets import WikiText2
# from torchtext.data.utils import get_tokenizer
import gradio as gr

# from language_model.data.utils.wiki2_transformer import get_wiki2_transformer_datapipe
# from language_model.models.transformer_from_scratch.transformer import Transformer

from language_model.api import LanguageModelAPI


if __name__ == "__main__":
    # # pylint: disable=too-many-locals

    # # Get vocabulary
    # train = WikiText2(root=".data", split=("train"))

    # vocabulary, _ = get_wiki2_transformer_datapipe(train)

    # # Create transformer
    # input_sequence_length = 5
    # encoder_layer_count = 3
    # decoder_layer_count = 3
    # word_embedding_vocabulary_size = len(vocabulary)
    # word_embedding_feature_count = 512
    # positional_encoding_max_sequence_length = 4096
    # positional_encoding_base = 1e4
    # encoder_block_head_count = 6
    # encoder_block_feed_forward_hidden_feature_count = 8
    # encoder_block_residual_dropout_rate = 0.1
    # decoder_block_head_count = 6
    # decoder_block_feed_forward_hidden_feature_count = 8
    # decoder_block_residual_dropout_rate = 0.1
    # inferred_max_length = 50

    # transformer = Transformer(
    #     encoder_layer_count=encoder_layer_count,
    #     decoder_layer_count=decoder_layer_count,
    #     word_embedding_vocabulary_size=word_embedding_vocabulary_size,
    #     word_embedding_feature_count=word_embedding_feature_count,
    #     positional_encoding_max_sequence_length=positional_encoding_max_sequence_length,
    #     positional_encoding_base=positional_encoding_base,
    #     encoder_block_head_count=encoder_block_head_count,
    #     encoder_block_feed_forward_hidden_feature_count=(
    #         encoder_block_feed_forward_hidden_feature_count
    #     ),
    #     encoder_block_residual_dropout_rate=encoder_block_residual_dropout_rate,
    #     decoder_block_head_count=decoder_block_head_count,
    #     decoder_block_feed_forward_hidden_feature_count=(
    #         decoder_block_feed_forward_hidden_feature_count
    #     ),
    #     decoder_block_residual_dropout_rate=decoder_block_residual_dropout_rate,
    # )

    # # Create tokenizer
    # tokenizer = get_tokenizer("basic_english")

    # # Tokenize and vectorize source
    # source_text = "The gorilla can"
    # source_tokens = tokenizer(source_text)
    # source_word_indices = T.tensor(
    #     [vocabulary.get_stoi()[token] for token in source_tokens], dtype=T.long
    # )

    # # Infer
    # for inferred_word_index in transformer.infer(
    #     source_word_indices, inferred_max_length
    # ):
    #     inferred_word_text = vocabulary.get_itos()[inferred_word_index]
    #     print(inferred_word_text, end=" ")

    # print()

    api = LanguageModelAPI(50)

    print("Loading API...")
    api.load()

    def finish_sentence_accumulated(prompt_text: str) -> Iterable[str]:
        accumulator = ""

        for token in api.finish_sentence(prompt_text):
            if len(accumulator) > 0:
                accumulator += " "

            accumulator += token

            yield accumulator

    interface = gr.Interface(
        finish_sentence_accumulated,
        inputs=gr.Textbox(
            lines=4, placeholder="Text to be completed...", label="Prompt text"
        ),
        outputs=gr.Textbox(lines=32, label="Generated text"),
    )

    interface.queue().launch()
