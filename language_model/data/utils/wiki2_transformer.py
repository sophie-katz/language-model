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

"""Create data pipeline for Wiki-2 corpus in use by transformers."""

from collections.abc import Iterable
import torchtext.vocab
from language_model.data.data_pipelines.build_vocabulary_from_tokens import (
    BuildVocabularyFromTokens,
)
from language_model.data.data_pipelines.filter_tokens import FilterTokens

from language_model.data.data_pipelines.simple_space_split import SimpleSpaceSplit
from language_model.data.data_pipelines.split_sentences_by_index import (
    SplitSentencesByIndex,
)


def get_wiki2_transformer_datapipe(
    strings: Iterable[str],
) -> tuple[torchtext.vocab.Vocab, Iterable[list[list[int]]]]:
    """Create data pipeline for Wiki-2 corpus in use by transformers."""
    datapipe_tokens_unfiltered = SimpleSpaceSplit(strings)
    datapipe_tokens_filtered = FilterTokens(datapipe_tokens_unfiltered, {"@-@"})
    datapipe_indices = BuildVocabularyFromTokens(
        datapipe_tokens_filtered, specials=["<unk>"]
    )
    datapipe_sentences = SplitSentencesByIndex(
        datapipe_indices, datapipe_indices.vocabulary["."]
    )

    return datapipe_indices.vocabulary, datapipe_sentences
