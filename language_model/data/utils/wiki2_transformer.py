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

from collections.abc import Iterable, Iterator
from typing import Optional

import torch as T
import torchtext.vocab
import torchdata.datapipes as dp

from language_model.data.data_pipelines.apply_vocabulary_to_tokens import (
    ApplyVocabularyToTokens,
)
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
    vocabulary: Optional[torchtext.vocab.Vocab] = None,
) -> tuple[torchtext.vocab.Vocab, Iterator[list[list[int]]]]:
    """Create data pipeline for Wiki-2 corpus in use by transformers."""
    # pylint: disable=consider-ternary-expression

    datapipe_tokens_unfiltered = SimpleSpaceSplit(strings)
    datapipe_tokens_filtered = FilterTokens(datapipe_tokens_unfiltered, {"@-@"})

    if vocabulary is None:
        datapipe_indices = BuildVocabularyFromTokens(
            datapipe_tokens_filtered, specials=["<unk>"]
        )

        vocabulary = datapipe_indices.vocabulary
    else:
        datapipe_indices = ApplyVocabularyToTokens(datapipe_tokens_filtered, vocabulary)

    datapipe_sentences_for_examples = SplitSentencesByIndex(
        datapipe_indices, vocabulary["."]
    )

    datapipe_sentences = dp.iter.FlatMapper(datapipe_sentences_for_examples)

    datapipe_sentence_tensors = dp.iter.Mapper(
        datapipe_sentences, lambda x: T.tensor(x, dtype=T.long)
    )

    return vocabulary, datapipe_sentence_tensors
