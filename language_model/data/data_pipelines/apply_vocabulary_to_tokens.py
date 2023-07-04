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

"""A data pipeline filters out unwanted tokens."""

from collections.abc import Iterator, Iterable

import torchdata.datapipes as dp
import torchtext.vocab


# pylint: disable-next=abstract-method
class ApplyVocabularyToTokens(dp.iter.IterDataPipe):  # type: ignore[misc]
    """A data pipeline that builds a vocabulary from tokens."""

    def __init__(
        self, datapipe: dp.iter.IterDataPipe, vocabulary: torchtext.vocab.Vocab
    ) -> None:
        """Initialize the data pipeline."""
        super().__init__()

        self.datapipe = datapipe
        self.vocabulary = vocabulary

    def __iter__(self) -> Iterator[Iterable[str]]:
        """Get the next example."""
        for example in self.datapipe:
            yield self.vocabulary.forward(
                example if isinstance(example, list) else list(example)
            )
