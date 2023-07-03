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

from collections.abc import Iterator

import torchdata.datapipes as dp


# pylint: disable-next=abstract-method
class SplitSentencesByIndex(dp.iter.IterDataPipe):  # type: ignore[misc]
    """A data pipeline that builds a vocabulary from tokens."""

    def __init__(self, datapipe: dp.iter.IterDataPipe, period_index: int) -> None:
        """Initialize the data pipeline."""
        super().__init__()

        self.datapipe = datapipe
        self.period_index = period_index

    def __iter__(self) -> Iterator[list[list[int]]]:
        """Get the next example."""
        for example in self.datapipe:
            sentences: list[list[int]] = []
            sentence: list[int] = []

            for index in example:
                if index == self.period_index:
                    if len(sentence) > 0:
                        sentences.append(sentence)
                        sentence = []
                else:
                    sentence.append(index)

            if len(sentence) > 0:
                sentences.append(sentence)

            yield sentences
