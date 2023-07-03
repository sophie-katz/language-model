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


# pylint: disable-next=abstract-method
class FilterTokens(dp.iter.IterDataPipe):  # type: ignore[misc]
    """A data pipeline filters out unwanted tokens."""

    def __init__(self, datapipe: dp.iter.IterDataPipe, unwanted: set[str]) -> None:
        """Initialize the data pipeline."""
        super().__init__()

        self.datapipe = datapipe
        self.unwanted = unwanted

    def __iter__(self) -> Iterator[Iterable[str]]:
        """Get the next example."""
        for example in self.datapipe:
            filtered = [token for token in example if token not in self.unwanted]

            if len(filtered) > 0:
                yield filtered
