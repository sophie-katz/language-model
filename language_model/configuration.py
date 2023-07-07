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

"""Code configuration."""

from typing import Optional
import dataclasses
import os

import dotenv


@dataclasses.dataclass
class Configuration:
    comet_api_key: Optional[str] = None
    comet_project: Optional[str] = None
    comet_workspace: Optional[str] = None

    def load_from_env(self) -> None:
        dotenv.load_dotenv()

        self.comet_api_key = Configuration._getenv_nonempty("COMET_API_KEY")
        self.comet_project = Configuration._getenv_nonempty("COMET_PROJECT")
        self.comet_workspace = Configuration._getenv_nonempty("COMET_WORKSPACE")

    @staticmethod
    def _getenv_nonempty(name: str) -> Optional[str]:
        result = os.getenv(name)

        if result is not None and len(result) == 0:
            return None
        else:
            return result
