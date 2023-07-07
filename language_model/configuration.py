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
    comet_enabled: Optional[bool] = None
    comet_api_key: Optional[str] = None
    comet_project: Optional[str] = None
    comet_workspace: Optional[str] = None

    def load_from_env(self) -> None:
        dotenv.load_dotenv()

        self.comet_enabled = Configuration._getenv_bool("COMET_ENABLED")
        self.comet_api_key = Configuration._getenv_nonempty("COMET_API_KEY")
        self.comet_project = Configuration._getenv_nonempty("COMET_PROJECT")
        self.comet_workspace = Configuration._getenv_nonempty("COMET_WORKSPACE")

        self._require_enabled()

    def _require_enabled(self) -> None:
        if self.comet_enabled is None:
            raise ValueError("Required environment variable COMET_ENABLED is not set")

    @staticmethod
    def _getenv_nonempty(name: str) -> Optional[str]:
        result = os.getenv(name)

        if result is not None and len(result) == 0:
            return None
        else:
            return result

    @staticmethod
    def _getenv_bool(name: str) -> Optional[bool]:
        result = os.getenv(name)

        if result is None:
            return None
        elif result.lower() in ["true", "yes", "on", "enabled", "1"]:
            return True
        elif result.lower() in ["false", "no", "off", "disabled", "0"]:
            return False
        else:
            raise ValueError(
                f"Unexpected value for a boolean environment variable \
                            {result!r} (expected true/false, yes/no, on/off, \
                            enabled/disabled, 1/0)"
            )
