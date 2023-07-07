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

"""Unit tests."""

from language_model.configuration import Configuration


def test_configuration_simple() -> None:
    """Test that the configuration can be loaded from the environment."""
    configuration = Configuration()
    configuration.load_from_env()

    assert (
        configuration.comet_api_key is not None and len(configuration.comet_api_key) > 0
    )
    assert (
        configuration.comet_project is not None and len(configuration.comet_project) > 0
    )
    assert (
        configuration.comet_workspace is not None
        and len(configuration.comet_workspace) > 0
    )
