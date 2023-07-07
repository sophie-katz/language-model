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

import comet_ml

import lightning as L
import torch.utils.data
from torchtext.datasets import WikiText2

from language_model.configuration import Configuration
from language_model.data.utils.wiki2_transformer import get_wiki2_transformer_datapipe
from language_model.models.transformer_from_scratch.transformer_module import (
    TransformerModule,
)

if __name__ == "__main__":
    # Load configuration
    configuration = Configuration()
    configuration.load_from_env()

    # Initialize Comet experiment
    comet_experiment = comet_ml.Experiment(
        api_key=configuration.comet_api_key,
        project_name=configuration.comet_project,
        workspace=configuration.comet_workspace,
    )

    # Create data loaders
    train, valid, test = WikiText2(root=".data", split=("train", "valid", "test"))

    vocabulary, train_datapipe = get_wiki2_transformer_datapipe(train)
    vocabulary.set_default_index(vocabulary["<unk>"])

    _, valid_datapipe = get_wiki2_transformer_datapipe(valid, vocabulary=vocabulary)
    _, test_datapipe = get_wiki2_transformer_datapipe(test, vocabulary=vocabulary)

    train_dataloader = torch.utils.data.DataLoader(train_datapipe)  # type: ignore

    # Create Lightning components
    transformer_module = TransformerModule(comet_experiment, len(vocabulary))

    trainer = L.Trainer(max_epochs=2, detect_anomaly=True, overfit_batches=10)

    # Train
    trainer.fit(transformer_module, train_dataloader)
