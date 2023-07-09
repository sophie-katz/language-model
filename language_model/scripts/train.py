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

import warnings
from typing import Optional

import comet_ml
import lightning as L
import torch.utils.data
from torchtext.datasets import WikiText2

from language_model.configuration import Configuration
from language_model.data.utils.wiki2_transformer import get_wiki2_transformer_datapipe
from language_model.models.transformer_from_scratch.transformer_module import (
    TransformerModule,
)
from language_model.lightning_checker import Checker

if __name__ == "__main__":
    # Disable warnings
    warnings.filterwarnings("ignore", ".*does not have many workers.*")
    warnings.filterwarnings("ignore", ".*Some child DataPipes are not exhausted.*")
    warnings.filterwarnings(
        "ignore", ".*are turning off the .* dataloader shuffling for you.*"
    )

    # Load configuration
    configuration = Configuration()
    configuration.load_from_env()

    # Initialize Comet experiment
    comet_experiment: Optional[comet_ml.Experiment] = None

    if configuration.comet_enabled:
        comet_experiment = comet_ml.Experiment(
            api_key=configuration.comet_api_key,
            project_name=configuration.comet_project_name,
            workspace=configuration.comet_workspace,
        )

    # Create data loaders
    train, valid, test = WikiText2(root=".data", split=("train", "valid", "test"))

    vocabulary, train_datapipe = get_wiki2_transformer_datapipe(train)
    vocabulary.set_default_index(vocabulary["<unk>"])

    _, valid_datapipe = get_wiki2_transformer_datapipe(valid, vocabulary=vocabulary)
    _, test_datapipe = get_wiki2_transformer_datapipe(test, vocabulary=vocabulary)

    train_dataloader = torch.utils.data.DataLoader(
        train_datapipe, shuffle=False, num_workers=0  # type: ignore
    )

    # Create Lightning components
    transformer_module = TransformerModule(
        token_embedding_vocabulary_size=len(vocabulary),
        comet_experiment=comet_experiment,
    )

    profiler = L.pytorch.profilers.AdvancedProfiler(dirpath=".", filename="perf_logs")

    MAX_EPOCHS = 100
    # OVERFIT_BATCHES = 100

    if comet_experiment is not None:
        comet_experiment.log_parameter("max_epochs", MAX_EPOCHS)
        # comet_experiment.log_parameter("overfit_batches", OVERFIT_BATCHES)

    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        # detect_anomaly=True,
        # overfit_batches=OVERFIT_BATCHES,
        logger=False,
        callbacks=[Checker()],
        # profiler=profiler,
        enable_checkpointing=False,
    )

    # Train
    trainer.fit(transformer_module, train_dataloader)
