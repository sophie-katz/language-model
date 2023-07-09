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

"""Lightning module for transformer."""

from typing import Any, Optional

import comet_ml
import lightning as L
import torch.optim
import torch as T
import torch.nn.functional as F
import torchmetrics

from language_model.models.transformer_from_scratch.transformer import Transformer


class TransformerModule(L.LightningModule):
    """Lightning module for transformer."""

    # pylint: disable=too-many-arguments,arguments-differ

    def __init__(
        self,
        word_embedding_vocabulary_size: int,
        encoder_layer_count: int = 1,
        decoder_layer_count: int = 1,
        word_embedding_feature_count: int = 512,
        positional_encoding_max_sequence_length: int = 4096,
        positional_encoding_base: float = 1e4,
        encoder_block_head_count: int = 3,
        encoder_block_feed_forward_hidden_feature_count: int = 128,
        encoder_block_residual_dropout_rate: float = 0.1,
        decoder_block_head_count: int = 3,
        decoder_block_feed_forward_hidden_feature_count: int = 128,
        decoder_block_residual_dropout_rate: float = 0.1,
        learning_rate: float = 1e-4,
        comet_experiment: Optional[comet_ml.Experiment] = None,
    ) -> None:
        """Initialize the module.

        Accepts hyperparameters as arguments.
        """
        super().__init__()

        self.hparams.word_embedding_vocabulary_size = (  # type: ignore
            word_embedding_vocabulary_size
        )
        self.hparams.encoder_layer_count = encoder_layer_count  # type: ignore
        self.hparams.decoder_layer_count = decoder_layer_count  # type: ignore
        self.hparams.word_embedding_feature_count = (  # type: ignore
            word_embedding_feature_count
        )
        self.hparams.positional_encoding_max_sequence_length = (  # type: ignore
            positional_encoding_max_sequence_length
        )
        self.hparams.positional_encoding_base = positional_encoding_base  # type: ignore
        self.hparams.encoder_block_head_count = encoder_block_head_count  # type: ignore
        self.hparams.encoder_block_feed_forward_hidden_feature_count = (  # type: ignore
            encoder_block_feed_forward_hidden_feature_count
        )
        self.hparams.encoder_block_residual_dropout_rate = (  # type: ignore
            encoder_block_residual_dropout_rate
        )
        self.hparams.decoder_block_head_count = decoder_block_head_count  # type: ignore
        self.hparams.decoder_block_feed_forward_hidden_feature_count = (  # type: ignore
            decoder_block_feed_forward_hidden_feature_count
        )
        self.hparams.decoder_block_residual_dropout_rate = (  # type: ignore
            decoder_block_residual_dropout_rate
        )
        self.hparams.learning_rate = learning_rate  # type: ignore

        self.comet_experiment = comet_experiment

        if self.comet_experiment is not None:
            self.comet_experiment.log_parameters(self.hparams)

        self.transformer = self._create_transformer()

        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=word_embedding_vocabulary_size)

    def training_step(self, batch: Any, _: int) -> T.Tensor:
        """Perform a training step."""
        assert isinstance(batch, T.Tensor)

        assert (
            batch.ndim == 2
        ), f"input sentence should be a batch of vectors of word indices, \
            not {batch.shape}"

        source = target = batch

        # if self.comet_experiment is not None and (self.global_step % 50) == 0:
        #     self.comet_experiment.log_histogram_3d(
        #         self.transformer.encoder.transformer_pass.word_embedding.embedding.weight.detach()
        #         .cpu()
        #         .numpy(),
        #         name="encoder/word_embedding/weight",
        #         step=self.global_step,
        #     )

        # if (
        #     self.comet_experiment is not None
        #     and (self.global_step % 50) == 0
        #     and self.transformer.encoder.transformer_pass.word_embedding.embedding.weight.grad
        #     is not None
        # ):
        #     self.comet_experiment.log_histogram_3d(
        #         self.transformer.encoder.transformer_pass.word_embedding.embedding.weight.grad.detach()
        #         .cpu()
        #         .numpy(),
        #         name="encoder/word_embedding/weight/grad",
        #         step=self.global_step,
        #     )

        # if self.comet_experiment is not None and (self.global_step % 50) == 0:
        #     self.comet_experiment.log_histogram_3d(
        #         self.transformer.decoder.linear.weight.detach().cpu().numpy(),
        #         name="decoder/linear/weight",
        #         step=self.global_step,
        #     )

        # if (
        #     self.comet_experiment is not None
        #     and (self.global_step % 50) == 0
        #     and self.transformer.decoder.linear.weight.grad is not None
        # ):
        #     self.comet_experiment.log_histogram_3d(
        #         self.transformer.decoder.linear.weight.grad.detach().cpu().numpy(),
        #         name="decoder/linear/weight/grad",
        #         step=self.global_step,
        #     )

        prediction = self.transformer(source, target)

        assert prediction.ndim == 3, "expected prediction to be a batch of sequences"
        assert prediction.size(0) == target.size(
            0
        ), "expected prediction to have same batch size as target"
        assert prediction.size(1) == target.size(
            1
        ), "expected prediction to have same sequence length as target"
        assert (
            prediction.size(2)
            == self.hparams.word_embedding_vocabulary_size  # type: ignore
        ), "expected prediction to be of vocabulary size"

        if self.comet_experiment is not None and (self.global_step % 50) == 0:
            self.comet_experiment.log_histogram_3d(
                prediction.squeeze().argmax(dim=-1).detach().cpu().numpy(),
                name="train/prediction",
                step=self.global_step,
            )

        target_for_loss = target.view(-1)
        prediction_for_loss = prediction.view(-1, prediction.size(2))

        # print()
        # print(f"Target: {target_for_loss} (shape: {target_for_loss.shape})")
        # print(f"Prediction: {prediction_for_loss} (shape: {prediction_for_loss.shape})")
        # print(
        #     f"  min: {prediction_for_loss.min()}, max: {prediction_for_loss.max()}, argmax: {prediction_for_loss.argmax(dim=-1)}"
        # )

        loss = F.cross_entropy(prediction_for_loss, target_for_loss)

        accuracy = self.accuracy(prediction_for_loss.argmax(dim=-1), target_for_loss) # type: ignore

        assert loss.ndim == 0, "expected loss to be a scalar"

        if self.comet_experiment is not None and (self.global_step % 50) == 0:
            self.comet_experiment.log_metric("train/loss", loss, step=self.global_step)
            self.comet_experiment.log_metric("train/accuracy", accuracy, step=self.global_step)

        return loss

    def configure_optimizers(self) -> Any:
        """Configure the optimizer."""
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,  # type: ignore
            betas=(0.9, 0.98),
            eps=1e-9,
        )

    # TODO: Needs to be implemented
    # https://www.notion.so/Implement-inference-70a9380b01774103b37dce5dfc398479?pvs=4
    def infer(self) -> None:
        """Inferer."""
        raise NotImplementedError()

    def _create_transformer(self) -> Transformer:
        """Create the transformer."""
        # fmt: off
        return Transformer(
            encoder_layer_count=self.hparams.encoder_layer_count,  # type: ignore
            decoder_layer_count=self.hparams.decoder_layer_count,  # type: ignore
            word_embedding_vocabulary_size=(
                self.hparams.word_embedding_vocabulary_size  # type: ignore
            ),
            word_embedding_feature_count=(
                self.hparams.word_embedding_feature_count  # type: ignore
            ),
            positional_encoding_max_sequence_length=(
                self.hparams.positional_encoding_max_sequence_length  # type: ignore
            ),
            positional_encoding_base=(
                self.hparams.positional_encoding_base  # type: ignore
            ),
            encoder_block_head_count=(
                self.hparams.encoder_block_head_count  # type: ignore
            ),
            encoder_block_feed_forward_hidden_feature_count=(
                self.hparams
                    .encoder_block_feed_forward_hidden_feature_count  # type: ignore
            ),
            encoder_block_residual_dropout_rate=(
                self.hparams.encoder_block_residual_dropout_rate  # type: ignore
            ),
            decoder_block_head_count=(
                self.hparams.decoder_block_head_count  # type: ignore
            ),
            decoder_block_feed_forward_hidden_feature_count=(
                self.hparams
                    .decoder_block_feed_forward_hidden_feature_count  # type: ignore
            ),
            decoder_block_residual_dropout_rate=(
                self.hparams.decoder_block_residual_dropout_rate  # type: ignore
            ),
        )
        # fmt: on
