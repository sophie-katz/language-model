# Copyright (c) 2023 Sophie Katz
#
# This file is part of Language Model.
#
# Language Model is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later version.
#
# Language Model is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Language
# Model. If not, see <https://www.gnu.org/licenses/>.

import lightning
from language_model.models.transformer_from_scratch.transformer import Transformer


class TransformerModule(lightning.LightningModule):
    def __init__(
        self,
        encoder_layer_count: int = 6,
        decoder_layer_count: int = 6,
        input_size: int = 512,
        head_count: int = 6,
        feed_forward_hidden_size: int = 2048,
        dropout_rate: float = 0.1,
        positional_encoding_base: float = 1e4,
    ) -> None:
        super().__init__()

        self.hparams.encoder_layer_count = encoder_layer_count  # type: ignore
        self.hparams.decoder_layer_count = decoder_layer_count  # type: ignore
        self.hparams.input_size = input_size  # type: ignore
        self.hparams.head_count = head_count  # type: ignore
        self.hparams.feed_forward_hidden_size = feed_forward_hidden_size  # type: ignore
        self.hparams.dropout_rate = dropout_rate  # type: ignore
        self.hparams.positional_encoding_base = positional_encoding_base  # type: ignore

        self.transformer = Transformer(
            encoder_layer_count=encoder_layer_count,
            decoder_layer_count=decoder_layer_count,
            input_size=input_size,
            head_count=head_count,
            feed_forward_hidden_size=feed_forward_hidden_size,
            dropout_rate=dropout_rate,
        )
