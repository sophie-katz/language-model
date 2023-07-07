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

"""Lightning callback to check for common problems."""

import warnings

import lightning as L


class Checker(L.Callback):
    """Lightning callback to check for common problems."""

    def __init__(self) -> None:
        super().__init__()

        self._params_grad_started_at_zero = set()

    def on_after_backward(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        """Check for common problems before zeroing gradients."""

        if pl_module.global_step == 0:
            for name, parameter in pl_module.named_parameters():
                if parameter.grad is None:
                    warnings.warn(
                        f"Gradients not flowing to parameter {name!r}", UserWarning
                    )
                elif parameter.grad.min() == 0 and parameter.grad.max() == 0:
                    self._params_grad_started_at_zero.add(name)
        elif pl_module.global_step == 1:
            for name, parameter in pl_module.named_parameters():
                if (
                    parameter.grad is not None
                    and name in self._params_grad_started_at_zero
                    and parameter.grad.min() == 0
                    and parameter.grad.max() == 0
                ):
                    warnings.warn(
                        f"Gradients for parameter {name!r} are stuck at 0", UserWarning
                    )
