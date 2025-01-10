# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Sequence
from types import MethodType
from typing import TYPE_CHECKING

import torch
from torch.ao.quantization.quantize_pt2e import prepare_qat_pt2e
from torch.ao.quantization.quantizer import Quantizer
from torch.ao.quantization.quantizer.xnnpack_quantizer import XNNPACKQuantizer, get_symmetric_quantization_config

from monai.utils import IgniteInfo, min_version, optional_import

Events, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Events")
Checkpoint, _ = optional_import("ignite.handlers", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Checkpoint")
if TYPE_CHECKING:
    from ignite.engine import Engine
else:
    Engine, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine")


class ModelQuantizer:
    """
    Model quantizer is for model quantization. It takes a model as input and convert it to a quantized
    model.

    Args:
        model: the model to be quantized.
        example_inputs: the example inputs for the model quantization. examples::
            (torch.randn(256,256,256),)
        quantizer: quantizer for the quantization job.

    """

    def __init__(
        self, model: torch.nn.Module, example_inputs: Sequence, export_path: str, quantizer: Quantizer | None = None
    ) -> None:
        self.model = model
        self.example_inputs = example_inputs
        self.export_path = export_path
        self.quantizer = (
            XNNPACKQuantizer().set_global(get_symmetric_quantization_config()) if quantizer is None else quantizer
        )

    def attach(self, engine: Engine) -> None:
        """
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        engine.add_event_handler(Events.STARTED, self.start)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.epoch)

    def start(self) -> None:
        self.model = torch.export.export_for_training(self.model, self.example_inputs).module()
        self.model = prepare_qat_pt2e(self.model, self.quantizer)
        self.model.train = MethodType(torch.ao.quantization.move_exported_model_to_train, self.model)
        self.model.eval = MethodType(torch.ao.quantization.move_exported_model_to_eval, self.model)

    def epoch(self) -> None:
        torch.save(self.model.state_dict(), self.export_path)
