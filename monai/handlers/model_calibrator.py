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

from functools import partial
from typing import TYPE_CHECKING

import modelopt.torch.quantization as mtq
import torch

from monai.utils import IgniteInfo, min_version, optional_import

Events, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Events")
Checkpoint, _ = optional_import("ignite.handlers", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Checkpoint")
if TYPE_CHECKING:
    from ignite.engine import Engine
else:
    Engine, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine")


class ModelCalibrater:
    """
    Model quantizer is for model quantization. It takes a model as input and convert it to a quantized
    model.

    Args:
        model: the model to be quantized.
        example_inputs: the example inputs for the model quantization. examples::
            (torch.randn(256,256,256),)
        config: the calibration config.

    """

    def __init__(self, model: torch.nn.Module, export_path: str, config: dict = mtq.INT8_SMOOTHQUANT_CFG) -> None:
        self.model = model
        self.export_path = export_path
        self.config = config

    def attach(self, engine: Engine) -> None:
        """
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        engine.add_event_handler(Events.STARTED, self)

    @staticmethod
    def _model_wrapper(engine, model):
        engine.run()

    def __call__(self, engine) -> None:
        quant_fun = partial(self._model_wrapper, engine)
        model = mtq.quantize(self.model, self.config, quant_fun)
        torch.save(model.state_dict(), self.export_path)
