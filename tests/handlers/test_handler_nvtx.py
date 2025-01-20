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

import unittest

import torch
from ignite.engine import Events
from parameterized import parameterized

from monai.engines import SupervisedEvaluator
from monai.handlers import StatsHandler, from_engine
from monai.handlers.nvtx_handlers import MarkHandler, RangeHandler, RangePopHandler, RangePushHandler
from monai.utils import optional_import
from tests.util import assert_allclose

_, has_nvtx = optional_import("torch._C._nvtx", descriptor="NVTX is not installed. Are you sure you have a CUDA build?")

TENSOR_0 = torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]])

TENSOR_1 = torch.tensor([[[[0.0], [-2.0]], [[-3.0], [4.0]]]])

TENSOR_1_EXPECTED = torch.tensor([[[1.0], [0.5]], [[0.25], [5.0]]])

TEST_CASE_0 = [[{"image": TENSOR_0}], TENSOR_0[0] + 1.0]
TEST_CASE_1 = [[{"image": TENSOR_1}], TENSOR_1_EXPECTED]


class TestHandlerDecollateBatch(unittest.TestCase):
    @parameterized.expand([TEST_CASE_0, TEST_CASE_1])
    @unittest.skipUnless(has_nvtx, "CUDA is required for NVTX!")
    def test_compute(self, data, expected):
        # Set up handlers
        handlers = [
            # Mark with Ignite Event
            MarkHandler(Events.STARTED),
            # Mark with literal
            MarkHandler("EPOCH_STARTED"),
            # Mark with literal and providing the message
            MarkHandler("EPOCH_STARTED", "Start of the epoch"),
            # Define a range using one prefix (between BATCH_STARTED and BATCH_COMPLETED)
            RangeHandler("Batch"),
            # Define a range using a pair of events
            RangeHandler((Events.STARTED, Events.COMPLETED)),
            # Define a range using a pair of literals
            RangeHandler(("GET_BATCH_STARTED", "GET_BATCH_COMPLETED"), msg="Batching!"),
            # Define a range using a pair of literal and events
            RangeHandler(("GET_BATCH_STARTED", Events.COMPLETED)),
            # Define the start of range using literal
            RangePushHandler("ITERATION_STARTED"),
            # Define the start of range using event
            RangePushHandler(Events.ITERATION_STARTED, "Iteration 2"),
            # Define the start of range using literals and providing message
            RangePushHandler("EPOCH_STARTED", "Epoch 2"),
            # Define the end of range using Ignite Event
            RangePopHandler(Events.ITERATION_COMPLETED),
            RangePopHandler(Events.EPOCH_COMPLETED),
            # Define the end of range using literal
            RangePopHandler("ITERATION_COMPLETED"),
            # Other handlers
            StatsHandler(tag_name="train", output_transform=from_engine(["label"], first=True)),
        ]

        # Set up an engine
        engine = SupervisedEvaluator(
            device=torch.device("cpu:0"),
            val_data_loader=data,
            epoch_length=1,
            network=torch.nn.PReLU(),
            postprocessing=lambda x: dict(pred=x["pred"] + 1.0),
            decollate=True,
            val_handlers=handlers,
        )
        # Run the engine
        engine.run()

        # Get the output from the engine
        output = engine.state.output[0]

        assert_allclose(output["pred"], expected)


if __name__ == "__main__":
    unittest.main()
