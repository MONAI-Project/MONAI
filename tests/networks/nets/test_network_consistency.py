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

import json
import os
import unittest
from collections.abc import Sequence
from glob import glob
from unittest.case import skipIf

import torch
from parameterized.parameterized import parameterized

import monai.networks.nets as nets
from monai.utils import set_determinism
from monai.utils.misc import MONAIEnvVars
from tests.util import assert_allclose

extra_test_data_dir = MONAIEnvVars.extra_test_data()

TESTS = []
if extra_test_data_dir is not None:
    for data_path in glob(os.path.join(extra_test_data_dir, "**", "*.pt")):
        json_path = data_path[:-3] + ".json"
        # net_name is filename until first underscore (e.g., unet_0.pt is unet)
        net_name = os.path.basename(data_path).split("_")[0]
        TESTS.append((net_name, data_path, json_path))


class TestNetworkConsistency(unittest.TestCase):
    def setUp(self):
        set_determinism(0)

    def tearDown(self):
        set_determinism(None)

    @skipIf(
        len(TESTS) == 0,
        "To run these tests, clone https://github.com/Project-MONAI/MONAI-extra-test-data and set MONAI_EXTRA_TEST_DATA",
    )
    @parameterized.expand(TESTS, skip_on_empty=True)
    def test_network_consistency(self, net_name, data_path, json_path):
        print("Net name: " + net_name)
        print("Data path: " + data_path)
        print("JSON path: " + json_path)

        # Load data
        loaded_data = torch.load(data_path)

        # Load json from file
        json_file = open(json_path)
        model_params = json.load(json_file)
        json_file.close()

        # Create model
        model = getattr(nets, net_name)(**model_params)
        model.load_state_dict(loaded_data["model"], strict=False)
        model.eval()

        in_data = loaded_data["in_data"]
        expected_out_data = loaded_data["out_data"]

        actual_out_data = model(in_data)

        self.check_output_consistency(actual_out_data, expected_out_data)

    def check_output_consistency(self, actual, expected):
        if isinstance(actual, Sequence):
            for a, e in zip(actual, expected):
                self.check_output_consistency(a, e)
        else:
            assert_allclose(actual, expected, rtol=5e-2, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
