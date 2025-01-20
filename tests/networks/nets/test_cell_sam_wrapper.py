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
from parameterized import parameterized

from monai.networks import eval_mode
from monai.networks.nets.cell_sam_wrapper import CellSamWrapper
from monai.utils import optional_import

build_sam_vit_b, has_sam = optional_import("segment_anything.build_sam", name="build_sam_vit_b")

device = "cuda" if torch.cuda.is_available() else "cpu"
TEST_CASE_CELLSEGWRAPPER = []
for dims in [128, 256, 512, 1024]:
    test_case = [
        {"auto_resize_inputs": True, "network_resize_roi": [1024, 1024], "checkpoint": None},
        (1, 3, *([dims] * 2)),
        (1, 3, *([dims] * 2)),
    ]
    TEST_CASE_CELLSEGWRAPPER.append(test_case)


@unittest.skipUnless(has_sam, "Requires SAM installation")
class TestResNetDS(unittest.TestCase):

    @parameterized.expand(TEST_CASE_CELLSEGWRAPPER)
    def test_shape(self, input_param, input_shape, expected_shape):
        net = CellSamWrapper(**input_param).to(device)
        with eval_mode(net):
            result = net(torch.randn(input_shape).to(device))
            self.assertEqual(result.shape, expected_shape, msg=str(input_param))

    def test_ill_arg0(self):
        with self.assertRaises(RuntimeError):
            net = CellSamWrapper(auto_resize_inputs=False, checkpoint=None).to(device)
            net(torch.randn([1, 3, 256, 256]).to(device))

    def test_ill_arg1(self):
        with self.assertRaises(RuntimeError):
            net = CellSamWrapper(network_resize_roi=[256, 256], checkpoint=None).to(device)
            net(torch.randn([1, 3, 1024, 1024]).to(device))


if __name__ == "__main__":
    unittest.main()
