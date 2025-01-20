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

from monai.transforms import ToDevice
from tests.util import assert_allclose, skip_if_no_cuda

TEST_CASE_1 = ["cuda:0"]

TEST_CASE_2 = ["cuda"]

TEST_CASE_3 = [torch.device("cpu:0")]

TEST_CASE_4 = ["cpu"]


@skip_if_no_cuda
class TestToDevice(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4])
    def test_value(self, device):
        converter = ToDevice(device=device, non_blocking=True)
        data = torch.tensor([1, 2, 3, 4])
        ret = converter(data)
        assert_allclose(ret, data.to(device))


if __name__ == "__main__":
    unittest.main()
