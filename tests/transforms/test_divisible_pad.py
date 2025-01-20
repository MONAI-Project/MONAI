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

from parameterized import parameterized

from monai.transforms import DivisiblePad
from monai.utils.enums import NumpyPadMode, PytorchPadMode
from tests.padders import PadTest

TESTS = []

# pad first dim to be divisible by 7, the second unchanged.
TESTS.append([{"k": (7, -1)}, (3, 8, 7), (3, 14, 7)])
# pad all dimensions to be divisible by 5
TESTS.append([{"k": 5, "method": "end"}, (3, 10, 5, 17), (3, 10, 5, 20)])


class TestDivisiblePad(PadTest):
    Padder = DivisiblePad

    @parameterized.expand(TESTS)
    def test_pad(self, input_param, input_shape, expected_shape):
        modes = ["constant", NumpyPadMode.CONSTANT, PytorchPadMode.CONSTANT]
        self.pad_test(input_param, input_shape, expected_shape, modes)

    def test_pad_kwargs(self):
        kwargs = {"k": 5, "method": "end"}
        unchanged_slices = [slice(None), slice(None, 8), slice(None, 4)]
        self.pad_test_kwargs(unchanged_slices, **kwargs)

    @parameterized.expand(TESTS)
    def test_pending_ops(self, input_param, input_shape, _):
        self.pad_test_pending_ops(input_param, input_shape)


if __name__ == "__main__":
    unittest.main()
