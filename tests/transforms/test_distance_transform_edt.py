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

import numpy as np
import torch
from parameterized import parameterized

from monai.transforms import DistanceTransformEDT, DistanceTransformEDTd
from tests.util import HAS_CUPY, assert_allclose, optional_import, skip_if_no_cuda

momorphology, has_cucim = optional_import("cucim.core.operations.morphology")
ndimage, has_ndimage = optional_import("scipy.ndimage")
cp, _ = optional_import("cupy")

TEST_CASES = [
    [
        np.array(
            ([[0, 1, 1, 1, 1], [0, 0, 1, 1, 1], [0, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 1, 1, 0, 0]],), dtype=np.float32
        ),
        np.array(
            [
                [
                    [0.0, 1.0, 1.4142, 2.2361, 3.0],
                    [0.0, 0.0, 1.0, 2.0, 2.0],
                    [0.0, 1.0, 1.4142, 1.4142, 1.0],
                    [0.0, 1.0, 1.4142, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 0.0, 0.0],
                ]
            ]
        ),
    ],
    [  # Example 4D input to test channel-wise CuPy
        np.array(
            [[[[0, 1, 1, 1, 1], [0, 0, 1, 1, 1], [0, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 1, 1, 0, 0]]]], dtype=np.float32
        ),
        np.array(
            [
                [
                    [
                        [0.0, 1.0, 1.4142, 2.2361, 3.0],
                        [0.0, 0.0, 1.0, 2.0, 2.0],
                        [0.0, 1.0, 1.4142, 1.4142, 1.0],
                        [0.0, 1.0, 1.4142, 1.0, 0.0],
                        [0.0, 1.0, 1.0, 0.0, 0.0],
                    ]
                ]
            ]
        ),
    ],
    [
        np.array(
            [
                [
                    [0.0, 1.0, 1.0, 1.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 0.0, 0.0],
                ],
                [
                    [0.0, 1.0, 1.0, 1.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 0.0, 0.0],
                ],
                [
                    [0.0, 1.0, 1.0, 1.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 0.0, 0.0],
                ],
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [
                    [0.0, 1.0, 1.4142135, 2.236068, 3.0],
                    [0.0, 0.0, 1.0, 2.0, 2.0],
                    [0.0, 1.0, 1.4142135, 1.4142135, 1.0],
                    [0.0, 1.0, 1.4142135, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 0.0, 0.0],
                ],
                [
                    [0.0, 1.0, 1.4142135, 2.236068, 3.0],
                    [0.0, 0.0, 1.0, 2.0, 2.0],
                    [0.0, 1.0, 1.4142135, 1.4142135, 1.0],
                    [0.0, 1.0, 1.4142135, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 0.0, 0.0],
                ],
                [
                    [0.0, 1.0, 1.4142135, 2.236068, 3.0],
                    [0.0, 0.0, 1.0, 2.0, 2.0],
                    [0.0, 1.0, 1.4142135, 1.4142135, 1.0],
                    [0.0, 1.0, 1.4142135, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 0.0, 0.0],
                ],
            ],
            dtype=np.float32,
        ),
    ],
]

SAMPLING_TEST_CASES = [
    [
        2,
        np.array(
            ([[0, 1, 1, 1, 1], [0, 0, 1, 1, 1], [0, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 1, 1, 0, 0]],), dtype=np.float32
        ),
        np.array(
            [
                [
                    [0.0, 2.0, 2.828427, 4.472136, 6.0],
                    [0.0, 0.0, 2.0, 4.0, 4.0],
                    [0.0, 2.0, 2.828427, 2.828427, 2.0],
                    [0.0, 2.0, 2.828427, 2.0, 0.0],
                    [0.0, 2.0, 2.0, 0.0, 0.0],
                ]
            ]
        ),
    ]
]

RAISES_TEST_CASES = (
    [  # Example 4D input. Should raise under CuPy
        np.array(
            [[[[[0, 1, 1, 1, 1], [0, 0, 1, 1, 1], [0, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 1, 1, 0, 0]]]]],
            dtype=np.float32,
        )
    ],
)


class TestDistanceTransformEDT(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_scipy_transform(self, input, expected_output):
        transform = DistanceTransformEDT()
        output = transform(input)
        assert_allclose(output, expected_output, atol=1e-4, rtol=1e-4, type_test=False)

    @parameterized.expand(TEST_CASES)
    def test_scipy_transformd(self, input, expected_output):
        transform = DistanceTransformEDTd(keys=("to_transform",))
        data = {"to_transform": input}
        data_ = transform(data)
        output = data_["to_transform"]
        assert_allclose(output, expected_output, atol=1e-4, rtol=1e-4, type_test=False)

    @parameterized.expand(SAMPLING_TEST_CASES)
    def test_scipy_sampling(self, sampling, input, expected_output):
        transform = DistanceTransformEDT(sampling=sampling)
        output = transform(input)
        assert_allclose(output, expected_output, atol=1e-4, rtol=1e-4, type_test=False)

    @parameterized.expand(TEST_CASES)
    @skip_if_no_cuda
    @unittest.skipUnless(HAS_CUPY, "CuPy is required.")
    @unittest.skipUnless(momorphology, "cuCIM transforms are required.")
    def test_cucim_transform(self, input, expected_output):
        input_ = torch.tensor(input, device="cuda")
        transform = DistanceTransformEDT()
        output = transform(input_)
        assert_allclose(cp.asnumpy(output), cp.asnumpy(expected_output), atol=1e-4, rtol=1e-4, type_test=False)

    @parameterized.expand(SAMPLING_TEST_CASES)
    @skip_if_no_cuda
    @unittest.skipUnless(HAS_CUPY, "CuPy is required.")
    @unittest.skipUnless(momorphology, "cuCIM transforms are required.")
    def test_cucim_sampling(self, sampling, input, expected_output):
        input_ = torch.tensor(input, device="cuda")
        transform = DistanceTransformEDT(sampling=sampling)
        output = transform(input_)
        assert_allclose(cp.asnumpy(output), cp.asnumpy(expected_output), atol=1e-4, rtol=1e-4, type_test=False)

    @parameterized.expand(RAISES_TEST_CASES)
    @skip_if_no_cuda
    @unittest.skipUnless(HAS_CUPY, "CuPy is required.")
    @unittest.skipUnless(momorphology, "cuCIM transforms are required.")
    def test_cucim_raises(self, raises):
        """Currently images of shape a certain shape are supported. This test checks for the according error message"""
        input_ = torch.tensor(raises, device="cuda")
        transform = DistanceTransformEDT()
        with self.assertRaises(RuntimeError):
            transform(input_)


if __name__ == "__main__":
    unittest.main()
