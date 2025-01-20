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
from copy import deepcopy

import numpy as np
from parameterized import parameterized

from monai.data.meta_tensor import MetaTensor
from monai.transforms import RandCropByPosNegLabel
from monai.transforms.lazy.functional import apply_pending
from tests.util import TEST_NDARRAYS_ALL, assert_allclose

TESTS = [
    [
        {
            "label": np.random.randint(0, 2, size=[3, 3, 3, 3]),
            "spatial_size": [2, 2, -1],
            "pos": 1,
            "neg": 1,
            "num_samples": 2,
            "image": np.random.randint(0, 2, size=[3, 3, 3, 3]),
            "image_threshold": 0,
        },
        {"img": np.random.randint(0, 2, size=[3, 3, 3, 3])},
        (3, 2, 2, 3),
    ],
    [
        {
            "label": np.random.randint(0, 2, size=[3, 3, 3, 3]),
            "spatial_size": [2, 2, 2],
            "pos": 1,
            "neg": 1,
            "num_samples": 2,
            "image": np.random.randint(0, 2, size=[3, 3, 3, 3]),
            "image_threshold": 0,
        },
        {"img": np.random.randint(0, 2, size=[3, 3, 3, 3])},
        (3, 2, 2, 2),
    ],
    [
        {
            "label": None,
            "spatial_size": [2, 2, 2],
            "pos": 1,
            "neg": 1,
            "num_samples": 2,
            "image": np.random.randint(0, 2, size=[3, 3, 3, 3]),
            "image_threshold": 0,
        },
        {
            "img": np.random.randint(0, 2, size=[3, 3, 3, 3]),
            "label": np.random.randint(0, 2, size=[3, 3, 3, 3]),
            "image": np.random.randint(0, 2, size=[3, 3, 3, 3]),
        },
        (3, 2, 2, 2),
    ],
    [
        {
            "label": np.random.randint(0, 2, size=[3, 3, 3, 3]),
            "spatial_size": [4, 4, 2],
            "pos": 1,
            "neg": 1,
            "num_samples": 2,
            "image": np.random.randint(0, 2, size=[3, 3, 3, 3]),
            "allow_smaller": True,
        },
        {"img": np.random.randint(0, 2, size=[3, 3, 3, 3])},
        (3, 3, 3, 2),
    ],
    [
        {
            "label": np.random.randint(0, 2, size=[3, 3, 3, 3]),
            "spatial_size": [4, 4, 4],
            "pos": 1,
            "neg": 1,
            "num_samples": 2,
            "image": np.random.randint(0, 2, size=[3, 3, 3, 3]),
            "allow_smaller": True,
        },
        {"img": np.random.randint(0, 2, size=[3, 3, 3, 3])},
        (3, 3, 3, 3),
    ],
]


class TestRandCropByPosNegLabel(unittest.TestCase):
    @staticmethod
    def convert_data_type(im_type, d, keys=("img", "image", "label")):
        out = deepcopy(d)
        for k, v in out.items():
            if k in keys and isinstance(v, np.ndarray):
                out[k] = im_type(v)
        return out

    @parameterized.expand(TESTS)
    def test_type_shape(self, input_param, input_data, expected_shape):
        results = []
        for p in TEST_NDARRAYS_ALL:
            input_param_mod = self.convert_data_type(p, input_param)
            input_data_mod = self.convert_data_type(p, input_data)
            cropper = RandCropByPosNegLabel(**input_param_mod)
            cropper.set_random_state(0)
            result = cropper(**input_data_mod)
            self.assertListEqual(cropper.spatial_size, input_param["spatial_size"])

            self.assertIsInstance(result, list)
            self.assertTupleEqual(result[0].shape, expected_shape)

            # check for same results across numpy, torch.Tensor and torch.cuda.Tensor
            result = np.asarray([i if isinstance(i, np.ndarray) else i.cpu().numpy() for i in result])
            results.append(np.asarray(result))
            if len(results) > 1:
                np.testing.assert_allclose(results[0], results[-1])

    @parameterized.expand(TESTS)
    def test_pending_ops(self, input_param, input_data, _expected_shape):
        for p in TEST_NDARRAYS_ALL:
            input_param_mod = self.convert_data_type(p, input_param)
            input_data_mod = self.convert_data_type(p, input_data)
            cropper = RandCropByPosNegLabel(**input_param_mod)
            # non-lazy
            cropper.set_random_state(0)
            expected = cropper(**input_data_mod)
            self.assertIsInstance(expected[0], MetaTensor)
            # lazy
            cropper.set_random_state(0)
            cropper.lazy = True
            pending_result = cropper(**input_data_mod)
            for i, _pending_result in enumerate(pending_result):
                self.assertIsInstance(_pending_result, MetaTensor)
                assert_allclose(_pending_result.peek_pending_affine(), expected[i].affine)
                assert_allclose(_pending_result.peek_pending_shape(), expected[i].shape[1:])
                # only support nearest
                result = apply_pending(_pending_result, overrides={"mode": "nearest", "align_corners": False})[0]
                # compare
                assert_allclose(result, expected[i], rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
