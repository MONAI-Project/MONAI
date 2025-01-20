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
from parameterized import parameterized

from monai.apps.reconstruction.transforms.dictionary import ReferenceBasedNormalizeIntensityd
from monai.utils.type_conversion import convert_to_numpy
from tests.util import TEST_NDARRAYS_NO_META_TENSOR, assert_allclose

# see test_normalize_intensityd for typical tests (like non-zero
# normalization, device test, etc.)
# here, we test DetailedNormalizeIntensityd's functionality
# which focuses on (1) automatic target normalization and (2) mean-std
# return values

TESTS = []
for p in TEST_NDARRAYS_NO_META_TENSOR:
    TESTS.append(
        [
            {"keys": ["kspace_masked_ifft", "target"], "ref_key": "kspace_masked_ifft", "channel_wise": True},
            {"kspace_masked_ifft": p(np.array([[-2.0, 0.0, 2.0]])), "target": p(np.array([[1.0, 2.0, 3.0]]))},
            p(np.array([[-1.225, 0.0, 1.225]])),  # normalized input
            p(np.array([[0.612, 1.225, 1.837]])),  # normalized target
            np.array([0.0]),  # mean
            np.array([1.633]),  # std
        ]
    )

    TESTS.append(
        [
            {"keys": ["kspace_masked_ifft", "target"], "ref_key": "kspace_masked_ifft", "channel_wise": False},
            {"kspace_masked_ifft": p(np.array([[-2.0, 0.0, 2.0]])), "target": p(np.array([[1.0, 2.0, 3.0]]))},
            p(np.array([[-1.225, 0.0, 1.225]])),  # normalized input
            p(np.array([[0.612, 1.225, 1.837]])),  # normalized target
            0.0,  # mean
            1.633,  # std
        ]
    )


class TestDetailedNormalizeIntensityd(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_target_mean_std(self, args, data, normalized_data, normalized_target, mean, std):
        dtype = data[args["keys"][0]].dtype
        normalizer = ReferenceBasedNormalizeIntensityd(
            keys=args["keys"], ref_key=args["ref_key"], channel_wise=args["channel_wise"], dtype=dtype
        )
        res_data = normalizer(data)

        img = np.round(convert_to_numpy(res_data[args["keys"][0]]), 3)
        normalized_data = np.round(convert_to_numpy(normalized_data), 3)

        target = np.round(convert_to_numpy(res_data[args["keys"][1]]), 3)
        normalized_target = np.round(convert_to_numpy(normalized_target), 3)

        assert_allclose(img, normalized_data)
        assert_allclose(target, normalized_target)

        assert_allclose(np.round(res_data["mean"], 3), mean)
        assert_allclose(np.round(res_data["std"], 3), std)


if __name__ == "__main__":
    unittest.main()
