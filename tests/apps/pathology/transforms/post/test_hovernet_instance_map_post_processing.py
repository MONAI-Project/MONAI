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

from monai.apps.pathology.transforms.post.array import HoVerNetInstanceMapPostProcessing
from monai.transforms import ComputeHoVerMaps, FillHoles, GaussianSmooth
from monai.utils import min_version, optional_import
from tests.util import TEST_NDARRAYS, assert_allclose

_, has_scipy = optional_import("scipy", "1.8.1", min_version)
_, has_skimage = optional_import("skimage", "0.19.3", min_version)

y, x = np.ogrid[0:30, 0:30]
image = (x - 10) ** 2 + (y - 10) ** 2 <= 5**2
image = image[None, ...].astype("uint8")

TEST_CASE_1 = [{}, {"1": {"centroid": 1, "bbox": 1.0}}, np.zeros_like(image)]
TEST_CASE_2 = [{"distance_smooth_fn": GaussianSmooth()}, {"1": {"type": 1, "type_prob": 1.0}}, np.zeros_like(image)]
TEST_CASE_3 = [{"marker_postprocess_fn": FillHoles()}, {"1": {"type": 1, "type_prob": 1.0}}, np.zeros_like(image)]

TEST_CASE = []
for p in TEST_NDARRAYS:
    TEST_CASE.append([p, image] + TEST_CASE_1)
    TEST_CASE.append([p, image] + TEST_CASE_2)
    TEST_CASE.append([p, image] + TEST_CASE_3)


@unittest.skipUnless(has_scipy, "Requires scipy library.")
@unittest.skipUnless(has_skimage, "Requires scikit-image library.")
class TestHoVerNetInstanceMapPostProcessing(unittest.TestCase):
    @parameterized.expand(TEST_CASE)
    def test_value(self, in_type, test_data, kwargs, expected_info, expected_map):
        nuclear_prediction = in_type(test_data.astype(float))
        hover_map = in_type(ComputeHoVerMaps()(test_data.astype(int)))

        inst_info, inst_map = HoVerNetInstanceMapPostProcessing(**kwargs)(nuclear_prediction, hover_map)

        # instance info
        for key in inst_info:
            assert_allclose(inst_info[key]["centroid"], expected_info[key]["centroid"], type_test=False)

        # instance map
        assert_allclose(inst_map, expected_map, type_test=False)


if __name__ == "__main__":
    unittest.main()
