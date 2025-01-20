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

from monai.apps.pathology.transforms.post.dictionary import (
    HoVerNetInstanceMapPostProcessingd,
    HoVerNetNuclearTypePostProcessingd,
)
from monai.transforms import ComputeHoVerMaps
from monai.utils import min_version, optional_import
from monai.utils.enums import HoVerNetBranch
from tests.util import TEST_NDARRAYS, assert_allclose

_, has_scipy = optional_import("scipy", "1.8.1", min_version)
_, has_skimage = optional_import("skimage", "0.19.3", min_version)

y, x = np.ogrid[0:30, 0:30]
image = (x - 10) ** 2 + (y - 10) ** 2 <= 5**2
image = image[None, ...].astype("uint8")

TEST_CASE_1 = [{}, [{"1": [10, 10]}, np.zeros_like(image), np.zeros_like(image)]]

TEST_CASE = []
for p in TEST_NDARRAYS:
    TEST_CASE.append([p, image] + TEST_CASE_1)


@unittest.skipUnless(has_scipy, "Requires scipy library.")
@unittest.skipUnless(has_skimage, "Requires scikit-image library.")
class TestHoVerNetNuclearTypePostProcessingd(unittest.TestCase):
    @parameterized.expand(TEST_CASE)
    def test_value(self, in_type, test_data, kwargs, expected):
        input = {
            HoVerNetBranch.NP.value: in_type(test_data.astype(float)),
            HoVerNetBranch.HV.value: in_type(ComputeHoVerMaps()(test_data.astype(int))),
            HoVerNetBranch.NC.value: in_type(test_data),
        }

        outputs = HoVerNetInstanceMapPostProcessingd()(input)
        outputs = HoVerNetNuclearTypePostProcessingd(**kwargs)(outputs)

        # instance prediction info
        for key in outputs["instance_info"]:
            assert_allclose(outputs["instance_info"][key]["centroid"], expected[0][key], type_test=False)

        # instance map
        assert_allclose(outputs["instance_map"], expected[1], type_test=False)

        # type map
        if expected[2] is None:
            self.assertIsNone(outputs["type_map"])
        else:
            assert_allclose(outputs["type_map"], expected[2], type_test=False)


if __name__ == "__main__":
    unittest.main()
