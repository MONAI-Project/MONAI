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

import unittest

import numpy as np
from parameterized import parameterized

from monai.apps.pathology.transforms.post.array import HoVerNetPostProcessing
from monai.apps.pathology.transforms.post.dictionary import (
    GenerateDistanceMapd,
    GenerateInstanceBorderd,
    GenerateWatershedMarkersd,
    GenerateWatershedMaskd,
    Watershedd,
)
from monai.transforms import Compose, ComputeHoVerMaps, FillHoles, GaussianSmooth
from monai.utils import min_version, optional_import
from monai.utils.enums import HoVerNetBranch
from tests.utils import TEST_NDARRAYS, assert_allclose

_, has_scipy = optional_import("scipy", "1.8.1", min_version)
_, has_skimage = optional_import("skimage", "0.19.3", min_version)

y, x = np.ogrid[0:30, 0:30]
image = (x - 10) ** 2 + (y - 10) ** 2 <= 5**2
hovermap = ComputeHoVerMaps()(image[None].astype(int))

seg_postpprocessing = Compose(
    [
        GenerateWatershedMaskd(
            keys=HoVerNetBranch.NP.value, sigmoid=True, softmax=False, threshold=0.7, remove_small_objects=False
        ),
        GenerateInstanceBorderd(
            keys="mask", hover_map_key=HoVerNetBranch.HV.value, kernel_size=3, remove_small_objects=False
        ),
        GenerateDistanceMapd(keys="mask", border_key="border", smooth_fn=GaussianSmooth()),
        GenerateWatershedMarkersd(
            keys="mask", border_key="border", threshold=0.9, radius=2, postprocess_fn=FillHoles()
        ),
        Watershedd(keys="dist", mask_key="mask", markers_key="markers"),
    ]
)
TEST_CASE_1 = [
    seg_postpprocessing,
    {"return_centroids": True, "return_centroids": True, "output_classes": 1},
    [image, [10, 10]],
]
TEST_CASE_2 = [
    seg_postpprocessing,
    {"return_centroids": False, "return_centroids": False, "output_classes": None},
    [image],
]


TEST_CASE = []
for p in TEST_NDARRAYS:
    pred = {
        HoVerNetBranch.NP.value: p(image[None].astype(float)),
        HoVerNetBranch.HV.value: p(hovermap),
        HoVerNetBranch.NC.value: p(image[None]),
    }
    TEST_CASE.append([pred] + TEST_CASE_1)
    TEST_CASE.append([pred] + TEST_CASE_2)


@unittest.skipUnless(has_skimage, "Requires scikit-image library.")
class TestHoVerNetPostProcessing(unittest.TestCase):
    @parameterized.expand(TEST_CASE)
    def test_value(self, test_data, seg_postpprocessing, kwargs, expected):
        post_transforms = HoVerNetPostProcessing(seg_postpprocessing=seg_postpprocessing, **kwargs)
        out = post_transforms(test_data)
        assert_allclose(out[HoVerNetBranch.NP.value].squeeze(), expected[0], type_test=False)
        if out["inst_info_dict"] is not None:
            assert_allclose(out["inst_info_dict"][1]["centroid"], expected[1], type_test=False)


if __name__ == "__main__":
    unittest.main()
