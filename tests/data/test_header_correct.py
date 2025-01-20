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

import nibabel as nib
import numpy as np

from monai.data import correct_nifti_header_if_necessary


class TestCorrection(unittest.TestCase):

    def test_correct(self):
        test_img = nib.Nifti1Image(np.zeros((1, 2, 3)), np.eye(4))
        test_img.header.set_zooms((100, 100, 100))
        test_img = correct_nifti_header_if_necessary(test_img)
        np.testing.assert_allclose(
            test_img.affine,
            np.array([[100.0, 0.0, 0.0, 0.0], [0.0, 100.0, 0.0, 0.0], [0.0, 0.0, 100.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
        )

    def test_affine(self):
        test_img = nib.Nifti1Image(np.zeros((1, 2, 3)), np.eye(4) * 20.0)
        test_img = correct_nifti_header_if_necessary(test_img)
        np.testing.assert_allclose(
            test_img.affine,
            np.array([[20.0, 0.0, 0.0, 0.0], [0.0, 20.0, 0.0, 0.0], [0.0, 0.0, 20.0, 0.0], [0.0, 0.0, 0.0, 20.0]]),
        )


if __name__ == "__main__":
    unittest.main()
