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

from monai.data import rectify_header_sform_qform


class TestRectifyHeaderSformQform(unittest.TestCase):

    def test_revise_q(self):
        img = nib.Nifti1Image(np.zeros((10, 10, 10)), np.eye(4))
        img.header.set_zooms((0.1, 0.2, 0.3))
        output = rectify_header_sform_qform(img)
        expected = np.diag([0.1, 0.2, 0.3, 1.0])
        np.testing.assert_allclose(output.affine, expected)

    def test_revise_both(self):
        img = nib.Nifti1Image(np.zeros((10, 10, 10)), np.eye(4))
        img.header.set_sform(np.diag([5, 3, 4, 1]))
        img.header.set_qform(np.diag([2, 3, 4, 1]))
        img.header.set_zooms((0.1, 0.2, 0.3))
        output = rectify_header_sform_qform(img)
        expected = np.diag([0.1, 0.2, 0.3, 1.0])
        np.testing.assert_allclose(output.affine, expected)


if __name__ == "__main__":
    unittest.main()
