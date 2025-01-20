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

import os
import tempfile
import unittest
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.case import skipUnless

import numpy as np
from parameterized import parameterized

from monai.data import DataLoader, Dataset, create_test_image_2d
from monai.data.image_reader import PILReader
from monai.transforms import LoadImage, LoadImaged
from monai.transforms.io.array import switch_endianness
from monai.utils.enums import PostFix
from monai.utils.module import optional_import

if TYPE_CHECKING:
    import nibabel as nib
    from PIL import Image as PILImage

    has_nib = True
    has_pil = True
else:
    nib, has_nib = optional_import("nibabel")
    PILImage, has_pil = optional_import("PIL.Image")

TESTS: list[tuple] = []
for endianness in ["<", ">"]:
    for use_array in [True, False]:
        for image_only in [True, False]:
            TESTS.append((endianness, use_array, image_only))


class TestNiftiEndianness(unittest.TestCase):

    def setUp(self):
        self.im, _ = create_test_image_2d(100, 100)
        self.fname = tempfile.NamedTemporaryFile(suffix=".nii.gz").name

    @parameterized.expand(TESTS)
    @skipUnless(has_nib, "Requires NiBabel")
    def test_endianness(self, endianness, use_array, image_only):
        hdr = nib.Nifti1Header(endianness=endianness)
        nii = nib.Nifti1Image(self.im, np.eye(4), header=hdr)
        nib.save(nii, self.fname)

        data = [self.fname] if use_array else [{"image": self.fname}]
        tr = LoadImage(image_only=image_only) if use_array else LoadImaged("image", image_only=image_only)
        check_ds = Dataset(data, tr)
        check_loader = DataLoader(check_ds, batch_size=1)
        ret = next(iter(check_loader))
        if isinstance(ret, dict) and PostFix.meta("image") in ret:
            np.testing.assert_allclose(ret[PostFix.meta("image")]["spatial_shape"], [[100, 100]])

    def test_switch(self):  # verify data types
        for data in (np.zeros((2, 1)), ("test",), [24, 42], {"foo": "bar"}, True, 42):
            output = switch_endianness(data, "<")
            self.assertEqual(type(data), type(output))

        before = np.array((20, 20), dtype=">i2")
        expected_float = before.astype(float)
        after = switch_endianness(before)
        np.testing.assert_allclose(after.astype(float), expected_float)
        self.assertEqual(after.dtype.byteorder, "<")

        before = np.array((20, 20), dtype="<i2")
        expected_float = before.astype(float)
        after = switch_endianness(before)
        np.testing.assert_allclose(after.astype(float), expected_float)

        before = np.array(["1.12", "-9.2", "42"], dtype=np.bytes_)
        after = switch_endianness(before)
        np.testing.assert_array_equal(before, after)

        with self.assertRaises(NotImplementedError):
            switch_endianness(np.zeros((2, 1)), "=")

        with self.assertRaises(RuntimeError):
            switch_endianness(Path("test"), "<")

    @skipUnless(has_pil, "Requires PIL")
    def test_pil(self):
        tempdir = tempfile.mkdtemp()
        test_image = np.random.randint(0, 256, size=[128, 256])
        filename = os.path.join(tempdir, "test_image.png")
        PILImage.fromarray(test_image.astype("uint8")).save(filename)

        loader = LoadImage(PILReader(converter=lambda image: image.convert("LA")))
        _ = loader(filename)


if __name__ == "__main__":
    unittest.main()
