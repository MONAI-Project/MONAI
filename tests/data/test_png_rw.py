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

import numpy as np
from PIL import Image

from monai.data.image_writer import PILWriter


class TestPngWrite(unittest.TestCase):

    def test_write_gray(self):
        with tempfile.TemporaryDirectory() as out_dir:
            image_name = os.path.join(out_dir, "test.png")
            img = np.random.rand(2, 3)
            img_save_val = (255 * img).astype(np.uint8)
            writer_obj = PILWriter(output_dtype=np.uint8)
            writer_obj.set_data_array(img, channel_dim=None)
            writer_obj.write(image_name, format="PNG")
            out = np.asarray(Image.open(image_name))
            out = np.moveaxis(out, 0, 1)
            np.testing.assert_allclose(out, img_save_val)

    def test_write_gray_1height(self):
        with tempfile.TemporaryDirectory() as out_dir:
            image_name = os.path.join(out_dir, "test.png")
            img = np.random.rand(1, 3)
            img_save_val = (65535 * img).astype(np.uint16)
            writer_obj = PILWriter(output_dtype=np.uint16, scale=65535)
            writer_obj.set_data_array(img, channel_dim=None)
            writer_obj.write(image_name, format="PNG")
            out = np.asarray(Image.open(image_name))
            out = np.moveaxis(out, 0, 1)
            np.testing.assert_allclose(out, img_save_val)

    def test_write_gray_1channel(self):
        with tempfile.TemporaryDirectory() as out_dir:
            image_name = os.path.join(out_dir, "test.png")
            img = np.random.rand(2, 3, 1)
            img_save_val = (255 * img).astype(np.uint8).squeeze(2)
            writer_obj = PILWriter(output_dtype=np.uint8, scale=255)
            writer_obj.set_data_array(img, channel_dim=None)
            writer_obj.write(image_name, format="PNG")
            out = np.asarray(Image.open(image_name))
            out = np.moveaxis(out, 0, 1)
            np.testing.assert_allclose(out, img_save_val)

    def test_write_rgb(self):
        """testing default kwargs and obj_kwargs"""
        with tempfile.TemporaryDirectory() as out_dir:
            image_name = os.path.join(out_dir, "test.png")
            img = np.random.rand(2, 3, 3)
            img_save_val = (255 * img).astype(np.uint8)
            writer_obj = PILWriter(output_dtype=np.uint8)
            writer_obj.set_data_array(img, channel_dim=-1)
            writer_obj.write(image_name)
            out = np.asarray(Image.open(image_name))
            out = np.moveaxis(out, 0, 1)
            np.testing.assert_allclose(out, img_save_val)

    def test_write_2channels(self):
        with tempfile.TemporaryDirectory() as out_dir:
            image_name = os.path.join(out_dir, "test.png")
            img = np.random.rand(2, 3, 2)
            img_save_val = (255 * img).astype(np.uint8)
            writer_obj = PILWriter(output_dtype=np.uint8)
            writer_obj.set_data_array(img, channel_dim=-1)
            writer_obj.write(image_name, format="PNG")
            out = np.asarray(Image.open(image_name))
            out = np.moveaxis(out, 0, 1)
            np.testing.assert_allclose(out, img_save_val)

    def test_write_output_shape(self):
        with tempfile.TemporaryDirectory() as out_dir:
            image_name = os.path.join(out_dir, "test.png")
            img = np.random.rand(2, 2, 3)
            writer_obj = PILWriter(output_dtype=np.uint8)
            writer_obj.set_data_array(img, channel_dim=-1)
            writer_obj.set_metadata({"spatial_shape": (4, 4)}, scale=255)
            writer_obj.write(image_name, format="PNG")
            out = np.asarray(Image.open(image_name))
            np.testing.assert_allclose(out.shape, (4, 4, 3))


if __name__ == "__main__":
    unittest.main()
