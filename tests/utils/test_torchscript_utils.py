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

import torch

from monai.config import get_config_values
from monai.data import load_net_with_metadata, save_net_with_metadata
from monai.utils import JITMetadataKeys


class TestModule(torch.nn.Module):
    __test__ = False  # indicate to pytest that this class is not intended for collection

    def forward(self, x):
        return x + 10


class TestTorchscript(unittest.TestCase):

    def test_save_net_with_metadata(self):
        """Save a network without metadata to a file."""
        m = torch.jit.script(TestModule())

        with tempfile.TemporaryDirectory() as tempdir:
            save_net_with_metadata(m, f"{tempdir}/test")

            self.assertTrue(os.path.isfile(f"{tempdir}/test.ts"))

    def test_save_net_with_metadata_ext(self):
        """Save a network without metadata to a file."""
        m = torch.jit.script(TestModule())

        with tempfile.TemporaryDirectory() as tempdir:
            save_net_with_metadata(m, f"{tempdir}/test.zip")

            self.assertTrue(os.path.isfile(f"{tempdir}/test.zip"))

    def test_save_net_with_metadata_with_extra(self):
        """Save a network with simple metadata to a file."""
        m = torch.jit.script(TestModule())

        test_metadata = {"foo": [1, 2], "bar": "string"}

        with tempfile.TemporaryDirectory() as tempdir:
            save_net_with_metadata(m, f"{tempdir}/test", meta_values=test_metadata)

            self.assertTrue(os.path.isfile(f"{tempdir}/test.ts"))

    def test_load_net_with_metadata(self):
        """Save then load a network with no metadata or other extra files."""
        m = torch.jit.script(TestModule())

        with tempfile.TemporaryDirectory() as tempdir:
            save_net_with_metadata(m, f"{tempdir}/test")
            _, meta, extra_files = load_net_with_metadata(f"{tempdir}/test.ts")

        del meta[JITMetadataKeys.TIMESTAMP.value]  # no way of knowing precisely what this value would be

        self.assertEqual(meta, get_config_values())
        self.assertEqual(extra_files, {})

    def test_load_net_with_metadata_with_extra(self):
        """Save then load a network with basic metadata."""
        m = torch.jit.script(TestModule())

        test_metadata = {"foo": [1, 2], "bar": "string"}

        with tempfile.TemporaryDirectory() as tempdir:
            save_net_with_metadata(m, f"{tempdir}/test", meta_values=test_metadata)
            _, meta, extra_files = load_net_with_metadata(f"{tempdir}/test.ts")

        del meta[JITMetadataKeys.TIMESTAMP.value]  # no way of knowing precisely what this value would be

        test_compare = get_config_values()
        test_compare.update(test_metadata)

        self.assertEqual(meta, test_compare)
        self.assertEqual(extra_files, {})

    def test_save_load_more_extra_files(self):
        """Save then load extra file data from a torchscript file."""
        m = torch.jit.script(TestModule())

        test_metadata = {"foo": [1, 2], "bar": "string"}

        more_extra_files = {"test.txt": b"This is test data"}

        with tempfile.TemporaryDirectory() as tempdir:
            save_net_with_metadata(m, f"{tempdir}/test", meta_values=test_metadata, more_extra_files=more_extra_files)

            self.assertTrue(os.path.isfile(f"{tempdir}/test.ts"))

            _, _, loaded_extra_files = load_net_with_metadata(f"{tempdir}/test.ts", more_extra_files=("test.txt",))

            self.assertEqual(more_extra_files["test.txt"], loaded_extra_files["test.txt"])


if __name__ == "__main__":
    unittest.main()
