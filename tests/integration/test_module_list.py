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

import glob
import inspect
import os
import pathlib
import unittest

import monai


class TestAllImport(unittest.TestCase):

    def test_public_api(self):
        """
        This is to check "monai.__all__" should be consistent with
        the top-level folders except for "__pycache__", "_extensions" and "csrc" (cpp/cuda src)
        """
        base_folder = os.path.dirname(monai.__file__)
        to_search = os.path.join(base_folder, "*", "")
        subfolders = [os.path.basename(x[:-1]) for x in glob.glob(to_search)]
        to_exclude = ("__pycache__", "_extensions", "csrc")
        mod = []
        for code_folder in subfolders:
            if code_folder in to_exclude:
                continue
            mod.append(code_folder)
        self.assertEqual(sorted(monai.__all__), sorted(mod))

    def test_transform_api(self):
        """monai subclasses of MapTransforms must have alias names ending with 'd', 'D', 'Dict'"""
        to_exclude = {"MapTransform"}  # except for these transforms
        to_exclude_docs = {"Decollate", "Ensemble", "Invert", "SaveClassification", "RandTorchVision", "RandCrop"}
        to_exclude_docs.update({"DeleteItems", "SelectItems", "FlattenSubKeys", "CopyItems", "ConcatItems"})
        to_exclude_docs.update({"ToMetaTensor", "FromMetaTensor"})
        xforms = {
            name: obj
            for name, obj in monai.transforms.__dict__.items()
            if inspect.isclass(obj) and issubclass(obj, monai.transforms.MapTransform)
        }
        names = sorted(x for x in xforms if x not in to_exclude)
        remained = set(names)
        doc_file = os.path.join(pathlib.Path(__file__).parent.parent, "docs", "source", "transforms.rst")
        contents = pathlib.Path(doc_file).read_text() if os.path.exists(doc_file) else None
        for n in names:
            if not n.endswith("d"):
                continue
            with self.subTest(n=n):
                basename = n[:-1]  # Transformd basename is Transform

                # remove aliases to check, do this before the assert below so that a failed assert does skip this
                for postfix in ("D", "d", "Dict"):
                    remained.remove(f"{basename}{postfix}")

                for docname in (f"{basename}", f"{basename}d"):
                    if docname in to_exclude_docs:
                        continue
                    if (contents is not None) and f"`{docname}`" not in f"{contents}":
                        self.assertTrue(False, f"please add `{docname}` to docs/source/transforms.rst")

        self.assertFalse(remained)


if __name__ == "__main__":
    unittest.main()
