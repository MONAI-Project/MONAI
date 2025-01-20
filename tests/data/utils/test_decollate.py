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

import sys
import unittest
from enum import Enum

import numpy as np
import torch
from parameterized import parameterized

from monai.data import CacheDataset, DataLoader, Dataset, create_test_image_2d
from monai.data.utils import decollate_batch
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    EnsureChannelFirstd,
    LoadImage,
    LoadImaged,
    RandAffine,
    RandFlip,
    RandFlipd,
    RandRotate90,
    SpatialPad,
    SpatialPadd,
    ToTensor,
    ToTensord,
)
from monai.transforms.inverse_batch_transform import Decollated
from monai.transforms.spatial.dictionary import RandAffined, RandRotate90d
from monai.utils import optional_import, set_determinism
from monai.utils.enums import PostFix, TraceKeys
from tests.util import make_nifti_image

_, has_nib = optional_import("nibabel")

KEYS = ["image"]

TESTS_DICT: list[tuple] = []
TESTS_DICT.append((SpatialPadd(KEYS, 150), RandFlipd(KEYS, prob=1.0, spatial_axis=1)))
TESTS_DICT.append((RandRotate90d(KEYS, prob=0.0, max_k=1),))
TESTS_DICT.append((RandAffined(KEYS, prob=0.0, translate_range=10),))

TESTS_LIST: list[tuple] = []
TESTS_LIST.append((SpatialPad(150), RandFlip(prob=1.0, spatial_axis=1)))
TESTS_LIST.append((RandRotate90(prob=0.0, max_k=1),))
TESTS_LIST.append((RandAffine(prob=0.0, translate_range=10),))

TEST_BASIC = [
    [("channel", "channel"), ["channel", "channel"]],
    [torch.Tensor([1, 2, 3]), [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)]],
    [
        [[torch.Tensor((1.0, 2.0, 3.0)), torch.Tensor((2.0, 3.0, 1.0))]],
        [
            [[torch.tensor(1.0), torch.tensor(2.0)]],
            [[torch.tensor(2.0), torch.tensor(3.0)]],
            [[torch.tensor(3.0), torch.tensor(1.0)]],
        ],
    ],
    [torch.Tensor((True, True, False, False)), [1.0, 1.0, 0.0, 0.0]],
    [
        [torch.Tensor([1]), torch.Tensor([2]), torch.Tensor([3])],
        [[torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)]],
    ],
    [[None, None], [None, None]],
    [["test"], ["test"]],
    [np.array([64, 64]), [64, 64]],
    [[], []],
    [[("ch1", "ch2"), ("ch3",)], [["ch1", "ch3"], ["ch2", None]]],  # default pad None
]


class TestDeCollate(unittest.TestCase):
    def setUp(self) -> None:
        set_determinism(seed=0)

        im = create_test_image_2d(100, 101)[0]
        self.data_dict = [{"image": make_nifti_image(im) if has_nib else im} for _ in range(6)]
        self.data_list = [make_nifti_image(im) if has_nib else im for _ in range(6)]

    def tearDown(self) -> None:
        set_determinism(None)

    def check_match(self, in1, in2):
        if isinstance(in1, dict):
            self.assertTrue(isinstance(in2, dict))
            for (k1, v1), (k2, v2) in zip(in1.items(), in2.items()):
                if isinstance(k1, Enum) and isinstance(k2, Enum):
                    k1, k2 = k1.value, k2.value
                self.check_match(k1, k2)
                # Transform ids won't match for windows with multiprocessing, so don't check values
                if k1 == TraceKeys.ID and sys.platform in ["darwin", "win32"]:
                    continue
                if not (isinstance(k1, str) and k1.endswith("_transforms")):
                    self.check_match(v1, v2)  # transform stack not necessarily match
        elif isinstance(in1, (list, tuple)):
            for l1, l2 in zip(in1, in2):
                self.check_match(l1, l2)
        elif isinstance(in1, (str, int)):
            self.assertEqual(in1, in2)
        elif isinstance(in1, (torch.Tensor, np.ndarray)):
            np.testing.assert_array_equal(in1, in2)
        else:
            raise RuntimeError(f"Not sure how to compare types. type(in1): {type(in1)}, type(in2): {type(in2)}")

    def check_decollate(self, dataset):
        batch_size = 2
        num_workers = 2 if sys.platform == "linux" else 0

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        for b, batch_data in enumerate(loader):
            decollated_1 = decollate_batch(batch_data)
            decollated_2 = Decollated(detach=True)(batch_data)

            for decollated in [decollated_1, decollated_2]:
                for i, d in enumerate(decollated):
                    self.check_match(dataset[b * batch_size + i], d)

    @parameterized.expand(TESTS_DICT)
    def test_decollation_dict(self, *transforms):
        t_compose = Compose([EnsureChannelFirstd(KEYS, channel_dim="no_channel"), Compose(transforms), ToTensord(KEYS)])
        # If nibabel present, read from disk
        if has_nib:
            t_compose = Compose([LoadImaged("image", image_only=True), t_compose])

        dataset = CacheDataset(self.data_dict, t_compose, progress=False)
        self.check_decollate(dataset=dataset)

    @parameterized.expand(TESTS_LIST)
    def test_decollation_tensor(self, *transforms):
        t_compose = Compose([EnsureChannelFirst(channel_dim="no_channel"), Compose(transforms), ToTensor()])
        # If nibabel present, read from disk
        if has_nib:
            t_compose = Compose([LoadImage(image_only=True), t_compose])

        dataset = Dataset(self.data_list, t_compose)
        self.check_decollate(dataset=dataset)

    @parameterized.expand(TESTS_LIST)
    def test_decollation_list(self, *transforms):
        t_compose = Compose([EnsureChannelFirst(channel_dim="no_channel"), Compose(transforms), ToTensor()])
        # If nibabel present, read from disk
        if has_nib:
            t_compose = Compose([LoadImage(image_only=True), t_compose])

        dataset = Dataset(self.data_list, t_compose)
        self.check_decollate(dataset=dataset)


class TestBasicDeCollate(unittest.TestCase):
    @parameterized.expand(TEST_BASIC)
    def test_decollation_examples(self, input_val, expected_out):
        out = decollate_batch(input_val)
        self.assertListEqual(expected_out, out)

    def test_dict_examples(self):
        test_case = {"meta": {"out": ["test", "test"]}, PostFix.meta("image"): {"scl_slope": torch.Tensor((0.0, 0.0))}}
        out = decollate_batch(test_case)
        self.assertEqual(out[0]["meta"]["out"], "test")
        self.assertEqual(out[0][PostFix.meta("image")]["scl_slope"], 0.0)

        test_case = [torch.ones((2, 1, 10, 10)), torch.ones((2, 3, 5, 5))]
        out = decollate_batch(test_case)
        self.assertTupleEqual(out[0][0].shape, (1, 10, 10))
        self.assertTupleEqual(out[0][1].shape, (3, 5, 5))

        test_case = torch.rand((2, 1, 10, 10))
        out = decollate_batch(test_case)
        self.assertTupleEqual(out[0].shape, (1, 10, 10))

        test_case = [torch.tensor(0), torch.tensor(0)]
        out = decollate_batch(test_case, detach=True)
        self.assertListEqual([0, 0], out)
        self.assertFalse(isinstance(out[0], torch.Tensor))

        test_case = {"a": [torch.tensor(0), torch.tensor(0)]}
        out = decollate_batch(test_case, detach=False)
        self.assertListEqual([{"a": torch.tensor(0)}, {"a": torch.tensor(0)}], out)
        self.assertTrue(isinstance(out[0]["a"], torch.Tensor))

        test_case = [torch.tensor(0), torch.tensor(0)]
        out = decollate_batch(test_case, detach=False)
        self.assertListEqual(test_case, out)

        test_case = {
            "image": torch.tensor([[[1, 2]], [[3, 4]]]),
            "label": torch.tensor([[[5, 6]], [[7, 8]]]),
            "pred": torch.tensor([[[9, 10]], [[11, 12]]]),
            "out": ["test"],
        }
        out = decollate_batch(test_case, detach=False)
        self.assertEqual(out[0]["out"], "test")

        test_case = {
            "image": torch.tensor([[[1, 2, 3]], [[3, 4, 5]]]),
            "label": torch.tensor([[[5]], [[7]]]),
            "out": ["test"],
        }
        out = decollate_batch(test_case, detach=False, pad=False)
        self.assertEqual(len(out), 1)  # no padding
        out = decollate_batch(test_case, detach=False, pad=True, fill_value=0)
        self.assertEqual(out[1]["out"], 0)  # verify padding fill_value

    def test_decollated(self):
        test_case = {
            "image": torch.tensor([[[1, 2]], [[3, 4]]]),
            "meta": {"out": ["test", "test"]},
            PostFix.meta("image"): {"scl_slope": torch.Tensor((0.0, 0.0))},
            "loss": 0.85,
        }
        transform = Decollated(keys=["meta", PostFix.meta("image")], detach=False)
        out = transform(test_case)
        self.assertFalse("loss" in out)
        self.assertEqual(out[0]["meta"]["out"], "test")
        self.assertEqual(out[0][PostFix.meta("image")]["scl_slope"], 0.0)
        self.assertTrue(isinstance(out[0][PostFix.meta("image")]["scl_slope"], torch.Tensor))
        # decollate all data with keys=None
        transform = Decollated(keys=None, detach=True)
        out = transform(test_case)
        self.assertEqual(out[1]["loss"], 0.85)
        self.assertEqual(out[0]["meta"]["out"], "test")
        self.assertEqual(out[0][PostFix.meta("image")]["scl_slope"], 0.0)
        self.assertTrue(isinstance(out[0][PostFix.meta("image")]["scl_slope"], float))

        # test list input
        test_case = [
            torch.tensor([[[1, 2]], [[3, 4]]]),
            {"out": ["test", "test"]},
            {"scl_slope": torch.Tensor((0.0, 0.0))},
            {"out2": ["test1"]},
            0.85,
            [],
        ]
        transform = Decollated(keys=None, detach=False, fill_value=-1)
        out = transform(test_case)

        self.assertEqual(out[0][-2], 0.85)  # scalar replicates
        self.assertEqual(out[1][-2], 0.85)  # scalar replicates
        self.assertEqual(out[1][-3], -1)  # fill value for the dictionary item
        self.assertEqual(out[0][1]["out"], "test")
        self.assertEqual(out[0][2]["scl_slope"], 0.0)
        self.assertTrue(isinstance(out[0][2]["scl_slope"], torch.Tensor))


if __name__ == "__main__":
    unittest.main()
