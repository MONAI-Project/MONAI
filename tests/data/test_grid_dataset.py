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

import numpy as np
from parameterized import parameterized

from monai.data import DataLoader, GridPatchDataset, PatchIter, PatchIterd, iter_patch
from monai.transforms import RandShiftIntensity, RandShiftIntensityd
from monai.utils import set_determinism
from tests.util import TEST_NDARRAYS, assert_allclose, get_arange_img


def identity_generator(x):
    # simple transform that returns the input itself
    for idx, item in enumerate(x):
        yield item, idx


TEST_CASES_ITER_PATCH = []
for p in TEST_NDARRAYS:
    TEST_CASES_ITER_PATCH.append([p, True])
    TEST_CASES_ITER_PATCH.append([p, False])

A = np.arange(16).repeat(3).reshape(4, 4, 3).transpose(2, 0, 1)
A11 = A[:, :2, :2]
A12 = A[:, :2, 2:]
A21 = A[:, 2:, :2]
A22 = A[:, 2:, 2:]
COORD11 = [[0, 3], [0, 2], [0, 2]]
COORD12 = [[0, 3], [0, 2], [2, 4]]
COORD21 = [[0, 3], [2, 4], [0, 2]]
COORD22 = [[0, 3], [2, 4], [2, 4]]

TEST_CASE_0 = [{"patch_size": (2, 2)}, A, [A11, A12, A21, A22], np.array([COORD11, COORD12, COORD21, COORD22])]
TEST_CASE_1 = [{"patch_size": (2, 2), "start_pos": (0, 2, 2)}, A, [A22], np.array([COORD22])]
TEST_CASE_2 = [{"patch_size": (2, 2), "start_pos": (0, 0, 2)}, A, [A12, A22], np.array([COORD12, COORD22])]
TEST_CASE_3 = [{"patch_size": (2, 2), "start_pos": (0, 2, 0)}, A, [A21, A22], np.array([COORD21, COORD22])]

TEST_CASES_PATCH_ITER = []
for p in TEST_NDARRAYS:
    TEST_CASES_PATCH_ITER.append([p, *TEST_CASE_0])
    TEST_CASES_PATCH_ITER.append([p, *TEST_CASE_1])
    TEST_CASES_PATCH_ITER.append([p, *TEST_CASE_2])
    TEST_CASES_PATCH_ITER.append([p, *TEST_CASE_3])


class TestGridPatchDataset(unittest.TestCase):
    def setUp(self):
        set_determinism(seed=1234)

    def tearDown(self):
        set_determinism(None)

    @parameterized.expand(TEST_CASES_ITER_PATCH)
    def test_iter_patch(self, in_type, cb):
        shape = (10, 30, 30)
        input_img = in_type(get_arange_img(shape))
        for p, _ in iter_patch(input_img, patch_size=(None, 10, 30, None), copy_back=cb):
            p += 1.0
            assert_allclose(p, in_type(get_arange_img(shape)) + 1.0, type_test=True, device_test=True)
        assert_allclose(
            input_img, in_type(get_arange_img(shape)) + (1.0 if cb else 0.0), type_test=True, device_test=True
        )

    @parameterized.expand(TEST_CASES_PATCH_ITER)
    def test_patch_iter(self, in_type, input_parameters, image, expected, coords):
        input_image = in_type(image)
        patch_iterator = PatchIter(**input_parameters)(input_image)
        for (result_image, result_loc), expected_patch, coord in zip(patch_iterator, expected, coords):
            assert_allclose(result_image, in_type(expected_patch), type_test=True, device_test=True)
            assert_allclose(result_loc, coord, type_test=True, device_test=True)

    @parameterized.expand(TEST_CASES_PATCH_ITER)
    def test_patch_iterd(self, in_type, input_parameters, image, expected, coords):
        image_key = "image"
        input_dict = {image_key: in_type(image)}
        patch_iterator = PatchIterd(keys=image_key, **input_parameters)(input_dict)
        for (result_image_dict, result_loc), expected_patch, coord in zip(patch_iterator, expected, coords):
            assert_allclose(result_image_dict[image_key], in_type(expected_patch), type_test=True, device_test=True)
            assert_allclose(result_loc, coord, type_test=True, device_test=True)

    def test_shape(self):
        # test Iterable input data
        test_dataset = iter(["vwxyz", "helloworld", "worldfoobar"])
        result = GridPatchDataset(data=test_dataset, patch_iter=identity_generator, with_coordinates=False)
        output = []
        n_workers = 0 if sys.platform == "win32" else 2
        for item in DataLoader(result, batch_size=3, num_workers=n_workers):
            output.append("".join(item))
        if sys.platform == "win32":
            expected = ["ar", "ell", "ldf", "oob", "owo", "rld", "vwx", "wor", "yzh"]
        else:
            expected = ["d", "dfo", "hel", "low", "oba", "orl", "orl", "r", "vwx", "yzw"]
            self.assertEqual(len("".join(expected)), len("".join(list(test_dataset))))
        self.assertEqual(sorted(output), sorted(expected))

    def test_loading_array(self):
        # test sequence input data with images
        images = [np.arange(16, dtype=float).reshape(1, 4, 4), np.arange(16, dtype=float).reshape(1, 4, 4)]
        # image level
        patch_intensity = RandShiftIntensity(offsets=1.0, prob=1.0).set_random_state(seed=1234)
        patch_iter = PatchIter(patch_size=(2, 2), start_pos=(0, 0))
        ds = GridPatchDataset(data=images, patch_iter=patch_iter, transform=patch_intensity)
        # use the grid patch dataset
        for item in DataLoader(ds, batch_size=2, shuffle=False, num_workers=0):
            np.testing.assert_equal(tuple(item[0].shape), (2, 1, 2, 2))
        np.testing.assert_allclose(
            item[0],
            np.array([[[[8.708934, 9.708934], [12.708934, 13.708934]]], [[[10.8683, 11.8683], [14.8683, 15.8683]]]]),
            rtol=1e-4,
        )
        np.testing.assert_allclose(item[1], np.array([[[0, 1], [2, 4], [0, 2]], [[0, 1], [2, 4], [2, 4]]]), rtol=1e-5)
        if sys.platform != "win32":
            for item in DataLoader(ds, batch_size=2, shuffle=False, num_workers=2):
                np.testing.assert_equal(tuple(item[0].shape), (2, 1, 2, 2))
            np.testing.assert_allclose(
                item[0],
                np.array([[[[7.27427, 8.27427], [11.27427, 12.27427]]], [[[9.4353, 10.4353], [13.4353, 14.4353]]]]),
                rtol=1e-3,
            )
            np.testing.assert_allclose(
                item[1], np.array([[[0, 1], [2, 4], [0, 2]], [[0, 1], [2, 4], [2, 4]]]), rtol=1e-5
            )

    def test_loading_dict(self):
        set_determinism(seed=1234)
        # test sequence input data with dict
        data = [
            {
                "image": np.arange(16, dtype=float).reshape(1, 4, 4),
                "label": np.arange(16, dtype=float).reshape(1, 4, 4),
                "metadata": "test string",
            },
            {
                "image": np.arange(16, dtype=float).reshape(1, 4, 4),
                "label": np.arange(16, dtype=float).reshape(1, 4, 4),
                "metadata": "test string",
            },
        ]
        # image level
        patch_intensity = RandShiftIntensityd(keys="image", offsets=1.0, prob=1.0)
        patch_iter = PatchIterd(keys=["image", "label"], patch_size=(2, 2), start_pos=(0, 0))
        ds = GridPatchDataset(data=data, patch_iter=patch_iter, transform=patch_intensity, with_coordinates=True)
        # use the grid patch dataset
        for item in DataLoader(ds, batch_size=2, shuffle=False, num_workers=0):
            np.testing.assert_equal(item[0]["image"].shape, (2, 1, 2, 2))
            np.testing.assert_equal(item[0]["label"].shape, (2, 1, 2, 2))
            self.assertListEqual(item[0]["metadata"], ["test string", "test string"])
        np.testing.assert_allclose(
            item[0]["image"],
            np.array([[[[8.708934, 9.708934], [12.708934, 13.708934]]], [[[10.8683, 11.8683], [14.8683, 15.8683]]]]),
            rtol=1e-4,
        )
        np.testing.assert_allclose(item[1], np.array([[[0, 1], [2, 4], [0, 2]], [[0, 1], [2, 4], [2, 4]]]), rtol=1e-5)
        if sys.platform != "win32":
            for item in DataLoader(ds, batch_size=2, shuffle=False, num_workers=2):
                np.testing.assert_equal(item[0]["image"].shape, (2, 1, 2, 2))
            np.testing.assert_allclose(
                item[0]["image"],
                np.array([[[[7.27427, 8.27427], [11.27427, 12.27427]]], [[[9.4353, 10.4353], [13.4353, 14.4353]]]]),
                rtol=1e-3,
            )
            np.testing.assert_allclose(
                item[1], np.array([[[0, 1], [2, 4], [0, 2]], [[0, 1], [2, 4], [2, 4]]]), rtol=1e-5
            )

    def test_set_data(self):
        from monai.transforms import Compose, Lambda, RandLambda

        images = [np.arange(2, 18, dtype=float).reshape(1, 4, 4), np.arange(16, dtype=float).reshape(1, 4, 4)]

        transform = Compose(
            [Lambda(func=lambda x: np.array(x * 10)), RandLambda(func=lambda x: x + 1)], map_items=False
        )
        patch_iter = PatchIter(patch_size=(2, 2), start_pos=(0, 0))
        dataset = GridPatchDataset(
            data=images,
            patch_iter=patch_iter,
            transform=transform,
            cache=True,
            cache_rate=1.0,
            copy_cache=not sys.platform == "linux",
        )

        num_workers = 2 if sys.platform == "linux" else 0
        for item in DataLoader(dataset, batch_size=2, shuffle=False, num_workers=num_workers):
            np.testing.assert_equal(tuple(item[0].shape), (2, 1, 2, 2))
        np.testing.assert_allclose(item[0], np.array([[[[81, 91], [121, 131]]], [[[101, 111], [141, 151]]]]), rtol=1e-4)
        np.testing.assert_allclose(item[1], np.array([[[0, 1], [2, 4], [0, 2]], [[0, 1], [2, 4], [2, 4]]]), rtol=1e-5)
        # simulate another epoch, the cache content should not be modified
        for item in DataLoader(dataset, batch_size=2, shuffle=False, num_workers=num_workers):
            np.testing.assert_equal(tuple(item[0].shape), (2, 1, 2, 2))
        np.testing.assert_allclose(item[0], np.array([[[[81, 91], [121, 131]]], [[[101, 111], [141, 151]]]]), rtol=1e-4)
        np.testing.assert_allclose(item[1], np.array([[[0, 1], [2, 4], [0, 2]], [[0, 1], [2, 4], [2, 4]]]), rtol=1e-5)

        # update the datalist and fill the cache content
        data_list2 = [np.arange(1, 17, dtype=float).reshape(1, 4, 4)]
        dataset.set_data(data=data_list2)
        # rerun with updated cache content
        for item in DataLoader(dataset, batch_size=2, shuffle=False, num_workers=num_workers):
            np.testing.assert_equal(tuple(item[0].shape), (2, 1, 2, 2))
        np.testing.assert_allclose(
            item[0], np.array([[[[91, 101], [131, 141]]], [[[111, 121], [151, 161]]]]), rtol=1e-4
        )
        np.testing.assert_allclose(item[1], np.array([[[0, 1], [2, 4], [0, 2]], [[0, 1], [2, 4], [2, 4]]]), rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
