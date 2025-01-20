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
import torch
from parameterized import parameterized

from monai.data import MetaTensor
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type, get_equivalent_dtype
from tests.util import TEST_NDARRAYS_ALL, assert_allclose

TESTS: list[tuple] = []
for in_type in TEST_NDARRAYS_ALL + (int, float):
    for out_type in TEST_NDARRAYS_ALL:
        TESTS.append((in_type(np.array(1.0)), out_type(np.array(1.0)), None, False))  # type: ignore
        if in_type is not float:
            TESTS.append((in_type(np.array(256)), out_type(np.array(255)), np.uint8, True))  # type: ignore

TESTS_LIST: list[tuple] = []
for in_type in TEST_NDARRAYS_ALL + (int, float):
    for out_type in TEST_NDARRAYS_ALL:
        TESTS_LIST.append(
            (
                [in_type(np.array(1.0)), in_type(np.array(1.0))],  # type: ignore
                out_type(np.array([1.0, 1.0])),
                True,
                None,
                False,
            )
        )
        TESTS_LIST.append(
            (
                [in_type(np.array(1.0)), in_type(np.array(1.0))],  # type: ignore
                [out_type(np.array(1.0)), out_type(np.array(1.0))],
                False,
                None,
                False,
            )
        )
        if in_type is not float:
            TESTS_LIST.append(
                (
                    [in_type(np.array(257)), in_type(np.array(1))],  # type: ignore
                    out_type(np.array([255, 1])),
                    True,
                    np.uint8,
                    True,
                )
            )
            TESTS_LIST.append(
                (
                    [in_type(np.array(257)), in_type(np.array(-12))],  # type: ignore
                    [out_type(np.array(255)), out_type(np.array(0))],
                    False,
                    np.uint8,
                    True,
                )
            )

UNSUPPORTED_TYPES = {np.dtype("uint16"): torch.int32, np.dtype("uint32"): torch.int64, np.dtype("uint64"): torch.int64}


class TestTensor(torch.Tensor):
    __test__ = False  # indicate to pytest that this class is not intended for collection
    pass


class TestConvertDataType(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_convert_data_type(self, in_image, im_out, out_dtype, safe):
        converted_im, orig_type, orig_device = convert_data_type(in_image, type(im_out), dtype=out_dtype, safe=safe)
        # check input is unchanged
        self.assertEqual(type(in_image), orig_type)
        if isinstance(in_image, torch.Tensor):
            self.assertEqual(in_image.device, orig_device)
        # check output is desired type
        self.assertEqual(type(converted_im), type(im_out))
        # check data has been clipped
        assert_allclose(converted_im, im_out)
        # check dtype is unchanged
        if out_dtype is None:
            if isinstance(in_image, (np.ndarray, torch.Tensor)):
                self.assertEqual(converted_im.dtype, im_out.dtype)

    def test_neg_stride(self):
        _ = convert_data_type(np.array((1, 2))[::-1], torch.Tensor)

    @parameterized.expand(list(UNSUPPORTED_TYPES.items()))
    def test_unsupported_np_types(self, np_type, pt_type):
        in_image = np.ones(13, dtype=np_type)  # choose a prime size so as to be indivisible by the size of any dtype
        converted_im, orig_type, orig_device = convert_data_type(in_image, torch.Tensor)

        self.assertEqual(converted_im.dtype, pt_type)

    @parameterized.expand(TESTS_LIST)
    def test_convert_list(self, in_image, im_out, wrap, out_dtype, safe):
        output_type = type(im_out) if wrap else type(im_out[0])
        converted_im, *_ = convert_data_type(in_image, output_type, wrap_sequence=wrap, dtype=out_dtype, safe=safe)
        # check output is desired type
        if not wrap:
            converted_im = converted_im[0]
            im_out = im_out[0]
        self.assertEqual(type(converted_im), type(im_out))
        assert_allclose(converted_im, im_out)
        # check dtype is unchanged
        if isinstance(in_image[0], (np.ndarray, torch.Tensor)):
            if out_dtype is None:
                self.assertEqual(converted_im.dtype, im_out.dtype)
            else:
                _out_dtype = get_equivalent_dtype(out_dtype, output_type)
                self.assertEqual(converted_im.dtype, _out_dtype)


class TestConvertDataSame(unittest.TestCase):
    # add test for subclass of Tensor
    @parameterized.expand(TESTS + [(np.array(256), TestTensor(np.array([255])), torch.uint8, True)])
    def test_convert_data_type(self, in_image, im_out, out_dtype, safe):
        converted_im, orig_type, orig_device = convert_to_dst_type(in_image, im_out, dtype=out_dtype, safe=safe)
        # check input is unchanged
        self.assertEqual(type(in_image), orig_type)
        assert_allclose(converted_im, im_out)
        if isinstance(in_image, torch.Tensor):
            self.assertEqual(in_image.device, orig_device)

        # check output is desired type
        if isinstance(im_out, MetaTensor):
            output_type = MetaTensor
        elif isinstance(im_out, torch.Tensor):
            output_type = torch.Tensor
        else:
            output_type = np.ndarray
        self.assertEqual(type(converted_im), output_type)
        # check dtype is unchanged
        if out_dtype is None:
            if isinstance(in_image, (np.ndarray, torch.Tensor, MetaTensor)):
                self.assertEqual(converted_im.dtype, im_out.dtype)


if __name__ == "__main__":
    unittest.main()
