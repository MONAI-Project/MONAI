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

from monai.data.meta_tensor import MetaTensor
from monai.transforms import Compose
from monai.transforms.lazy.functional import apply_pending
from monai.transforms.transform import MapTransform
from monai.utils.enums import NumpyPadMode, PytorchPadMode
from tests.util import TEST_NDARRAYS_ALL, assert_allclose

MODES = []
# Test modes
NP_MODES: list = [
    "constant",
    "edge",
    # `reflect` mode is not supported in some PyTorch versions, skip the test
    # "reflect",
    "wrap",
    "median",
    "mean",
]
MODES += NP_MODES
MODES += [NumpyPadMode(i) for i in NP_MODES]

PT_MODES: list = [
    "constant",
    "replicate",
    "circular",
    # `reflect` mode is not supported in some PyTorch versions, skip the test
    # "reflect",
]
MODES += PT_MODES
MODES += [PytorchPadMode(i) for i in PT_MODES]

TESTS_PENDING_MODE = [["constant", "zeros"], ["edge", "border"]]


class PadTest(unittest.TestCase):
    @staticmethod
    def get_arr(shape):
        return np.random.randint(100, size=shape).astype(float)

    def pad_test(self, input_param, input_shape, expected_shape, modes=None):
        # loop over each mode
        for mode in modes or MODES:
            with self.subTest(mode=mode):
                base_comparison = None
                im = self.get_arr(input_shape)
                padder = self.Padder(mode=mode, **input_param)
                is_map = isinstance(padder, MapTransform)
                # check result is the same regardless of input type
                for im_type in TEST_NDARRAYS_ALL:
                    with self.subTest(im_type=im_type):
                        input_image = im_type(im)
                        input_data = {"img": im_type(im)} if is_map else im_type(im)
                        # our array transforms can also take `mode` as an argument to `__call__`
                        # Check this gives equivalent results
                        for call_extra_args in [{}] if is_map else [{}, {"mode": mode}]:
                            with self.subTest(call_extra_args=call_extra_args):
                                r_out = padder(input_data, **call_extra_args)
                                r_im = r_out["img"] if is_map else r_out
                                # check shape, type, etc.
                                np.testing.assert_allclose(r_im.shape, expected_shape)
                                self.assertIsInstance(r_im, MetaTensor)
                                self.assertEqual(len(r_im.applied_operations), 1)
                                # check results are same regardless of input type
                                if base_comparison is None:
                                    base_comparison = r_im
                                else:
                                    assert_allclose(r_im, base_comparison)
                                # test inverse
                                if isinstance(r_im, MetaTensor):
                                    r_out = padder.inverse(r_out)
                                    r_im = r_out["img"] if is_map else r_out
                                    self.assertIsInstance(r_im, MetaTensor)
                                    assert_allclose(r_im, input_image, type_test=False)
                                    self.assertEqual(r_im.applied_operations, [])

    def pad_test_kwargs(self, unchanged_slices, **input_param):
        for im_type in TEST_NDARRAYS_ALL:
            with self.subTest(im_type=im_type):
                for kwargs in ({"value": 2}, {"constant_values": ((0, 0), (1, 1), (2, 2))}):
                    with self.subTest(kwargs=kwargs):
                        im = im_type(np.random.randint(-100, -10, size=(3, 8, 4)))
                        padder = self.Padder(**input_param, **kwargs)
                        result = padder(im)
                        if isinstance(result, torch.Tensor):
                            result = result.cpu()
                        assert_allclose(result[unchanged_slices], im, type_test=False)
                        # we should have the same as the input plus some 2s (if value) or 1s and 2s (if constant_values)
                        if isinstance(im, torch.Tensor):
                            im = im.detach().cpu().numpy()
                        expected_vals = np.unique(im).tolist()
                        expected_vals += [2] if "value" in kwargs else [1, 2]
                        assert_allclose(np.unique(result), expected_vals, type_test=False)
                        # check inverse
                        if isinstance(result, MetaTensor):
                            inv = padder.inverse(result)
                            assert_allclose(im, inv, type_test=False)
                            self.assertEqual(inv.applied_operations, [])

    def pad_test_pending_ops(self, input_param, input_shape):
        for mode in TESTS_PENDING_MODE:
            # TODO: One of the dim in the input data contains 1 report error.
            pad_fn = self.Padder(mode=mode[0], **input_param)
            data = self.get_arr(input_shape)
            is_map = isinstance(pad_fn, MapTransform)
            im = MetaTensor(data, meta={"a": "b", "affine": np.eye(len(input_shape))})
            input_data = {"img": im} if is_map else im
            # non-lazy
            result_non_lazy = pad_fn(input_data)
            expected = result_non_lazy["img"] if is_map else result_non_lazy
            self.assertIsInstance(expected, MetaTensor)
            # lazy
            pad_fn.lazy = True
            pending_result = pad_fn(input_data)
            pending_result = pending_result["img"] if is_map else pending_result
            self.assertIsInstance(pending_result, MetaTensor)
            assert_allclose(pending_result.peek_pending_affine(), expected.affine)
            assert_allclose(pending_result.peek_pending_shape(), expected.shape[1:])
            # TODO: mode="bilinear" may report error
            overrides = {"mode": "nearest", "padding_mode": mode[1], "align_corners": False}
            result = apply_pending(pending_result, overrides=overrides)[0]
            # lazy in constructor
            pad_fn_lazy = self.Padder(mode=mode[0], lazy=True, **input_param)
            self.assertTrue(pad_fn_lazy.lazy)
            # compare
            assert_allclose(result, expected, rtol=1e-5)
            if isinstance(result, MetaTensor) and not isinstance(pad_fn, MapTransform):
                pad_fn.lazy = False
                inverted = pad_fn.inverse(result)
                self.assertTrue((not inverted.pending_operations) and (not inverted.applied_operations))
                self.assertEqual(inverted.shape, im.shape)

    def pad_test_combine_ops(self, funcs, input_shape, expected_shape):
        for mode in TESTS_PENDING_MODE:
            # non-lazy
            _funcs = []
            for func in funcs:
                for _func, _params in func.items():
                    _funcs.append(_func(mode=mode[0], **_params))
            trans = Compose(_funcs)
            data = self.get_arr(input_shape)
            is_map = isinstance(_funcs[0], MapTransform)
            im = MetaTensor(data, meta={"a": "b", "affine": np.eye(len(input_shape))})
            input_data = {"img": im} if is_map else im
            result_non_lazy = trans(input_data)
            expected = result_non_lazy["img"] if is_map else result_non_lazy
            self.assertIsInstance(expected, MetaTensor)
            # lazy
            pending_result = input_data
            for _func in _funcs:
                _func.lazy = True
                pending_result = _func(pending_result)
            pending_result = pending_result["img"] if is_map else pending_result
            self.assertIsInstance(pending_result, MetaTensor)
            assert_allclose(pending_result.peek_pending_affine(), expected.affine)
            assert_allclose(pending_result.peek_pending_shape(), expected.shape[1:])
            # TODO: mode="bilinear" may report error
            overrides = {"mode": "nearest", "padding_mode": mode[1], "align_corners": False}
            result = apply_pending(pending_result, overrides=overrides)[0]
            # compare
            assert_allclose(result, expected, rtol=1e-5)
