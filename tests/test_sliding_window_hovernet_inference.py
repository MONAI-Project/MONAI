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

import itertools
import unittest

import numpy as np
import torch
from parameterized import parameterized

from monai.apps.pathology.inferers import SlidingWindowHoVerNetInferer
from monai.data.utils import list_data_collate
from monai.inferers import sliding_window_inference
from monai.utils import optional_import
from tests.utils import TEST_TORCH_AND_META_TENSORS, skip_if_no_cuda

_, has_tqdm = optional_import("tqdm")

TEST_CASES_PADDING = [
    [None, (1, 3, 16, 8), (4, 4), 7, 0.5, "constant", torch.device("cpu:0"), None, True],
    ["hover", (1, 3, 16, 8), (4, 4), 7, 0.5, "constant", torch.device("cpu:0"), None, True],
    [None, (1, 3, 16, 8), (4, 4), 7, 0.5, "constant", torch.device("cpu:0"), (1,) * 4, True],
    ["hover", (1, 3, 16, 8), (4, 4), 7, 0.5, "constant", torch.device("cpu:0"), (1,) * 4, True],
]

TEST_CASES = [
    [(2, 3, 16), (4,), 3, 0.25, "constant", torch.device("cpu:0")],  # 1D small roi
    [(2, 3, 16, 15, 7, 9), 4, 3, 0.25, "constant", torch.device("cpu:0")],  # 4D small roi
    [(1, 3, 16, 15, 7), (4, -1, 7), 3, 0.25, "constant", torch.device("cpu:0")],  # 3D small roi
    [(2, 3, 16, 15, 7), (4, -1, 7), 3, 0.25, "constant", torch.device("cpu:0")],  # 3D small roi
    [(3, 3, 16, 15, 7), (4, -1, 7), 3, 0.25, "constant", torch.device("cpu:0")],  # 3D small roi
    [(2, 3, 16, 15, 7), (4, -1, 7), 3, 0.25, "constant", torch.device("cpu:0")],  # 3D small roi
    [(1, 3, 16, 15, 7), (4, 10, 7), 3, 0.25, "constant", torch.device("cpu:0")],  # 3D small roi
    [(1, 3, 16, 15, 7), (20, 22, 23), 10, 0.25, "constant", torch.device("cpu:0")],  # 3D large roi
    [(2, 3, 15, 7), (2, 6), 1000, 0.25, "constant", torch.device("cpu:0")],  # 2D small roi, large batch
    [(1, 3, 16, 7), (80, 50), 7, 0.25, "constant", torch.device("cpu:0")],  # 2D large roi
    [(1, 3, 16, 15, 7), (20, 22, 23), 10, 0.5, "constant", torch.device("cpu:0")],  # 3D large overlap
    [(1, 3, 16, 7), (80, 50), 7, 0.5, "gaussian", torch.device("cpu:0")],  # 2D large overlap, gaussian
    [(1, 3, 16, 15, 7), (4, 10, 7), 3, 0.25, "gaussian", torch.device("cpu:0")],  # 3D small roi, gaussian
    [(3, 3, 16, 15, 7), (4, 10, 7), 3, 0.25, "gaussian", torch.device("cpu:0")],  # 3D small roi, gaussian
    [(1, 3, 16, 15, 7), (4, 10, 7), 3, 0.25, "gaussian", torch.device("cuda:0")],  # test inference on gpu if availabe
    [(1, 3, 16, 15, 7), (4, 1, 7), 3, 0.25, "constant", torch.device("cpu:0")],  # 3D small roi
    [(5, 3, 16, 15, 7), (4, 1, 7), 3, 0.25, "constant", torch.device("cpu:0")],  # 3D small roi
]


class TestSlidingWindowInference(unittest.TestCase):
    @parameterized.expand(TEST_CASES_PADDING)
    def test_sliding_window_with_padding(
        self, key, image_shape, roi_shape, sw_batch_size, overlap, mode, device, extra_input_padding, pad_output
    ):
        n_total = np.prod(image_shape)
        if mode == "constant":
            inputs = torch.arange(n_total, dtype=torch.float).reshape(*image_shape)
        else:
            inputs = torch.ones(*image_shape, dtype=torch.float)
        if device.type == "cuda" and not torch.cuda.is_available():
            device = torch.device("cpu:0")

        def compute(data):
            if key:
                return {key: torch.clone(data[..., 1:-1, 1:-1]) + 1}
            else:
                return torch.clone(data[..., 1:-1, 1:-1]) + 1

        if mode == "constant":
            expected_val = np.arange(n_total, dtype=np.float32).reshape(*image_shape) + 1.0
        else:
            expected_val = np.ones(image_shape, dtype=np.float32) + 1.0

        if extra_input_padding is None:
            expected_val[..., 0, :] = expected_val[..., -1, :] = None
            expected_val[..., 0] = expected_val[..., -1] = None

        sliding_inference = SlidingWindowHoVerNetInferer(
            roi_shape, sw_batch_size, overlap, mode, extra_input_padding=extra_input_padding, pad_output=pad_output
        )
        result = sliding_inference(inputs.to(device), compute)
        result = result[key] if key else result
        np.testing.assert_string_equal(device.type, result.device.type)
        np.testing.assert_allclose(result.cpu().numpy(), expected_val)

    @parameterized.expand(TEST_CASES)
    def test_sliding_window_default(self, image_shape, roi_shape, sw_batch_size, overlap, mode, device):
        n_total = np.prod(image_shape)
        if mode == "constant":
            inputs = torch.arange(n_total, dtype=torch.float).reshape(*image_shape)
        else:
            inputs = torch.ones(*image_shape, dtype=torch.float)
        if device.type == "cuda" and not torch.cuda.is_available():
            device = torch.device("cpu:0")

        def compute(data):
            return data + 1

        if mode == "constant":
            expected_val = np.arange(n_total, dtype=np.float32).reshape(*image_shape) + 1.0
        else:
            expected_val = np.ones(image_shape, dtype=np.float32) + 1.0
        result = sliding_window_inference(inputs.to(device), roi_shape, sw_batch_size, compute, overlap, mode=mode)
        np.testing.assert_string_equal(device.type, result.device.type)
        np.testing.assert_allclose(result.cpu().numpy(), expected_val)

        result = SlidingWindowHoVerNetInferer(roi_shape, sw_batch_size, overlap, mode)(inputs.to(device), compute)
        np.testing.assert_string_equal(device.type, result.device.type)
        np.testing.assert_allclose(result.cpu().numpy(), expected_val)

    @parameterized.expand([[x] for x in TEST_TORCH_AND_META_TENSORS])
    def test_default_device(self, data_type):
        device = "cuda" if torch.cuda.is_available() else "cpu:0"
        inputs = data_type(torch.ones((3, 16, 15, 7))).to(device=device)
        inputs = list_data_collate([inputs])  # make a proper batch
        roi_shape = (4, 10, 7)
        sw_batch_size = 10

        def compute(data):
            return data + 1

        result = sliding_window_inference(inputs, roi_shape, sw_batch_size, compute)
        np.testing.assert_string_equal(inputs.device.type, result.device.type)
        expected_val = np.ones((1, 3, 16, 15, 7), dtype=np.float32) + 1
        np.testing.assert_allclose(result.cpu().numpy(), expected_val)

    @parameterized.expand(list(itertools.product(TEST_TORCH_AND_META_TENSORS, ("cpu", "cuda"), ("cpu", "cuda", None))))
    @skip_if_no_cuda
    def test_sw_device(self, data_type, device, sw_device):
        inputs = data_type(torch.ones((3, 16, 15, 7))).to(device=device)
        inputs = list_data_collate([inputs])  # make a proper batch
        roi_shape = (4, 10, 7)
        sw_batch_size = 10

        def compute(data):
            self.assertEqual(data.device.type, sw_device or device)
            return data + torch.tensor(1, device=sw_device or device)

        result = sliding_window_inference(inputs, roi_shape, sw_batch_size, compute, sw_device=sw_device, device="cpu")
        np.testing.assert_string_equal("cpu", result.device.type)
        expected_val = np.ones((1, 3, 16, 15, 7), dtype=np.float32) + 1
        np.testing.assert_allclose(result.cpu().numpy(), expected_val)

    def test_sigma(self):
        device = "cuda" if torch.cuda.is_available() else "cpu:0"
        inputs = torch.ones((1, 1, 7, 7)).to(device=device)
        roi_shape = (3, 3)
        sw_batch_size = 10

        class _Pred:
            add = 1

            def compute(self, data):
                self.add += 1
                return data + self.add

        result = sliding_window_inference(
            inputs,
            roi_shape,
            sw_batch_size,
            _Pred().compute,
            overlap=0.5,
            padding_mode="constant",
            cval=-1,
            mode="constant",
            sigma_scale=1.0,
        )

        expected = np.array(
            [
                [
                    [
                        [3.0000, 3.0000, 3.0000, 3.0000, 3.0000, 3.0000, 3.0000],
                        [3.0000, 3.0000, 3.0000, 3.0000, 3.0000, 3.0000, 3.0000],
                        [3.3333, 3.3333, 3.3333, 3.3333, 3.3333, 3.3333, 3.3333],
                        [3.6667, 3.6667, 3.6667, 3.6667, 3.6667, 3.6667, 3.6667],
                        [4.3333, 4.3333, 4.3333, 4.3333, 4.3333, 4.3333, 4.3333],
                        [4.5000, 4.5000, 4.5000, 4.5000, 4.5000, 4.5000, 4.5000],
                        [5.0000, 5.0000, 5.0000, 5.0000, 5.0000, 5.0000, 5.0000],
                    ]
                ]
            ]
        )
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-4)
        result = sliding_window_inference(
            inputs,
            roi_shape,
            sw_batch_size,
            _Pred().compute,
            overlap=0.5,
            padding_mode="constant",
            cval=-1,
            mode="gaussian",
            sigma_scale=1.0,
            progress=has_tqdm,
        )
        expected = np.array(
            [
                [
                    [
                        [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
                        [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
                        [3.3271625, 3.3271623, 3.3271623, 3.3271623, 3.3271623, 3.3271623, 3.3271625],
                        [3.6728377, 3.6728377, 3.6728377, 3.6728377, 3.6728377, 3.6728377, 3.6728377],
                        [4.3271623, 4.3271623, 4.3271627, 4.3271627, 4.3271627, 4.3271623, 4.3271623],
                        [4.513757, 4.513757, 4.513757, 4.513757, 4.513757, 4.513757, 4.513757],
                        [4.9999995, 5.0, 5.0, 5.0, 5.0, 5.0, 4.9999995],
                    ]
                ]
            ]
        )
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-4)

        result = SlidingWindowHoVerNetInferer(roi_shape, sw_batch_size, overlap=0.5, mode="gaussian", sigma_scale=1.0)(
            inputs, _Pred().compute
        )
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-4)

        result = SlidingWindowHoVerNetInferer(
            roi_shape, sw_batch_size, overlap=0.5, mode="gaussian", sigma_scale=[1.0, 1.0]
        )(inputs, _Pred().compute)
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-4)

        result = SlidingWindowHoVerNetInferer(
            roi_shape, sw_batch_size, overlap=0.5, mode="gaussian", sigma_scale=[1.0, 1.0], cache_roi_weight_map=True
        )(inputs, _Pred().compute)
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-4)

    def test_cval(self):
        device = "cuda" if torch.cuda.is_available() else "cpu:0"
        inputs = torch.ones((1, 1, 3, 3)).to(device=device)
        roi_shape = (5, 5)
        sw_batch_size = 10

        def compute(data):
            return data + data.sum()

        result = sliding_window_inference(
            inputs,
            roi_shape,
            sw_batch_size,
            compute,
            overlap=0.5,
            padding_mode="constant",
            cval=-1,
            mode="constant",
            sigma_scale=1.0,
        )
        expected = np.ones((1, 1, 3, 3)) * -6.0
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-4)

        result = SlidingWindowHoVerNetInferer(roi_shape, sw_batch_size, overlap=0.5, mode="constant", cval=-1)(
            inputs, compute
        )
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-4)

    def test_args_kwargs(self):
        device = "cuda" if torch.cuda.is_available() else "cpu:0"
        inputs = torch.ones((1, 1, 3, 3)).to(device=device)
        t1 = torch.ones(1).to(device=device)
        t2 = torch.ones(1).to(device=device)
        roi_shape = (5, 5)
        sw_batch_size = 10

        def compute(data, test1, test2):
            return data + test1 + test2

        result = sliding_window_inference(
            inputs,
            roi_shape,
            sw_batch_size,
            compute,
            0.5,
            "constant",
            1.0,
            "constant",
            0.0,
            device,
            device,
            has_tqdm,
            None,
            False,
            t1,
            test2=t2,
        )
        expected = np.ones((1, 1, 3, 3)) + 2.0
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-4)

        result = SlidingWindowHoVerNetInferer(
            roi_shape, sw_batch_size, overlap=0.5, mode="constant", cval=-1, progress=has_tqdm
        )(inputs, compute, t1, test2=t2)
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-4)

    def test_multioutput(self):
        device = "cuda" if torch.cuda.is_available() else "cpu:0"
        inputs = torch.ones((1, 6, 20, 20)).to(device=device)
        roi_shape = (8, 8)
        sw_batch_size = 10

        def compute(data):
            return data + 1, data[:, ::3, ::2, ::2] + 2, data[:, ::2, ::4, ::4] + 3

        def compute_dict(data):
            return {1: data + 1, 2: data[:, ::3, ::2, ::2] + 2, 3: data[:, ::2, ::4, ::4] + 3}

        result = sliding_window_inference(
            inputs,
            roi_shape,
            sw_batch_size,
            compute,
            0.5,
            "constant",
            1.0,
            "constant",
            0.0,
            device,
            device,
            has_tqdm,
            None,
        )
        result_dict = sliding_window_inference(
            inputs,
            roi_shape,
            sw_batch_size,
            compute_dict,
            0.5,
            "constant",
            1.0,
            "constant",
            0.0,
            device,
            device,
            has_tqdm,
            None,
        )
        expected = (np.ones((1, 6, 20, 20)) + 1, np.ones((1, 2, 10, 10)) + 2, np.ones((1, 3, 5, 5)) + 3)
        expected_dict = {1: np.ones((1, 6, 20, 20)) + 1, 2: np.ones((1, 2, 10, 10)) + 2, 3: np.ones((1, 3, 5, 5)) + 3}
        for rr, ee in zip(result, expected):
            np.testing.assert_allclose(rr.cpu().numpy(), ee, rtol=1e-4)
        for rr, _ in zip(result_dict, expected_dict):
            np.testing.assert_allclose(result_dict[rr].cpu().numpy(), expected_dict[rr], rtol=1e-4)

        result = SlidingWindowHoVerNetInferer(
            roi_shape, sw_batch_size, overlap=0.5, mode="constant", cval=-1, progress=has_tqdm
        )(inputs, compute)
        for rr, ee in zip(result, expected):
            np.testing.assert_allclose(rr.cpu().numpy(), ee, rtol=1e-4)

        result_dict = SlidingWindowHoVerNetInferer(
            roi_shape, sw_batch_size, overlap=0.5, mode="constant", cval=-1, progress=has_tqdm
        )(inputs, compute_dict)
        for rr, _ in zip(result_dict, expected_dict):
            np.testing.assert_allclose(result_dict[rr].cpu().numpy(), expected_dict[rr], rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
