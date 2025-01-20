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

import itertools
import unittest

import numpy as np
import torch
from parameterized import parameterized

from monai.data.utils import list_data_collate
from monai.inferers import SlidingWindowInferer, SlidingWindowInfererAdapt, sliding_window_inference
from monai.utils import optional_import
from tests.util import TEST_TORCH_AND_META_TENSORS, skip_if_no_cuda, test_is_quick

_, has_tqdm = optional_import("tqdm")

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

_devices = [["cpu", "cuda:0"]] if torch.cuda.is_available() else [["cpu"]]
_windows = [
    [(2, 3, 10, 11), (7, 10), 0.8, 5],
    [(2, 3, 10, 11), (15, 12), 0, 2],
    [(2, 3, 10, 11), (10, 11), 0, 3],
    [(2, 3, 511, 237), (96, 80), 0.4, 5],
    [(2, 3, 512, 245), (96, 80), 0, 5],
    [(2, 3, 512, 245), (512, 80), 0.125, 5],
    [(2, 3, 10, 11, 12), (7, 8, 10), 0.2, 2],
]
if not test_is_quick():
    _windows += [
        [(2, 1, 125, 512, 200), (96, 97, 98), (0.4, 0.32, 0), 20],
        [(2, 1, 10, 512, 200), (96, 97, 98), (0.4, 0.12, 0), 21],
        [(2, 3, 100, 100, 200), (50, 50, 100), 0, 8],
    ]

BUFFER_CASES: list = []
for x in _windows:
    for s in (1, 3, 4):
        for d in (-1, 0, 1):
            BUFFER_CASES.extend([x, s, d, dev] for dev in itertools.product(*_devices * 3))


class TestSlidingWindowInference(unittest.TestCase):
    @parameterized.expand(BUFFER_CASES)
    def test_buffers(self, size_params, buffer_steps, buffer_dim, device_params):
        def mult_two(patch, *args, **kwargs):
            return 2.0 * patch

        img_size, roi_size, overlap, sw_batch_size = size_params
        img_device, device, sw_device = device_params
        dtype = [torch.float, torch.double][roi_size[0] % 2]  # test different input dtype
        mode = ["constant", "gaussian"][img_size[1] % 2]
        image = torch.randint(0, 255, size=img_size, dtype=dtype, device=img_device)
        sw = sliding_window_inference(
            image,
            roi_size,
            sw_batch_size,
            mult_two,
            overlap,
            mode=mode,
            sw_device=sw_device,
            device=device,
            buffer_steps=buffer_steps,
            buffer_dim=buffer_dim,
        )
        max_diff = torch.max(torch.abs(image.to(sw) - 0.5 * sw)).item()
        self.assertGreater(0.001, max_diff)

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

        result = SlidingWindowInferer(roi_shape, sw_batch_size, overlap, mode)(inputs.to(device), compute)
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

        inputs.requires_grad = True
        result = sliding_window_inference(inputs, roi_shape, sw_batch_size, compute)
        self.assertTrue(result.requires_grad)
        np.testing.assert_string_equal(inputs.device.type, result.device.type)
        expected_val = np.ones((1, 3, 16, 15, 7), dtype=np.float32) + 1
        np.testing.assert_allclose(result.detach().cpu().numpy(), expected_val)

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

        result = SlidingWindowInferer(roi_shape, sw_batch_size, overlap=0.5, mode="gaussian", sigma_scale=1.0)(
            inputs, _Pred().compute
        )
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-4)

        result = SlidingWindowInferer(roi_shape, sw_batch_size, overlap=0.5, mode="gaussian", sigma_scale=[1.0, 1.0])(
            inputs, _Pred().compute
        )
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-4)

        result = SlidingWindowInferer(
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

        result = SlidingWindowInferer(roi_shape, sw_batch_size, overlap=0.5, mode="constant", cval=-1)(inputs, compute)
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
            None,
            None,
            0,
            False,
            t1,
            test2=t2,
        )
        expected = np.ones((1, 1, 3, 3)) + 2.0
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-4)

        result = SlidingWindowInferer(
            roi_shape, sw_batch_size, overlap=0.5, mode="constant", cval=-1, progress=has_tqdm
        )(inputs, compute, t1, test2=t2)
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-4)

        result = SlidingWindowInfererAdapt(
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

        result = SlidingWindowInferer(
            roi_shape, sw_batch_size, overlap=0.5, mode="constant", cval=-1, progress=has_tqdm
        )(inputs, compute)
        for rr, ee in zip(result, expected):
            np.testing.assert_allclose(rr.cpu().numpy(), ee, rtol=1e-4)

        result_dict = SlidingWindowInferer(
            roi_shape, sw_batch_size, overlap=0.5, mode="constant", cval=-1, progress=has_tqdm
        )(inputs, compute_dict)
        for rr, _ in zip(result_dict, expected_dict):
            np.testing.assert_allclose(result_dict[rr].cpu().numpy(), expected_dict[rr], rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
