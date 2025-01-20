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

from monai.networks.layers import HilbertTransform
from monai.utils import OptionalImportError
from tests.util import SkipIfModule, SkipIfNoModule


def create_expected_numpy_output(input_datum, **kwargs):
    x = np.fft.fft(input_datum.cpu().numpy(), **kwargs)
    f = np.fft.fftfreq(x.shape[kwargs["axis"]])
    u = np.heaviside(f, 0.5)
    new_dims_before = kwargs["axis"]
    new_dims_after = len(x.shape) - kwargs["axis"] - 1
    for _ in range(new_dims_before):
        u = np.expand_dims(u, 0)
    for _ in range(new_dims_after):
        u = np.expand_dims(u, -1)
    ht = np.fft.ifft(x * 2 * u, axis=kwargs["axis"])

    return ht


cpu = torch.device("cpu")
n_samples = 500
hann_windowed_sine = np.sin(2 * np.pi * 10 * np.linspace(0, 1, n_samples)) * np.hanning(n_samples)

# CPU TEST DATA

cpu_input_data = {}
cpu_input_data["1D"] = torch.as_tensor(hann_windowed_sine, device=cpu)[None, None]
cpu_input_data["2D"] = torch.as_tensor(np.stack([hann_windowed_sine] * 10, axis=1), device=cpu)[None, None]
cpu_input_data["3D"] = torch.as_tensor(
    np.stack([np.stack([hann_windowed_sine] * 10, axis=1)] * 10, axis=2), device=cpu
)[None, None]
cpu_input_data["1D 2CH"] = torch.as_tensor(np.stack([hann_windowed_sine] * 10, axis=1), device=cpu)[None]
cpu_input_data["2D 2CH"] = torch.as_tensor(
    np.stack([np.stack([hann_windowed_sine] * 10, axis=1)] * 10, axis=2), device=cpu
)[None]

# SINGLE-CHANNEL CPU VALUE TESTS

TEST_CASE_1D_SINE_CPU = [
    {},  # args (empty, so use default)
    cpu_input_data["1D"],  # Input data: Random 1D signal
    create_expected_numpy_output(cpu_input_data["1D"], axis=2),  # Expected output: FFT of signal
    1e-5,  # absolute tolerance
]

TEST_CASE_2D_SINE_CPU = [
    {},  # args (empty, so use default)
    cpu_input_data["2D"],  # Input data: Random 1D signal
    create_expected_numpy_output(cpu_input_data["2D"], axis=2),  # Expected output: FFT of signal
    1e-5,  # absolute tolerance
]

TEST_CASE_3D_SINE_CPU = [
    {},  # args (empty, so use default)
    cpu_input_data["3D"],  # Input data: Random 1D signal
    create_expected_numpy_output(cpu_input_data["3D"], axis=2),
    1e-5,  # absolute tolerance
]

# MULTICHANNEL CPU VALUE TESTS, PROCESS ALONG FIRST SPATIAL AXIS

TEST_CASE_1D_2CH_SINE_CPU = [
    {},  # args (empty, so use default)
    cpu_input_data["1D 2CH"],  # Input data: Random 1D signal
    create_expected_numpy_output(cpu_input_data["1D 2CH"], axis=2),
    1e-5,  # absolute tolerance
]

TEST_CASE_2D_2CH_SINE_CPU = [
    {},  # args (empty, so use default)
    cpu_input_data["2D 2CH"],  # Input data: Random 1D signal
    create_expected_numpy_output(cpu_input_data["2D 2CH"], axis=2),
    1e-5,  # absolute tolerance
]

TEST_CASES_CPU = [
    TEST_CASE_1D_SINE_CPU,
    TEST_CASE_2D_SINE_CPU,
    TEST_CASE_3D_SINE_CPU,
    TEST_CASE_1D_2CH_SINE_CPU,
    TEST_CASE_2D_2CH_SINE_CPU,
]

# GPU TEST DATA

if torch.cuda.is_available():
    gpu = torch.device("cuda")
    TEST_CASES_GPU = [[args, image.to(gpu), exp_data, atol] for args, image, exp_data, atol in TEST_CASES_CPU]
else:
    TEST_CASES_GPU = []

# TESTS CHECKING PADDING, AXIS SELECTION ETC ARE COVERED BY test_detect_envelope.py


@SkipIfNoModule("torch.fft")
class TestHilbertTransformCPU(unittest.TestCase):
    @parameterized.expand(TEST_CASES_CPU + TEST_CASES_GPU)
    def test_value(self, arguments, image, expected_data, atol):
        result = HilbertTransform(**arguments)(image)
        result = np.squeeze(result.cpu().numpy())
        np.testing.assert_allclose(result, expected_data.squeeze(), atol=atol)


@SkipIfModule("torch.fft")
class TestHilbertTransformNoFFTMod(unittest.TestCase):
    def test_no_fft_module_error(self):
        self.assertRaises(OptionalImportError, HilbertTransform(), torch.randn(1, 1, 10))


if __name__ == "__main__":
    unittest.main()
