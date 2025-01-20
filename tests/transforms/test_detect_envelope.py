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

from monai.transforms import DetectEnvelope
from monai.utils import OptionalImportError
from tests.util import TEST_NDARRAYS, SkipIfModule, SkipIfNoModule, assert_allclose

n_samples = 500
hann_windowed_sine = np.sin(2 * np.pi * 10 * np.linspace(0, 1, n_samples)) * np.hanning(n_samples)

# SINGLE-CHANNEL VALUE TESTS
# using np.expand_dims() to add length 1 channel dimension at dimension 0

TEST_CASE_1D_SINE = [
    {},  # args (empty, so use default)
    np.expand_dims(hann_windowed_sine, 0),  # Input data: Hann windowed sine wave
    np.expand_dims(np.hanning(n_samples), 0),  # Expected output: the Hann window
    1e-4,  # absolute tolerance
]

TEST_CASE_2D_SINE = [
    {},  # args (empty, so use default (i.e. process along first spatial dimension, axis=1)
    # Create 10 identical windowed sine waves as a 2D numpy array
    np.expand_dims(np.stack([hann_windowed_sine] * 10, axis=1), 0),
    # Expected output: Set of 10 identical Hann windows
    np.expand_dims(np.stack([np.hanning(n_samples)] * 10, axis=1), 0),
    1e-4,  # absolute tolerance
]

TEST_CASE_3D_SINE = [
    {},  # args (empty, so use default (i.e. process along first spatial dimension, axis=1)
    # Create 100 identical windowed sine waves as a (n_samples x 10 x 10) 3D numpy array
    np.expand_dims(np.stack([np.stack([hann_windowed_sine] * 10, axis=1)] * 10, axis=2), 0),
    # Expected output: Set of 100 identical Hann windows in (n_samples x 10 x 10) 3D numpy array
    np.expand_dims(np.stack([np.stack([np.hanning(n_samples)] * 10, axis=1)] * 10, axis=2), 0),
    1e-4,  # absolute tolerance
]

TEST_CASE_2D_SINE_AXIS_1 = [
    {"axis": 2},  # set axis argument to 1
    # Create 10 identical windowed sine waves as a 2D numpy array
    np.expand_dims(np.stack([hann_windowed_sine] * 10, axis=1), 0),
    # Expected output: absolute value of each sample of the waveform, repeated (i.e. flat envelopes)
    np.expand_dims(np.abs(np.repeat(hann_windowed_sine, 10).reshape((n_samples, 10))), 0),
    1e-4,  # absolute tolerance
]

TEST_CASE_1D_SINE_PADDING_N = [
    {"n": 512},  # args (empty, so use default)
    np.expand_dims(hann_windowed_sine, 0),  # Input data: Hann windowed sine wave
    np.expand_dims(np.concatenate([np.hanning(500), np.zeros(12)]), 0),  # Expected output: the Hann window
    1e-3,  # absolute tolerance
]

# MULTI-CHANNEL VALUE TEST

TEST_CASE_2_CHAN_3D_SINE = [
    {},  # args (empty, so use default (i.e. process along first spatial dimension, axis=1)
    # Create 100 identical windowed sine waves as a (n_samples x 10 x 10) 3D numpy array, twice (2 channels)
    torch.as_tensor(np.stack([np.stack([np.stack([hann_windowed_sine] * 10, axis=1)] * 10, axis=2)] * 2, axis=0)),
    # Expected output: Set of 100 identical Hann windows in (n_samples x 10 x 10) 3D numpy array, twice (2 channels)
    torch.as_tensor(np.stack([np.stack([np.stack([np.hanning(n_samples)] * 10, axis=1)] * 10, axis=2)] * 2, axis=0)),
    1e-4,  # absolute tolerance
]

# EXCEPTION TESTS

TEST_CASE_INVALID_AXIS_1 = [
    {"axis": 3},  # set axis argument to 3 when only 3 dimensions (1 channel + 2 spatial)
    np.expand_dims(np.stack([hann_windowed_sine] * 10, axis=1), 0),  # Create 2D dataset
    "__call__",  # method expected to raise exception
]

TEST_CASE_INVALID_AXIS_2 = [
    {"axis": -1},  # set axis argument negative
    np.expand_dims(np.stack([hann_windowed_sine] * 10, axis=1), 0),  # Create 2D dataset
    "__init__",  # method expected to raise exception
]

TEST_CASE_INVALID_N = [
    {"n": 0},  # set FFT length to zero
    np.expand_dims(np.stack([hann_windowed_sine] * 10, axis=1), 0),  # Create 2D dataset
    "__call__",  # method expected to raise exception
]

TEST_CASE_INVALID_DTYPE = [
    {},
    np.expand_dims(np.array(hann_windowed_sine, dtype=complex), 0),  # complex numbers are invalid
    "__call__",  # method expected to raise exception
]

TEST_CASE_INVALID_IMG_LEN = [
    {},
    np.expand_dims(np.array([]), 0),  # empty array is invalid
    "__call__",  # method expected to raise exception
]

TEST_CASE_INVALID_OBJ = [{}, "a string", "__call__"]  # method expected to raise exception


@SkipIfNoModule("torch.fft")
class TestDetectEnvelope(unittest.TestCase):
    @parameterized.expand(
        [
            TEST_CASE_1D_SINE,
            TEST_CASE_2D_SINE,
            TEST_CASE_3D_SINE,
            TEST_CASE_2D_SINE_AXIS_1,
            TEST_CASE_1D_SINE_PADDING_N,
            TEST_CASE_2_CHAN_3D_SINE,
        ]
    )
    def test_value(self, arguments, image, expected_data, atol):
        for p in TEST_NDARRAYS:
            result = DetectEnvelope(**arguments)(p(image))
            assert_allclose(result, p(expected_data), atol=atol, type_test="tensor")

    @parameterized.expand(
        [
            TEST_CASE_INVALID_AXIS_1,
            TEST_CASE_INVALID_AXIS_2,
            TEST_CASE_INVALID_N,
            TEST_CASE_INVALID_DTYPE,
            TEST_CASE_INVALID_IMG_LEN,
        ]
    )
    def test_value_error(self, arguments, image, method):
        if method == "__init__":
            self.assertRaises(ValueError, DetectEnvelope, **arguments)
        elif method == "__call__":
            self.assertRaises(ValueError, DetectEnvelope(**arguments), image)
        else:
            self.fail("Expected raising method invalid. Should be __init__ or __call__.")


@SkipIfModule("torch.fft")
class TestHilbertTransformNoFFTMod(unittest.TestCase):
    def test_no_fft_module_error(self):
        self.assertRaises(OptionalImportError, DetectEnvelope(), np.random.rand(1, 10))


if __name__ == "__main__":
    unittest.main()
