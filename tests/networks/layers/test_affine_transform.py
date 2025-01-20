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

from monai.networks import normalize_transform, to_norm_affine
from monai.networks.layers import AffineTransform
from tests.util import is_tf32_env

_rtol = 1e-4 if not is_tf32_env() else 5e-3

TEST_NORM_CASES = [
    [(4, 5), True, [[[0.666667, 0, -1], [0, 0.5, -1], [0, 0, 1]]]],
    [(4, 5), True, [[[0.5, 0, 0], [0, 0.4, 0], [0, 0, 1]]], True],
    [
        (2, 4, 5),
        True,
        [[[2.0, 0.0, 0.0, -1.0], [0.0, 0.6666667, 0.0, -1.0], [0.0, 0.0, 0.5, -1.0], [0.0, 0.0, 0.0, 1.0]]],
    ],
    [(4, 5), False, [[[0.5, 0.0, -0.75], [0.0, 0.4, -0.8], [0.0, 0.0, 1.0]]]],
    [(4, 5), False, [[[0.6666667, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]]], True],
    [(2, 4, 5), False, [[[1.0, 0.0, 0.0, -0.5], [0.0, 0.5, 0.0, -0.75], [0.0, 0.0, 0.4, -0.8], [0.0, 0.0, 0.0, 1.0]]]],
]

TEST_TO_NORM_AFFINE_CASES = [
    [
        [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]],
        (4, 6),
        (5, 3),
        True,
        [[[1.3333334, 0.0, 0.33333337], [0.0, 0.4, -0.6], [0.0, 0.0, 1.0]]],
    ],
    [
        [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]],
        (4, 6),
        (5, 3),
        False,
        [[[1.25, 0.0, 0.25], [0.0, 0.5, -0.5], [0.0, 0.0, 1.0]]],
    ],
    [
        [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]],
        (2, 4, 6),
        (3, 5, 3),
        True,
        [[[2.0, 0.0, 0.0, 1.0], [0.0, 1.3333334, 0.0, 0.33333337], [0.0, 0.0, 0.4, -0.6], [0.0, 0.0, 0.0, 1.0]]],
    ],
    [
        [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]],
        (2, 4, 6),
        (3, 5, 3),
        False,
        [[[1.5, 0.0, 0.0, 0.5], [0.0, 1.25, 0.0, 0.25], [0.0, 0.0, 0.5, -0.5], [0.0, 0.0, 0.0, 1.0]]],
    ],
    [
        [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]],
        (2, 4, 6),
        (3, 5, 3),
        False,
        [[[2.0, 0.0, 0.0, 0.0], [0.0, 1.3333334, 0.0, 0.0], [0.0, 0.0, 0.4, 0.0], [0.0, 0.0, 0.0, 1.0]]],
        True,
    ],
]

TEST_ILL_TO_NORM_AFFINE_CASES = [
    [[[[1, 0, 0], [0, 1, 0], [0, 0, 1]]], (3, 4, 6), (3, 5, 3), False],
    [[[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]], (4, 6), (3, 5, 3), True],
    [[[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]], (4, 6), (3, 5, 3), True],
]


class TestNormTransform(unittest.TestCase):
    @parameterized.expand(TEST_NORM_CASES)
    def test_norm_xform(self, input_shape, align_corners, expected, zero_centered=False):
        norm = normalize_transform(
            input_shape,
            device=torch.device("cpu:0"),
            dtype=torch.float32,
            align_corners=align_corners,
            zero_centered=zero_centered,
        )
        norm = norm.detach().cpu().numpy()
        np.testing.assert_allclose(norm, expected, atol=1e-6)
        if torch.cuda.is_available():
            norm = normalize_transform(
                input_shape,
                device=torch.device("cuda:0"),
                dtype=torch.float32,
                align_corners=align_corners,
                zero_centered=zero_centered,
            )
            norm = norm.detach().cpu().numpy()
            np.testing.assert_allclose(norm, expected, atol=1e-4)


class TestToNormAffine(unittest.TestCase):
    @parameterized.expand(TEST_TO_NORM_AFFINE_CASES)
    def test_to_norm_affine(self, affine, src_size, dst_size, align_corners, expected, zero_centered=False):
        affine = torch.as_tensor(affine, device=torch.device("cpu:0"), dtype=torch.float32)
        new_affine = to_norm_affine(affine, src_size, dst_size, align_corners, zero_centered)
        new_affine = new_affine.detach().cpu().numpy()
        np.testing.assert_allclose(new_affine, expected, atol=1e-6)

        if torch.cuda.is_available():
            affine = torch.as_tensor(affine, device=torch.device("cuda:0"), dtype=torch.float32)
            new_affine = to_norm_affine(affine, src_size, dst_size, align_corners, zero_centered)
            new_affine = new_affine.detach().cpu().numpy()
            np.testing.assert_allclose(new_affine, expected, atol=1e-5, rtol=_rtol)

    @parameterized.expand(TEST_ILL_TO_NORM_AFFINE_CASES)
    def test_to_norm_affine_ill(self, affine, src_size, dst_size, align_corners):
        with self.assertRaises(TypeError):
            to_norm_affine(affine, src_size, dst_size, align_corners)
        with self.assertRaises(ValueError):
            affine = torch.as_tensor(affine, device=torch.device("cpu:0"), dtype=torch.float32)
            to_norm_affine(affine, src_size, dst_size, align_corners)


class TestAffineTransform(unittest.TestCase):
    @parameterized.expand(
        [
            (torch.as_tensor([[1.0, 0.0, 0.0], [0.0, 1.0, -1.0]]), [[[[0, 4, 1, 3], [0, 7, 6, 8], [0, 3, 5, 3]]]]),
            (torch.as_tensor([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0]]), [[[[0, 0, 0, 0], [0, 4, 1, 3], [0, 7, 6, 8]]]]),
            (torch.as_tensor([[1.0, 0.0, -1.0], [0.0, 1.0, 0.0]]), [[[[0, 0, 0, 0], [4, 1, 3, 2], [7, 6, 8, 5]]]]),
        ]
    )
    def test_affine_transforms(self, affine, expected):
        image = torch.as_tensor([[[[4.0, 1.0, 3.0, 2.0], [7.0, 6.0, 8.0, 5.0], [3.0, 5.0, 3.0, 6.0]]]])
        out = AffineTransform(align_corners=False)(image, affine)
        out = out.detach().cpu().numpy()
        np.testing.assert_allclose(out, expected, atol=1e-5, rtol=_rtol)

    def test_zoom(self):
        affine = torch.as_tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        image = torch.arange(1.0, 13.0).view(1, 1, 3, 4).to(device=torch.device("cpu:0"))
        out = AffineTransform((3, 2), align_corners=False)(image, affine)
        expected = [[[[1, 3], [5, 7], [9, 11]]]]
        np.testing.assert_allclose(out, expected, atol=1e-5, rtol=_rtol)

    def test_zoom_1(self):
        affine = torch.as_tensor([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        image = torch.arange(1.0, 13.0).view(1, 1, 3, 4).to(device=torch.device("cpu:0"))
        out = AffineTransform()(image, affine, (1, 4))
        expected = [[[[2.333333, 3.333333, 4.333333, 5.333333]]]]
        np.testing.assert_allclose(out, expected, atol=_rtol)

    def test_zoom_2(self):
        affine = torch.as_tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=torch.float32)
        image = torch.arange(1.0, 13.0).view(1, 1, 3, 4).to(device=torch.device("cpu:0"))
        out = AffineTransform((1, 2))(image, affine)
        expected = [[[[1.458333, 4.958333]]]]
        np.testing.assert_allclose(out, expected, atol=1e-5, rtol=_rtol)

    def test_zoom_zero_center(self):
        affine = torch.as_tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=torch.float32)
        image = torch.arange(1.0, 13.0).view(1, 1, 3, 4).to(device=torch.device("cpu:0"))
        out = AffineTransform((1, 2), zero_centered=True)(image, affine)
        expected = [[[[5.5, 7.5]]]]
        np.testing.assert_allclose(out, expected, atol=1e-5, rtol=_rtol)

    def test_affine_transform_minimum(self):
        t = np.pi / 3
        affine = [[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]]
        affine = torch.as_tensor(affine, device=torch.device("cpu:0"), dtype=torch.float32)
        image = torch.arange(24.0).view(1, 1, 4, 6).to(device=torch.device("cpu:0"))
        out = AffineTransform(align_corners=False)(image, affine)
        out = out.detach().cpu().numpy()
        expected = [
            [
                [
                    [0.0, 0.06698727, 0.0, 0.0, 0.0, 0.0],
                    [3.8660254, 0.86602557, 0.0, 0.0, 0.0, 0.0],
                    [7.732051, 3.035899, 0.73205125, 0.0, 0.0, 0.0],
                    [11.598076, 6.901923, 2.7631402, 0.0, 0.0, 0.0],
                ]
            ]
        ]
        np.testing.assert_allclose(out, expected, atol=1e-3, rtol=_rtol)

    def test_affine_transform_2d(self):
        t = np.pi / 3
        affine = [[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]]
        affine = torch.as_tensor(affine, device=torch.device("cpu:0"), dtype=torch.float32)
        image = torch.arange(24.0).view(1, 1, 4, 6).to(device=torch.device("cpu:0"))
        xform = AffineTransform((3, 4), padding_mode="border", align_corners=False, mode="bilinear")
        out = xform(image, affine)
        out = out.detach().cpu().numpy()
        expected = [
            [
                [
                    [7.1525574e-07, 4.9999994e-01, 1.0000000e00, 1.4999999e00],
                    [3.8660259e00, 1.3660253e00, 1.8660252e00, 2.3660252e00],
                    [7.7320518e00, 3.0358994e00, 2.7320509e00, 3.2320507e00],
                ]
            ]
        ]
        np.testing.assert_allclose(out, expected, atol=1e-3, rtol=_rtol)

        if torch.cuda.is_available():
            affine = torch.as_tensor(affine, device=torch.device("cuda:0"), dtype=torch.float32)
            image = torch.arange(24.0).view(1, 1, 4, 6).to(device=torch.device("cuda:0"))
            xform = AffineTransform(padding_mode="border", align_corners=False, mode="bilinear")
            out = xform(image, affine, (3, 4))
            out = out.detach().cpu().numpy()
            expected = [
                [
                    [
                        [7.1525574e-07, 4.9999994e-01, 1.0000000e00, 1.4999999e00],
                        [3.8660259e00, 1.3660253e00, 1.8660252e00, 2.3660252e00],
                        [7.7320518e00, 3.0358994e00, 2.7320509e00, 3.2320507e00],
                    ]
                ]
            ]
            np.testing.assert_allclose(out, expected, atol=5e-3)

    def test_affine_transform_3d(self):
        t = np.pi / 3
        affine = [[1, 0, 0, 0], [0.0, np.cos(t), -np.sin(t), 0], [0, np.sin(t), np.cos(t), 0], [0, 0, 0, 1]]
        affine = torch.as_tensor(affine, device=torch.device("cpu:0"), dtype=torch.float32)
        image = torch.arange(48.0).view(2, 1, 4, 2, 3).to(device=torch.device("cpu:0"))
        xform = AffineTransform((3, 4, 2), padding_mode="border", align_corners=False, mode="bilinear")
        out = xform(image, affine)
        out = out.detach().cpu().numpy()
        expected = [
            [
                [
                    [[0.00000006, 0.5000001], [2.3660254, 1.3660254], [4.732051, 2.4019241], [5.0, 3.9019237]],
                    [[6.0, 6.5], [8.366026, 7.3660254], [10.732051, 8.401924], [11.0, 9.901924]],
                    [[12.0, 12.5], [14.366026, 13.366025], [16.732052, 14.401924], [17.0, 15.901923]],
                ]
            ],
            [
                [
                    [[24.0, 24.5], [26.366024, 25.366024], [28.732052, 26.401924], [29.0, 27.901924]],
                    [[30.0, 30.5], [32.366028, 31.366026], [34.732048, 32.401924], [35.0, 33.901924]],
                    [[36.0, 36.5], [38.366024, 37.366024], [40.73205, 38.401924], [41.0, 39.901924]],
                ]
            ],
        ]
        np.testing.assert_allclose(out, expected, atol=1e-4, rtol=_rtol)

        if torch.cuda.is_available():
            affine = torch.as_tensor(affine, device=torch.device("cuda:0"), dtype=torch.float32)
            image = torch.arange(48.0).view(2, 1, 4, 2, 3).to(device=torch.device("cuda:0"))
            xform = AffineTransform(padding_mode="border", align_corners=False, mode="bilinear")
            out = xform(image, affine, (3, 4, 2))
            out = out.detach().cpu().numpy()
            expected = [
                [
                    [
                        [[0.00000006, 0.5000001], [2.3660254, 1.3660254], [4.732051, 2.4019241], [5.0, 3.9019237]],
                        [[6.0, 6.5], [8.366026, 7.3660254], [10.732051, 8.401924], [11.0, 9.901924]],
                        [[12.0, 12.5], [14.366026, 13.366025], [16.732052, 14.401924], [17.0, 15.901923]],
                    ]
                ],
                [
                    [
                        [[24.0, 24.5], [26.366024, 25.366024], [28.732052, 26.401924], [29.0, 27.901924]],
                        [[30.0, 30.5], [32.366028, 31.366026], [34.732048, 32.401924], [35.0, 33.901924]],
                        [[36.0, 36.5], [38.366024, 37.366024], [40.73205, 38.401924], [41.0, 39.901924]],
                    ]
                ],
            ]
            np.testing.assert_allclose(out, expected, atol=5e-3)

    def test_ill_affine_transform(self):
        with self.assertRaises(ValueError):  # image too small
            t = np.pi / 3
            affine = [[1, 0, 0, 0], [0.0, np.cos(t), -np.sin(t), 0], [0, np.sin(t), np.cos(t), 0], [0, 0, 0, 1]]
            affine = torch.as_tensor(affine, device=torch.device("cpu:0"), dtype=torch.float32)
            xform = AffineTransform((3, 4, 2), padding_mode="border", align_corners=False, mode="bilinear")
            xform(torch.as_tensor([1.0, 2.0, 3.0]), affine)

        with self.assertRaises(ValueError):  # output shape too small
            t = np.pi / 3
            affine = [[1, 0, 0, 0], [0.0, np.cos(t), -np.sin(t), 0], [0, np.sin(t), np.cos(t), 0], [0, 0, 0, 1]]
            affine = torch.as_tensor(affine, device=torch.device("cpu:0"), dtype=torch.float32)
            image = torch.arange(48).view(2, 1, 4, 2, 3).to(device=torch.device("cpu:0"))
            xform = AffineTransform((3, 4), padding_mode="border", align_corners=False, mode="bilinear")
            xform(image, affine)

        with self.assertRaises(ValueError):  # incorrect affine
            t = np.pi / 3
            affine = [[1, 0, 0, 0], [0.0, np.cos(t), -np.sin(t), 0], [0, np.sin(t), np.cos(t), 0], [0, 0, 0, 1]]
            affine = torch.as_tensor(affine, device=torch.device("cpu:0"), dtype=torch.float32)
            affine = affine.unsqueeze(0).unsqueeze(0)
            image = torch.arange(48).view(2, 1, 4, 2, 3).to(device=torch.device("cpu:0"))
            xform = AffineTransform((2, 3, 4), padding_mode="border", align_corners=False, mode="bilinear")
            xform(image, affine)

        with self.assertRaises(ValueError):  # batch doesn't match
            t = np.pi / 3
            affine = [[1, 0, 0, 0], [0.0, np.cos(t), -np.sin(t), 0], [0, np.sin(t), np.cos(t), 0], [0, 0, 0, 1]]
            affine = torch.as_tensor(affine, device=torch.device("cpu:0"), dtype=torch.float32)
            affine = affine.unsqueeze(0)
            affine = affine.repeat(3, 1, 1)
            image = torch.arange(48).view(2, 1, 4, 2, 3).to(device=torch.device("cpu:0"))
            xform = AffineTransform((2, 3, 4), padding_mode="border", align_corners=False, mode="bilinear")
            xform(image, affine)

        with self.assertRaises(RuntimeError):  # input grid dtypes different
            t = np.pi / 3
            affine = [[1, 0, 0, 0], [0.0, np.cos(t), -np.sin(t), 0], [0, np.sin(t), np.cos(t), 0], [0, 0, 0, 1]]
            affine = torch.as_tensor(affine, device=torch.device("cpu:0"), dtype=torch.float32)
            affine = affine.unsqueeze(0)
            affine = affine.repeat(2, 1, 1)
            image = torch.arange(48).view(2, 1, 4, 2, 3).to(device=torch.device("cpu:0"), dtype=torch.int32)
            xform = AffineTransform((2, 3, 4), padding_mode="border", mode="bilinear", normalized=True)
            xform(image, affine)

        with self.assertRaises(ValueError):  # wrong affine
            affine = torch.as_tensor([[1, 0, 0, 0], [0, 0, 0, 1]])
            image = torch.arange(48).view(2, 1, 4, 2, 3).to(device=torch.device("cpu:0"))
            xform = AffineTransform((2, 3, 4), padding_mode="border", align_corners=False, mode="bilinear")
            xform(image, affine)

        with self.assertRaises(RuntimeError):  # dtype doesn't match
            affine = torch.as_tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=torch.float64)
            image = torch.arange(1.0, 13.0).view(1, 1, 3, 4).to(device=torch.device("cpu:0"))
            AffineTransform((1, 2))(image, affine)

    def test_forward_2d(self):
        x = torch.rand(2, 1, 4, 4)
        theta = torch.Tensor([[[0, -1, 0], [1, 0, 0]]]).repeat(2, 1, 1)
        grid = torch.nn.functional.affine_grid(theta, x.size(), align_corners=False)
        expected = torch.nn.functional.grid_sample(x, grid, align_corners=False)
        expected = expected.detach().cpu().numpy()

        actual = AffineTransform(normalized=True, reverse_indexing=False, align_corners=False)(x, theta)
        actual = actual.detach().cpu().numpy()
        np.testing.assert_allclose(actual, expected)
        np.testing.assert_allclose(list(theta.shape), [2, 2, 3])

        theta = torch.Tensor([[0, -1, 0], [1, 0, 0]])
        actual = AffineTransform(normalized=True, reverse_indexing=False, align_corners=False)(x, theta)
        actual = actual.detach().cpu().numpy()
        np.testing.assert_allclose(actual, expected)
        np.testing.assert_allclose(list(theta.shape), [2, 3])

        theta = torch.Tensor([[[0, -1, 0], [1, 0, 0]]])
        actual = AffineTransform(normalized=True, reverse_indexing=False, align_corners=False)(x, theta)
        actual = actual.detach().cpu().numpy()
        np.testing.assert_allclose(actual, expected)
        np.testing.assert_allclose(list(theta.shape), [1, 2, 3])

    def test_forward_3d(self):
        x = torch.rand(2, 1, 4, 4, 4)
        theta = torch.Tensor([[[0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 1, 0]]]).repeat(2, 1, 1)
        grid = torch.nn.functional.affine_grid(theta, x.size(), align_corners=False)
        expected = torch.nn.functional.grid_sample(x, grid, align_corners=False)
        expected = expected.detach().cpu().numpy()

        actual = AffineTransform(normalized=True, reverse_indexing=False, align_corners=False)(x, theta)
        actual = actual.detach().cpu().numpy()
        np.testing.assert_allclose(actual, expected)
        np.testing.assert_allclose(list(theta.shape), [2, 3, 4])

        theta = torch.Tensor([[0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 1, 0]])
        actual = AffineTransform(normalized=True, reverse_indexing=False, align_corners=False)(x, theta)
        actual = actual.detach().cpu().numpy()
        np.testing.assert_allclose(actual, expected)
        np.testing.assert_allclose(list(theta.shape), [3, 4])

        theta = torch.Tensor([[[0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 1, 0]]])
        actual = AffineTransform(normalized=True, reverse_indexing=False, align_corners=False)(x, theta)
        actual = actual.detach().cpu().numpy()
        np.testing.assert_allclose(actual, expected)
        np.testing.assert_allclose(list(theta.shape), [1, 3, 4])


if __name__ == "__main__":
    unittest.main()
