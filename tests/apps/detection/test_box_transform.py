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
from copy import deepcopy

import numpy as np
import torch
from parameterized import parameterized

from monai.apps.detection.transforms.box_ops import convert_mask_to_box
from monai.apps.detection.transforms.dictionary import (
    AffineBoxToImageCoordinated,
    AffineBoxToWorldCoordinated,
    BoxToMaskd,
    ClipBoxToImaged,
    ConvertBoxModed,
    FlipBoxd,
    MaskToBoxd,
    RandCropBoxByPosNegLabeld,
    RandFlipBoxd,
    RandRotateBox90d,
    RandZoomBoxd,
    RotateBox90d,
    ZoomBoxd,
)
from monai.data.meta_tensor import MetaTensor
from monai.transforms import CastToTyped, Invertd
from tests.util import TEST_NDARRAYS, assert_allclose

TESTS_3D = []
boxes = [[0, 0, 0, 0, 0, 0], [0, 1, 0, 2, 3, 3], [0, 1, 1, 2, 3, 4]]
labels = [1, 1, 0]
scores = [[0.2, 0.8], [0.3, 0.7], [0.6, 0.4]]
image_size = [1, 4, 6, 4]
image = np.zeros(image_size)

for p in TEST_NDARRAYS:
    TESTS_3D.append(
        [
            {"box_keys": "boxes", "dst_mode": "xyzwhd"},
            {"boxes": p(boxes), "image": p(image), "labels": p(labels), "scores": p(scores)},
            p([[0, 0, 0, 0, 0, 0], [0, 1, 0, 2, 2, 3], [0, 1, 1, 2, 2, 3]]),
            p([[0, 0, 0, 0, 0, 0], [0, 3, 0, 1, 9, 4.5], [0, 3, 1.5, 1, 9, 6]]),
            p([[1, -6, -1, 1, -6, -1], [1, -3, -1, 2, 3, 3.5], [1, -3, 0.5, 2, 3, 5]]),
            p([[4, 6, 4, 4, 6, 4], [2, 3, 1, 4, 5, 4], [2, 3, 0, 4, 5, 3]]),
            p([[0, 1, 0, 2, 3, 3], [0, 1, 1, 2, 3, 4]]),
            p([[6, 0, 0, 6, 0, 0], [3, 0, 0, 5, 2, 3], [3, 0, 1, 5, 2, 4]]),
        ]
    )

TESTS_2D = []
boxes = [[0, 1, 2, 2], [0, 0, 1, 1]]
labels = [1, 0]
image_size = [1, 2, 2]
image = np.zeros(image_size)
for p in TEST_NDARRAYS:
    TESTS_2D.append(
        [{"boxes": p(boxes), "image": p(image), "labels": p(labels)}, p([[[0, 2], [0, 2]], [[1, 0], [0, 0]]])]
    )

TESTS_2D_mask = []
boxes_mask = [[[-1, 0], [0, -1]]]
for p in TEST_NDARRAYS:
    TESTS_2D_mask.append([p(boxes_mask), (p([[0.0, 0.0, 2.0, 2.0]]), p([0]))])
boxes_mask = [[[-1, 0], [0, -1]], [[-1, 1], [1, -1]]]
for p in TEST_NDARRAYS:
    TESTS_2D_mask.append([p(boxes_mask), (p([[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 2.0, 2.0]]), p([0, 1]))])


class TestBoxTransform(unittest.TestCase):
    @parameterized.expand(TESTS_2D_mask)
    def test_value_2d_mask(self, mask, expected_box_label):
        box_label = convert_mask_to_box(mask)
        assert_allclose(box_label[0], expected_box_label[0], type_test=True, device_test=True, atol=1e-3)
        assert_allclose(box_label[1], expected_box_label[1], type_test=True, device_test=True, atol=1e-3)

    @parameterized.expand(TESTS_2D)
    def test_value_2d(self, data, expected_mask):
        test_dtype = [torch.float32, torch.float16]
        for dtype in test_dtype:
            data = CastToTyped(keys=["image", "boxes"], dtype=dtype)(data)
            transform_to_mask = BoxToMaskd(
                box_keys="boxes",
                box_mask_keys="box_mask",
                box_ref_image_keys="image",
                label_keys="labels",
                min_fg_label=0,
                ellipse_mask=False,
            )
            transform_to_box = MaskToBoxd(
                box_keys="boxes", box_mask_keys="box_mask", label_keys="labels", min_fg_label=0
            )
            data_mask = transform_to_mask(data)
            assert_allclose(data_mask["box_mask"], expected_mask, type_test=True, device_test=True, atol=1e-3)
            data_back = transform_to_box(data_mask)
            assert_allclose(data_back["boxes"], data["boxes"], type_test=False, device_test=False, atol=1e-3)
            assert_allclose(data_back["labels"], data["labels"], type_test=False, device_test=False, atol=1e-3)

    def test_value_3d_mask(self):
        test_dtype = [torch.float32, torch.float16]
        image = np.zeros((1, 32, 33, 34))
        boxes = np.array([[7, 8, 9, 10, 12, 13], [1, 3, 5, 2, 5, 9], [0, 0, 0, 1, 1, 1]])
        data = {"image": image, "boxes": boxes, "labels": np.array((1, 0, 3))}
        for dtype in test_dtype:
            data = CastToTyped(keys=["image", "boxes"], dtype=dtype)(data)
            transform_to_mask = BoxToMaskd(
                box_keys="boxes",
                box_mask_keys="box_mask",
                box_ref_image_keys="image",
                label_keys="labels",
                min_fg_label=0,
                ellipse_mask=False,
            )
            transform_to_box = MaskToBoxd(
                box_keys="boxes", box_mask_keys="box_mask", label_keys="labels", min_fg_label=0
            )
            data_mask = transform_to_mask(data)
            assert_allclose(data_mask["box_mask"].shape, (3, 32, 33, 34), type_test=True, device_test=True, atol=1e-3)
            data_back = transform_to_box(data_mask)
            assert_allclose(data_back["boxes"], data["boxes"], type_test=False, device_test=False, atol=1e-3)
            assert_allclose(data_back["labels"], data["labels"], type_test=False, device_test=False, atol=1e-3)

    def test_shape_assertion(self):
        test_dtype = torch.float32
        image = np.zeros((1, 10, 10, 10))
        boxes = np.array([[7, 8, 9, 10, 12, 13]])
        data = {"image": image, "boxes": boxes, "labels": np.array((1,))}
        data = CastToTyped(keys=["image", "boxes"], dtype=test_dtype)(data)
        transform_to_mask = BoxToMaskd(
            box_keys="boxes",
            box_mask_keys="box_mask",
            box_ref_image_keys="image",
            label_keys="labels",
            min_fg_label=0,
            ellipse_mask=False,
        )
        with self.assertRaises(ValueError) as context:
            transform_to_mask(data)
        self.assertTrue("Some boxes are larger than the image." in str(context.exception))

    @parameterized.expand(TESTS_3D)
    def test_value_3d(
        self,
        keys,
        data,
        expected_convert_result,
        expected_zoom_result,
        expected_zoom_keepsize_result,
        expected_flip_result,
        expected_clip_result,
        expected_rotate_result,
    ):
        test_dtype = [torch.float32]
        for dtype in test_dtype:
            data = CastToTyped(keys=["image", "boxes"], dtype=dtype)(data)
            # test ConvertBoxToStandardModed
            transform_convert_mode = ConvertBoxModed(**keys)
            convert_result = transform_convert_mode(data)
            assert_allclose(
                convert_result["boxes"], expected_convert_result, type_test=True, device_test=True, atol=1e-3
            )

            invert_transform_convert_mode = Invertd(
                keys=["boxes"], transform=transform_convert_mode, orig_keys=["boxes"]
            )
            data_back = invert_transform_convert_mode(convert_result)
            if "boxes_transforms" in data_back:  # if the transform is tracked in dict:
                self.assertEqual(data_back["boxes_transforms"], [])  # it should be updated
            assert_allclose(data_back["boxes"], data["boxes"], type_test=False, device_test=False, atol=1e-3)

            # test ZoomBoxd
            transform_zoom = ZoomBoxd(
                image_keys="image", box_keys="boxes", box_ref_image_keys="image", zoom=[0.5, 3, 1.5], keep_size=False
            )
            zoom_result = transform_zoom(data)
            self.assertEqual(len(zoom_result["image"].applied_operations), 1)
            assert_allclose(zoom_result["boxes"], expected_zoom_result, type_test=True, device_test=True, atol=1e-3)
            invert_transform_zoom = Invertd(
                keys=["image", "boxes"], transform=transform_zoom, orig_keys=["image", "boxes"]
            )
            data_back = invert_transform_zoom(zoom_result)
            self.assertEqual(data_back["image"].applied_operations, [])
            assert_allclose(data_back["boxes"], data["boxes"], type_test=False, device_test=False, atol=1e-3)
            assert_allclose(data_back["image"], data["image"], type_test=False, device_test=False, atol=1e-3)

            transform_zoom = ZoomBoxd(
                image_keys="image", box_keys="boxes", box_ref_image_keys="image", zoom=[0.5, 3, 1.5], keep_size=True
            )
            zoom_result = transform_zoom(data)
            self.assertEqual(len(zoom_result["image"].applied_operations), 1)
            assert_allclose(
                zoom_result["boxes"], expected_zoom_keepsize_result, type_test=True, device_test=True, atol=1e-3
            )

            # test RandZoomBoxd
            transform_zoom = RandZoomBoxd(
                image_keys="image",
                box_keys="boxes",
                box_ref_image_keys="image",
                prob=1.0,
                min_zoom=(0.3,) * 3,
                max_zoom=(3.0,) * 3,
                keep_size=False,
            )
            zoom_result = transform_zoom(data)
            self.assertEqual(len(zoom_result["image"].applied_operations), 1)
            invert_transform_zoom = Invertd(
                keys=["image", "boxes"], transform=transform_zoom, orig_keys=["image", "boxes"]
            )
            data_back = invert_transform_zoom(zoom_result)
            self.assertEqual(data_back["image"].applied_operations, [])
            assert_allclose(data_back["boxes"], data["boxes"], type_test=False, device_test=False, atol=0.01)
            assert_allclose(data_back["image"], data["image"], type_test=False, device_test=False, atol=1e-3)

            # test AffineBoxToImageCoordinated, AffineBoxToWorldCoordinated
            transform_affine = AffineBoxToImageCoordinated(box_keys="boxes", box_ref_image_keys="image")
            if not isinstance(data["image"], MetaTensor):  # metadict should be undefined and it's an exception
                with self.assertRaises(Exception) as context:
                    transform_affine(deepcopy(data))
                self.assertTrue("Please check whether it is the correct the image meta key." in str(context.exception))

            data["image"] = MetaTensor(data["image"], meta={"affine": torch.diag(1.0 / torch.Tensor([0.5, 3, 1.5, 1]))})
            affine_result = transform_affine(data)
            if "boxes_transforms" in affine_result:
                self.assertEqual(len(affine_result["boxes_transforms"]), 1)
            assert_allclose(affine_result["boxes"], expected_zoom_result, type_test=True, device_test=True, atol=0.01)
            invert_transform_affine = Invertd(keys=["boxes"], transform=transform_affine, orig_keys=["boxes"])
            data_back = invert_transform_affine(affine_result)
            if "boxes_transforms" in data_back:
                self.assertEqual(data_back["boxes_transforms"], [])
            assert_allclose(data_back["boxes"], data["boxes"], type_test=False, device_test=False, atol=0.01)
            invert_transform_affine = AffineBoxToWorldCoordinated(box_keys="boxes", box_ref_image_keys="image")
            data_back = invert_transform_affine(affine_result)
            assert_allclose(data_back["boxes"], data["boxes"], type_test=False, device_test=False, atol=0.01)

            # test FlipBoxd
            transform_flip = FlipBoxd(
                image_keys="image", box_keys="boxes", box_ref_image_keys="image", spatial_axis=[0, 1, 2]
            )
            flip_result = transform_flip(data)
            if "boxes_transforms" in flip_result:
                self.assertEqual(len(flip_result["boxes_transforms"]), 1)
            assert_allclose(flip_result["boxes"], expected_flip_result, type_test=True, device_test=True, atol=1e-3)
            invert_transform_flip = Invertd(
                keys=["image", "boxes"], transform=transform_flip, orig_keys=["image", "boxes"]
            )
            data_back = invert_transform_flip(flip_result)
            if "boxes_transforms" in data_back:
                self.assertEqual(data_back["boxes_transforms"], [])
            assert_allclose(data_back["boxes"], data["boxes"], type_test=False, device_test=False, atol=1e-3)
            assert_allclose(data_back["image"], data["image"], type_test=False, device_test=False, atol=1e-3)

            # test RandFlipBoxd
            for spatial_axis in [(0,), (1,), (2,), (0, 1), (1, 2)]:
                transform_flip = RandFlipBoxd(
                    image_keys="image",
                    box_keys="boxes",
                    box_ref_image_keys="image",
                    prob=1.0,
                    spatial_axis=spatial_axis,
                )
                flip_result = transform_flip(data)
                if "boxes_transforms" in flip_result:
                    self.assertEqual(len(flip_result["boxes_transforms"]), 1)
                invert_transform_flip = Invertd(
                    keys=["image", "boxes"], transform=transform_flip, orig_keys=["image", "boxes"]
                )
                data_back = invert_transform_flip(flip_result)
                if "boxes_transforms" in data_back:
                    self.assertEqual(data_back["boxes_transforms"], [])
                assert_allclose(data_back["boxes"], data["boxes"], type_test=False, device_test=False, atol=1e-3)
                assert_allclose(data_back["image"], data["image"], type_test=False, device_test=False, atol=1e-3)

            # test ClipBoxToImaged
            transform_clip = ClipBoxToImaged(
                box_keys="boxes", box_ref_image_keys="image", label_keys=["labels", "scores"], remove_empty=True
            )
            clip_result = transform_clip(data)
            assert_allclose(clip_result["boxes"], expected_clip_result, type_test=True, device_test=True, atol=1e-3)
            assert_allclose(clip_result["labels"], data["labels"][1:], type_test=True, device_test=True, atol=1e-3)
            assert_allclose(clip_result["scores"], data["scores"][1:], type_test=True, device_test=True, atol=1e-3)

            transform_clip = ClipBoxToImaged(
                box_keys="boxes", box_ref_image_keys="image", label_keys=[], remove_empty=True
            )  # corner case when label_keys is empty
            clip_result = transform_clip(data)
            assert_allclose(clip_result["boxes"], expected_clip_result, type_test=True, device_test=True, atol=1e-3)

            # test RandCropBoxByPosNegLabeld
            transform_crop = RandCropBoxByPosNegLabeld(
                image_keys="image", box_keys="boxes", label_keys=["labels", "scores"], spatial_size=2, num_samples=3
            )
            crop_result = transform_crop(data)
            assert len(crop_result) == 3
            for ll in range(3):
                assert_allclose(
                    crop_result[ll]["boxes"].shape[0],
                    crop_result[ll]["labels"].shape[0],
                    type_test=True,
                    device_test=True,
                    atol=1e-3,
                )
                assert_allclose(
                    crop_result[ll]["boxes"].shape[0],
                    crop_result[ll]["scores"].shape[0],
                    type_test=True,
                    device_test=True,
                    atol=1e-3,
                )

            # test RotateBox90d
            transform_rotate = RotateBox90d(
                image_keys="image", box_keys="boxes", box_ref_image_keys="image", k=1, spatial_axes=[0, 1]
            )
            rotate_result = transform_rotate(data)
            self.assertEqual(len(rotate_result["image"].applied_operations), 1)
            assert_allclose(rotate_result["boxes"], expected_rotate_result, type_test=True, device_test=True, atol=1e-3)
            invert_transform_rotate = Invertd(
                keys=["image", "boxes"], transform=transform_rotate, orig_keys=["image", "boxes"]
            )
            data_back = invert_transform_rotate(rotate_result)
            self.assertEqual(data_back["image"].applied_operations, [])
            assert_allclose(data_back["boxes"], data["boxes"], type_test=False, device_test=False, atol=1e-3)
            assert_allclose(data_back["image"], data["image"], type_test=False, device_test=False, atol=1e-3)

            transform_rotate = RandRotateBox90d(
                image_keys="image", box_keys="boxes", box_ref_image_keys="image", prob=1.0, max_k=3, spatial_axes=[0, 1]
            )
            rotate_result = transform_rotate(data)
            self.assertEqual(len(rotate_result["image"].applied_operations), 1)
            invert_transform_rotate = Invertd(
                keys=["image", "boxes"], transform=transform_rotate, orig_keys=["image", "boxes"]
            )
            data_back = invert_transform_rotate(rotate_result)
            self.assertEqual(data_back["image"].applied_operations, [])
            assert_allclose(data_back["boxes"], data["boxes"], type_test=False, device_test=False, atol=1e-3)
            assert_allclose(data_back["image"], data["image"], type_test=False, device_test=False, atol=1e-3)

    def test_crop_shape(self):
        tt = RandCropBoxByPosNegLabeld(
            image_keys=["image"],
            box_keys="box",
            label_keys="label",
            spatial_size=[10, 7, -1],
            whole_box=True,
            num_samples=1,
            pos=1,
            neg=0,
        )
        iii = {
            "image": torch.rand(1, 10, 8, 7),
            "box": torch.tensor(((1.0, 2.0, 3.0, 4.0, 5.0, 6.0),)),
            "label": torch.tensor((1,)).long(),
        }
        self.assertEqual(tt(iii)[0]["image"].shape, (1, 10, 7, 7))


if __name__ == "__main__":
    unittest.main()
