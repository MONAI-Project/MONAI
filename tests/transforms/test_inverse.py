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

import random
import sys
import unittest
from copy import deepcopy
from functools import partial
from typing import TYPE_CHECKING
from unittest.case import skipUnless

import numpy as np
import torch
from parameterized import parameterized

from monai.data import CacheDataset, DataLoader, MetaTensor, create_test_image_2d, create_test_image_3d, decollate_batch
from monai.networks.nets import UNet
from monai.transforms import (
    Affined,
    BorderPadd,
    CenterScaleCropd,
    CenterSpatialCropd,
    Compose,
    CropForegroundd,
    DivisiblePadd,
    EnsureChannelFirstd,
    Flipd,
    FromMetaTensord,
    InvertibleTransform,
    Lambdad,
    LoadImaged,
    Orientationd,
    RandAffined,
    RandAxisFlipd,
    RandCropByLabelClassesd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandLambdad,
    Randomizable,
    RandRotate90d,
    RandRotated,
    RandSpatialCropd,
    RandSpatialCropSamplesd,
    RandWeightedCropd,
    RandZoomd,
    Resized,
    ResizeWithPadOrCrop,
    ResizeWithPadOrCropd,
    Rotate90d,
    Rotated,
    Spacingd,
    SpatialCropd,
    SpatialPadd,
    ToMetaTensord,
    Transposed,
    Zoomd,
    allow_missing_keys_mode,
    convert_applied_interp_mode,
    reset_ops_id,
)
from monai.utils import first, get_seed, optional_import, set_determinism
from tests.util import make_nifti_image, make_rand_affine

if TYPE_CHECKING:
    has_nib = True
else:
    _, has_nib = optional_import("nibabel")

KEYS = ["image", "label"]

TESTS: list[tuple] = []

# For pad, start with odd/even images and add odd/even amounts
for name in ("1D even", "1D odd"):
    for val in (3, 4):
        for t in (
            partial(SpatialPadd, spatial_size=val, method="symmetric"),
            partial(SpatialPadd, spatial_size=val, method="end"),
            partial(BorderPadd, spatial_border=[val, val + 1]),
            partial(DivisiblePadd, k=val),
            partial(ResizeWithPadOrCropd, spatial_size=20 + val),
            partial(CenterSpatialCropd, roi_size=10 + val),
            partial(CenterScaleCropd, roi_scale=0.8),
            partial(CropForegroundd, source_key="label"),
            partial(SpatialCropd, roi_center=10, roi_size=10 + val),
            partial(SpatialCropd, roi_center=11, roi_size=10 + val),
            partial(SpatialCropd, roi_start=val, roi_end=17),
            partial(SpatialCropd, roi_start=val, roi_end=16),
            partial(RandSpatialCropd, roi_size=12 + val),
            partial(ResizeWithPadOrCropd, spatial_size=21 - val),
        ):
            TESTS.append((t.func.__name__ + name, name, 0, True, t(KEYS)))  # type: ignore

# non-sensical tests: crop bigger or pad smaller or -ve values
for t in (
    partial(DivisiblePadd, k=-3),
    partial(CenterSpatialCropd, roi_size=-3),
    partial(RandSpatialCropd, roi_size=-3),
    partial(SpatialPadd, spatial_size=15),
    partial(BorderPadd, spatial_border=[15, 16]),
    partial(CenterSpatialCropd, roi_size=30),
    partial(SpatialCropd, roi_center=10, roi_size=100),
    partial(SpatialCropd, roi_start=3, roi_end=100),
):
    TESTS.append((t.func.__name__ + "bad 1D even", "1D even", 0, True, t(KEYS)))  # type: ignore

TESTS.append(
    (
        "SpatialPadd (x2) 2d",
        "2D",
        0,
        True,
        SpatialPadd(KEYS, spatial_size=[111, 113], method="end"),
        SpatialPadd(KEYS, spatial_size=[118, 117]),
    )
)

TESTS.append(("SpatialPadd 3d", "3D", 0, True, SpatialPadd(KEYS, spatial_size=[112, 113, 116])))

TESTS.append(("SpatialCropd 2d", "2D", 0, True, SpatialCropd(KEYS, [49, 51], [90, 89])))

TESTS.append(
    (
        "SpatialCropd 3d",
        "3D",
        0,
        True,
        SpatialCropd(KEYS, roi_slices=[slice(s, e) for s, e in zip([None, None, -99], [None, -2, None])]),
    )
)

TESTS.append(("SpatialCropd 2d", "2D", 0, True, SpatialCropd(KEYS, [49, 51], [390, 89])))

TESTS.append(("SpatialCropd 3d", "3D", 0, True, SpatialCropd(KEYS, [49, 51, 44], [90, 89, 93])))

TESTS.append(("RandSpatialCropd 2d", "2D", 0, True, RandSpatialCropd(KEYS, [96, 93], None, True, False)))

TESTS.append(("RandSpatialCropd 3d", "3D", 0, True, RandSpatialCropd(KEYS, [96, 93, 92], None, False, False)))

TESTS.append(("BorderPadd 2d", "2D", 0, True, BorderPadd(KEYS, [3, 7, 2, 5])))

TESTS.append(("BorderPadd 2d", "2D", 0, True, BorderPadd(KEYS, [3, 7])))

TESTS.append(("BorderPadd 3d", "3D", 0, True, BorderPadd(KEYS, [4])))

TESTS.append(("DivisiblePadd 2d", "2D", 0, True, DivisiblePadd(KEYS, k=4)))

TESTS.append(("DivisiblePadd 3d", "3D", 0, True, DivisiblePadd(KEYS, k=[4, 8, 11])))

TESTS.append(("CenterSpatialCropd 2d", "2D", 0, True, CenterSpatialCropd(KEYS, roi_size=95)))

TESTS.append(("CenterSpatialCropd 3d", "3D", 0, True, CenterSpatialCropd(KEYS, roi_size=[95, 97, 98])))

TESTS.append(("CropForegroundd 2d", "2D", 0, True, CropForegroundd(KEYS, source_key="label", margin=2)))

TESTS.append(("CropForegroundd 3d", "3D", 0, True, CropForegroundd(KEYS, source_key="label", k_divisible=[5, 101, 2])))

TESTS.append(("ResizeWithPadOrCropd 3d", "3D", 0, True, ResizeWithPadOrCropd(KEYS, [201, 150, 105])))

TESTS.append(("Flipd 3d", "3D", 0, True, Flipd(KEYS, [1, 2])))
TESTS.append(("Flipd 3d", "3D", 0, True, Flipd(KEYS, [1, 2])))

TESTS.append(("RandFlipd 3d", "3D", 0, True, RandFlipd(KEYS, 1, [1, 2])))

TESTS.append(("RandAxisFlipd 3d", "3D", 0, True, RandAxisFlipd(KEYS, 1)))
TESTS.append(("RandAxisFlipd 3d", "3D", 0, True, RandAxisFlipd(KEYS, 1)))

for acc in [True, False]:
    TESTS.append(("Orientationd 3d", "3D", 0, True, Orientationd(KEYS, "RAS", as_closest_canonical=acc)))

TESTS.append(("Rotate90d 2d", "2D", 0, True, Rotate90d(KEYS)))

TESTS.append(("Rotate90d 3d", "3D", 0, True, Rotate90d(KEYS, k=2, spatial_axes=(1, 2))))

TESTS.append(("RandRotate90d 3d", "3D", 0, True, RandRotate90d(KEYS, prob=1, spatial_axes=(1, 2))))

TESTS.append(("Spacingd 3d", "3D", 3e-2, True, Spacingd(KEYS, [0.5, 0.7, 0.9], diagonal=False)))

TESTS.append(("Resized 2d", "2D", 2e-1, True, Resized(KEYS, [50, 47])))

TESTS.append(("Resized 3d", "3D", 5e-2, True, Resized(KEYS, [201, 150, 78])))

TESTS.append(("Resized longest 2d", "2D", 2e-1, True, Resized(KEYS, 47, "longest", "area")))

TESTS.append(("Resized longest 3d", "3D", 5e-2, True, Resized(KEYS, 201, "longest", "trilinear", True)))

TESTS.append(
    ("Lambdad 2d", "2D", 5e-2, False, Lambdad(KEYS, func=lambda x: x + 5, inv_func=lambda x: x - 5, overwrite=True))
)

TESTS.append(
    (
        "RandLambdad 3d",
        "3D",
        5e-2,
        False,
        RandLambdad(KEYS, func=lambda x: x * 10, inv_func=lambda x: x / 10, overwrite=True, prob=0.5),
    )
)

TESTS.append(("Zoomd 1d", "1D odd", 0, True, Zoomd(KEYS, zoom=2, keep_size=False)))

TESTS.append(("Zoomd 2d", "2D", 2e-1, True, Zoomd(KEYS, zoom=0.9)))

TESTS.append(("Zoomd 3d", "3D", 3e-2, True, Zoomd(KEYS, zoom=[2.5, 1, 3], keep_size=False)))

TESTS.append(("RandZoom 3d", "3D", 9e-2, True, RandZoomd(KEYS, 1, [0.5, 0.6, 0.9], [1.1, 1, 1.05], keep_size=True)))

TESTS.append(("RandRotated, prob 0", "2D", 0, True, RandRotated(KEYS, prob=0, dtype=np.float64)))

TESTS.append(
    (
        "Rotated 2d",
        "2D",
        8e-2,
        True,
        Rotated(KEYS, random.uniform(np.pi / 6, np.pi), keep_size=True, align_corners=False, dtype=np.float64),
    )
)

TESTS.append(
    (
        "Rotated 3d",
        "3D",
        1e-1,
        True,
        Rotated(KEYS, [random.uniform(np.pi / 6, np.pi) for _ in range(3)], True, dtype=np.float64),
    )
)

TESTS.append(
    (
        "RandRotated 3d",
        "3D",
        1e-1,
        True,
        RandRotated(KEYS, *[random.uniform(np.pi / 6, np.pi) for _ in range(3)], 1, dtype=np.float64),  # type: ignore
    )
)

TESTS.append(("Transposed 2d", "2D", 0, False, Transposed(KEYS, [0, 2, 1])))  # channel=0

TESTS.append(("Transposed 3d", "3D", 0, False, Transposed(KEYS, [0, 3, 1, 2])))  # channel=0

TESTS.append(
    (
        "Affine 3d",
        "3D",
        1e-1,
        True,
        Affined(
            KEYS,
            spatial_size=[155, 179, 192],
            rotate_params=[np.pi / 6, -np.pi / 5, np.pi / 7],
            shear_params=[0.5, 0.5],
            translate_params=[10, 5, -4],
            scale_params=[0.8, 1.3],
        ),
    )
)

TESTS.append(
    (
        "RandAffine 3d",
        "3D",
        1e-1,
        True,
        RandAffined(
            KEYS,
            [155, 179, 192],
            prob=1,
            padding_mode="zeros",
            rotate_range=[np.pi / 6, -np.pi / 5, np.pi / 7],
            shear_range=[(0.5, 0.5)],
            translate_range=[10, 5, -4],
            scale_range=[(0.8, 1.2), (0.9, 1.3)],
        ),
    )
)

TESTS.append(("RandAffine 3d", "3D", 0, True, RandAffined(KEYS, spatial_size=None, prob=0)))

TESTS.append(
    (
        "RandCropByLabelClassesd 2d",
        "2D",
        1e-7,
        True,
        RandCropByLabelClassesd(KEYS, "label", (99, 96), ratios=[1, 2, 3, 4, 5], num_classes=5, num_samples=10),
    )
)

TESTS.append(
    ("RandCropByPosNegLabeld 2d", "2D", 1e-7, True, RandCropByPosNegLabeld(KEYS, "label", (99, 96), num_samples=10))
)

TESTS.append(("RandSpatialCropSamplesd 2d", "2D", 1e-7, True, RandSpatialCropSamplesd(KEYS, (90, 91), num_samples=10)))

TESTS.append(("RandWeightedCropd 2d", "2D", 1e-7, True, RandWeightedCropd(KEYS, "label", (90, 91), num_samples=10)))

TESTS_COMPOSE_X2 = [(t[0] + " Compose", t[1], t[2], t[3], Compose(Compose(t[4:]))) for t in TESTS]

TESTS = TESTS + TESTS_COMPOSE_X2

NUM_SAMPLES = 5
N_SAMPLES_TESTS = [
    [RandCropByLabelClassesd(KEYS, "label", (110, 99), [1, 2, 3, 4, 5], num_classes=5, num_samples=NUM_SAMPLES)],
    [RandCropByPosNegLabeld(KEYS, "label", (110, 99), num_samples=NUM_SAMPLES)],
    [RandSpatialCropSamplesd(KEYS, (90, 91), num_samples=NUM_SAMPLES, random_size=False)],
    [RandWeightedCropd(KEYS, "label", (90, 91), num_samples=NUM_SAMPLES)],
]


def no_collation(x):
    return x


class TestInverse(unittest.TestCase):
    """Test inverse methods.

    If tests are failing, the following function might be useful for displaying
    `x`, `fx`, `f⁻¹fx` and `x - f⁻¹fx`.

    .. code-block:: python

        def plot_im(orig, fwd_bck, fwd):
            import matplotlib.pyplot as plt
            diff_orig_fwd_bck = orig - fwd_bck
            ims_to_show = [orig, fwd, fwd_bck, diff_orig_fwd_bck]
            titles = ["x", "fx", "f⁻¹fx", "x - f⁻¹fx"]
            fig, axes = plt.subplots(1, 4, gridspec_kw={"width_ratios": [i.shape[1] for i in ims_to_show]})
            vmin = min(np.array(i).min() for i in [orig, fwd_bck, fwd])
            vmax = max(np.array(i).max() for i in [orig, fwd_bck, fwd])
            for im, title, ax in zip(ims_to_show, titles, axes):
                _vmin, _vmax = (vmin, vmax) if id(im) != id(diff_orig_fwd_bck) else (None, None)
                im = np.squeeze(np.array(im))
                while im.ndim > 2:
                    im = im[..., im.shape[-1] // 2]
                im_show = ax.imshow(np.squeeze(im), vmin=_vmin, vmax=_vmax)
                ax.set_title(title, fontsize=25)
                ax.axis("off")
                fig.colorbar(im_show, ax=ax)
            plt.show()

    This can then be added to the exception:

    .. code-block:: python

        except AssertionError:
            print(
                f"Failed: {name}. Mean diff = {mean_diff} (expected <= {acceptable_diff}), unmodified diff: {unmodded_diff}"
            )
            if orig[0].ndim > 1:
                plot_im(orig, fwd_bck, unmodified)
    """

    def setUp(self):
        if not has_nib:
            self.skipTest("nibabel required for test_inverse")

        set_determinism(seed=0)

        self.all_data = {}

        affine = make_rand_affine()
        affine[0] *= 2

        for size in [10, 11]:
            # pad 5 onto both ends so that cropping can be lossless
            im_1d = np.pad(np.arange(size), 5)[None]
            name = "1D even" if size % 2 == 0 else "1D odd"
            self.all_data[name] = {
                "image": torch.as_tensor(np.array(im_1d, copy=True)),
                "label": torch.as_tensor(np.array(im_1d, copy=True)),
                "other": torch.as_tensor(np.array(im_1d, copy=True)),
            }

        im_2d_fname, seg_2d_fname = (make_nifti_image(i) for i in create_test_image_2d(101, 100))
        im_3d_fname, seg_3d_fname = (make_nifti_image(i, affine) for i in create_test_image_3d(100, 101, 107))

        load_ims = Compose(
            [LoadImaged(KEYS), EnsureChannelFirstd(KEYS, channel_dim="no_channel"), FromMetaTensord(KEYS)]
        )
        self.all_data["2D"] = load_ims({"image": im_2d_fname, "label": seg_2d_fname})
        self.all_data["3D"] = load_ims({"image": im_3d_fname, "label": seg_3d_fname})

    def tearDown(self):
        set_determinism(seed=None)

    def check_inverse(self, name, keys, orig_d, fwd_bck_d, unmodified_d, acceptable_diff):
        for key in keys:
            orig = orig_d[key]
            fwd_bck = fwd_bck_d[key]
            if isinstance(fwd_bck, torch.Tensor):
                fwd_bck = fwd_bck.cpu().numpy()
            unmodified = unmodified_d[key]
            if isinstance(orig, np.ndarray):
                mean_diff = np.mean(np.abs(orig - fwd_bck))
                resized = ResizeWithPadOrCrop(orig.shape[1:])(unmodified)
                if isinstance(resized, torch.Tensor):
                    resized = resized.detach().cpu().numpy()
                unmodded_diff = np.mean(np.abs(orig - resized))
                try:
                    self.assertLessEqual(mean_diff, acceptable_diff)
                except AssertionError:
                    print(
                        f"Failed: {name}. Mean diff = {mean_diff} (expected <= {acceptable_diff}), unmodified diff: {unmodded_diff}"
                    )
                    if orig[0].ndim == 1:
                        print("orig", orig[0])
                        print("fwd_bck", fwd_bck[0])
                        print("unmod", unmodified[0])
                    raise

    @parameterized.expand(TESTS)
    def test_inverse(self, _, data_name, acceptable_diff, is_meta, *transforms):
        name = _

        data = self.all_data[data_name]
        if is_meta:
            data = ToMetaTensord(KEYS)(data)

        forwards = [data.copy()]

        # Apply forwards
        for t in transforms:
            if isinstance(t, Randomizable):
                t.set_random_state(seed=get_seed())
            forwards.append(t(forwards[-1]))

        # Apply inverses
        fwd_bck = forwards[-1].copy()
        for i, t in enumerate(reversed(transforms)):
            if isinstance(t, InvertibleTransform):
                if isinstance(fwd_bck, list):
                    for j, _fwd_bck in enumerate(fwd_bck):
                        fwd_bck = t.inverse(_fwd_bck)
                        self.check_inverse(
                            name, data.keys(), forwards[-i - 2], fwd_bck, forwards[-1][j], acceptable_diff
                        )
                else:
                    fwd_bck = t.inverse(fwd_bck)
                    self.check_inverse(name, data.keys(), forwards[-i - 2], fwd_bck, forwards[-1], acceptable_diff)

    # skip this test if multiprocessing uses 'spawn', as the check is only basic anyway
    @skipUnless(torch.multiprocessing.get_start_method() == "spawn", "requires spawn")
    def test_fail(self):
        t1 = SpatialPadd("image", [10, 5])
        data = t1(self.all_data["2D"])

        # Check that error is thrown when inverse are used out of order.
        t2 = ResizeWithPadOrCropd("image", [10, 5])
        with self.assertRaises(RuntimeError):
            t2.inverse(data)

    @parameterized.expand(N_SAMPLES_TESTS)
    def test_inverse_inferred_seg(self, extra_transform):
        test_data = []
        for _ in range(20):
            image, label = create_test_image_2d(100, 101)
            test_data.append({"image": image, "label": label.astype(np.float32)})

        batch_size = 10
        # num workers = 0 for mac
        num_workers = 2 if sys.platform == "linux" else 0
        transforms = Compose(
            [EnsureChannelFirstd(KEYS, channel_dim="no_channel"), SpatialPadd(KEYS, (150, 153)), extra_transform]
        )

        dataset = CacheDataset(test_data, transform=transforms, progress=False)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = UNet(spatial_dims=2, in_channels=1, out_channels=1, channels=(2, 4), strides=(1,)).to(device)

        data = first(loader)
        self.assertEqual(data["image"].shape[0], batch_size * NUM_SAMPLES)

        labels = data["label"].to(device)
        self.assertIsInstance(labels, MetaTensor)
        segs = model(labels).detach().cpu()
        segs_decollated = decollate_batch(segs)
        self.assertIsInstance(segs_decollated[0], MetaTensor)
        # inverse of individual segmentation
        seg_metatensor = first(segs_decollated)
        # test to convert interpolation mode for 1 data of model output batch
        convert_applied_interp_mode(seg_metatensor.applied_operations, mode="nearest", align_corners=None)

        # manually invert the last crop samples
        xform = seg_metatensor.applied_operations.pop(-1)
        shape_before_extra_xform = xform["orig_size"]
        resizer = ResizeWithPadOrCrop(spatial_size=shape_before_extra_xform)
        with resizer.trace_transform(False):
            seg_metatensor = resizer(seg_metatensor)
        no_ops_id_tensor = reset_ops_id(deepcopy(seg_metatensor))

        with allow_missing_keys_mode(transforms):
            inv_seg = transforms.inverse({"label": seg_metatensor})["label"]
            inv_seg_1 = transforms.inverse({"label": no_ops_id_tensor})["label"]
        self.assertEqual(inv_seg.shape[1:], test_data[0]["label"].shape)
        self.assertEqual(inv_seg_1.shape[1:], test_data[0]["label"].shape)

        # # Inverse of batch
        # batch_inverter = BatchInverseTransform(transforms, loader, collate_fn=no_collation, detach=True)
        # with allow_missing_keys_mode(transforms):
        #     inv_batch = batch_inverter(first(loader))
        # self.assertEqual(inv_batch[0]["label"].shape[1:], test_data[0]["label"].shape)


if __name__ == "__main__":
    unittest.main()
