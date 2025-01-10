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
"""
A collection of dictionary-based wrappers around the "vanilla" transforms for box operations
defined in :py:class:`monai.apps.detection.transforms.array`.

Class names are ended with 'd' to denote dictionary-based transforms.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from copy import deepcopy
from typing import Any

import numpy as np
import torch

from monai.apps.detection.transforms.array import (
    AffineBox,
    BoxToMask,
    ClipBoxToImage,
    ConvertBoxMode,
    ConvertBoxToStandardMode,
    FlipBox,
    MaskToBox,
    RotateBox90,
    SpatialCropBox,
    StandardizeEmptyBox,
    ZoomBox,
)
from monai.apps.detection.transforms.box_ops import convert_box_to_mask
from monai.config import KeysCollection, SequenceStr
from monai.config.type_definitions import DtypeLike, NdarrayOrTensor
from monai.data.box_utils import COMPUTE_DTYPE, BoxMode, clip_boxes_to_image
from monai.data.meta_tensor import MetaTensor, get_track_meta
from monai.data.utils import orientation_ras_lps
from monai.transforms import Flip, RandFlip, RandZoom, Rotate90, SpatialCrop, Zoom
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.transform import MapTransform, Randomizable, RandomizableTransform
from monai.transforms.utils import generate_pos_neg_label_crop_centers, map_binary_to_indices
from monai.utils import InterpolateMode, NumpyPadMode, ensure_tuple, ensure_tuple_rep, fall_back_tuple
from monai.utils.enums import PostFix, TraceKeys
from monai.utils.type_conversion import convert_data_type, convert_to_tensor

__all__ = [
    "StandardizeEmptyBoxd",
    "StandardizeEmptyBoxD",
    "StandardizeEmptyBoxDict",
    "ConvertBoxModed",
    "ConvertBoxModeD",
    "ConvertBoxModeDict",
    "ConvertBoxToStandardModed",
    "ConvertBoxToStandardModeD",
    "ConvertBoxToStandardModeDict",
    "AffineBoxToImageCoordinated",
    "AffineBoxToImageCoordinateD",
    "AffineBoxToImageCoordinateDict",
    "ZoomBoxd",
    "ZoomBoxD",
    "ZoomBoxDict",
    "RandZoomBoxd",
    "RandZoomBoxD",
    "RandZoomBoxDict",
    "FlipBoxd",
    "FlipBoxD",
    "FlipBoxDict",
    "RandFlipBoxd",
    "RandFlipBoxD",
    "RandFlipBoxDict",
    "ClipBoxToImaged",
    "ClipBoxToImageD",
    "ClipBoxToImageDict",
    "BoxToMaskd",
    "BoxToMaskD",
    "BoxToMaskDict",
    "MaskToBoxd",
    "MaskToBoxD",
    "MaskToBoxDict",
    "RandCropBoxByPosNegLabeld",
    "RandCropBoxByPosNegLabelD",
    "RandCropBoxByPosNegLabelDict",
    "RotateBox90d",
    "RotateBox90D",
    "RotateBox90Dict",
    "RandRotateBox90d",
    "RandRotateBox90D",
    "RandRotateBox90Dict",
]

DEFAULT_POST_FIX = PostFix.meta()


class StandardizeEmptyBoxd(MapTransform, InvertibleTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.apps.detection.transforms.array.StandardizeEmptyBox`.

    When boxes are empty, this transform standardize it to shape of (0,4) or (0,6).

    Example:
        .. code-block:: python

            data = {"boxes": torch.ones(0,), "image": torch.ones(1, 128, 128, 128)}
            box_converter = StandardizeEmptyBoxd(box_keys=["boxes"], box_ref_image_keys="image")
            box_converter(data)
    """

    def __init__(self, box_keys: KeysCollection, box_ref_image_keys: str, allow_missing_keys: bool = False) -> None:
        """
        Args:
            box_keys: Keys to pick data for transformation.
            box_ref_image_keys: The single key that represents the reference image to which ``box_keys`` are attached.
            allow_missing_keys: don't raise exception if key is missing.

        See also :py:class:`monai.apps.detection,transforms.array.ConvertBoxToStandardMode`
        """
        super().__init__(box_keys, allow_missing_keys)
        box_ref_image_keys_tuple = ensure_tuple(box_ref_image_keys)
        if len(box_ref_image_keys_tuple) > 1:
            raise ValueError(
                "Please provide a single key for box_ref_image_keys.\
                All boxes of box_keys are attached to box_ref_image_keys."
            )
        self.box_ref_image_keys = box_ref_image_keys

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        spatial_dims = len(d[self.box_ref_image_keys].shape) - 1
        self.converter = StandardizeEmptyBox(spatial_dims=spatial_dims)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d

    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        return dict(data)


class ConvertBoxModed(MapTransform, InvertibleTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.apps.detection.transforms.array.ConvertBoxMode`.

    This transform converts the boxes in src_mode to the dst_mode.

    Example:
        .. code-block:: python

            data = {"boxes": torch.ones(10,4)}
            # convert boxes with format [xmin, ymin, xmax, ymax] to [xcenter, ycenter, xsize, ysize].
            box_converter = ConvertBoxModed(box_keys=["boxes"], src_mode="xyxy", dst_mode="ccwh")
            box_converter(data)
    """

    def __init__(
        self,
        box_keys: KeysCollection,
        src_mode: str | BoxMode | type[BoxMode] | None = None,
        dst_mode: str | BoxMode | type[BoxMode] | None = None,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            box_keys: Keys to pick data for transformation.
            src_mode: source box mode. If it is not given, this func will assume it is ``StandardMode()``.
                It follows the same format with ``src_mode`` in :class:`~monai.apps.detection.transforms.array.ConvertBoxMode` .
            dst_mode: target box mode. If it is not given, this func will assume it is ``StandardMode()``.
                It follows the same format with ``src_mode`` in :class:`~monai.apps.detection.transforms.array.ConvertBoxMode` .
            allow_missing_keys: don't raise exception if key is missing.

        See also :py:class:`monai.apps.detection,transforms.array.ConvertBoxMode`
        """
        super().__init__(box_keys, allow_missing_keys)
        self.converter = ConvertBoxMode(src_mode=src_mode, dst_mode=dst_mode)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
            self.push_transform(d, key, extra_info={"src": self.converter.src_mode, "dst": self.converter.dst_mode})
        return d

    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            tr = self.get_most_recent_transform(d, key)
            src_mode, dst_mode = tr[TraceKeys.EXTRA_INFO]["src"], tr[TraceKeys.EXTRA_INFO]["dst"]
            inverse_converter = ConvertBoxMode(src_mode=dst_mode, dst_mode=src_mode)
            # Inverse is same as forward
            d[key] = inverse_converter(d[key])
            # Remove the applied transform
            self.pop_transform(d, key)
        return d


class ConvertBoxToStandardModed(MapTransform, InvertibleTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.apps.detection.transforms.array.ConvertBoxToStandardMode`.

    Convert given boxes to standard mode.
    Standard mode is "xyxy" or "xyzxyz",
    representing box format of [xmin, ymin, xmax, ymax] or [xmin, ymin, zmin, xmax, ymax, zmax].

    Example:
        .. code-block:: python

            data = {"boxes": torch.ones(10,6)}
            # convert boxes with format [xmin, xmax, ymin, ymax, zmin, zmax] to [xmin, ymin, zmin, xmax, ymax, zmax]
            box_converter = ConvertBoxToStandardModed(box_keys=["boxes"], mode="xxyyzz")
            box_converter(data)
    """

    def __init__(
        self,
        box_keys: KeysCollection,
        mode: str | BoxMode | type[BoxMode] | None = None,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            box_keys: Keys to pick data for transformation.
            mode: source box mode. If it is not given, this func will assume it is ``StandardMode()``.
                It follows the same format with ``src_mode`` in :class:`~monai.apps.detection.transforms.array.ConvertBoxMode` .
            allow_missing_keys: don't raise exception if key is missing.

        See also :py:class:`monai.apps.detection,transforms.array.ConvertBoxToStandardMode`
        """
        super().__init__(box_keys, allow_missing_keys)
        self.converter = ConvertBoxToStandardMode(mode=mode)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
            self.push_transform(d, key, extra_info={"mode": self.converter.mode})
        return d

    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            tr = self.get_most_recent_transform(d, key)
            original_mode = tr[TraceKeys.EXTRA_INFO]["mode"]
            inverse_converter = ConvertBoxMode(src_mode=None, dst_mode=original_mode)
            # Inverse is same as forward
            d[key] = inverse_converter(d[key])
            # Remove the applied transform
            self.pop_transform(d, key)
        return d


class AffineBoxToImageCoordinated(MapTransform, InvertibleTransform):
    """
    Dictionary-based transform that converts box in world coordinate to image coordinate.

    Args:
        box_keys: Keys to pick box data for transformation. The box mode is assumed to be ``StandardMode``.
        box_ref_image_keys: The single key that represents the reference image to which ``box_keys`` are attached.
        remove_empty: whether to remove the boxes that are actually empty
        allow_missing_keys: don't raise exception if key is missing.
        image_meta_key: explicitly indicate the key of the corresponding metadata dictionary.
            for example, for data with key `image`, the metadata by default is in `image_meta_dict`.
            the metadata is a dictionary object which contains: filename, affine, original_shape, etc.
            it is a string, map to the `box_ref_image_key`.
            if None, will try to construct meta_keys by `box_ref_image_key_{meta_key_postfix}`.
        image_meta_key_postfix: if image_meta_keys=None, use `box_ref_image_key_{postfix}` to fetch the metadata according
            to the key data, default is `meta_dict`, the metadata is a dictionary object.
            For example, to handle key `image`,  read/write affine matrices from the
            metadata `image_meta_dict` dictionary's `affine` field.
        affine_lps_to_ras: default ``False``. Yet if 1) the image is read by ITKReader,
            and 2) the ITKReader has affine_lps_to_ras=True, and 3) the box is in world coordinate,
            then set ``affine_lps_to_ras=True``.
    """

    def __init__(
        self,
        box_keys: KeysCollection,
        box_ref_image_keys: str,
        allow_missing_keys: bool = False,
        image_meta_key: str | None = None,
        image_meta_key_postfix: str | None = DEFAULT_POST_FIX,
        affine_lps_to_ras: bool = False,
    ) -> None:
        super().__init__(box_keys, allow_missing_keys)
        box_ref_image_keys_tuple = ensure_tuple(box_ref_image_keys)
        if len(box_ref_image_keys_tuple) > 1:
            raise ValueError(
                "Please provide a single key for box_ref_image_keys.\
                All boxes of box_keys are attached to box_ref_image_keys."
            )
        self.box_ref_image_keys = box_ref_image_keys
        self.image_meta_key = image_meta_key or f"{box_ref_image_keys}_{image_meta_key_postfix}"
        self.converter_to_image_coordinate = AffineBox()
        self.affine_lps_to_ras = affine_lps_to_ras

    def extract_affine(self, data: Mapping[Hashable, torch.Tensor]) -> tuple[NdarrayOrTensor, torch.Tensor]:
        d = dict(data)

        meta_key = self.image_meta_key
        # extract affine matrix from metadata
        if isinstance(d[self.box_ref_image_keys], MetaTensor):
            meta_dict = d[self.box_ref_image_keys].meta  # type: ignore
        elif meta_key in d:
            meta_dict = d[meta_key]
        else:
            raise ValueError(f"{meta_key} is not found. Please check whether it is the correct the image meta key.")
        if "affine" not in meta_dict:
            raise ValueError(
                f"'affine' is not found in {meta_key}. \
                Please check whether it is the correct the image meta key."
            )
        affine: NdarrayOrTensor = meta_dict["affine"]

        if self.affine_lps_to_ras:  # RAS affine
            affine = orientation_ras_lps(affine)

        # when convert boxes from world coordinate to image coordinate,
        # we apply inverse affine transform
        affine_t, *_ = convert_data_type(affine, torch.Tensor)
        # torch.inverse should not run in half precision
        inv_affine_t = torch.inverse(affine_t.to(COMPUTE_DTYPE))
        return affine, inv_affine_t

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)

        affine, inv_affine_t = self.extract_affine(data)  # type: ignore

        for key in self.key_iterator(d):
            d[key] = self.converter_to_image_coordinate(d[key], affine=inv_affine_t)
            self.push_transform(d, key, extra_info={"affine": affine})
        return d

    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key)
            affine = transform["extra_info"]["affine"]
            d[key] = AffineBox()(d[key], affine=affine)
            self.pop_transform(d, key)
        return d


class AffineBoxToWorldCoordinated(AffineBoxToImageCoordinated):
    """
    Dictionary-based transform that converts box in image coordinate to world coordinate.

    Args:
        box_keys: Keys to pick box data for transformation. The box mode is assumed to be ``StandardMode``.
        box_ref_image_keys: The single key that represents the reference image to which ``box_keys`` are attached.
        remove_empty: whether to remove the boxes that are actually empty
        allow_missing_keys: don't raise exception if key is missing.
        image_meta_key: explicitly indicate the key of the corresponding metadata dictionary.
            for example, for data with key `image`, the metadata by default is in `image_meta_dict`.
            the metadata is a dictionary object which contains: filename, affine, original_shape, etc.
            it is a string, map to the `box_ref_image_key`.
            if None, will try to construct meta_keys by `box_ref_image_key_{meta_key_postfix}`.
        image_meta_key_postfix: if image_meta_keys=None, use `box_ref_image_key_{postfix}` to fetch the metadata according
            to the key data, default is `meta_dict`, the metadata is a dictionary object.
            For example, to handle key `image`,  read/write affine matrices from the
            metadata `image_meta_dict` dictionary's `affine` field.
        affine_lps_to_ras: default ``False``. Yet if 1) the image is read by ITKReader,
            and 2) the ITKReader has affine_lps_to_ras=True, and 3) the box is in world coordinate,
            then set ``affine_lps_to_ras=True``.
    """

    def __init__(
        self,
        box_keys: KeysCollection,
        box_ref_image_keys: str,
        allow_missing_keys: bool = False,
        image_meta_key: str | None = None,
        image_meta_key_postfix: str | None = DEFAULT_POST_FIX,
        affine_lps_to_ras: bool = False,
    ) -> None:
        super().__init__(
            box_keys, box_ref_image_keys, allow_missing_keys, image_meta_key, image_meta_key_postfix, affine_lps_to_ras
        )
        self.converter_to_world_coordinate = AffineBox()

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)

        affine, inv_affine_t = self.extract_affine(data)  # type: ignore

        for key in self.key_iterator(d):
            d[key] = self.converter_to_world_coordinate(d[key], affine=affine)
            self.push_transform(d, key, extra_info={"affine": inv_affine_t})
        return d


class ZoomBoxd(MapTransform, InvertibleTransform):
    """
    Dictionary-based transform that zooms input boxes and images with the given zoom scale.

    Args:
        image_keys: Keys to pick image data for transformation.
        box_keys: Keys to pick box data for transformation. The box mode is assumed to be ``StandardMode``.
        box_ref_image_keys: Keys that represent the reference images to which ``box_keys`` are attached.
        zoom: The zoom factor along the spatial axes.
            If a float, zoom is the same for each spatial axis.
            If a sequence, zoom should contain one value for each spatial axis.
        mode: {``"nearest"``, ``"nearest-exact"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
            The interpolation mode. Defaults to ``"area"``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
            It also can be a sequence of string, each element corresponds to a key in ``keys``.
        padding_mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
            ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
            available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
            One of the listed string values or a user supplied function. Defaults to ``"constant"``.
            The mode to pad data after zooming.
            See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
            https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        align_corners: This only has an effect when mode is
            'linear', 'bilinear', 'bicubic' or 'trilinear'. Default: None.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
            It also can be a sequence of bool or None, each element corresponds to a key in ``keys``.
        keep_size: Should keep original size (pad if needed), default is True.
        allow_missing_keys: don't raise exception if key is missing.
        kwargs: other arguments for the `np.pad` or `torch.pad` function.
            note that `np.pad` treats channel dimension as the first dimension.
    """

    def __init__(
        self,
        image_keys: KeysCollection,
        box_keys: KeysCollection,
        box_ref_image_keys: KeysCollection,
        zoom: Sequence[float] | float,
        mode: SequenceStr = InterpolateMode.AREA,
        padding_mode: SequenceStr = NumpyPadMode.EDGE,
        align_corners: Sequence[bool | None] | bool | None = None,
        keep_size: bool = True,
        allow_missing_keys: bool = False,
        **kwargs: Any,
    ) -> None:
        self.image_keys = ensure_tuple(image_keys)
        self.box_keys = ensure_tuple(box_keys)
        super().__init__(self.image_keys + self.box_keys, allow_missing_keys)
        self.box_ref_image_keys = ensure_tuple_rep(box_ref_image_keys, len(self.box_keys))

        self.mode = ensure_tuple_rep(mode, len(self.image_keys))
        self.padding_mode = ensure_tuple_rep(padding_mode, len(self.image_keys))
        self.align_corners = ensure_tuple_rep(align_corners, len(self.image_keys))
        self.zoomer = Zoom(zoom=zoom, keep_size=keep_size, **kwargs)
        self.keep_size = keep_size

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        d: dict[Hashable, torch.Tensor] = dict(data)

        # zoom box
        for box_key, box_ref_image_key in zip(self.box_keys, self.box_ref_image_keys):
            src_spatial_size = d[box_ref_image_key].shape[1:]
            dst_spatial_size = [int(round(z * ss)) for z, ss in zip(self.zoomer.zoom, src_spatial_size)]  # type: ignore
            self.zoomer.zoom = [ds / float(ss) for ss, ds in zip(src_spatial_size, dst_spatial_size)]
            d[box_key] = ZoomBox(zoom=self.zoomer.zoom, keep_size=self.keep_size)(
                d[box_key], src_spatial_size=src_spatial_size
            )
            self.push_transform(
                d,
                box_key,
                extra_info={"zoom": self.zoomer.zoom, "src_spatial_size": src_spatial_size, "type": "box_key"},
            )

        # zoom image
        for key, mode, padding_mode, align_corners in zip(
            self.image_keys, self.mode, self.padding_mode, self.align_corners
        ):
            d[key] = self.zoomer(d[key], mode=mode, padding_mode=padding_mode, align_corners=align_corners)

        return d

    def inverse(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        d: dict[Hashable, torch.Tensor] = dict(data)

        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key, check=False)
            key_type = transform[TraceKeys.EXTRA_INFO].get("type", "image_key")
            # zoom image, copied from monai.transforms.spatial.dictionary.Zoomd
            if key_type == "image_key":
                d[key] = self.zoomer.inverse(d[key])

            # zoom boxes
            if key_type == "box_key":
                zoom = np.array(transform[TraceKeys.EXTRA_INFO]["zoom"])
                src_spatial_size = transform[TraceKeys.EXTRA_INFO]["src_spatial_size"]
                box_inverse_transform = ZoomBox(zoom=(1 / zoom).tolist(), keep_size=self.zoomer.keep_size)
                d[key] = box_inverse_transform(d[key], src_spatial_size=src_spatial_size)
                # Remove the applied transform
                self.pop_transform(d, key)

        return d


class RandZoomBoxd(RandomizableTransform, MapTransform, InvertibleTransform):
    """
    Dictionary-based transform that randomly zooms input boxes and images with given probability within given zoom range.

    Args:
        image_keys: Keys to pick image data for transformation.
        box_keys: Keys to pick box data for transformation. The box mode is assumed to be ``StandardMode``.
        box_ref_image_keys: Keys that represent the reference images to which ``box_keys`` are attached.
        prob: Probability of zooming.
        min_zoom: Min zoom factor. Can be float or sequence same size as image.
            If a float, select a random factor from `[min_zoom, max_zoom]` then apply to all spatial dims
            to keep the original spatial shape ratio.
            If a sequence, min_zoom should contain one value for each spatial axis.
            If 2 values provided for 3D data, use the first value for both H & W dims to keep the same zoom ratio.
        max_zoom: Max zoom factor. Can be float or sequence same size as image.
            If a float, select a random factor from `[min_zoom, max_zoom]` then apply to all spatial dims
            to keep the original spatial shape ratio.
            If a sequence, max_zoom should contain one value for each spatial axis.
            If 2 values provided for 3D data, use the first value for both H & W dims to keep the same zoom ratio.
        mode: {``"nearest"``, ``"nearest-exact"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
            The interpolation mode. Defaults to ``"area"``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
            It also can be a sequence of string, each element corresponds to a key in ``keys``.
        padding_mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
            ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
            available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
            One of the listed string values or a user supplied function. Defaults to ``"constant"``.
            The mode to pad data after zooming.
            See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
            https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        align_corners: This only has an effect when mode is
            'linear', 'bilinear', 'bicubic' or 'trilinear'. Default: None.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
            It also can be a sequence of bool or None, each element corresponds to a key in ``keys``.
        keep_size: Should keep original size (pad if needed), default is True.
        allow_missing_keys: don't raise exception if key is missing.
        kwargs: other args for `np.pad` API, note that `np.pad` treats channel dimension as the first dimension.
            more details: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
    """

    backend = RandZoom.backend

    def __init__(
        self,
        image_keys: KeysCollection,
        box_keys: KeysCollection,
        box_ref_image_keys: KeysCollection,
        prob: float = 0.1,
        min_zoom: Sequence[float] | float = 0.9,
        max_zoom: Sequence[float] | float = 1.1,
        mode: SequenceStr = InterpolateMode.AREA,
        padding_mode: SequenceStr = NumpyPadMode.EDGE,
        align_corners: Sequence[bool | None] | bool | None = None,
        keep_size: bool = True,
        allow_missing_keys: bool = False,
        **kwargs: Any,
    ) -> None:
        self.image_keys = ensure_tuple(image_keys)
        self.box_keys = ensure_tuple(box_keys)
        MapTransform.__init__(self, self.image_keys + self.box_keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.box_ref_image_keys = ensure_tuple_rep(box_ref_image_keys, len(self.box_keys))

        self.rand_zoom = RandZoom(prob=1.0, min_zoom=min_zoom, max_zoom=max_zoom, keep_size=keep_size, **kwargs)
        self.mode = ensure_tuple_rep(mode, len(self.image_keys))
        self.padding_mode = ensure_tuple_rep(padding_mode, len(self.image_keys))
        self.align_corners = ensure_tuple_rep(align_corners, len(self.image_keys))
        self.keep_size = keep_size

    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None) -> RandZoomBoxd:
        super().set_random_state(seed, state)
        self.rand_zoom.set_random_state(seed, state)
        return self

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        first_key: Hashable = self.first_key(d)
        if first_key == ():
            return d

        self.randomize(None)

        # all the keys share the same random zoom factor
        self.rand_zoom.randomize(d[first_key])

        # zoom box
        for box_key, box_ref_image_key in zip(self.box_keys, self.box_ref_image_keys):
            if self._do_transform:
                src_spatial_size = d[box_ref_image_key].shape[1:]
                dst_spatial_size = [int(round(z * ss)) for z, ss in zip(self.rand_zoom._zoom, src_spatial_size)]
                self.rand_zoom._zoom = [ds / float(ss) for ss, ds in zip(src_spatial_size, dst_spatial_size)]

                d[box_key] = ZoomBox(zoom=self.rand_zoom._zoom, keep_size=self.keep_size)(
                    d[box_key], src_spatial_size=src_spatial_size
                )
                self.push_transform(
                    d,
                    box_key,
                    extra_info={"zoom": self.rand_zoom._zoom, "src_spatial_size": src_spatial_size, "type": "box_key"},
                )

        # zoom image, copied from monai.transforms.spatial.dictionary.RandZoomd
        for key, mode, padding_mode, align_corners in zip(
            self.image_keys, self.mode, self.padding_mode, self.align_corners
        ):
            if self._do_transform:
                d[key] = self.rand_zoom(
                    d[key], mode=mode, padding_mode=padding_mode, align_corners=align_corners, randomize=False
                )
            else:
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            if get_track_meta():
                xform = self.pop_transform(d[key], check=False) if self._do_transform else {}
                self.push_transform(d[key], extra_info=xform)

        return d

    def inverse(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        d = dict(data)

        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key, check=False)
            key_type = transform[TraceKeys.EXTRA_INFO].get("type", "image_key")
            # Check if random transform was actually performed (based on `prob`)
            if transform[TraceKeys.DO_TRANSFORM]:
                # zoom image, copied from monai.transforms.spatial.dictionary.Zoomd
                if key_type == "image_key":
                    xform = self.pop_transform(d[key])
                    d[key].applied_operations.append(xform[TraceKeys.EXTRA_INFO])  # type: ignore
                    d[key] = self.rand_zoom.inverse(d[key])

                # zoom boxes
                if key_type == "box_key":
                    # Create inverse transform
                    zoom = np.array(transform[TraceKeys.EXTRA_INFO]["zoom"])
                    src_spatial_size = transform[TraceKeys.EXTRA_INFO]["src_spatial_size"]
                    box_inverse_transform = ZoomBox(zoom=(1.0 / zoom).tolist(), keep_size=self.rand_zoom.keep_size)
                    d[key] = box_inverse_transform(d[key], src_spatial_size=src_spatial_size)
                    # Remove the applied transform
                    self.pop_transform(d, key)
        return d


class FlipBoxd(MapTransform, InvertibleTransform):
    """
    Dictionary-based transform that flip boxes and images.

    Args:
        image_keys: Keys to pick image data for transformation.
        box_keys: Keys to pick box data for transformation. The box mode is assumed to be ``StandardMode``.
        box_ref_image_keys: Keys that represent the reference images to which ``box_keys`` are attached.
        spatial_axis: Spatial axes along which to flip over. Default is None.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = Flip.backend

    def __init__(
        self,
        image_keys: KeysCollection,
        box_keys: KeysCollection,
        box_ref_image_keys: KeysCollection,
        spatial_axis: Sequence[int] | int | None = None,
        allow_missing_keys: bool = False,
    ) -> None:
        self.image_keys = ensure_tuple(image_keys)
        self.box_keys = ensure_tuple(box_keys)
        super().__init__(self.image_keys + self.box_keys, allow_missing_keys)
        self.box_ref_image_keys = ensure_tuple_rep(box_ref_image_keys, len(self.box_keys))

        self.flipper = Flip(spatial_axis=spatial_axis)
        self.box_flipper = FlipBox(spatial_axis=self.flipper.spatial_axis)

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        d = dict(data)

        for key in self.image_keys:
            d[key] = self.flipper(d[key])

        for box_key, box_ref_image_key in zip(self.box_keys, self.box_ref_image_keys):
            spatial_size = d[box_ref_image_key].shape[1:]
            d[box_key] = self.box_flipper(d[box_key], spatial_size)
            self.push_transform(d, box_key, extra_info={"spatial_size": spatial_size, "type": "box_key"})
        return d

    def inverse(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        d = dict(data)

        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key, check=False)
            key_type = transform.get(TraceKeys.EXTRA_INFO, {}).get("type", "image_key")

            # flip image, copied from monai.transforms.spatial.dictionary.Flipd
            if key_type == "image_key":
                d[key] = self.flipper.inverse(d[key])

            # flip boxes
            if key_type == "box_key":
                spatial_size = transform[TraceKeys.EXTRA_INFO]["spatial_size"]
                d[key] = self.box_flipper(d[key], spatial_size)
                # Remove the applied transform
                self.pop_transform(d, key)
        return d


class RandFlipBoxd(RandomizableTransform, MapTransform, InvertibleTransform):
    """
    Dictionary-based transform that randomly flip boxes and images with the given probabilities.

    Args:
        image_keys: Keys to pick image data for transformation.
        box_keys: Keys to pick box data for transformation. The box mode is assumed to be ``StandardMode``.
        box_ref_image_keys: Keys that represent the reference images to which ``box_keys`` are attached.
        prob: Probability of flipping.
        spatial_axis: Spatial axes along which to flip over. Default is None.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = RandFlip.backend

    def __init__(
        self,
        image_keys: KeysCollection,
        box_keys: KeysCollection,
        box_ref_image_keys: KeysCollection,
        prob: float = 0.1,
        spatial_axis: Sequence[int] | int | None = None,
        allow_missing_keys: bool = False,
    ) -> None:
        self.image_keys = ensure_tuple(image_keys)
        self.box_keys = ensure_tuple(box_keys)
        MapTransform.__init__(self, self.image_keys + self.box_keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.box_ref_image_keys = ensure_tuple_rep(box_ref_image_keys, len(self.box_keys))

        self.flipper = Flip(spatial_axis=spatial_axis)
        self.box_flipper = FlipBox(spatial_axis=spatial_axis)

    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None) -> RandFlipBoxd:
        super().set_random_state(seed, state)
        return self

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        self.randomize(None)

        for key in self.image_keys:
            if self._do_transform:
                d[key] = self.flipper(d[key])
            else:
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            if get_track_meta():
                xform_info = self.pop_transform(d[key], check=False) if self._do_transform else {}
                self.push_transform(d[key], extra_info=xform_info)

        for box_key, box_ref_image_key in zip(self.box_keys, self.box_ref_image_keys):
            spatial_size = d[box_ref_image_key].shape[1:]
            if self._do_transform:
                d[box_key] = self.box_flipper(d[box_key], spatial_size)
            self.push_transform(d, box_key, extra_info={"spatial_size": spatial_size, "type": "box_key"})
        return d

    def inverse(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        d = dict(data)

        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key, check=False)
            key_type = transform[TraceKeys.EXTRA_INFO].get("type", "image_key")
            # Check if random transform was actually performed (based on `prob`)
            if transform[TraceKeys.DO_TRANSFORM]:
                # flip image, copied from monai.transforms.spatial.dictionary.RandFlipd
                if key_type == "image_key":
                    with self.flipper.trace_transform(False):
                        d[key] = self.flipper(d[key])

                # flip boxes
                if key_type == "box_key":
                    spatial_size = transform[TraceKeys.EXTRA_INFO]["spatial_size"]
                    d[key] = self.box_flipper(d[key], spatial_size)

            # Remove the applied transform
            self.pop_transform(d, key, check=False)
        return d


class ClipBoxToImaged(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.apps.detection.transforms.array.ClipBoxToImage`.

    Clip the bounding boxes and the associated labels/scores to makes sure they are within the image.
    There might be multiple keys of labels/scores associated with one key of boxes.

    Args:
        box_keys: The single key to pick box data for transformation. The box mode is assumed to be ``StandardMode``.
        label_keys: Keys that represent the labels corresponding to the ``box_keys``. Multiple keys are allowed.
        box_ref_image_keys: The single key that represents the reference image
            to which ``box_keys`` and ``label_keys`` are attached.
        remove_empty: whether to remove the boxes that are actually empty
        allow_missing_keys: don't raise exception if key is missing.

    Example:
        .. code-block:: python

            ClipBoxToImaged(
                box_keys="boxes", box_ref_image_keys="image", label_keys=["labels", "scores"], remove_empty=True
            )
    """

    def __init__(
        self,
        box_keys: KeysCollection,
        label_keys: KeysCollection,
        box_ref_image_keys: KeysCollection,
        remove_empty: bool = True,
        allow_missing_keys: bool = False,
    ) -> None:
        box_keys_tuple = ensure_tuple(box_keys)
        if len(box_keys_tuple) != 1:
            raise ValueError(
                "Please provide a single key for box_keys.\
                All label_keys are attached to this box_keys."
            )
        box_ref_image_keys_tuple = ensure_tuple(box_ref_image_keys)
        if len(box_ref_image_keys_tuple) != 1:
            raise ValueError(
                "Please provide a single key for box_ref_image_keys.\
                All box_keys and label_keys are attached to this box_ref_image_keys."
            )
        self.label_keys = ensure_tuple(label_keys)
        super().__init__(box_keys_tuple, allow_missing_keys)

        self.box_keys = box_keys_tuple[0]
        self.box_ref_image_keys = box_ref_image_keys_tuple[0]
        self.clipper = ClipBoxToImage(remove_empty=remove_empty)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        spatial_size = d[self.box_ref_image_keys].shape[1:]
        labels = [d[label_key] for label_key in self.label_keys]  # could be multiple arrays
        d[self.box_keys], clipped_labels = self.clipper(d[self.box_keys], labels, spatial_size)

        for label_key, clipped_labels_i in zip(self.label_keys, clipped_labels):
            d[label_key] = clipped_labels_i
        return d


class BoxToMaskd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.apps.detection.transforms.array.BoxToMask`.
    Pairs with :py:class:`monai.apps.detection.transforms.dictionary.MaskToBoxd` .
    Please make sure the same ``min_fg_label`` is used when using the two transforms in pairs.
    The output ``d[box_mask_key]`` will have background intensity 0, since the following operations
    may pad 0 on the border.

    This is the general solution for transforms that need to be applied on images and boxes simultaneously.
    It is performed with the following steps.

        1) use ``BoxToMaskd`` to covert boxes and labels to box_masks;
        2) do transforms, e.g., rotation or cropping, on images and box_masks together;
        3) use ``MaskToBoxd`` to convert box_masks back to boxes and labels.

    Args:
        box_keys: Keys to pick box data for transformation. The box mode is assumed to be ``StandardMode``.
        box_mask_keys: Keys to store output box mask results for transformation. Same length with ``box_keys``.
        label_keys: Keys that represent the labels corresponding to the ``box_keys``. Same length with ``box_keys``.
        box_ref_image_keys: Keys that represent the reference images to which ``box_keys`` are attached.
        min_fg_label: min foreground box label.
        ellipse_mask: bool.

            - If True, it assumes the object shape is close to ellipse or ellipsoid.
            - If False, it assumes the object shape is close to rectangle or cube and well occupies the bounding box.
            - If the users are going to apply random rotation as data augmentation, we suggest setting ellipse_mask=True
              See also Kalra et al. "Towards Rotation Invariance in Object Detection", ICCV 2021.
        allow_missing_keys: don't raise exception if key is missing.

    Example:
        .. code-block:: python

            # This code snippet creates transforms (random rotation and cropping) on boxes, labels, and image together.
            import numpy as np
            from monai.transforms import Compose, RandRotated, RandSpatialCropd, DeleteItemsd
            transforms = Compose(
                [
                    BoxToMaskd(
                        box_keys="boxes", label_keys="labels",
                        box_mask_keys="box_mask", box_ref_image_keys="image",
                        min_fg_label=0, ellipse_mask=True
                    ),
                    RandRotated(keys=["image","box_mask"],mode=["nearest","nearest"],
                        prob=0.2,range_x=np.pi/6,range_y=np.pi/6,range_z=np.pi/6,
                        keep_size=True,padding_mode="zeros"
                    ),
                    RandSpatialCropd(keys=["image","box_mask"],roi_size=128, random_size=False),
                    MaskToBoxd(
                        box_mask_keys="box_mask", box_keys="boxes",
                        label_keys="labels", min_fg_label=0
                    )
                    DeleteItemsd(keys=["box_mask"]),
                ]
            )

    """

    def __init__(
        self,
        box_keys: KeysCollection,
        box_mask_keys: KeysCollection,
        label_keys: KeysCollection,
        box_ref_image_keys: KeysCollection,
        min_fg_label: int,
        ellipse_mask: bool = False,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(box_keys, allow_missing_keys)
        self.box_keys = ensure_tuple(box_keys)
        self.label_keys = ensure_tuple(label_keys)
        self.box_mask_keys = ensure_tuple(box_mask_keys)
        if not len(self.label_keys) == len(self.box_keys) == len(self.box_mask_keys):
            raise ValueError("Please make sure len(label_keys)==len(box_keys)==len(box_mask_keys)!")
        self.box_ref_image_keys = ensure_tuple_rep(box_ref_image_keys, len(self.box_keys))
        self.bg_label = min_fg_label - 1  # make sure background label is always smaller than fg labels.
        self.converter = BoxToMask(bg_label=self.bg_label, ellipse_mask=ellipse_mask)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)

        for box_key, label_key, box_mask_key, box_ref_image_key in zip(
            self.box_keys, self.label_keys, self.box_mask_keys, self.box_ref_image_keys
        ):
            spatial_size = d[box_ref_image_key].shape[1:]
            d[box_mask_key] = self.converter(d[box_key], d[label_key], spatial_size)
            # make box mask background intensity to be 0, since the following operations may pad 0 on the border.
            d[box_mask_key] -= self.bg_label
        return d


class MaskToBoxd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.apps.detection.transforms.array.MaskToBox`.
    Pairs with :py:class:`monai.apps.detection.transforms.dictionary.BoxToMaskd` .
    Please make sure the same ``min_fg_label`` is used when using the two transforms in pairs.

    This is the general solution for transforms that need to be applied on images and boxes simultaneously.
    It is performed with the following steps.

        1) use ``BoxToMaskd`` to covert boxes and labels to box_masks;
        2) do transforms, e.g., rotation or cropping, on images and box_masks together;
        3) use ``MaskToBoxd`` to convert box_masks back to boxes and labels.

    Args:
        box_keys: Keys to pick box data for transformation. The box mode is assumed to be ``StandardMode``.
        box_mask_keys: Keys to store output box mask results for transformation. Same length with ``box_keys``.
        label_keys: Keys that represent the labels corresponding to the ``box_keys``. Same length with ``box_keys``.
        min_fg_label: min foreground box label.
        box_dtype: output dtype for box_keys
        label_dtype: output dtype for label_keys
        allow_missing_keys: don't raise exception if key is missing.

    Example:
        .. code-block:: python

            # This code snippet creates transforms (random rotation and cropping) on boxes, labels, and images together.
            import numpy as np
            from monai.transforms import Compose, RandRotated, RandSpatialCropd, DeleteItemsd
            transforms = Compose(
                [
                    BoxToMaskd(
                        box_keys="boxes", label_keys="labels",
                        box_mask_keys="box_mask", box_ref_image_keys="image",
                        min_fg_label=0, ellipse_mask=True
                    ),
                    RandRotated(keys=["image","box_mask"],mode=["nearest","nearest"],
                        prob=0.2,range_x=np.pi/6,range_y=np.pi/6,range_z=np.pi/6,
                        keep_size=True,padding_mode="zeros"
                    ),
                    RandSpatialCropd(keys=["image","box_mask"],roi_size=128, random_size=False),
                    MaskToBoxd(
                        box_mask_keys="box_mask", box_keys="boxes",
                        label_keys="labels", min_fg_label=0
                    )
                    DeleteItemsd(keys=["box_mask"]),
                ]
            )
    """

    def __init__(
        self,
        box_keys: KeysCollection,
        box_mask_keys: KeysCollection,
        label_keys: KeysCollection,
        min_fg_label: int,
        box_dtype: DtypeLike | torch.dtype = torch.float32,
        label_dtype: DtypeLike | torch.dtype = torch.long,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(box_keys, allow_missing_keys)
        self.box_keys = ensure_tuple(box_keys)
        self.label_keys = ensure_tuple(label_keys)
        self.box_mask_keys = ensure_tuple(box_mask_keys)
        if not len(self.label_keys) == len(self.box_keys) == len(self.box_mask_keys):
            raise ValueError("Please make sure len(label_keys)==len(box_keys)==len(box_mask_keys)!")
        self.bg_label = min_fg_label - 1  # make sure background label is always smaller than fg labels.
        self.converter = MaskToBox(bg_label=self.bg_label, box_dtype=box_dtype, label_dtype=label_dtype)
        self.box_dtype = box_dtype

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)

        for box_key, label_key, box_mask_key in zip(self.box_keys, self.label_keys, self.box_mask_keys):
            d[box_mask_key] += self.bg_label  # pairs with the operation in BoxToMaskd
            d[box_key], d[label_key] = self.converter(d[box_mask_key])
        return d


class RandCropBoxByPosNegLabeld(Randomizable, MapTransform):
    """
    Crop random fixed sized regions that contains foreground boxes.
    Suppose all the expected fields specified by `image_keys` have same shape,
    and add `patch_index` to the corresponding meta data.
    And will return a list of dictionaries for all the cropped images.
    If a dimension of the expected spatial size is bigger than the input image size,
    will not crop that dimension. So the cropped result may be smaller than the expected size,
    and the cropped results of several images may not have exactly the same shape.

    Args:
        image_keys: Keys to pick image data for transformation. They need to have the same spatial size.
        box_keys: The single key to pick box data for transformation. The box mode is assumed to be ``StandardMode``.
        label_keys: Keys that represent the labels corresponding to the ``box_keys``. Multiple keys are allowed.
        spatial_size: the spatial size of the crop region e.g. [224, 224, 128].
            if a dimension of ROI size is bigger than image size, will not crop that dimension of the image.
            if its components have non-positive values, the corresponding size of `data[label_key]` will be used.
            for example: if the spatial size of input data is [40, 40, 40] and `spatial_size=[32, 64, -1]`,
            the spatial size of output data will be [32, 40, 40].
        pos: used with `neg` together to calculate the ratio ``pos / (pos + neg)`` for the probability
            to pick a foreground voxel as a center rather than a background voxel.
        neg: used with `pos` together to calculate the ratio ``pos / (pos + neg)`` for the probability
            to pick a foreground voxel as a center rather than a background voxel.
        num_samples: number of samples (crop regions) to take in each list.
        whole_box: Bool, default True, whether we prefer to contain at least one whole box in the cropped foreground patch.
            Even if True, it is still possible to get partial box if there are multiple boxes in the image.
        thresh_image_key: if thresh_image_key is not None, use ``label == 0 & thresh_image > image_threshold`` to select
            the negative sample(background) center. so the crop center will only exist on valid image area.
        image_threshold: if enabled thresh_image_key, use ``thresh_image > image_threshold`` to determine
            the valid image content area.
        fg_indices_key: if provided pre-computed foreground indices of `label`, will ignore above `image_key` and
            `image_threshold`, and randomly select crop centers based on them, need to provide `fg_indices_key`
            and `bg_indices_key` together, expect to be 1 dim array of spatial indices after flattening.
            a typical usage is to call `FgBgToIndicesd` transform first and cache the results.
        bg_indices_key: if provided pre-computed background indices of `label`, will ignore above `image_key` and
            `image_threshold`, and randomly select crop centers based on them, need to provide `fg_indices_key`
            and `bg_indices_key` together, expect to be 1 dim array of spatial indices after flattening.
            a typical usage is to call `FgBgToIndicesd` transform first and cache the results.
        meta_keys: explicitly indicate the key of the corresponding metadata dictionary.
            used to add `patch_index` to the meta dict.
            for example, for data with key `image`, the metadata by default is in `image_meta_dict`.
            the metadata is a dictionary object which contains: filename, original_shape, etc.
            it can be a sequence of string, map to the `keys`.
            if None, will try to construct meta_keys by `key_{meta_key_postfix}`.
        meta_key_postfix: if meta_keys is None, use `key_{postfix}` to fetch the metadata according
            to the key data, default is `meta_dict`, the metadata is a dictionary object.
            used to add `patch_index` to the meta dict.
        allow_smaller: if `False`, an exception will be raised if the image is smaller than
            the requested ROI in any dimension. If `True`, any smaller dimensions will be set to
            match the cropped size (i.e., no cropping in that dimension).
        allow_missing_keys: don't raise exception if key is missing.
    """

    def __init__(
        self,
        image_keys: KeysCollection,
        box_keys: str,
        label_keys: KeysCollection,
        spatial_size: Sequence[int] | int,
        pos: float = 1.0,
        neg: float = 1.0,
        num_samples: int = 1,
        whole_box: bool = True,
        thresh_image_key: str | None = None,
        image_threshold: float = 0.0,
        fg_indices_key: str | None = None,
        bg_indices_key: str | None = None,
        meta_keys: KeysCollection | None = None,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        allow_smaller: bool = False,
        allow_missing_keys: bool = False,
    ) -> None:
        self.image_keys = ensure_tuple(image_keys)
        if len(self.image_keys) < 1:
            raise ValueError("At least one image_keys should be provided.")

        MapTransform.__init__(self, self.image_keys, allow_missing_keys)

        box_keys_tuple = ensure_tuple(box_keys)
        if len(box_keys_tuple) != 1:
            raise ValueError(
                "Please provide a single key for box_keys.\
                All label_keys are attached to this box_keys."
            )
        self.box_keys = box_keys_tuple[0]
        self.label_keys = ensure_tuple(label_keys)

        self.spatial_size_: tuple[int, ...] | Sequence[int] | int = spatial_size

        if pos < 0 or neg < 0:
            raise ValueError(f"pos and neg must be nonnegative, got pos={pos} neg={neg}.")
        if pos + neg == 0:
            raise ValueError("Incompatible values: pos=0 and neg=0.")
        self.pos_ratio = pos / (pos + neg)
        if num_samples < 1:
            raise ValueError(f"num_samples needs to be positive int, got num_samples={num_samples}.")
        self.num_samples = num_samples
        self.whole_box = whole_box

        self.thresh_image_key = thresh_image_key
        self.image_threshold = image_threshold
        self.fg_indices_key = fg_indices_key
        self.bg_indices_key = bg_indices_key

        self.meta_keys = ensure_tuple_rep(None, len(self.image_keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.image_keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.image_keys))
        self.centers: tuple[tuple] | None = None
        self.allow_smaller = allow_smaller

    def generate_fg_center_boxes_np(self, boxes: NdarrayOrTensor, image_size: Sequence[int]) -> np.ndarray:
        # We don't require crop center to be within the boxes.
        # As along as the cropped patch contains a box, it is considered as a foreground patch.
        # Positions within extended_boxes are crop centers for foreground patches
        spatial_dims = len(image_size)
        boxes_np, *_ = convert_data_type(boxes, np.ndarray)

        extended_boxes = np.zeros_like(boxes_np, dtype=int)
        boxes_start = np.ceil(boxes_np[:, :spatial_dims]).astype(int)
        boxes_stop = np.floor(boxes_np[:, spatial_dims:]).astype(int)
        for axis in range(spatial_dims):
            if not self.whole_box:
                extended_boxes[:, axis] = boxes_start[:, axis] - self.spatial_size[axis] // 2 + 1
                extended_boxes[:, axis + spatial_dims] = boxes_stop[:, axis] + self.spatial_size[axis] // 2 - 1
            else:
                # the cropper will extend an additional pixel to the left side when the size is even
                radius_left = self.spatial_size[axis] // 2
                radius_right = self.spatial_size[axis] - radius_left - 1  # we subtract 1 for the center voxel
                # extended box start
                extended_boxes[:, axis] = boxes_stop[:, axis] - radius_right
                extended_boxes[:, axis] = np.minimum(extended_boxes[:, axis], boxes_start[:, axis])
                # extended box stop
                extended_boxes[:, axis + spatial_dims] = boxes_start[:, axis] + radius_left
                extended_boxes[:, axis + spatial_dims] = np.maximum(
                    extended_boxes[:, axis + spatial_dims], boxes_stop[:, axis]
                )
        extended_boxes, _ = clip_boxes_to_image(extended_boxes, image_size, remove_empty=True)  # type: ignore
        return extended_boxes

    def randomize(  # type: ignore
        self,
        boxes: NdarrayOrTensor,
        image_size: Sequence[int],
        fg_indices: NdarrayOrTensor | None = None,
        bg_indices: NdarrayOrTensor | None = None,
        thresh_image: NdarrayOrTensor | None = None,
    ) -> None:
        if fg_indices is None or bg_indices is None:
            # We don't require crop center to be within the boxes.
            # As along as the cropped patch contains a box, it is considered as a foreground patch.
            # Positions within extended_boxes are crop centers for foreground patches
            extended_boxes_np = self.generate_fg_center_boxes_np(boxes, image_size)
            mask_img = convert_box_to_mask(
                extended_boxes_np, np.ones(extended_boxes_np.shape[0]), image_size, bg_label=0, ellipse_mask=False
            )
            mask_img = np.amax(mask_img, axis=0, keepdims=True)[0:1, ...]
            fg_indices_, bg_indices_ = map_binary_to_indices(mask_img, thresh_image, self.image_threshold)
        else:
            fg_indices_ = fg_indices
            bg_indices_ = bg_indices

        self.centers = generate_pos_neg_label_crop_centers(
            self.spatial_size,
            self.num_samples,
            self.pos_ratio,
            image_size,
            fg_indices_,
            bg_indices_,
            self.R,
            self.allow_smaller,
        )

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> list[dict[Hashable, torch.Tensor]]:
        d = dict(data)
        image_size = d[self.image_keys[0]].shape[1:]
        self.spatial_size = fall_back_tuple(self.spatial_size_, image_size)

        # randomly sample crop centers
        boxes = d[self.box_keys]
        labels = [d[label_key] for label_key in self.label_keys]  # could be multiple arrays
        fg_indices = d.pop(self.fg_indices_key, None) if self.fg_indices_key is not None else None
        bg_indices = d.pop(self.bg_indices_key, None) if self.bg_indices_key is not None else None
        thresh_image = d[self.thresh_image_key] if self.thresh_image_key else None
        self.randomize(boxes, image_size, fg_indices, bg_indices, thresh_image)

        if self.centers is None:
            raise ValueError("no available ROI centers to crop.")

        # initialize returned list with shallow copy to preserve key ordering
        results: list[dict[Hashable, torch.Tensor]] = [dict(d) for _ in range(self.num_samples)]

        # crop images and boxes for each center.
        for i, center in enumerate(self.centers):
            results[i] = deepcopy(d)
            # compute crop start and end, always crop, no padding
            cropper = SpatialCrop(roi_center=tuple(center), roi_size=self.spatial_size)
            crop_start = [max(s.start, 0) for s in cropper.slices]
            crop_end = [min(s.stop, image_size_a) for s, image_size_a in zip(cropper.slices, image_size)]
            crop_slices = [slice(int(s), int(e)) for s, e in zip(crop_start, crop_end)]

            # crop images
            cropper = SpatialCrop(roi_slices=crop_slices)
            for image_key in self.image_keys:
                results[i][image_key] = cropper(d[image_key])

            # crop boxes and labels
            boxcropper = SpatialCropBox(roi_slices=crop_slices)
            results[i][self.box_keys], cropped_labels = boxcropper(boxes, labels)
            for label_key, cropped_labels_i in zip(self.label_keys, cropped_labels):
                results[i][label_key] = cropped_labels_i

        return results


class RotateBox90d(MapTransform, InvertibleTransform):
    """
    Input boxes and images are rotated by 90 degrees
    in the plane specified by ``spatial_axes`` for ``k`` times

    Args:
        image_keys: Keys to pick image data for transformation.
        box_keys: Keys to pick box data for transformation. The box mode is assumed to be ``StandardMode``.
        box_ref_image_keys: Keys that represent the reference images to which ``box_keys`` are attached.
        k: number of times to rotate by 90 degrees.
        spatial_axes: 2 int numbers, defines the plane to rotate with 2 spatial axes.
            Default (0, 1), this is the first two axis in spatial dimensions.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = RotateBox90.backend

    def __init__(
        self,
        image_keys: KeysCollection,
        box_keys: KeysCollection,
        box_ref_image_keys: KeysCollection,
        k: int = 1,
        spatial_axes: tuple[int, int] = (0, 1),
        allow_missing_keys: bool = False,
    ) -> None:
        self.image_keys = ensure_tuple(image_keys)
        self.box_keys = ensure_tuple(box_keys)
        super().__init__(self.image_keys + self.box_keys, allow_missing_keys)
        self.box_ref_image_keys = ensure_tuple_rep(box_ref_image_keys, len(self.box_keys))
        self.img_rotator = Rotate90(k, spatial_axes)
        self.box_rotator = RotateBox90(k, spatial_axes)

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Mapping[Hashable, torch.Tensor]:
        d = dict(data)
        for key, box_ref_image_key in zip(self.box_keys, self.box_ref_image_keys):
            spatial_size = list(d[box_ref_image_key].shape[1:])
            d[key] = self.box_rotator(d[key], spatial_size)
            if self.img_rotator.k % 2 == 1:
                # if k = 1 or 3, spatial_size will be transposed
                spatial_size[self.img_rotator.spatial_axes[0]], spatial_size[self.img_rotator.spatial_axes[1]] = (
                    spatial_size[self.img_rotator.spatial_axes[1]],
                    spatial_size[self.img_rotator.spatial_axes[0]],
                )
            self.push_transform(d, key, extra_info={"spatial_size": spatial_size, "type": "box_key"})

        for key in self.image_keys:
            d[key] = self.img_rotator(d[key])
        return d

    def inverse(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        d = dict(data)

        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key, check=False)
            key_type = transform[TraceKeys.EXTRA_INFO].get("type", "image_key")
            num_times_to_rotate = 4 - self.img_rotator.k

            if key_type == "image_key":
                d[key] = self.img_rotator.inverse(d[key])
            if key_type == "box_key":
                spatial_size = transform[TraceKeys.EXTRA_INFO]["spatial_size"]
                inverse_transform = RotateBox90(num_times_to_rotate, self.box_rotator.spatial_axes)
                d[key] = inverse_transform(d[key], spatial_size)
                self.pop_transform(d, key)
        return d


class RandRotateBox90d(RandomizableTransform, MapTransform, InvertibleTransform):
    """
    With probability `prob`, input boxes and images are rotated by 90 degrees
    in the plane specified by `spatial_axes`.

    Args:
        image_keys: Keys to pick image data for transformation.
        box_keys: Keys to pick box data for transformation. The box mode is assumed to be ``StandardMode``.
        box_ref_image_keys: Keys that represent the reference images to which ``box_keys`` are attached.
        prob: probability of rotating.
            (Default 0.1, with 10% probability it returns a rotated array.)
        max_k: number of rotations will be sampled from `np.random.randint(max_k) + 1`.
            (Default 3)
        spatial_axes: 2 int numbers, defines the plane to rotate with 2 spatial axes.
            Default: (0, 1), this is the first two axis in spatial dimensions.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = RotateBox90.backend

    def __init__(
        self,
        image_keys: KeysCollection,
        box_keys: KeysCollection,
        box_ref_image_keys: KeysCollection,
        prob: float = 0.1,
        max_k: int = 3,
        spatial_axes: tuple[int, int] = (0, 1),
        allow_missing_keys: bool = False,
    ) -> None:
        self.image_keys = ensure_tuple(image_keys)
        self.box_keys = ensure_tuple(box_keys)

        MapTransform.__init__(self, self.image_keys + self.box_keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)

        self.max_k = max_k
        self.spatial_axes = spatial_axes
        self._rand_k = 0
        self.box_ref_image_keys = ensure_tuple_rep(box_ref_image_keys, len(self.box_keys))

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Mapping[Hashable, torch.Tensor]:
        self.randomize()
        d = dict(data)

        if self._rand_k % 4 == 0:
            return d

        # FIXME: here we didn't use array version `RandRotate90` transform as others, because we need
        # to be compatible with the random status of some previous integration tests
        box_rotator = RotateBox90(self._rand_k, self.spatial_axes)
        img_rotator = Rotate90(self._rand_k, self.spatial_axes)

        for key, box_ref_image_key in zip(self.box_keys, self.box_ref_image_keys):
            if self._do_transform:
                spatial_size = list(d[box_ref_image_key].shape[1:])
                d[key] = box_rotator(d[key], spatial_size)
                if self._rand_k % 2 == 1:
                    # if k = 1 or 3, spatial_size will be transposed
                    spatial_size[self.spatial_axes[0]], spatial_size[self.spatial_axes[1]] = (
                        spatial_size[self.spatial_axes[1]],
                        spatial_size[self.spatial_axes[0]],
                    )
                self.push_transform(
                    d, key, extra_info={"rand_k": self._rand_k, "spatial_size": spatial_size, "type": "box_key"}
                )

        for key in self.image_keys:
            if self._do_transform:
                d[key] = (
                    img_rotator(d[key])
                    if self._do_transform
                    else convert_to_tensor(d[key], track_meta=get_track_meta())
                )
                if get_track_meta():
                    xform = self.pop_transform(d[key], check=False) if self._do_transform else {}
                    self.push_transform(d[key], extra_info=xform)
        return d

    def randomize(self, data: Any | None = None) -> None:
        self._rand_k = self.R.randint(self.max_k) + 1
        super().randomize(None)

    def inverse(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        if self._rand_k % 4 == 0:
            return d

        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key, check=False)
            key_type = transform[TraceKeys.EXTRA_INFO].get("type", "image_key")
            # Check if random transform was actually performed (based on `prob`)
            if transform[TraceKeys.DO_TRANSFORM]:
                # flip image, copied from monai.transforms.spatial.dictionary.RandFlipd
                if key_type == "image_key":
                    xform = self.pop_transform(d, key, check=False)
                    d[key] = Rotate90().inverse_transform(d[key], xform[TraceKeys.EXTRA_INFO])
                if key_type == "box_key":
                    num_times_rotated = transform[TraceKeys.EXTRA_INFO]["rand_k"]
                    num_times_to_rotate = 4 - num_times_rotated
                    spatial_size = transform[TraceKeys.EXTRA_INFO]["spatial_size"]
                    inverse_transform = RotateBox90(num_times_to_rotate, self.spatial_axes)
                    d[key] = inverse_transform(d[key], spatial_size)
                    self.pop_transform(d, key)
        return d


ConvertBoxModeD = ConvertBoxModeDict = ConvertBoxModed
ConvertBoxToStandardModeD = ConvertBoxToStandardModeDict = ConvertBoxToStandardModed
ZoomBoxD = ZoomBoxDict = ZoomBoxd
RandZoomBoxD = RandZoomBoxDict = RandZoomBoxd
AffineBoxToImageCoordinateD = AffineBoxToImageCoordinateDict = AffineBoxToImageCoordinated
FlipBoxD = FlipBoxDict = FlipBoxd
RandFlipBoxD = RandFlipBoxDict = RandFlipBoxd
ClipBoxToImageD = ClipBoxToImageDict = ClipBoxToImaged
BoxToMaskD = BoxToMaskDict = BoxToMaskd
MaskToBoxD = MaskToBoxDict = MaskToBoxd
RandCropBoxByPosNegLabelD = RandCropBoxByPosNegLabelDict = RandCropBoxByPosNegLabeld
RotateBox90D = RotateBox90Dict = RotateBox90d
RandRotateBox90D = RandRotateBox90Dict = RandRotateBox90d
StandardizeEmptyBoxD = StandardizeEmptyBoxDict = StandardizeEmptyBoxd
