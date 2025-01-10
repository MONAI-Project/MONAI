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
A collection of dictionary-based wrappers around the "vanilla" transforms for spatial operations
defined in :py:class:`monai.transforms.spatial.array`.

Class names are ended with 'd' to denote dictionary-based transforms.
"""

from __future__ import annotations

import warnings
from collections.abc import Hashable, Mapping, Sequence
from typing import Any, cast

import numpy as np
import torch

import monai.transforms as transforms
from monai.config import DtypeLike, KeysCollection, SequenceStr
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.box_utils import BoxMode, StandardMode
from monai.data.meta_obj import get_track_meta
from monai.data.meta_tensor import MetaTensor
from monai.data.utils import is_supported_format
from monai.networks.layers.simplelayers import GaussianFilter
from monai.transforms.croppad.array import CenterSpatialCrop
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.spatial.array import (
    Affine,
    ConvertBoxToPoints,
    ConvertPointsToBoxes,
    Flip,
    GridDistortion,
    GridPatch,
    GridSplit,
    Orientation,
    Rand2DElastic,
    Rand3DElastic,
    RandAffine,
    RandAxisFlip,
    RandGridDistortion,
    RandGridPatch,
    RandRotate,
    RandSimulateLowResolution,
    RandZoom,
    ResampleToMatch,
    Resize,
    Rotate,
    Rotate90,
    Spacing,
    SpatialResample,
    Zoom,
)
from monai.transforms.traits import MultiSampleTrait
from monai.transforms.transform import LazyTransform, MapTransform, RandomizableTransform
from monai.transforms.utils import create_grid
from monai.utils import (
    GridSampleMode,
    GridSamplePadMode,
    InterpolateMode,
    NumpyPadMode,
    convert_to_tensor,
    ensure_tuple,
    ensure_tuple_rep,
    fall_back_tuple,
)
from monai.utils.enums import TraceKeys
from monai.utils.module import optional_import

nib, _ = optional_import("nibabel")

__all__ = [
    "SpatialResampled",
    "ResampleToMatchd",
    "Spacingd",
    "Orientationd",
    "Rotate90d",
    "RandRotate90d",
    "Resized",
    "Affined",
    "RandAffined",
    "Rand2DElasticd",
    "Rand3DElasticd",
    "Flipd",
    "RandFlipd",
    "GridDistortiond",
    "RandGridDistortiond",
    "RandAxisFlipd",
    "Rotated",
    "RandRotated",
    "Zoomd",
    "RandZoomd",
    "SpatialResampleD",
    "SpatialResampleDict",
    "SpacingD",
    "SpacingDict",
    "OrientationD",
    "OrientationDict",
    "Rotate90D",
    "Rotate90Dict",
    "RandRotate90D",
    "RandRotate90Dict",
    "ResizeD",
    "ResizeDict",
    "AffineD",
    "AffineDict",
    "RandAffineD",
    "RandAffineDict",
    "Rand2DElasticD",
    "Rand2DElasticDict",
    "Rand3DElasticD",
    "Rand3DElasticDict",
    "FlipD",
    "FlipDict",
    "RandFlipD",
    "RandFlipDict",
    "GridDistortionD",
    "GridDistortionDict",
    "RandGridDistortionD",
    "RandGridDistortionDict",
    "RandAxisFlipD",
    "RandAxisFlipDict",
    "RotateD",
    "RotateDict",
    "RandRotateD",
    "RandRotateDict",
    "ZoomD",
    "ZoomDict",
    "RandZoomD",
    "RandZoomDict",
    "GridSplitd",
    "GridSplitD",
    "GridSplitDict",
    "GridPatchd",
    "GridPatchD",
    "GridPatchDict",
    "RandGridPatchd",
    "RandGridPatchD",
    "RandGridPatchDict",
    "RandSimulateLowResolutiond",
    "RandSimulateLowResolutionD",
    "RandSimulateLowResolutionDict",
]


class SpatialResampled(MapTransform, InvertibleTransform, LazyTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.SpatialResample`.

    This transform assumes the ``data`` dictionary has a key for the input
    data's metadata and contains ``src_affine`` and ``dst_affine`` required by
    `SpatialResample`. The key is formed by ``key_{meta_key_postfix}``.  The
    transform will swap ``src_affine`` and ``dst_affine`` affine (with potential data type
    changes) in the dictionary so that ``src_affine`` always refers to the current
    status of affine.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    See also:
        :py:class:`monai.transforms.SpatialResample`
    """

    backend = SpatialResample.backend

    def __init__(
        self,
        keys: KeysCollection,
        mode: SequenceStr = GridSampleMode.BILINEAR,
        padding_mode: SequenceStr = GridSamplePadMode.BORDER,
        align_corners: Sequence[bool] | bool = False,
        dtype: Sequence[DtypeLike] | DtypeLike = np.float64,
        dst_keys: KeysCollection | None = "dst_affine",
        allow_missing_keys: bool = False,
        lazy: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
            mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
                and the value represents the order of the spline interpolation.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
                It also can be a sequence, each element corresponds to a key in ``keys``.
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"border"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `mode` is an integer, using numpy/cupy backends, this argument accepts
                {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
                It also can be a sequence, each element corresponds to a key in ``keys``.
            align_corners: Geometrically, we consider the pixels of the input as squares rather than points.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
                It also can be a sequence of bool, each element corresponds to a key in ``keys``.
            dtype: data type for resampling computation. Defaults to ``float64`` for best precision.
                If None, use the data type of input data. To be compatible with other modules,
                the output data type is always ``float32``.
                It also can be a sequence of dtypes, each element corresponds to a key in ``keys``.
            dst_keys: the key of the corresponding ``dst_affine`` in the metadata dictionary.
            allow_missing_keys: don't raise exception if key is missing.
            lazy: a flag to indicate whether this transform should execute lazily or not.
                Defaults to False.
        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        LazyTransform.__init__(self, lazy=lazy)
        self.sp_transform = SpatialResample(lazy=lazy)
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.padding_mode = ensure_tuple_rep(padding_mode, len(self.keys))
        self.align_corners = ensure_tuple_rep(align_corners, len(self.keys))
        self.dtype = ensure_tuple_rep(dtype, len(self.keys))
        self.dst_keys = ensure_tuple_rep(dst_keys, len(self.keys))

    @LazyTransform.lazy.setter  # type: ignore
    def lazy(self, val: bool) -> None:
        self._lazy = val
        self.sp_transform.lazy = val

    def __call__(self, data: Mapping[Hashable, torch.Tensor], lazy: bool | None = None) -> dict[Hashable, torch.Tensor]:
        """
        Args:
            data: a dictionary containing the tensor-like data to be processed. The ``keys`` specified
                in this dictionary must be tensor like arrays that are channel first and have at most
                three spatial dimensions
            lazy: a flag to indicate whether this transform should execute lazily or not
                during this call. Setting this to False or True overrides the ``lazy`` flag set
                during initialization for this call. Defaults to None.

        Returns:
            a dictionary containing the transformed data, as well as any other data present in the dictionary
        """
        lazy_ = self.lazy if lazy is None else lazy
        d: dict = dict(data)
        for key, mode, padding_mode, align_corners, dtype, dst_key in self.key_iterator(
            d, self.mode, self.padding_mode, self.align_corners, self.dtype, self.dst_keys
        ):
            d[key] = self.sp_transform(
                img=d[key],
                dst_affine=d[dst_key],
                spatial_size=None,  # None means shape auto inferred
                mode=mode,
                padding_mode=padding_mode,
                align_corners=align_corners,
                dtype=dtype,
                lazy=lazy_,
            )
        return d

    def inverse(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.sp_transform.inverse(d[key])
        return d


class ResampleToMatchd(MapTransform, InvertibleTransform, LazyTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ResampleToMatch`.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    """

    backend = ResampleToMatch.backend

    def __init__(
        self,
        keys: KeysCollection,
        key_dst: str,
        mode: SequenceStr = GridSampleMode.BILINEAR,
        padding_mode: SequenceStr = GridSamplePadMode.BORDER,
        align_corners: Sequence[bool] | bool = False,
        dtype: Sequence[DtypeLike] | DtypeLike = np.float64,
        allow_missing_keys: bool = False,
        lazy: bool = False,
    ):
        """
        Args:
            keys: keys of the corresponding items to be transformed.
            key_dst: key of image to resample to match.
            mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
                and the value represents the order of the spline interpolation.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
                It also can be a sequence, each element corresponds to a key in ``keys``.
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"border"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `mode` is an integer, using numpy/cupy backends, this argument accepts
                {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
                It also can be a sequence, each element corresponds to a key in ``keys``.
            align_corners: Geometrically, we consider the pixels of the input as squares rather than points.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
                It also can be a sequence of bool, each element corresponds to a key in ``keys``.
            dtype: data type for resampling computation. Defaults to ``float64`` for best precision.
                If None, use the data type of input data. To be compatible with other modules,
                the output data type is always ``float32``.
                It also can be a sequence of dtypes, each element corresponds to a key in ``keys``.
            allow_missing_keys: don't raise exception if key is missing.
            lazy: a flag to indicate whether this transform should execute lazily or not.
                Defaults to False
        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        LazyTransform.__init__(self, lazy=lazy)
        self.key_dst = key_dst
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.padding_mode = ensure_tuple_rep(padding_mode, len(self.keys))
        self.align_corners = ensure_tuple_rep(align_corners, len(self.keys))
        self.dtype = ensure_tuple_rep(dtype, len(self.keys))
        self.resampler = ResampleToMatch(lazy=lazy)

    @LazyTransform.lazy.setter  # type: ignore
    def lazy(self, val: bool) -> None:
        self._lazy = val
        self.resampler.lazy = val

    def __call__(self, data: Mapping[Hashable, torch.Tensor], lazy: bool | None = None) -> dict[Hashable, torch.Tensor]:
        """
        Args:
            data: a dictionary containing the tensor-like data to be processed. The ``keys`` specified
                in this dictionary must be tensor like arrays that are channel first and have at most
                three spatial dimensions
            lazy: a flag to indicate whether this transform should execute lazily or not
                during this call. Setting this to False or True overrides the ``lazy`` flag set
                during initialization for this call. Defaults to None.

        Returns:
            a dictionary containing the transformed data, as well as any other data present in the dictionary
        """
        lazy_ = self.lazy if lazy is None else lazy
        d = dict(data)
        for key, mode, padding_mode, align_corners, dtype in self.key_iterator(
            d, self.mode, self.padding_mode, self.align_corners, self.dtype
        ):
            d[key] = self.resampler(
                img=d[key],
                img_dst=d[self.key_dst],
                mode=mode,
                padding_mode=padding_mode,
                align_corners=align_corners,
                dtype=dtype,
                lazy=lazy_,
            )
        return d

    def inverse(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.resampler.inverse(d[key])
        return d


class Spacingd(MapTransform, InvertibleTransform, LazyTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.Spacing`.

    This transform assumes the ``data`` dictionary has a key for the input
    data's metadata and contains `affine` field.  The key is formed by ``key_{meta_key_postfix}``.

    After resampling the input array, this transform will write the new affine
    to the `affine` field of metadata which is formed by ``key_{meta_key_postfix}``.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    see also:
        :py:class:`monai.transforms.Spacing`
    """

    backend = Spacing.backend

    def __init__(
        self,
        keys: KeysCollection,
        pixdim: Sequence[float] | float,
        diagonal: bool = False,
        mode: SequenceStr = GridSampleMode.BILINEAR,
        padding_mode: SequenceStr = GridSamplePadMode.BORDER,
        align_corners: Sequence[bool] | bool = False,
        dtype: Sequence[DtypeLike] | DtypeLike = np.float64,
        scale_extent: bool = False,
        recompute_affine: bool = False,
        min_pixdim: Sequence[float] | float | None = None,
        max_pixdim: Sequence[float] | float | None = None,
        ensure_same_shape: bool = True,
        allow_missing_keys: bool = False,
        lazy: bool = False,
    ) -> None:
        """
        Args:
            pixdim: output voxel spacing. if providing a single number, will use it for the first dimension.
                items of the pixdim sequence map to the spatial dimensions of input image, if length
                of pixdim sequence is longer than image spatial dimensions, will ignore the longer part,
                if shorter, will pad with `1.0`.
                if the components of the `pixdim` are non-positive values, the transform will use the
                corresponding components of the original pixdim, which is computed from the `affine`
                matrix of input image.
            diagonal: whether to resample the input to have a diagonal affine matrix.
                If True, the input data is resampled to the following affine::

                    np.diag((pixdim_0, pixdim_1, pixdim_2, 1))

                This effectively resets the volume to the world coordinate system (RAS+ in nibabel).
                The original orientation, rotation, shearing are not preserved.

                If False, the axes orientation, orthogonal rotation and
                translations components from the original affine will be
                preserved in the target affine. This option will not flip/swap
                axes against the original ones.
            mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
                and the value represents the order of the spline interpolation.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
                It also can be a sequence, each element corresponds to a key in ``keys``.
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"border"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `mode` is an integer, using numpy/cupy backends, this argument accepts
                {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
                It also can be a sequence, each element corresponds to a key in ``keys``.
            align_corners: Geometrically, we consider the pixels of the input as squares rather than points.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                It also can be a sequence of bool, each element corresponds to a key in ``keys``.
            dtype: data type for resampling computation. Defaults to ``float64`` for best precision.
                If None, use the data type of input data. To be compatible with other modules,
                the output data type is always ``float32``.
                It also can be a sequence of dtypes, each element corresponds to a key in ``keys``.
            scale_extent: whether the scale is computed based on the spacing or the full extent of voxels,
                default False. The option is ignored if output spatial size is specified when calling this transform.
                See also: :py:func:`monai.data.utils.compute_shape_offset`. When this is True, `align_corners`
                should be `True` because `compute_shape_offset` already provides the corner alignment shift/scaling.
            recompute_affine: whether to recompute affine based on the output shape. The affine computed
                analytically does not reflect the potential quantization errors in terms of the output shape.
                Set this flag to True to recompute the output affine based on the actual pixdim. Default to ``False``.
            min_pixdim: minimal input spacing to be resampled. If provided, input image with a larger spacing than this
                value will be kept in its original spacing (not be resampled to `pixdim`). Set it to `None` to use the
                value of `pixdim`. Default to `None`.
            max_pixdim: maximal input spacing to be resampled. If provided, input image with a smaller spacing than this
                value will be kept in its original spacing (not be resampled to `pixdim`). Set it to `None` to use the
                value of `pixdim`. Default to `None`.
            ensure_same_shape: when the inputs have the same spatial shape, and almost the same pixdim,
                whether to ensure exactly the same output spatial shape.  Default to True.
            allow_missing_keys: don't raise exception if key is missing.
            lazy: a flag to indicate whether this transform should execute lazily or not.
                Defaults to False
        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        LazyTransform.__init__(self, lazy=lazy)
        self.spacing_transform = Spacing(
            pixdim,
            diagonal=diagonal,
            recompute_affine=recompute_affine,
            min_pixdim=min_pixdim,
            max_pixdim=max_pixdim,
            lazy=lazy,
        )
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.padding_mode = ensure_tuple_rep(padding_mode, len(self.keys))
        self.align_corners = ensure_tuple_rep(align_corners, len(self.keys))
        self.dtype = ensure_tuple_rep(dtype, len(self.keys))
        self.scale_extent = ensure_tuple_rep(scale_extent, len(self.keys))
        self.ensure_same_shape = ensure_same_shape

    @LazyTransform.lazy.setter  # type: ignore
    def lazy(self, val: bool) -> None:
        self._lazy = val
        self.spacing_transform.lazy = val

    def __call__(self, data: Mapping[Hashable, torch.Tensor], lazy: bool | None = None) -> dict[Hashable, torch.Tensor]:
        """
        Args:
            data: a dictionary containing the tensor-like data to be processed. The ``keys`` specified
                in this dictionary must be tensor like arrays that are channel first and have at most
                three spatial dimensions
            lazy: a flag to indicate whether this transform should execute lazily or not
                during this call. Setting this to False or True overrides the ``lazy`` flag set
                during initialization for this call. Defaults to None.

        Returns:
            a dictionary containing the transformed data, as well as any other data present in the dictionary
        """
        d: dict = dict(data)

        _init_shape, _pixdim, should_match = None, None, False
        output_shape_k = None  # tracking output shape
        lazy_ = self.lazy if lazy is None else lazy

        for key, mode, padding_mode, align_corners, dtype, scale_extent in self.key_iterator(
            d, self.mode, self.padding_mode, self.align_corners, self.dtype, self.scale_extent
        ):
            if self.ensure_same_shape and isinstance(d[key], MetaTensor):
                if _init_shape is None and _pixdim is None:
                    _init_shape, _pixdim = d[key].peek_pending_shape(), d[key].pixdim
                else:
                    should_match = np.allclose(_init_shape, d[key].peek_pending_shape()) and np.allclose(
                        _pixdim, d[key].pixdim, atol=1e-3
                    )
            d[key] = self.spacing_transform(
                data_array=d[key],
                mode=mode,
                padding_mode=padding_mode,
                align_corners=align_corners,
                dtype=dtype,
                scale_extent=scale_extent,
                output_spatial_shape=output_shape_k if should_match else None,
                lazy=lazy_,
            )
            if isinstance(d[key], MetaTensor) and f"{key}_meta_dict" in d:
                if "filename_or_obj" in d[key].meta and is_supported_format(
                    d[key].meta["filename_or_obj"], ["nii", "nii.gz"]
                ):
                    d = transforms.sync_meta_info(key, d)
            if output_shape_k is None:
                output_shape_k = d[key].peek_pending_shape() if isinstance(d[key], MetaTensor) else d[key].shape[1:]
        return d

    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.spacing_transform.inverse(cast(torch.Tensor, d[key]))
        return d


class Orientationd(MapTransform, InvertibleTransform, LazyTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.Orientation`.

    This transform assumes the channel-first input format.
    In the case of using this transform for normalizing the orientations of images,
    it should be used before any anisotropic spatial transforms.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.
    """

    backend = Orientation.backend

    def __init__(
        self,
        keys: KeysCollection,
        axcodes: str | None = None,
        as_closest_canonical: bool = False,
        labels: Sequence[tuple[str, str]] | None = (("L", "R"), ("P", "A"), ("I", "S")),
        allow_missing_keys: bool = False,
        lazy: bool = False,
    ) -> None:
        """
        Args:
            axcodes: N elements sequence for spatial ND input's orientation.
                e.g. axcodes='RAS' represents 3D orientation:
                (Left, Right), (Posterior, Anterior), (Inferior, Superior).
                default orientation labels options are: 'L' and 'R' for the first dimension,
                'P' and 'A' for the second, 'I' and 'S' for the third.
            as_closest_canonical: if True, load the image as closest to canonical axis format.
            labels: optional, None or sequence of (2,) sequences
                (2,) sequences are labels for (beginning, end) of output axis.
                Defaults to ``(('L', 'R'), ('P', 'A'), ('I', 'S'))``.
            allow_missing_keys: don't raise exception if key is missing.
            lazy: a flag to indicate whether this transform should execute lazily or not.
                Defaults to False

        See Also:
            `nibabel.orientations.ornt2axcodes`.

        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        LazyTransform.__init__(self, lazy=lazy)
        self.ornt_transform = Orientation(
            axcodes=axcodes, as_closest_canonical=as_closest_canonical, labels=labels, lazy=lazy
        )

    @LazyTransform.lazy.setter  # type: ignore
    def lazy(self, val: bool) -> None:
        self._lazy = val
        self.ornt_transform.lazy = val

    def __call__(self, data: Mapping[Hashable, torch.Tensor], lazy: bool | None = None) -> dict[Hashable, torch.Tensor]:
        """
        Args:
            data: a dictionary containing the tensor-like data to be processed. The ``keys`` specified
                in this dictionary must be tensor like arrays that are channel first and have at most
                three spatial dimensions
            lazy: a flag to indicate whether this transform should execute lazily or not
                during this call. Setting this to False or True overrides the ``lazy`` flag set
                during initialization for this call. Defaults to None.

        Returns:
            a dictionary containing the transformed data, as well as any other data present in the dictionary
        """
        d: dict = dict(data)
        lazy_ = self.lazy if lazy is None else lazy
        for key in self.key_iterator(d):
            d[key] = self.ornt_transform(d[key], lazy=lazy_)
        return d

    def inverse(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.ornt_transform.inverse(d[key])
        return d


class Rotate90d(MapTransform, InvertibleTransform, LazyTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.Rotate90`.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.
    """

    backend = Rotate90.backend

    def __init__(
        self,
        keys: KeysCollection,
        k: int = 1,
        spatial_axes: tuple[int, int] = (0, 1),
        allow_missing_keys: bool = False,
        lazy: bool = False,
    ) -> None:
        """
        Args:
            k: number of times to rotate by 90 degrees.
            spatial_axes: 2 int numbers, defines the plane to rotate with 2 spatial axes.
                Default: (0, 1), this is the first two axis in spatial dimensions.
            allow_missing_keys: don't raise exception if key is missing.
            lazy: a flag to indicate whether this transform should execute lazily or not.
                Defaults to False
        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        LazyTransform.__init__(self, lazy=lazy)
        self.rotator = Rotate90(k, spatial_axes, lazy=lazy)

    @LazyTransform.lazy.setter  # type: ignore
    def lazy(self, val: bool) -> None:
        self._lazy = val
        self.rotator.lazy = val

    def __call__(self, data: Mapping[Hashable, torch.Tensor], lazy: bool | None = None) -> dict[Hashable, torch.Tensor]:
        """
        Args:
            data: a dictionary containing the tensor-like data to be processed. The ``keys`` specified
                in this dictionary must be tensor like arrays that are channel first and have at most
                three spatial dimensions
            lazy: a flag to indicate whether this transform should execute lazily or not
                during this call. Setting this to False or True overrides the ``lazy`` flag set
                during initialization for this call. Defaults to None.

        Returns:
            a dictionary containing the transformed data, as well as any other data present in the dictionary
        """
        d = dict(data)
        lazy_ = self.lazy if lazy is None else lazy
        for key in self.key_iterator(d):
            d[key] = self.rotator(d[key], lazy=lazy_)
        return d

    def inverse(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.rotator.inverse(d[key])
        return d


class RandRotate90d(RandomizableTransform, MapTransform, InvertibleTransform, LazyTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandRotate90`.
    With probability `prob`, input arrays are rotated by 90 degrees
    in the plane specified by `spatial_axes`.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.
    """

    backend = Rotate90.backend

    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.1,
        max_k: int = 3,
        spatial_axes: tuple[int, int] = (0, 1),
        allow_missing_keys: bool = False,
        lazy: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            prob: probability of rotating.
                (Default 0.1, with 10% probability it returns a rotated array.)
            max_k: number of rotations will be sampled from `np.random.randint(max_k) + 1`.
                (Default 3)
            spatial_axes: 2 int numbers, defines the plane to rotate with 2 spatial axes.
                Default: (0, 1), this is the first two axis in spatial dimensions.
            allow_missing_keys: don't raise exception if key is missing.
            lazy: a flag to indicate whether this transform should execute lazily or not.
                Defaults to False
        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        LazyTransform.__init__(self, lazy=lazy)

        self.max_k = max_k
        self.spatial_axes = spatial_axes

        self._rand_k = 0

    def randomize(self, data: Any | None = None) -> None:
        self._rand_k = self.R.randint(self.max_k) + 1
        super().randomize(None)

    def __call__(
        self, data: Mapping[Hashable, torch.Tensor], lazy: bool | None = None
    ) -> Mapping[Hashable, torch.Tensor]:
        """
        Args:
            data: a dictionary containing the tensor-like data to be processed. The ``keys`` specified
                in this dictionary must be tensor like arrays that are channel first and have at most
                three spatial dimensions
            lazy: a flag to indicate whether this transform should execute lazily or not
                during this call. Setting this to False or True overrides the ``lazy`` flag set
                during initialization for this call. Defaults to None.

        Returns:
            a dictionary containing the transformed data, as well as any other data present in the dictionary
        """
        self.randomize()
        d = dict(data)

        # FIXME: here we didn't use array version `RandRotate90` transform as others, because we need
        # to be compatible with the random status of some previous integration tests
        lazy_ = self.lazy if lazy is None else lazy
        rotator = Rotate90(self._rand_k, self.spatial_axes, lazy=lazy_)
        for key in self.key_iterator(d):
            d[key] = rotator(d[key]) if self._do_transform else convert_to_tensor(d[key], track_meta=get_track_meta())
            self.push_transform(d[key], replace=True, lazy=lazy_)
        return d

    def inverse(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            if not isinstance(d[key], MetaTensor):
                continue
            xform = self.pop_transform(d[key])
            if xform[TraceKeys.DO_TRANSFORM]:
                d[key] = Rotate90().inverse_transform(d[key], xform[TraceKeys.EXTRA_INFO])
        return d


class Resized(MapTransform, InvertibleTransform, LazyTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.Resize`.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        spatial_size: expected shape of spatial dimensions after resize operation.
            if some components of the `spatial_size` are non-positive values, the transform will use the
            corresponding components of img size. For example, `spatial_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        size_mode: should be "all" or "longest", if "all", will use `spatial_size` for all the spatial dims,
            if "longest", rescale the image so that only the longest side is equal to specified `spatial_size`,
            which must be an int number in this case, keeping the aspect ratio of the initial image, refer to:
            https://albumentations.ai/docs/api_reference/augmentations/geometric/resize/
            #albumentations.augmentations.geometric.resize.LongestMaxSize.
        mode: {``"nearest"``, ``"nearest-exact"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
            The interpolation mode. Defaults to ``"area"``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
            It also can be a sequence of string, each element corresponds to a key in ``keys``.
        align_corners: This only has an effect when mode is
            'linear', 'bilinear', 'bicubic' or 'trilinear'. Default: None.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
            It also can be a sequence of bool or None, each element corresponds to a key in ``keys``.
        anti_aliasing: bool
            Whether to apply a Gaussian filter to smooth the image prior
            to downsampling. It is crucial to filter when downsampling
            the image to avoid aliasing artifacts. See also ``skimage.transform.resize``
        anti_aliasing_sigma: {float, tuple of floats}, optional
            Standard deviation for Gaussian filtering used when anti-aliasing.
            By default, this value is chosen as (s - 1) / 2 where s is the
            downsampling factor, where s > 1. For the up-size case, s < 1, no
            anti-aliasing is performed prior to rescaling.
        dtype: data type for resampling computation. Defaults to ``float32``.
            If None, use the data type of input data.
        allow_missing_keys: don't raise exception if key is missing.
        lazy: a flag to indicate whether this transform should execute lazily or not.
            Defaults to False
    """

    backend = Resize.backend

    def __init__(
        self,
        keys: KeysCollection,
        spatial_size: Sequence[int] | int,
        size_mode: str = "all",
        mode: SequenceStr = InterpolateMode.AREA,
        align_corners: Sequence[bool | None] | bool | None = None,
        anti_aliasing: Sequence[bool] | bool = False,
        anti_aliasing_sigma: Sequence[Sequence[float] | float | None] | Sequence[float] | float | None = None,
        dtype: Sequence[DtypeLike | torch.dtype] | DtypeLike | torch.dtype = np.float32,
        allow_missing_keys: bool = False,
        lazy: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        LazyTransform.__init__(self, lazy=lazy)
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.align_corners = ensure_tuple_rep(align_corners, len(self.keys))
        self.dtype = ensure_tuple_rep(dtype, len(self.keys))
        self.anti_aliasing = ensure_tuple_rep(anti_aliasing, len(self.keys))
        self.anti_aliasing_sigma = ensure_tuple_rep(anti_aliasing_sigma, len(self.keys))
        self.resizer = Resize(spatial_size=spatial_size, size_mode=size_mode, lazy=lazy)

    @LazyTransform.lazy.setter  # type: ignore
    def lazy(self, val: bool) -> None:
        self._lazy = val
        self.resizer.lazy = val

    def __call__(self, data: Mapping[Hashable, torch.Tensor], lazy: bool | None = None) -> dict[Hashable, torch.Tensor]:
        """
        Args:
            data: a dictionary containing the tensor-like data to be processed. The ``keys`` specified
                in this dictionary must be tensor like arrays that are channel first and have at most
                three spatial dimensions
            lazy: a flag to indicate whether this transform should execute lazily or not
                during this call. Setting this to False or True overrides the ``lazy`` flag set
                during initialization for this call. Defaults to None.

        Returns:
            a dictionary containing the transformed data, as well as any other data present in the dictionary
        """
        d = dict(data)
        lazy_ = self.lazy if lazy is None else lazy
        for key, mode, align_corners, anti_aliasing, anti_aliasing_sigma, dtype in self.key_iterator(
            d, self.mode, self.align_corners, self.anti_aliasing, self.anti_aliasing_sigma, self.dtype
        ):
            d[key] = self.resizer(
                d[key],
                mode=mode,
                align_corners=align_corners,
                anti_aliasing=anti_aliasing,
                anti_aliasing_sigma=anti_aliasing_sigma,
                dtype=dtype,
                lazy=lazy_,
            )
        return d

    def inverse(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.resizer.inverse(d[key])
        return d


class Affined(MapTransform, InvertibleTransform, LazyTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.Affine`.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.
    """

    backend = Affine.backend

    def __init__(
        self,
        keys: KeysCollection,
        rotate_params: Sequence[float] | float | None = None,
        shear_params: Sequence[float] | float | None = None,
        translate_params: Sequence[float] | float | None = None,
        scale_params: Sequence[float] | float | None = None,
        affine: NdarrayOrTensor | None = None,
        spatial_size: Sequence[int] | int | None = None,
        mode: SequenceStr = GridSampleMode.BILINEAR,
        padding_mode: SequenceStr = GridSamplePadMode.REFLECTION,
        device: torch.device | None = None,
        dtype: DtypeLike | torch.dtype = np.float32,
        align_corners: bool = False,
        allow_missing_keys: bool = False,
        lazy: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
            rotate_params: a rotation angle in radians, a scalar for 2D image, a tuple of 3 floats for 3D.
                Defaults to no rotation.
            shear_params: shearing factors for affine matrix, take a 3D affine as example::

                [
                    [1.0, params[0], params[1], 0.0],
                    [params[2], 1.0, params[3], 0.0],
                    [params[4], params[5], 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]

                a tuple of 2 floats for 2D, a tuple of 6 floats for 3D. Defaults to no shearing.
            translate_params: a tuple of 2 floats for 2D, a tuple of 3 floats for 3D. Translation is in
                pixel/voxel relative to the center of the input image. Defaults to no translation.
            scale_params: scale factor for every spatial dims. a tuple of 2 floats for 2D,
                a tuple of 3 floats for 3D. Defaults to `1.0`.
            affine: if applied, ignore the params (`rotate_params`, etc.) and use the
                supplied matrix. Should be square with each side = num of image spatial
                dimensions + 1.
            spatial_size: output image spatial size.
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
                if some components of the `spatial_size` are non-positive values, the transform will use the
                corresponding components of img size. For example, `spatial_size=(32, -1)` will be adapted
                to `(32, 64)` if the second spatial dimension size of img is `64`.
            mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
                and the value represents the order of the spline interpolation.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
                It also can be a sequence, each element corresponds to a key in ``keys``.
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"reflection"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `mode` is an integer, using numpy/cupy backends, this argument accepts
                {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
                It also can be a sequence, each element corresponds to a key in ``keys``.
            device: device on which the tensor will be allocated.
            dtype: data type for resampling computation. Defaults to ``float32``.
                If ``None``, use the data type of input data. To be compatible with other modules,
                the output data type is always `float32`.
            align_corners: Defaults to False.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            allow_missing_keys: don't raise exception if key is missing.
            lazy: a flag to indicate whether this transform should execute lazily or not.
                Defaults to False

        See also:
            - :py:class:`monai.transforms.compose.MapTransform`
            - :py:class:`RandAffineGrid` for the random affine parameters configurations.

        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        LazyTransform.__init__(self, lazy=lazy)
        self.affine = Affine(
            rotate_params=rotate_params,
            shear_params=shear_params,
            translate_params=translate_params,
            scale_params=scale_params,
            affine=affine,
            spatial_size=spatial_size,
            device=device,
            dtype=dtype,  # type: ignore
            align_corners=align_corners,
            lazy=lazy,
        )
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.padding_mode = ensure_tuple_rep(padding_mode, len(self.keys))

    @LazyTransform.lazy.setter  # type: ignore
    def lazy(self, val: bool) -> None:
        self._lazy = val
        self.affine.lazy = val

    def __call__(self, data: Mapping[Hashable, torch.Tensor], lazy: bool | None = None) -> dict[Hashable, torch.Tensor]:
        """
        Args:
            data: a dictionary containing the tensor-like data to be processed. The ``keys`` specified
                in this dictionary must be tensor like arrays that are channel first and have at most
                three spatial dimensions
            lazy: a flag to indicate whether this transform should execute lazily or not
                during this call. Setting this to False or True overrides the ``lazy`` flag set
                during initialization for this call. Defaults to None.

        Returns:
            a dictionary containing the transformed data, as well as any other data present in the dictionary
        """
        lazy_ = self.lazy if lazy is None else lazy
        d = dict(data)
        for key, mode, padding_mode in self.key_iterator(d, self.mode, self.padding_mode):
            d[key], _ = self.affine(d[key], mode=mode, padding_mode=padding_mode, lazy=lazy_)
        return d

    def inverse(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.affine.inverse(d[key])
        return d


class RandAffined(RandomizableTransform, MapTransform, InvertibleTransform, LazyTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.RandAffine`.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.
    """

    backend = RandAffine.backend

    def __init__(
        self,
        keys: KeysCollection,
        spatial_size: Sequence[int] | int | None = None,
        prob: float = 0.1,
        rotate_range: Sequence[tuple[float, float] | float] | float | None = None,
        shear_range: Sequence[tuple[float, float] | float] | float | None = None,
        translate_range: Sequence[tuple[float, float] | float] | float | None = None,
        scale_range: Sequence[tuple[float, float] | float] | float | None = None,
        mode: SequenceStr = GridSampleMode.BILINEAR,
        padding_mode: SequenceStr = GridSamplePadMode.REFLECTION,
        cache_grid: bool = False,
        device: torch.device | None = None,
        allow_missing_keys: bool = False,
        lazy: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
            spatial_size: output image spatial size.
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
                if some components of the `spatial_size` are non-positive values, the transform will use the
                corresponding components of img size. For example, `spatial_size=(32, -1)` will be adapted
                to `(32, 64)` if the second spatial dimension size of img is `64`.
            prob: probability of returning a randomized affine grid.
                defaults to 0.1, with 10% chance returns a randomized grid.
            rotate_range: angle range in radians. If element `i` is a pair of (min, max) values, then
                `uniform[-rotate_range[i][0], rotate_range[i][1])` will be used to generate the rotation parameter
                for the `i`th spatial dimension. If not, `uniform[-rotate_range[i], rotate_range[i])` will be used.
                This can be altered on a per-dimension basis. E.g., `((0,3), 1, ...)`: for dim0, rotation will be
                in range `[0, 3]`, and for dim1 `[-1, 1]` will be used. Setting a single value will use `[-x, x]`
                for dim0 and nothing for the remaining dimensions.
            shear_range: shear range with format matching `rotate_range`, it defines the range to randomly select
                shearing factors(a tuple of 2 floats for 2D, a tuple of 6 floats for 3D) for affine matrix,
                take a 3D affine as example::

                    [
                        [1.0, params[0], params[1], 0.0],
                        [params[2], 1.0, params[3], 0.0],
                        [params[4], params[5], 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]

            translate_range: translate range with format matching `rotate_range`, it defines the range to randomly
                select pixel/voxel to translate for every spatial dims.
            scale_range: scaling range with format matching `rotate_range`. it defines the range to randomly select
                the scale factor to translate for every spatial dims. A value of 1.0 is added to the result.
                This allows 0 to correspond to no change (i.e., a scaling of 1.0).
            mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
                and the value represents the order of the spline interpolation.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
                It also can be a sequence, each element corresponds to a key in ``keys``.
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"reflection"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `mode` is an integer, using numpy/cupy backends, this argument accepts
                {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
                It also can be a sequence, each element corresponds to a key in ``keys``.
            cache_grid: whether to cache the identity sampling grid.
                If the spatial size is not dynamically defined by input image, enabling this option could
                accelerate the transform.
            device: device on which the tensor will be allocated.
            allow_missing_keys: don't raise exception if key is missing.
            lazy: a flag to indicate whether this transform should execute lazily or not.
                Defaults to False

        See also:
            - :py:class:`monai.transforms.compose.MapTransform`
            - :py:class:`RandAffineGrid` for the random affine parameters configurations.

        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        LazyTransform.__init__(self, lazy=lazy)
        self.rand_affine = RandAffine(
            prob=1.0,  # because probability handled in this class
            rotate_range=rotate_range,
            shear_range=shear_range,
            translate_range=translate_range,
            scale_range=scale_range,
            spatial_size=spatial_size,
            cache_grid=cache_grid,
            device=device,
            lazy=lazy,
        )
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.padding_mode = ensure_tuple_rep(padding_mode, len(self.keys))

    @LazyTransform.lazy.setter  # type: ignore
    def lazy(self, val: bool) -> None:
        self._lazy = val
        self.rand_affine.lazy = val

    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None) -> RandAffined:
        self.rand_affine.set_random_state(seed, state)
        super().set_random_state(seed, state)
        return self

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor], lazy: bool | None = None
    ) -> dict[Hashable, NdarrayOrTensor]:
        """
        Args:
            data: a dictionary containing the tensor-like data to be processed. The ``keys`` specified
                in this dictionary must be tensor like arrays that are channel first and have at most
                three spatial dimensions
            lazy: a flag to indicate whether this transform should execute lazily or not
                during this call. Setting this to False or True overrides the ``lazy`` flag set
                during initialization for this call. Defaults to None.

        Returns:
            a dictionary containing the transformed data, as well as any other data present in the dictionary
        """
        d = dict(data)
        first_key: Hashable = self.first_key(d)
        if first_key == ():
            out: dict[Hashable, NdarrayOrTensor] = convert_to_tensor(d, track_meta=get_track_meta())
            return out

        self.randomize(None)
        # all the keys share the same random Affine factor
        self.rand_affine.randomize()

        item = d[first_key]
        spatial_size = item.peek_pending_shape() if isinstance(item, MetaTensor) else item.shape[1:]
        lazy_ = self.lazy if lazy is None else lazy

        sp_size = fall_back_tuple(self.rand_affine.spatial_size, spatial_size)
        # change image size or do random transform
        do_resampling = self._do_transform or (sp_size != ensure_tuple(spatial_size))
        # converting affine to tensor because the resampler currently only support torch backend
        grid = None
        if do_resampling:  # need to prepare grid
            grid = self.rand_affine.get_identity_grid(sp_size, lazy=lazy_)
            if self._do_transform:  # add some random factors
                grid = self.rand_affine.rand_affine_grid(sp_size, grid=grid, lazy=lazy_)
        grid = 0 if grid is None else grid  # always provide a grid to self.rand_affine

        for key, mode, padding_mode in self.key_iterator(d, self.mode, self.padding_mode):
            # do the transform
            if do_resampling:
                d[key] = self.rand_affine(d[key], None, mode, padding_mode, True, grid, lazy=lazy_)  # type: ignore
            else:
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta(), dtype=torch.float32)
            self._do_transform = do_resampling  # TODO: unify self._do_transform and do_resampling
            self.push_transform(d[key], replace=True, lazy=lazy_)
        return d

    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            tr = self.pop_transform(d[key])
            if TraceKeys.EXTRA_INFO not in tr[TraceKeys.EXTRA_INFO]:
                continue
            do_resampling = tr[TraceKeys.EXTRA_INFO][TraceKeys.EXTRA_INFO]["do_resampling"]
            if do_resampling:
                d[key].applied_operations.append(tr[TraceKeys.EXTRA_INFO])  # type: ignore
                d[key] = self.rand_affine.inverse(d[key])  # type: ignore

        return d


class Rand2DElasticd(RandomizableTransform, MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.Rand2DElastic`.
    """

    backend = Rand2DElastic.backend

    def __init__(
        self,
        keys: KeysCollection,
        spacing: tuple[float, float] | float,
        magnitude_range: tuple[float, float],
        spatial_size: tuple[int, int] | int | None = None,
        prob: float = 0.1,
        rotate_range: Sequence[tuple[float, float] | float] | float | None = None,
        shear_range: Sequence[tuple[float, float] | float] | float | None = None,
        translate_range: Sequence[tuple[float, float] | float] | float | None = None,
        scale_range: Sequence[tuple[float, float] | float] | float | None = None,
        mode: SequenceStr = GridSampleMode.BILINEAR,
        padding_mode: SequenceStr = GridSamplePadMode.REFLECTION,
        device: torch.device | None = None,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
            spacing: distance in between the control points.
            magnitude_range: 2 int numbers, the random offsets will be generated from
                ``uniform[magnitude[0], magnitude[1])``.
            spatial_size: specifying output image spatial size [h, w].
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
                if some components of the `spatial_size` are non-positive values, the transform will use the
                corresponding components of img size. For example, `spatial_size=(32, -1)` will be adapted
                to `(32, 64)` if the second spatial dimension size of img is `64`.
            prob: probability of returning a randomized affine grid.
                defaults to 0.1, with 10% chance returns a randomized grid,
                otherwise returns a ``spatial_size`` centered area extracted from the input image.
            rotate_range: angle range in radians. If element `i` is a pair of (min, max) values, then
                `uniform[-rotate_range[i][0], rotate_range[i][1])` will be used to generate the rotation parameter
                for the `i`th spatial dimension. If not, `uniform[-rotate_range[i], rotate_range[i])` will be used.
                This can be altered on a per-dimension basis. E.g., `((0,3), 1, ...)`: for dim0, rotation will be
                in range `[0, 3]`, and for dim1 `[-1, 1]` will be used. Setting a single value will use `[-x, x]`
                for dim0 and nothing for the remaining dimensions.
            shear_range: shear range with format matching `rotate_range`, it defines the range to randomly select
                shearing factors(a tuple of 2 floats for 2D) for affine matrix, take a 2D affine as example::

                    [
                        [1.0, params[0], 0.0],
                        [params[1], 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ]

            translate_range: translate range with format matching `rotate_range`, it defines the range to randomly
                select pixel to translate for every spatial dims.
            scale_range: scaling range with format matching `rotate_range`. it defines the range to randomly select
                the scale factor to translate for every spatial dims. A value of 1.0 is added to the result.
                This allows 0 to correspond to no change (i.e., a scaling of 1.0).
            mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
                and the value represents the order of the spline interpolation.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
                It also can be a sequence, each element corresponds to a key in ``keys``.
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"reflection"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `mode` is an integer, using numpy/cupy backends, this argument accepts
                {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
                It also can be a sequence, each element corresponds to a key in ``keys``.
            device: device on which the tensor will be allocated.
            allow_missing_keys: don't raise exception if key is missing.

        See also:
            - :py:class:`RandAffineGrid` for the random affine parameters configurations.
            - :py:class:`Affine` for the affine transformation parameters configurations.

        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.rand_2d_elastic = Rand2DElastic(
            spacing=spacing,
            magnitude_range=magnitude_range,
            prob=1.0,  # because probability controlled by this class
            rotate_range=rotate_range,
            shear_range=shear_range,
            translate_range=translate_range,
            scale_range=scale_range,
            spatial_size=spatial_size,
            device=device,
        )
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.padding_mode = ensure_tuple_rep(padding_mode, len(self.keys))

    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None) -> Rand2DElasticd:
        self.rand_2d_elastic.set_random_state(seed, state)
        super().set_random_state(seed, state)
        return self

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        """
        Args:
            data: a dictionary containing the tensor-like data to be processed. The ``keys`` specified
                in this dictionary must be tensor like arrays that are channel first and have at most
                three spatial dimensions

        Returns:
            a dictionary containing the transformed data, as well as any other data present in the dictionary
        """
        d = dict(data)
        first_key: Hashable = self.first_key(d)

        if first_key == ():
            out: dict[Hashable, NdarrayOrTensor] = convert_to_tensor(d, track_meta=get_track_meta())
            return out

        self.randomize(None)
        device = self.rand_2d_elastic.device
        if device is None and isinstance(d[first_key], torch.Tensor):
            device = d[first_key].device  # type: ignore
            self.rand_2d_elastic.set_device(device)
        if isinstance(d[first_key], MetaTensor) and d[first_key].pending_operations:  # type: ignore
            warnings.warn(f"data['{first_key}'] has pending operations, transform may return incorrect results.")
        sp_size = fall_back_tuple(self.rand_2d_elastic.spatial_size, d[first_key].shape[1:])

        # all the keys share the same random elastic factor
        self.rand_2d_elastic.randomize(sp_size)

        if self._do_transform:
            grid = self.rand_2d_elastic.deform_grid(spatial_size=sp_size)
            grid = self.rand_2d_elastic.rand_affine_grid(grid=grid)
            grid = torch.nn.functional.interpolate(
                recompute_scale_factor=True,
                input=grid.unsqueeze(0),
                scale_factor=ensure_tuple_rep(self.rand_2d_elastic.deform_grid.spacing, 2),
                mode=InterpolateMode.BICUBIC.value,
                align_corners=False,
            )
            grid = CenterSpatialCrop(roi_size=sp_size)(grid[0])
        else:
            grid = cast(torch.Tensor, create_grid(spatial_size=sp_size, device=device, backend="torch"))

        for key, mode, padding_mode in self.key_iterator(d, self.mode, self.padding_mode):
            d[key] = self.rand_2d_elastic.resampler(d[key], grid, mode=mode, padding_mode=padding_mode)  # type: ignore
        return d


class Rand3DElasticd(RandomizableTransform, MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.Rand3DElastic`.
    """

    backend = Rand3DElastic.backend

    def __init__(
        self,
        keys: KeysCollection,
        sigma_range: tuple[float, float],
        magnitude_range: tuple[float, float],
        spatial_size: tuple[int, int, int] | int | None = None,
        prob: float = 0.1,
        rotate_range: Sequence[tuple[float, float] | float] | float | None = None,
        shear_range: Sequence[tuple[float, float] | float] | float | None = None,
        translate_range: Sequence[tuple[float, float] | float] | float | None = None,
        scale_range: Sequence[tuple[float, float] | float] | float | None = None,
        mode: SequenceStr = GridSampleMode.BILINEAR,
        padding_mode: SequenceStr = GridSamplePadMode.REFLECTION,
        device: torch.device | None = None,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
            sigma_range: a Gaussian kernel with standard deviation sampled from
                ``uniform[sigma_range[0], sigma_range[1])`` will be used to smooth the random offset grid.
            magnitude_range: the random offsets on the grid will be generated from
                ``uniform[magnitude[0], magnitude[1])``.
            spatial_size: specifying output image spatial size [h, w, d].
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
                if some components of the `spatial_size` are non-positive values, the transform will use the
                corresponding components of img size. For example, `spatial_size=(32, 32, -1)` will be adapted
                to `(32, 32, 64)` if the third spatial dimension size of img is `64`.
            prob: probability of returning a randomized affine grid.
                defaults to 0.1, with 10% chance returns a randomized grid,
                otherwise returns a ``spatial_size`` centered area extracted from the input image.
            rotate_range: angle range in radians. If element `i` is a pair of (min, max) values, then
                `uniform[-rotate_range[i][0], rotate_range[i][1])` will be used to generate the rotation parameter
                for the `i`th spatial dimension. If not, `uniform[-rotate_range[i], rotate_range[i])` will be used.
                This can be altered on a per-dimension basis. E.g., `((0,3), 1, ...)`: for dim0, rotation will be
                in range `[0, 3]`, and for dim1 `[-1, 1]` will be used. Setting a single value will use `[-x, x]`
                for dim0 and nothing for the remaining dimensions.
            shear_range: shear range with format matching `rotate_range`, it defines the range to randomly select
                shearing factors(a tuple of 6 floats for 3D) for affine matrix, take a 3D affine as example::

                    [
                        [1.0, params[0], params[1], 0.0],
                        [params[2], 1.0, params[3], 0.0],
                        [params[4], params[5], 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]

            translate_range: translate range with format matching `rotate_range`, it defines the range to randomly
                select voxel to translate for every spatial dims.
            scale_range: scaling range with format matching `rotate_range`. it defines the range to randomly select
                the scale factor to translate for every spatial dims. A value of 1.0 is added to the result.
                This allows 0 to correspond to no change (i.e., a scaling of 1.0).
            mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
                and the value represents the order of the spline interpolation.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
                It also can be a sequence, each element corresponds to a key in ``keys``.
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"reflection"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `mode` is an integer, using numpy/cupy backends, this argument accepts
                {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
                It also can be a sequence, each element corresponds to a key in ``keys``.
            device: device on which the tensor will be allocated.
            allow_missing_keys: don't raise exception if key is missing.

        See also:
            - :py:class:`RandAffineGrid` for the random affine parameters configurations.
            - :py:class:`Affine` for the affine transformation parameters configurations.

        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.rand_3d_elastic = Rand3DElastic(
            sigma_range=sigma_range,
            magnitude_range=magnitude_range,
            prob=1.0,  # because probability controlled by this class
            rotate_range=rotate_range,
            shear_range=shear_range,
            translate_range=translate_range,
            scale_range=scale_range,
            spatial_size=spatial_size,
            device=device,
        )
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.padding_mode = ensure_tuple_rep(padding_mode, len(self.keys))

    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None) -> Rand3DElasticd:
        self.rand_3d_elastic.set_random_state(seed, state)
        super().set_random_state(seed, state)
        return self

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        """
        Args:
            data: a dictionary containing the tensor-like data to be processed. The ``keys`` specified
                in this dictionary must be tensor like arrays that are channel first and have at most
                three spatial dimensions

        Returns:
            a dictionary containing the transformed data, as well as any other data present in the dictionary
        """
        d = dict(data)
        first_key: Hashable = self.first_key(d)

        if first_key == ():
            out: dict[Hashable, torch.Tensor] = convert_to_tensor(d, track_meta=get_track_meta())
            return out

        self.randomize(None)
        if isinstance(d[first_key], MetaTensor) and d[first_key].pending_operations:  # type: ignore
            warnings.warn(f"data['{first_key}'] has pending operations, transform may return incorrect results.")
        sp_size = fall_back_tuple(self.rand_3d_elastic.spatial_size, d[first_key].shape[1:])

        # all the keys share the same random elastic factor
        self.rand_3d_elastic.randomize(sp_size)

        device = self.rand_3d_elastic.device
        if device is None and isinstance(d[first_key], torch.Tensor):
            device = d[first_key].device
            self.rand_3d_elastic.set_device(device)
        grid = create_grid(spatial_size=sp_size, device=device, backend="torch")
        if self._do_transform:
            gaussian = GaussianFilter(spatial_dims=3, sigma=self.rand_3d_elastic.sigma, truncated=3.0).to(device)
            offset = torch.as_tensor(self.rand_3d_elastic.rand_offset, device=device).unsqueeze(0)
            grid[:3] += gaussian(offset)[0] * self.rand_3d_elastic.magnitude
            grid = self.rand_3d_elastic.rand_affine_grid(grid=grid)

        for key, mode, padding_mode in self.key_iterator(d, self.mode, self.padding_mode):
            d[key] = self.rand_3d_elastic.resampler(d[key], grid, mode=mode, padding_mode=padding_mode)  # type: ignore
        return d


class Flipd(MapTransform, InvertibleTransform, LazyTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.Flip`.

    See `numpy.flip` for additional details.
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    Args:
        keys: Keys to pick data for transformation.
        spatial_axis: Spatial axes along which to flip over. Default is None.
        allow_missing_keys: don't raise exception if key is missing.
        lazy: a flag to indicate whether this transform should execute lazily or not.
            Defaults to False
    """

    backend = Flip.backend

    def __init__(
        self,
        keys: KeysCollection,
        spatial_axis: Sequence[int] | int | None = None,
        allow_missing_keys: bool = False,
        lazy: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        LazyTransform.__init__(self, lazy=lazy)
        self.flipper = Flip(spatial_axis=spatial_axis)

    @LazyTransform.lazy.setter  # type: ignore
    def lazy(self, val: bool):
        self.flipper.lazy = val
        self._lazy = val

    def __call__(self, data: Mapping[Hashable, torch.Tensor], lazy: bool | None = None) -> dict[Hashable, torch.Tensor]:
        """
        Args:
            data: a dictionary containing the tensor-like data to be processed. The ``keys`` specified
                in this dictionary must be tensor like arrays that are channel first and have at most
                three spatial dimensions
            lazy: a flag to indicate whether this transform should execute lazily or not
                during this call. Setting this to False or True overrides the ``lazy`` flag set
                during initialization for this call. Defaults to None.

        Returns:
            a dictionary containing the transformed data, as well as any other data present in the dictionary
        """
        d = dict(data)
        lazy_ = self.lazy if lazy is None else lazy
        for key in self.key_iterator(d):
            d[key] = self.flipper(d[key], lazy=lazy_)
        return d

    def inverse(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.flipper.inverse(d[key])
        return d


class RandFlipd(RandomizableTransform, MapTransform, InvertibleTransform, LazyTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandFlip`.

    See `numpy.flip` for additional details.
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    Args:
        keys: Keys to pick data for transformation.
        prob: Probability of flipping.
        spatial_axis: Spatial axes along which to flip over. Default is None.
        allow_missing_keys: don't raise exception if key is missing.
        lazy: a flag to indicate whether this transform should execute lazily or not.
            Defaults to False
    """

    backend = Flip.backend

    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.1,
        spatial_axis: Sequence[int] | int | None = None,
        allow_missing_keys: bool = False,
        lazy: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        LazyTransform.__init__(self, lazy=lazy)
        self.flipper = Flip(spatial_axis=spatial_axis, lazy=lazy)

    @LazyTransform.lazy.setter  # type: ignore
    def lazy(self, val: bool):
        self.flipper.lazy = val
        self._lazy = val

    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None) -> RandFlipd:
        super().set_random_state(seed, state)
        return self

    def __call__(self, data: Mapping[Hashable, torch.Tensor], lazy: bool | None = None) -> dict[Hashable, torch.Tensor]:
        """
        Args:
            data: a dictionary containing the tensor-like data to be processed. The ``keys`` specified
                in this dictionary must be tensor like arrays that are channel first and have at most
                three spatial dimensions
            lazy: a flag to indicate whether this transform should execute lazily or not
                during this call. Setting this to False or True overrides the ``lazy`` flag set
                during initialization for this call. Defaults to None.

        Returns:
            a dictionary containing the transformed data, as well as any other data present in the dictionary
        """
        d = dict(data)
        self.randomize(None)

        lazy_ = self.lazy if lazy is None else lazy
        for key in self.key_iterator(d):
            if self._do_transform:
                d[key] = self.flipper(d[key], lazy=lazy_)
            else:
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            self.push_transform(d[key], replace=True, lazy=lazy_)
        return d

    def inverse(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            xform = self.pop_transform(d[key])
            if not xform[TraceKeys.DO_TRANSFORM]:
                continue
            with self.flipper.trace_transform(False):
                d[key] = self.flipper(d[key])
        return d


class RandAxisFlipd(RandomizableTransform, MapTransform, InvertibleTransform, LazyTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandAxisFlip`.

    See `numpy.flip` for additional details.
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    Args:
        keys: Keys to pick data for transformation.
        prob: Probability of flipping.
        allow_missing_keys: don't raise exception if key is missing.
        lazy: a flag to indicate whether this transform should execute lazily or not.
            Defaults to False
    """

    backend = RandAxisFlip.backend

    def __init__(
        self, keys: KeysCollection, prob: float = 0.1, allow_missing_keys: bool = False, lazy: bool = False
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        LazyTransform.__init__(self, lazy=lazy)
        self.flipper = RandAxisFlip(prob=1.0, lazy=lazy)

    @LazyTransform.lazy.setter  # type: ignore
    def lazy(self, val: bool):
        self.flipper.lazy = val
        self._lazy = val

    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None) -> RandAxisFlipd:
        super().set_random_state(seed, state)
        self.flipper.set_random_state(seed, state)
        return self

    def __call__(self, data: Mapping[Hashable, torch.Tensor], lazy: bool | None = None) -> dict[Hashable, torch.Tensor]:
        """
        Args:
            data: a dictionary containing the tensor-like data to be processed. The ``keys`` specified
                in this dictionary must be tensor like arrays that are channel first and have at most
                three spatial dimensions
            lazy: a flag to indicate whether this transform should execute lazily or not
                during this call. Setting this to False or True overrides the ``lazy`` flag set
                during initialization for this call. Defaults to None.

        Returns:
            a dictionary containing the transformed data, as well as any other data present in the dictionary
        """
        d = dict(data)
        first_key: Hashable = self.first_key(d)
        if first_key == ():
            return d

        self.randomize(None)

        # all the keys share the same random selected axis
        self.flipper.randomize(d[first_key])

        lazy_ = self.lazy if lazy is None else lazy
        for key in self.key_iterator(d):
            if self._do_transform:
                d[key] = self.flipper(d[key], randomize=False, lazy=lazy_)
            else:
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            self.push_transform(d[key], replace=True, lazy=lazy_)
        return d

    def inverse(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            xform = self.pop_transform(d[key])
            if xform[TraceKeys.DO_TRANSFORM]:
                d[key].applied_operations.append(xform[TraceKeys.EXTRA_INFO])  # type: ignore
                d[key] = self.flipper.inverse(d[key])
        return d


class Rotated(MapTransform, InvertibleTransform, LazyTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.Rotate`.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    Args:
        keys: Keys to pick data for transformation.
        angle: Rotation angle(s) in radians.
        keep_size: If it is False, the output shape is adapted so that the
            input array is contained completely in the output.
            If it is True, the output shape is the same as the input. Default is True.
        mode: {``"bilinear"``, ``"nearest"``}
            Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            It also can be a sequence of string, each element corresponds to a key in ``keys``.
        padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
            Padding mode for outside grid values. Defaults to ``"border"``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            It also can be a sequence of string, each element corresponds to a key in ``keys``.
        align_corners: Defaults to False.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            It also can be a sequence of bool, each element corresponds to a key in ``keys``.
        dtype: data type for resampling computation. Defaults to ``float32``.
            If None, use the data type of input data. To be compatible with other modules,
            the output data type is always ``float32``.
            It also can be a sequence of dtype or None, each element corresponds to a key in ``keys``.
        allow_missing_keys: don't raise exception if key is missing.
        lazy: a flag to indicate whether this transform should execute lazily or not.
            Defaults to False
    """

    backend = Rotate.backend

    def __init__(
        self,
        keys: KeysCollection,
        angle: Sequence[float] | float,
        keep_size: bool = True,
        mode: SequenceStr = GridSampleMode.BILINEAR,
        padding_mode: SequenceStr = GridSamplePadMode.BORDER,
        align_corners: Sequence[bool] | bool = False,
        dtype: Sequence[DtypeLike | torch.dtype] | DtypeLike | torch.dtype = np.float32,
        allow_missing_keys: bool = False,
        lazy: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        LazyTransform.__init__(self, lazy=lazy)
        self.rotator = Rotate(angle=angle, keep_size=keep_size, lazy=lazy)

        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.padding_mode = ensure_tuple_rep(padding_mode, len(self.keys))
        self.align_corners = ensure_tuple_rep(align_corners, len(self.keys))
        self.dtype = ensure_tuple_rep(dtype, len(self.keys))

    @LazyTransform.lazy.setter  # type: ignore
    def lazy(self, val: bool):
        self.rotator.lazy = val
        self._lazy = val

    def __call__(self, data: Mapping[Hashable, torch.Tensor], lazy: bool | None = None) -> dict[Hashable, torch.Tensor]:
        """
        Args:
            data: a dictionary containing the tensor-like data to be processed. The ``keys`` specified
                in this dictionary must be tensor like arrays that are channel first and have at most
                three spatial dimensions
            lazy: a flag to indicate whether this transform should execute lazily or not
                during this call. Setting this to False or True overrides the ``lazy`` flag set
                during initialization for this call. Defaults to None.

        Returns:
            a dictionary containing the transformed data, as well as any other data present in the dictionary
        """
        d = dict(data)
        lazy_ = self.lazy if lazy is None else lazy
        for key, mode, padding_mode, align_corners, dtype in self.key_iterator(
            d, self.mode, self.padding_mode, self.align_corners, self.dtype
        ):
            d[key] = self.rotator(
                d[key], mode=mode, padding_mode=padding_mode, align_corners=align_corners, dtype=dtype, lazy=lazy_
            )
        return d

    def inverse(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.rotator.inverse(d[key])
        return d


class RandRotated(RandomizableTransform, MapTransform, InvertibleTransform, LazyTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandRotate`
    Randomly rotates the input arrays.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    Args:
        keys: Keys to pick data for transformation.
        range_x: Range of rotation angle in radians in the plane defined by the first and second axes.
            If single number, angle is uniformly sampled from (-range_x, range_x).
        range_y: Range of rotation angle in radians in the plane defined by the first and third axes.
            If single number, angle is uniformly sampled from (-range_y, range_y). only work for 3D data.
        range_z: Range of rotation angle in radians in the plane defined by the second and third axes.
            If single number, angle is uniformly sampled from (-range_z, range_z). only work for 3D data.
        prob: Probability of rotation.
        keep_size: If it is False, the output shape is adapted so that the
            input array is contained completely in the output.
            If it is True, the output shape is the same as the input. Default is True.
        mode: {``"bilinear"``, ``"nearest"``}
            Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            It also can be a sequence of string, each element corresponds to a key in ``keys``.
        padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
            Padding mode for outside grid values. Defaults to ``"border"``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            It also can be a sequence of string, each element corresponds to a key in ``keys``.
        align_corners: Defaults to False.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
            It also can be a sequence of bool, each element corresponds to a key in ``keys``.
        dtype: data type for resampling computation. Defaults to ``float64`` for best precision.
            If None, use the data type of input data. To be compatible with other modules,
            the output data type is always ``float32``.
            It also can be a sequence of dtype or None, each element corresponds to a key in ``keys``.
        allow_missing_keys: don't raise exception if key is missing.
        lazy: a flag to indicate whether this transform should execute lazily or not.
            Defaults to False
    """

    backend = RandRotate.backend

    def __init__(
        self,
        keys: KeysCollection,
        range_x: tuple[float, float] | float = 0.0,
        range_y: tuple[float, float] | float = 0.0,
        range_z: tuple[float, float] | float = 0.0,
        prob: float = 0.1,
        keep_size: bool = True,
        mode: SequenceStr = GridSampleMode.BILINEAR,
        padding_mode: SequenceStr = GridSamplePadMode.BORDER,
        align_corners: Sequence[bool] | bool = False,
        dtype: Sequence[DtypeLike | torch.dtype] | DtypeLike | torch.dtype = np.float32,
        allow_missing_keys: bool = False,
        lazy: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        LazyTransform.__init__(self, lazy=lazy)
        self.rand_rotate = RandRotate(
            range_x=range_x, range_y=range_y, range_z=range_z, prob=1.0, keep_size=keep_size, lazy=lazy
        )
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.padding_mode = ensure_tuple_rep(padding_mode, len(self.keys))
        self.align_corners = ensure_tuple_rep(align_corners, len(self.keys))
        self.dtype = ensure_tuple_rep(dtype, len(self.keys))

    @LazyTransform.lazy.setter  # type: ignore
    def lazy(self, val: bool):
        self.rand_rotate.lazy = val
        self._lazy = val

    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None) -> RandRotated:
        super().set_random_state(seed, state)
        self.rand_rotate.set_random_state(seed, state)
        return self

    def __call__(self, data: Mapping[Hashable, torch.Tensor], lazy: bool | None = None) -> dict[Hashable, torch.Tensor]:
        """
        Args:
            data: a dictionary containing the tensor-like data to be processed. The ``keys`` specified
                in this dictionary must be tensor like arrays that are channel first and have at most
                three spatial dimensions
            lazy: a flag to indicate whether this transform should execute lazily or not
                during this call. Setting this to False or True overrides the ``lazy`` flag set
                during initialization for this call. Defaults to None.

        Returns:
            a dictionary containing the transformed data, as well as any other data present in the dictionary
        """
        d = dict(data)
        self.randomize(None)

        # all the keys share the same random rotate angle
        self.rand_rotate.randomize()
        lazy_ = self.lazy if lazy is None else lazy

        for key, mode, padding_mode, align_corners, dtype in self.key_iterator(
            d, self.mode, self.padding_mode, self.align_corners, self.dtype
        ):
            if self._do_transform:
                d[key] = self.rand_rotate(
                    d[key],
                    mode=mode,
                    padding_mode=padding_mode,
                    align_corners=align_corners,
                    dtype=dtype,
                    randomize=False,
                    lazy=lazy_,
                )
            else:
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta(), dtype=torch.float32)
            self.push_transform(d[key], replace=True, lazy=lazy_)
        return d

    def inverse(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            xform = self.pop_transform(d[key])
            if xform[TraceKeys.DO_TRANSFORM]:
                d[key].applied_operations.append(xform[TraceKeys.EXTRA_INFO])  # type: ignore
                d[key] = self.rand_rotate.inverse(d[key])
        return d


class Zoomd(MapTransform, InvertibleTransform, LazyTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.Zoom`.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    Args:
        keys: Keys to pick data for transformation.
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
            One of the listed string values or a user supplied function. Defaults to ``"edge"``.
            The mode to pad data after zooming.
            See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
            https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        align_corners: This only has an effect when mode is
            'linear', 'bilinear', 'bicubic' or 'trilinear'. Default: None.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
            It also can be a sequence of bool or None, each element corresponds to a key in ``keys``.
        dtype: data type for resampling computation. Defaults to ``float32``.
            If None, use the data type of input data.
        keep_size: Should keep original size (pad if needed), default is True.
        allow_missing_keys: don't raise exception if key is missing.
        lazy: a flag to indicate whether this transform should execute lazily or not.
            Defaults to False
        kwargs: other arguments for the `np.pad` or `torch.pad` function.
            note that `np.pad` treats channel dimension as the first dimension.

    """

    backend = Zoom.backend

    def __init__(
        self,
        keys: KeysCollection,
        zoom: Sequence[float] | float,
        mode: SequenceStr = InterpolateMode.AREA,
        padding_mode: SequenceStr = NumpyPadMode.EDGE,
        align_corners: Sequence[bool | None] | bool | None = None,
        dtype: Sequence[DtypeLike | torch.dtype] | DtypeLike | torch.dtype = np.float32,
        keep_size: bool = True,
        allow_missing_keys: bool = False,
        lazy: bool = False,
        **kwargs,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        LazyTransform.__init__(self, lazy=lazy)

        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.padding_mode = ensure_tuple_rep(padding_mode, len(self.keys))
        self.align_corners = ensure_tuple_rep(align_corners, len(self.keys))
        self.dtype = ensure_tuple_rep(dtype, len(self.keys))
        self.zoomer = Zoom(zoom=zoom, keep_size=keep_size, lazy=lazy, **kwargs)

    @LazyTransform.lazy.setter  # type: ignore
    def lazy(self, val: bool):
        self.zoomer.lazy = val
        self._lazy = val

    def __call__(self, data: Mapping[Hashable, torch.Tensor], lazy: bool | None = None) -> dict[Hashable, torch.Tensor]:
        """
        Args:
            data: a dictionary containing the tensor-like data to be processed. The ``keys`` specified
                in this dictionary must be tensor like arrays that are channel first and have at most
                three spatial dimensions
            lazy: a flag to indicate whether this transform should execute lazily or not
                during this call. Setting this to False or True overrides the ``lazy`` flag set
                during initialization for this call. Defaults to None.

        Returns:
            a dictionary containing the transformed data, as well as any other data present in the dictionary
        """
        d = dict(data)
        lazy_ = self.lazy if lazy is None else lazy
        for key, mode, padding_mode, align_corners, dtype in self.key_iterator(
            d, self.mode, self.padding_mode, self.align_corners, self.dtype
        ):
            d[key] = self.zoomer(
                d[key], mode=mode, padding_mode=padding_mode, align_corners=align_corners, dtype=dtype, lazy=lazy_
            )
        return d

    def inverse(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.zoomer.inverse(d[key])
        return d


class RandZoomd(RandomizableTransform, MapTransform, InvertibleTransform, LazyTransform):
    """
    Dict-based version :py:class:`monai.transforms.RandZoom`.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    Args:
        keys: Keys to pick data for transformation.
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
            One of the listed string values or a user supplied function. Defaults to ``"edge"``.
            The mode to pad data after zooming.
            See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
            https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        align_corners: This only has an effect when mode is
            'linear', 'bilinear', 'bicubic' or 'trilinear'. Default: None.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
            It also can be a sequence of bool or None, each element corresponds to a key in ``keys``.
        dtype: data type for resampling computation. Defaults to ``float32``.
            If None, use the data type of input data.
        keep_size: Should keep original size (pad if needed), default is True.
        allow_missing_keys: don't raise exception if key is missing.
        lazy: a flag to indicate whether this transform should execute lazily or not.
            Defaults to False
        kwargs: other args for `np.pad` API, note that `np.pad` treats channel dimension as the first dimension.
            more details: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
    """

    backend = RandZoom.backend

    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.1,
        min_zoom: Sequence[float] | float = 0.9,
        max_zoom: Sequence[float] | float = 1.1,
        mode: SequenceStr = InterpolateMode.AREA,
        padding_mode: SequenceStr = NumpyPadMode.EDGE,
        align_corners: Sequence[bool | None] | bool | None = None,
        dtype: Sequence[DtypeLike | torch.dtype] | DtypeLike | torch.dtype = np.float32,
        keep_size: bool = True,
        allow_missing_keys: bool = False,
        lazy: bool = False,
        **kwargs,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        LazyTransform.__init__(self, lazy=lazy)
        self.rand_zoom = RandZoom(
            prob=1.0, min_zoom=min_zoom, max_zoom=max_zoom, keep_size=keep_size, lazy=lazy, **kwargs
        )
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.padding_mode = ensure_tuple_rep(padding_mode, len(self.keys))
        self.align_corners = ensure_tuple_rep(align_corners, len(self.keys))
        self.dtype = ensure_tuple_rep(dtype, len(self.keys))

    @LazyTransform.lazy.setter  # type: ignore
    def lazy(self, val: bool):
        self.rand_zoom.lazy = val
        self._lazy = val

    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None) -> RandZoomd:
        super().set_random_state(seed, state)
        self.rand_zoom.set_random_state(seed, state)
        return self

    def __call__(self, data: Mapping[Hashable, torch.Tensor], lazy: bool | None = None) -> dict[Hashable, torch.Tensor]:
        """
        Args:
            data: a dictionary containing the tensor-like data to be processed. The ``keys`` specified
                in this dictionary must be tensor like arrays that are channel first and have at most
                three spatial dimensions
            lazy: a flag to indicate whether this transform should execute lazily or not
                during this call. Setting this to False or True overrides the ``lazy`` flag set
                during initialization for this call. Defaults to None.

        Returns:
            a dictionary containing the transformed data, as well as any other data present in the dictionary
        """
        d = dict(data)
        first_key: Hashable = self.first_key(d)
        if first_key == ():
            out: dict[Hashable, torch.Tensor] = convert_to_tensor(d, track_meta=get_track_meta())
            return out

        self.randomize(None)

        # all the keys share the same random zoom factor
        self.rand_zoom.randomize(d[first_key])
        lazy_ = self.lazy if lazy is None else lazy

        for key, mode, padding_mode, align_corners, dtype in self.key_iterator(
            d, self.mode, self.padding_mode, self.align_corners, self.dtype
        ):
            if self._do_transform:
                d[key] = self.rand_zoom(
                    d[key],
                    mode=mode,
                    padding_mode=padding_mode,
                    align_corners=align_corners,
                    dtype=dtype,
                    randomize=False,
                    lazy=lazy_,
                )
            else:
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta(), dtype=torch.float32)
            self.push_transform(d[key], replace=True, lazy=lazy_)
        return d

    def inverse(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            xform = self.pop_transform(d[key])
            if xform[TraceKeys.DO_TRANSFORM]:
                d[key].applied_operations.append(xform[TraceKeys.EXTRA_INFO])  # type: ignore
                d[key] = self.rand_zoom.inverse(d[key])
        return d


class GridDistortiond(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.GridDistortion`.
    """

    backend = GridDistortion.backend

    def __init__(
        self,
        keys: KeysCollection,
        num_cells: tuple[int] | int,
        distort_steps: list[tuple],
        mode: str = GridSampleMode.BILINEAR,
        padding_mode: str = GridSamplePadMode.BORDER,
        device: torch.device | None = None,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
            num_cells: number of grid cells on each dimension.
            distort_steps: This argument is a list of tuples, where each tuple contains the distort steps of the
                corresponding dimensions (in the order of H, W[, D]). The length of each tuple equals to `num_cells + 1`.
                Each value in the tuple represents the distort step of the related cell.
            mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
                and the value represents the order of the spline interpolation.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
                It also can be a sequence, each element corresponds to a key in ``keys``.
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"border"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `mode` is an integer, using numpy/cupy backends, this argument accepts
                {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
                It also can be a sequence, each element corresponds to a key in ``keys``.
            device: device on which the tensor will be allocated.
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.grid_distortion = GridDistortion(num_cells=num_cells, distort_steps=distort_steps, device=device)
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.padding_mode = ensure_tuple_rep(padding_mode, len(self.keys))

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        """
        Args:
            data: a dictionary containing the tensor-like data to be processed. The ``keys`` specified
                in this dictionary must be tensor like arrays that are channel first and have at most
                three spatial dimensions

        Returns:
            a dictionary containing the transformed data, as well as any other data present in the dictionary
        """
        d = dict(data)
        for key, mode, padding_mode in self.key_iterator(d, self.mode, self.padding_mode):
            d[key] = self.grid_distortion(d[key], mode=mode, padding_mode=padding_mode)
        return d


class RandGridDistortiond(RandomizableTransform, MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.RandGridDistortion`.
    """

    backend = RandGridDistortion.backend

    def __init__(
        self,
        keys: KeysCollection,
        num_cells: tuple[int] | int = 5,
        prob: float = 0.1,
        distort_limit: tuple[float, float] | float = (-0.03, 0.03),
        mode: str = GridSampleMode.BILINEAR,
        padding_mode: str = GridSamplePadMode.BORDER,
        device: torch.device | None = None,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
            num_cells: number of grid cells on each dimension.
            prob: probability of returning a randomized grid distortion transform. Defaults to 0.1.
            distort_limit: range to randomly distort.
                If single number, distort_limit is picked from (-distort_limit, distort_limit).
                Defaults to (-0.03, 0.03).
            mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
                and the value represents the order of the spline interpolation.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
                It also can be a sequence, each element corresponds to a key in ``keys``.
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"border"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `mode` is an integer, using numpy/cupy backends, this argument accepts
                {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
                It also can be a sequence, each element corresponds to a key in ``keys``.
            device: device on which the tensor will be allocated.
            allow_missing_keys: don't raise exception if key is missing.

        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.rand_grid_distortion = RandGridDistortion(
            num_cells=num_cells, prob=1.0, distort_limit=distort_limit, device=device
        )
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.padding_mode = ensure_tuple_rep(padding_mode, len(self.keys))

    def set_random_state(
        self, seed: int | None = None, state: np.random.RandomState | None = None
    ) -> RandGridDistortiond:
        super().set_random_state(seed, state)
        self.rand_grid_distortion.set_random_state(seed, state)
        return self

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        """
        Args:
            data: a dictionary containing the tensor-like data to be processed. The ``keys`` specified
                in this dictionary must be tensor like arrays that are channel first and have at most
                three spatial dimensions

        Returns:
            a dictionary containing the transformed data, as well as any other data present in the dictionary
        """
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            out: dict[Hashable, torch.Tensor] = convert_to_tensor(d, track_meta=get_track_meta())
            return out

        first_key: Hashable = self.first_key(d)
        if first_key == ():
            out = convert_to_tensor(d, track_meta=get_track_meta())
            return out
        if isinstance(d[first_key], MetaTensor) and d[first_key].pending_operations:  # type: ignore
            warnings.warn(f"data['{first_key}'] has pending operations, transform may return incorrect results.")
        self.rand_grid_distortion.randomize(d[first_key].shape[1:])

        for key, mode, padding_mode in self.key_iterator(d, self.mode, self.padding_mode):
            d[key] = self.rand_grid_distortion(d[key], mode=mode, padding_mode=padding_mode, randomize=False)
        return d


class GridSplitd(MapTransform, MultiSampleTrait):
    """
    Split the image into patches based on the provided grid in 2D.

    Args:
        keys: keys of the corresponding items to be transformed.
        grid: a tuple define the shape of the grid upon which the image is split. Defaults to (2, 2)
        size: a tuple or an integer that defines the output patch sizes,
            or a dictionary that define it separately for each key, like {"image": 3, "mask", (2, 2)}.
            If it's an integer, the value will be repeated for each dimension.
            The default is None, where the patch size will be inferred from the grid shape.
        allow_missing_keys: don't raise exception if key is missing.

    Note: This transform currently support only image with two spatial dimensions.
    """

    backend = GridSplit.backend

    def __init__(
        self,
        keys: KeysCollection,
        grid: tuple[int, int] = (2, 2),
        size: int | tuple[int, int] | dict[Hashable, int | tuple[int, int] | None] | None = None,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.grid = grid
        self.size = size if isinstance(size, dict) else {key: size for key in self.keys}
        self.splitter = GridSplit(grid=grid)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> list[dict[Hashable, NdarrayOrTensor]]:
        """
        Args:
            data: a dictionary containing the tensor-like data to be processed. The ``keys`` specified
                in this dictionary must be tensor like arrays that are channel first and have at most
                three spatial dimensions

        Returns:
            a dictionary containing the transformed data, as well as any other data present in the dictionary
        """
        d = dict(data)
        n_outputs = np.prod(self.grid)
        output: list[dict[Hashable, NdarrayOrTensor]] = [dict(d) for _ in range(n_outputs)]
        for key in self.key_iterator(d):
            result = self.splitter(d[key], self.size[key])
            for i in range(n_outputs):
                output[i][key] = result[i]
        return output


class GridPatchd(MapTransform, MultiSampleTrait):
    """
    Extract all the patches sweeping the entire image in a row-major sliding-window manner with possible overlaps.
    It can sort the patches and return all or a subset of them.

    Args:
        keys: keys of the corresponding items to be transformed.
        patch_size: size of patches to generate slices for, 0 or None selects whole dimension
        offset: starting position in the array, default is 0 for each dimension.
            np.random.randint(0, patch_size, 2) creates random start between 0 and `patch_size` for a 2D image.
        num_patches: number of patches (or maximum number of patches) to return.
            If the requested number of patches is greater than the number of available patches,
            padding will be applied to provide exactly `num_patches` patches unless `threshold` is set.
            When `threshold` is set, this value is treated as the maximum number of patches.
            Defaults to None, which does not limit number of the patches.
        overlap: amount of overlap between patches in each dimension. Default to 0.0.
        sort_fn: when `num_patches` is provided, it determines if keep patches with highest values (`"max"`),
            lowest values (`"min"`), or in their default order (`None`). Default to None.
        threshold: a value to keep only the patches whose sum of intensities are less than the threshold.
            Defaults to no filtering.
        pad_mode: the  mode for padding the input image by `patch_size` to include patches that cross boundaries.
            Available modes: (Numpy) {``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
            ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
            (PyTorch) {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
            One of the listed string values or a user supplied function.
            Defaults to `None`, which means no padding will be applied.
            See also: https://numpy.org/doc/stable/reference/generated/numpy.pad.html
            https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
            requires pytorch >= 1.10 for best compatibility.
        allow_missing_keys: don't raise exception if key is missing.
        pad_kwargs: other arguments for the `np.pad` or `torch.pad` function.
            note that `np.pad` treats channel dimension as the first dimension.

    Returns:
        dictionary, contains the all the original key/value with the values for `keys`
            replaced by the patches, a MetaTensor with following metadata:

            - `PatchKeys.LOCATION`: the starting location of the patch in the image,
            - `PatchKeys.COUNT`: total number of patches in the image,
            - "spatial_shape": spatial size of the extracted patch, and
            - "offset": the amount of offset for the patches in the image (starting position of the first patch)
    """

    backend = GridPatch.backend

    def __init__(
        self,
        keys: KeysCollection,
        patch_size: Sequence[int],
        offset: Sequence[int] | None = None,
        num_patches: int | None = None,
        overlap: float = 0.0,
        sort_fn: str | None = None,
        threshold: float | None = None,
        pad_mode: str | None = None,
        allow_missing_keys: bool = False,
        **pad_kwargs,
    ):
        super().__init__(keys, allow_missing_keys)
        self.patcher = GridPatch(
            patch_size=patch_size,
            offset=offset,
            num_patches=num_patches,
            overlap=overlap,
            sort_fn=sort_fn,
            threshold=threshold,
            pad_mode=pad_mode,
            **pad_kwargs,
        )

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        """
        Args:
            data: a dictionary containing the tensor-like data to be processed. The ``keys`` specified
                in this dictionary must be tensor like arrays that are channel first and have at most
                three spatial dimensions

        Returns:
            a dictionary containing the transformed data, as well as any other data present in the dictionary
        """
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.patcher(d[key])
        return d


class RandGridPatchd(RandomizableTransform, MapTransform, MultiSampleTrait):
    """
    Extract all the patches sweeping the entire image in a row-major sliding-window manner with possible overlaps,
    and with random offset for the minimal corner of the image, (0,0) for 2D and (0,0,0) for 3D.
    It can sort the patches and return all or a subset of them.

    Args:
        keys: keys of the corresponding items to be transformed.
        patch_size: size of patches to generate slices for, 0 or None selects whole dimension
        min_offset: the minimum range of starting position to be selected randomly. Defaults to 0.
        max_offset: the maximum range of starting position to be selected randomly.
            Defaults to image size modulo patch size.
        num_patches: number of patches (or maximum number of patches) to return.
            If the requested number of patches is greater than the number of available patches,
            padding will be applied to provide exactly `num_patches` patches unless `threshold` is set.
            When `threshold` is set, this value is treated as the maximum number of patches.
            Defaults to None, which does not limit number of the patches.
        overlap: the amount of overlap of neighboring patches in each dimension (a value between 0.0 and 1.0).
            If only one float number is given, it will be applied to all dimensions. Defaults to 0.0.
        sort_fn: when `num_patches` is provided, it determines if keep patches with highest values (`"max"`),
            lowest values (`"min"`), in random ("random"), or in their default order (`None`). Default to None.
        threshold: a value to keep only the patches whose sum of intensities are less than the threshold.
            Defaults to no filtering.
        pad_mode: the  mode for padding the input image by `patch_size` to include patches that cross boundaries.
            Available modes: (Numpy) {``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
            ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
            (PyTorch) {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
            One of the listed string values or a user supplied function.
            Defaults to `None`, which means no padding will be applied.
            See also: https://numpy.org/doc/stable/reference/generated/numpy.pad.html
            https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
            requires pytorch >= 1.10 for best compatibility.
        allow_missing_keys: don't raise exception if key is missing.
        pad_kwargs: other arguments for the `np.pad` or `torch.pad` function.
            note that `np.pad` treats channel dimension as the first dimension.

    Returns:
        dictionary, contains the all the original key/value with the values for `keys`
            replaced by the patches, a MetaTensor with following metadata:

            - `PatchKeys.LOCATION`: the starting location of the patch in the image,
            - `PatchKeys.COUNT`: total number of patches in the image,
            - "spatial_shape": spatial size of the extracted patch, and
            - "offset": the amount of offset for the patches in the image (starting position of the first patch)

    """

    backend = RandGridPatch.backend

    def __init__(
        self,
        keys: KeysCollection,
        patch_size: Sequence[int],
        min_offset: Sequence[int] | int | None = None,
        max_offset: Sequence[int] | int | None = None,
        num_patches: int | None = None,
        overlap: float = 0.0,
        sort_fn: str | None = None,
        threshold: float | None = None,
        pad_mode: str | None = None,
        allow_missing_keys: bool = False,
        **pad_kwargs,
    ):
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.patcher = RandGridPatch(
            patch_size=patch_size,
            min_offset=min_offset,
            max_offset=max_offset,
            num_patches=num_patches,
            overlap=overlap,
            sort_fn=sort_fn,
            threshold=threshold,
            pad_mode=pad_mode,
            **pad_kwargs,
        )

    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None) -> RandGridPatchd:
        super().set_random_state(seed, state)
        self.patcher.set_random_state(seed, state)
        return self

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        """
        Args:
            data: a dictionary containing the tensor-like data to be processed. The ``keys`` specified
                in this dictionary must be tensor like arrays that are channel first and have at most
                three spatial dimensions

        Returns:
            a dictionary containing the transformed data, as well as any other data present in the dictionary
        """
        d = dict(data)
        # All the keys share the same random noise
        for key in self.key_iterator(d):
            self.patcher.randomize(d[key])
            break
        for key in self.key_iterator(d):
            d[key] = self.patcher(d[key], randomize=False)
        return d


class RandSimulateLowResolutiond(RandomizableTransform, MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.RandSimulateLowResolution`.
    Random simulation of low resolution corresponding to nnU-Net's SimulateLowResolutionTransform
    (https://github.com/MIC-DKFZ/batchgenerators/blob/7651ece69faf55263dd582a9f5cbd149ed9c3ad0/batchgenerators/transforms/resample_transforms.py#L23)
    First, the array/tensor is resampled at lower resolution as determined by the zoom_factor which is uniformly sampled
    from the `zoom_range`. Then, the array/tensor is resampled at the original resolution.
    """

    backend = RandAffine.backend

    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.1,
        downsample_mode: InterpolateMode | str = InterpolateMode.NEAREST,
        upsample_mode: InterpolateMode | str = InterpolateMode.TRILINEAR,
        zoom_range=(0.5, 1.0),
        align_corners=False,
        allow_missing_keys: bool = False,
        device: torch.device | None = None,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
            prob: probability of performing this augmentation
            downsample_mode: interpolation mode for downsampling operation
            upsample_mode: interpolation mode for upsampling operation
            zoom_range: range from which the random zoom factor for the downsampling and upsampling operation is
            sampled. It determines the shape of the downsampled tensor.
            align_corners: This only has an effect when downsample_mode or upsample_mode  is 'linear', 'bilinear',
                'bicubic' or 'trilinear'. Default: False
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
            allow_missing_keys: don't raise exception if key is missing.
            device: device on which the tensor will be allocated.

        See also:
            - :py:class:`monai.transforms.compose.MapTransform`

        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)

        self.downsample_mode = downsample_mode
        self.upsample_mode = upsample_mode
        self.zoom_range = zoom_range
        self.align_corners = align_corners
        self.device = device

        self.sim_lowres_tfm = RandSimulateLowResolution(
            prob=1.0,  # probability is handled by dictionary class
            downsample_mode=self.downsample_mode,
            upsample_mode=self.upsample_mode,
            zoom_range=self.zoom_range,
            align_corners=self.align_corners,
            device=self.device,
        )

    def set_random_state(
        self, seed: int | None = None, state: np.random.RandomState | None = None
    ) -> RandSimulateLowResolutiond:
        super().set_random_state(seed, state)
        self.sim_lowres_tfm.set_random_state(seed, state)
        return self

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        """
        Args:
            data: a dictionary containing the tensor-like data to be transformed. The ``keys`` specified
                in this dictionary must be tensor like arrays that are channel first and have at most
                three spatial dimensions
        """
        d = dict(data)
        first_key: Hashable = self.first_key(d)
        if first_key == ():
            out: dict[Hashable, NdarrayOrTensor] = convert_to_tensor(d, track_meta=get_track_meta())
            return out

        self.randomize(None)

        for key in self.key_iterator(d):
            # do the transform
            if self._do_transform:
                d[key] = self.sim_lowres_tfm(d[key])  # type: ignore
            else:
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta(), dtype=torch.float32)
        return d


class ConvertBoxToPointsd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ConvertBoxToPoints`.
    """

    backend = ConvertBoxToPoints.backend

    def __init__(
        self,
        keys: KeysCollection,
        point_key="points",
        mode: str | BoxMode | type[BoxMode] | None = StandardMode,
        allow_missing_keys: bool = False,
    ):
        """
        Args:
            keys: keys of the corresponding items to be transformed.
            point_key: key to store the point data.
            mode: the mode of the input boxes. Defaults to StandardMode.
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.point_key = point_key
        self.converter = ConvertBoxToPoints(mode=mode)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            data[self.point_key] = self.converter(d[key])
        return data


class ConvertPointsToBoxesd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ConvertPointsToBoxes`.
    """

    def __init__(self, keys: KeysCollection, box_key="box", allow_missing_keys: bool = False):
        """
        Args:
            keys: keys of the corresponding items to be transformed.
            box_key: key to store the box data.
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.box_key = box_key
        self.converter = ConvertPointsToBoxes()

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            data[self.box_key] = self.converter(d[key])
        return data


SpatialResampleD = SpatialResampleDict = SpatialResampled
ResampleToMatchD = ResampleToMatchDict = ResampleToMatchd
SpacingD = SpacingDict = Spacingd
OrientationD = OrientationDict = Orientationd
Rotate90D = Rotate90Dict = Rotate90d
RandRotate90D = RandRotate90Dict = RandRotate90d
ResizeD = ResizeDict = Resized
AffineD = AffineDict = Affined
RandAffineD = RandAffineDict = RandAffined
Rand2DElasticD = Rand2DElasticDict = Rand2DElasticd
Rand3DElasticD = Rand3DElasticDict = Rand3DElasticd
FlipD = FlipDict = Flipd
RandFlipD = RandFlipDict = RandFlipd
GridDistortionD = GridDistortionDict = GridDistortiond
RandGridDistortionD = RandGridDistortionDict = RandGridDistortiond
RandAxisFlipD = RandAxisFlipDict = RandAxisFlipd
RotateD = RotateDict = Rotated
RandRotateD = RandRotateDict = RandRotated
ZoomD = ZoomDict = Zoomd
RandZoomD = RandZoomDict = RandZoomd
GridSplitD = GridSplitDict = GridSplitd
GridPatchD = GridPatchDict = GridPatchd
RandGridPatchD = RandGridPatchDict = RandGridPatchd
RandSimulateLowResolutionD = RandSimulateLowResolutionDict = RandSimulateLowResolutiond
ConvertBoxToPointsD = ConvertBoxToPointsDict = ConvertBoxToPointsd
ConvertPointsToBoxesD = ConvertPointsToBoxesDict = ConvertPointsToBoxesd
