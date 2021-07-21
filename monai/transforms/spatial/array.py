# Copyright 2020 - 2021 MONAI Consortium
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
A collection of "vanilla" transforms for spatial operations
https://github.com/Project-MONAI/MONAI/wiki/MONAI_Design
"""

import warnings
from copy import deepcopy
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from monai.config import USE_COMPILED, DtypeLike
from monai.data.utils import compute_shape_offset, to_affine_nd, zoom_affine
from monai.networks.layers import AffineTransform, GaussianFilter, grid_pull
from monai.transforms.croppad.array import CenterSpatialCrop, Pad
from monai.transforms.transform import NumpyTransform, Randomizable, RandomizableTransform, ThreadUnsafe, TorchTransform
from monai.transforms.utils import (
    create_control_grid,
    create_grid,
    create_rotate,
    create_scale,
    create_shear,
    create_translate,
    map_spatial_axes,
)
from monai.utils import (
    GridSampleMode,
    GridSamplePadMode,
    InterpolateMode,
    NumpyPadMode,
    ensure_tuple,
    ensure_tuple_rep,
    ensure_tuple_size,
    fall_back_tuple,
    issequenceiterable,
    optional_import,
)
from monai.utils.enums import DataObjects
from monai.utils.misc import convert_data_type
from monai.utils.module import look_up_option

nib, _ = optional_import("nibabel")

__all__ = [
    "Spacing",
    "Orientation",
    "Flip",
    "Resize",
    "Rotate",
    "Zoom",
    "Rotate90",
    "RandRotate90",
    "RandRotate",
    "RandFlip",
    "RandAxisFlip",
    "RandZoom",
    "AffineGrid",
    "RandAffineGrid",
    "RandDeformGrid",
    "Resample",
    "Affine",
    "RandAffine",
    "Rand2DElastic",
    "Rand3DElastic",
    "AddCoordinateChannels",
]

RandRange = Optional[Union[Sequence[Union[Tuple[float, float], float]], float]]


class Spacing(TorchTransform):
    """
    Resample input image into the specified `pixdim`.
    """

    def __init__(
        self,
        pixdim: Union[Sequence[float], float],
        diagonal: bool = False,
        mode: Union[GridSampleMode, str] = GridSampleMode.BILINEAR,
        padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.BORDER,
        align_corners: bool = False,
        dtype: DtypeLike = np.float64,
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

                    np.diag((pixdim_0, pixdim_1, ..., pixdim_n, 1))

                This effectively resets the volume to the world coordinate system (RAS+ in nibabel).
                The original orientation, rotation, shearing are not preserved.

                If False, this transform preserves the axes orientation, orthogonal rotation and
                translation components from the original affine. This option will not flip/swap axes
                of the original data.
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"border"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            align_corners: Geometrically, we consider the pixels of the input as squares rather than points.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            dtype: data type for resampling computation. Defaults to ``np.float64`` for best precision.
                If None, use the data type of input data. To be compatible with other modules,
                the output data type is always ``np.float32``.

        """
        self.pixdim = np.array(ensure_tuple(pixdim), dtype=np.float64)
        self.diagonal = diagonal
        self.mode: GridSampleMode = look_up_option(mode, GridSampleMode)
        self.padding_mode: GridSamplePadMode = look_up_option(padding_mode, GridSamplePadMode)
        self.align_corners = align_corners
        self.dtype = dtype

    def __call__(
        self,
        data_array: DataObjects.Images,
        affine: Optional[DataObjects.Images] = None,
        mode: Optional[Union[GridSampleMode, str]] = None,
        padding_mode: Optional[Union[GridSamplePadMode, str]] = None,
        align_corners: Optional[bool] = None,
        dtype: DtypeLike = None,
        output_spatial_shape: Optional[np.ndarray] = None,
    ) -> Tuple[DataObjects.Images, DataObjects.Images, DataObjects.Images]:
        """
        Args:
            data_array: in shape (num_channels, H[, W, ...]).
            affine (matrix): (N+1)x(N+1) original affine matrix for spatially ND `data_array`. Defaults to identity.
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``self.padding_mode``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            align_corners: Geometrically, we consider the pixels of the input as squares rather than points.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            dtype: data type for resampling computation. Defaults to ``self.dtype``.
                If None, use the data type of input data. To be compatible with other modules,
                the output data type is always ``np.float32``.
            output_spatial_shape: specify the shape of the output data_array. This is typically useful for
                the inverse of `Spacingd` where sometimes we could not compute the exact shape due to the quantization
                error with the affine.

        Raises:
            ValueError: When ``data_array`` has no spatial dimensions.
            ValueError: When ``pixdim`` is nonpositive.

        Returns:
            data_array (resampled into `self.pixdim`), original affine, current affine.

        """
        _dtype = dtype or self.dtype or data_array.dtype
        data_array_torch: torch.Tensor
        data_array_torch, orig_type, orig_device = convert_data_type(data_array, torch.Tensor, dtype=_dtype)  # type: ignore

        sr = data_array_torch.ndim - 1
        if sr <= 0:
            raise ValueError("data_array must have at least one spatial dimension.")
        if affine is None:
            # default to identity
            affine = np.eye(sr + 1, dtype=np.float64)
            affine_ = np.eye(sr + 1, dtype=np.float64)
        else:
            affine_ = to_affine_nd(sr, affine)  # type: ignore
            affine_, *_ = convert_data_type(affine_, np.ndarray)  # type: ignore

        out_d = self.pixdim[:sr]
        if out_d.size < sr:
            out_d = np.append(out_d, [1.0] * (sr - out_d.size))

        # compute output affine, shape and offset
        new_affine = zoom_affine(affine_, out_d, diagonal=self.diagonal)
        output_shape, offset = compute_shape_offset(data_array_torch.shape[1:], affine_, new_affine)
        new_affine[:sr, -1] = offset[:sr]
        affine_inv = np.linalg.inv(affine_)
        transform = affine_inv @ new_affine
        # adapt to the actual rank
        transform = to_affine_nd(sr, transform)

        # no resampling if it's identity transform
        if np.allclose(transform, np.diag(np.ones(len(transform))), atol=1e-3):
            output_data, *_ = convert_data_type(deepcopy(data_array), dtype=_dtype)
            new_affine = to_affine_nd(affine, new_affine)
        else:
            # resample
            affine_xform = AffineTransform(
                normalized=False,
                mode=look_up_option(mode or self.mode, GridSampleMode),
                padding_mode=look_up_option(padding_mode or self.padding_mode, GridSamplePadMode),
                align_corners=self.align_corners if align_corners is None else align_corners,
                reverse_indexing=True,
            )
            output_data = affine_xform(
                # AffineTransform requires a batch dim
                data_array_torch.unsqueeze(0),
                convert_data_type(transform, torch.Tensor, data_array_torch.device, dtype=_dtype)[0],
                spatial_size=output_shape if output_spatial_shape is None else output_spatial_shape,
            ).squeeze(0)
            output_data, *_ = convert_data_type(output_data, orig_type, dtype=np.float32)  # type: ignore
            new_affine = to_affine_nd(affine, new_affine)

        return output_data, affine, new_affine


class Orientation(NumpyTransform):
    """
    Change the input image's orientation into the specified based on `axcodes`.
    """

    def __init__(
        self,
        axcodes: Optional[str] = None,
        as_closest_canonical: bool = False,
        labels: Optional[Sequence[Tuple[str, str]]] = tuple(zip("LPI", "RAS")),
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

        Raises:
            ValueError: When ``axcodes=None`` and ``as_closest_canonical=True``. Incompatible values.

        See Also: `nibabel.orientations.ornt2axcodes`.

        """
        if axcodes is None and not as_closest_canonical:
            raise ValueError("Incompatible values: axcodes=None and as_closest_canonical=True.")
        if axcodes is not None and as_closest_canonical:
            warnings.warn("using as_closest_canonical=True, axcodes ignored.")
        self.axcodes = axcodes
        self.as_closest_canonical = as_closest_canonical
        self.labels = labels

    def __call__(
        self, data_array: DataObjects.Images, affine: Optional[DataObjects.Images] = None
    ) -> Tuple[DataObjects.Images, DataObjects.Images, np.ndarray]:
        """
        original orientation of `data_array` is defined by `affine`.

        Args:
            data_array: in shape (num_channels, H[, W, ...]).
            affine (matrix): (N+1)x(N+1) original affine matrix for spatially ND `data_array`. Defaults to identity.

        Raises:
            ValueError: When ``data_array`` has no spatial dimensions.
            ValueError: When ``axcodes`` spatiality differs from ``data_array``.

        Returns:
            data_array (reoriented in `self.axcodes`), original axcodes, current axcodes.

        """
        data_np: np.ndarray
        data_np, orig_type, orig_device = convert_data_type(data_array, np.ndarray)  # type: ignore
        sr = data_np.ndim - 1
        if sr <= 0:
            raise ValueError("data_array must have at least one spatial dimension.")
        if affine is None:
            affine = np.eye(sr + 1, dtype=np.float64)
            affine_ = np.eye(sr + 1, dtype=np.float64)
        else:
            affine, *_ = convert_data_type(affine, np.ndarray)
            affine_ = to_affine_nd(sr, affine)  # type: ignore
        src = nib.io_orientation(affine_)
        if self.as_closest_canonical:
            spatial_ornt = src
        else:
            if self.axcodes is None:
                raise AssertionError
            dst = nib.orientations.axcodes2ornt(self.axcodes[:sr], labels=self.labels)
            if len(dst) < sr:
                raise ValueError(
                    f"axcodes must match data_array spatially, got axcodes={len(self.axcodes)}D data_array={sr}D"
                )
            spatial_ornt = nib.orientations.ornt_transform(src, dst)
        ornt = spatial_ornt.copy()
        ornt[:, 0] += 1  # skip channel dim
        ornt = np.concatenate([np.array([[0, 1]]), ornt])
        shape = data_np.shape[1:]
        data_np = np.ascontiguousarray(nib.orientations.apply_orientation(data_np, ornt))
        new_affine = affine_ @ nib.orientations.inv_ornt_aff(spatial_ornt, shape)
        new_affine = to_affine_nd(affine, new_affine)

        data_out, *_ = convert_data_type(data_np, orig_type, orig_device)

        return data_out, affine, new_affine


class Flip(TorchTransform):
    """
    Reverses the order of elements along the given spatial axis. Preserves shape.
    Uses ``np.flip`` in practice. See numpy.flip for additional details:
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html.

    Args:
        spatial_axis: spatial axes along which to flip over. Default is None.
            The default `axis=None` will flip over all of the axes of the input array.
            If axis is negative it counts from the last to the first axis.
            If axis is a tuple of ints, flipping is performed on all of the axes
            specified in the tuple.

    """

    def __init__(self, spatial_axis: Optional[Union[Sequence[int], int]] = None) -> None:
        self.spatial_axis = spatial_axis

    def __call__(self, img: DataObjects.Images) -> DataObjects.Images:
        """
        Args:
            img: channel first array, must have shape: (num_channels, H[, W, ..., ]),
        """
        img_t: torch.Tensor
        img_t, orig_type, orig_device = convert_data_type(img, torch.Tensor)  # type: ignore

        result_t = torch.flip(img_t, map_spatial_axes(img.ndim, self.spatial_axis))

        result, *_ = convert_data_type(result_t, orig_type, orig_device)
        return result


class Resize(TorchTransform):
    """
    Resize the input image to given spatial size (with scaling, not cropping/padding).
    Implemented using :py:class:`torch.nn.functional.interpolate`.

    Args:
        spatial_size: expected shape of spatial dimensions after resize operation.
            if the components of the `spatial_size` are non-positive values, the transform will use the
            corresponding components of img size. For example, `spatial_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
            The interpolation mode. Defaults to ``"area"``.
            See also: https://pytorch.org/docs/stable/nn.functional.html#interpolate
        align_corners: This only has an effect when mode is
            'linear', 'bilinear', 'bicubic' or 'trilinear'. Default: None.
            See also: https://pytorch.org/docs/stable/nn.functional.html#interpolate
    """

    def __init__(
        self,
        spatial_size: Union[Sequence[int], int],
        mode: Union[InterpolateMode, str] = InterpolateMode.AREA,
        align_corners: Optional[bool] = None,
    ) -> None:
        self.spatial_size = ensure_tuple(spatial_size)
        self.mode: InterpolateMode = look_up_option(mode, InterpolateMode)
        self.align_corners = align_corners

    def __call__(
        self,
        img: DataObjects.Images,
        mode: Optional[Union[InterpolateMode, str]] = None,
        align_corners: Optional[bool] = None,
    ) -> DataObjects.Images:
        """
        Args:
            img: channel first array, must have shape: (num_channels, H[, W, ..., ]).
            mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
                The interpolation mode. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#interpolate
            align_corners: This only has an effect when mode is
                'linear', 'bilinear', 'bicubic' or 'trilinear'. Defaults to ``self.align_corners``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#interpolate

        Raises:
            ValueError: When ``self.spatial_size`` length is less than ``img`` spatial dimensions.

        """
        img_t: torch.Tensor
        img_t, orig_type, orig_device = convert_data_type(img, torch.Tensor, dtype=float)  # type: ignore
        input_ndim = img_t.ndim - 1  # spatial ndim
        output_ndim = len(self.spatial_size)
        if output_ndim > input_ndim:
            input_shape = ensure_tuple_size(img_t.shape, output_ndim + 1, 1)
            img_t = img_t.reshape(input_shape)
        elif output_ndim < input_ndim:
            raise ValueError(
                "len(spatial_size) must be greater or equal to img spatial dimensions, "
                f"got spatial_size={output_ndim} img={input_ndim}."
            )
        spatial_size = fall_back_tuple(self.spatial_size, img_t.shape[1:])
        resized = torch.nn.functional.interpolate(
            input=img_t.unsqueeze(0),
            size=spatial_size,
            mode=look_up_option(self.mode if mode is None else mode, InterpolateMode).value,
            align_corners=self.align_corners if align_corners is None else align_corners,
        )
        resized = resized.squeeze(0)
        out, *_ = convert_data_type(resized, orig_type, orig_device)
        return out


class Rotate(TorchTransform, ThreadUnsafe):
    """
    Rotates an input image by given angle using :py:class:`monai.networks.layers.AffineTransform`.

    Args:
        angle: Rotation angle(s) in radians. should a float for 2D, three floats for 3D.
        keep_size: If it is True, the output shape is kept the same as the input.
            If it is False, the output shape is adapted so that the
            input array is contained completely in the output. Default is True.
        mode: {``"bilinear"``, ``"nearest"``}
            Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
            See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
        padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
            Padding mode for outside grid values. Defaults to ``"border"``.
            See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
        align_corners: Defaults to False.
            See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
        dtype: data type for resampling computation. Defaults to ``np.float64`` for best precision.
            If None, use the data type of input data. To be compatible with other modules,
            the output data type is always ``np.float32``.
    """

    def __init__(
        self,
        angle: Union[Sequence[float], float],
        keep_size: bool = True,
        mode: Union[GridSampleMode, str] = GridSampleMode.BILINEAR,
        padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.BORDER,
        align_corners: bool = False,
        dtype: DtypeLike = np.float64,
    ) -> None:
        self.angle = angle
        self.keep_size = keep_size
        self.mode: GridSampleMode = look_up_option(mode, GridSampleMode)
        self.padding_mode: GridSamplePadMode = look_up_option(padding_mode, GridSamplePadMode)
        self.align_corners = align_corners
        self.dtype = dtype
        self._rotation_matrix: Optional[np.ndarray] = None

    def __call__(
        self,
        img: DataObjects.Images,
        mode: Optional[Union[GridSampleMode, str]] = None,
        padding_mode: Optional[Union[GridSamplePadMode, str]] = None,
        align_corners: Optional[bool] = None,
        dtype: DtypeLike = None,
    ) -> DataObjects.Images:
        """
        Args:
            img: channel first array, must have shape: [chns, H, W] or [chns, H, W, D].
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``self.padding_mode``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
                align_corners: Defaults to ``self.align_corners``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            align_corners: Defaults to ``self.align_corners``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            dtype: data type for resampling computation. Defaults to ``self.dtype``.
                If None, use the data type of input data. To be compatible with other modules,
                the output data type is always ``np.float32``.

        Raises:
            ValueError: When ``img`` spatially is not one of [2D, 3D].

        """
        _dtype = dtype or self.dtype or img.dtype
        img_t: torch.Tensor
        img_t, orig_type, orig_device = convert_data_type(img, torch.Tensor, dtype=_dtype)  # type: ignore

        im_shape = np.asarray(img_t.shape[1:])  # spatial dimensions
        input_ndim = len(im_shape)
        if input_ndim not in (2, 3):
            raise ValueError(f"Unsupported img dimension: {input_ndim}, available options are [2, 3].")
        _angle = ensure_tuple_rep(self.angle, 1 if input_ndim == 2 else 3)
        transform = create_rotate(input_ndim, _angle)
        shift = create_translate(input_ndim, ((im_shape - 1) / 2).tolist())
        if self.keep_size:
            output_shape = im_shape
        else:
            corners = np.asarray(np.meshgrid(*[(0, dim) for dim in im_shape], indexing="ij")).reshape(
                (len(im_shape), -1)
            )
            corners = transform[:-1, :-1] @ corners
            output_shape = np.asarray(corners.ptp(axis=1) + 0.5, dtype=int)
        shift_1 = create_translate(input_ndim, (-(output_shape - 1) / 2).tolist())
        transform = shift @ transform @ shift_1
        transform_t: torch.Tensor
        transform_t, *_ = convert_data_type(transform, torch.Tensor, dtype=_dtype, device=img_t.device)  # type: ignore

        xform = AffineTransform(
            normalized=False,
            mode=look_up_option(mode or self.mode, GridSampleMode),
            padding_mode=look_up_option(padding_mode or self.padding_mode, GridSamplePadMode),
            align_corners=self.align_corners if align_corners is None else align_corners,
            reverse_indexing=True,
        )
        output = xform(
            img_t.unsqueeze(0),
            transform_t,
            spatial_size=output_shape,
        )
        self._rotation_matrix = transform
        out, *_ = convert_data_type(output.squeeze(0).float(), orig_type, orig_device)
        return out

    def get_rotation_matrix(self) -> Optional[np.ndarray]:
        """
        Get the most recently applied rotation matrix
        This is not thread-safe.
        """
        return self._rotation_matrix


class Zoom(TorchTransform):
    """
    Zooms an ND image using :py:class:`torch.nn.functional.interpolate`.
    For details, please see https://pytorch.org/docs/stable/nn.functional.html#interpolate.

    Different from :py:class:`monai.transforms.resize`, this transform takes scaling factors
    as input, and provides an option of preserving the input spatial size.

    Args:
        zoom: The zoom factor along the spatial axes.
            If a float, zoom is the same for each spatial axis.
            If a sequence, zoom should contain one value for each spatial axis.
        mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
            The interpolation mode. Defaults to ``"area"``.
            See also: https://pytorch.org/docs/stable/nn.functional.html#interpolate
        padding_mode: {``"constant"``, ``"edge``", ``"linear_ramp``", ``"maximum``", ``"mean``", `"median``",
            ``"minimum``", `"reflect``", ``"symmetric``", ``"wrap``", ``"empty``", ``"<function>``"}
            The mode to pad data after zooming.
            See also: https://numpy.org/doc/stable/reference/generated/numpy.pad.html
        align_corners: This only has an effect when mode is
            'linear', 'bilinear', 'bicubic' or 'trilinear'. Default: None.
            See also: https://pytorch.org/docs/stable/nn.functional.html#interpolate
        keep_size: Should keep original size (padding/slicing if needed), default is True.
    """

    def __init__(
        self,
        zoom: Union[Sequence[float], float],
        mode: Union[InterpolateMode, str] = InterpolateMode.AREA,
        padding_mode: Union[NumpyPadMode, str] = NumpyPadMode.EDGE,
        align_corners: Optional[bool] = None,
        keep_size: bool = True,
    ) -> None:
        self.zoom = zoom
        self.mode: InterpolateMode = InterpolateMode(mode)
        self.padding_mode: NumpyPadMode = NumpyPadMode(padding_mode)
        self.align_corners = align_corners
        self.keep_size = keep_size

    def __call__(
        self,
        img: DataObjects.Images,
        mode: Optional[Union[InterpolateMode, str]] = None,
        padding_mode: Optional[Union[NumpyPadMode, str]] = None,
        align_corners: Optional[bool] = None,
    ) -> DataObjects.Images:
        """
        Args:
            img: channel first array, must have shape: (num_channels, H[, W, ..., ]).
            mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
                The interpolation mode. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#interpolate
            padding_mode: {``"constant"``, ``"edge``", ``"linear_ramp``", ``"maximum``", ``"mean``", `"median``",
                ``"minimum``", `"reflect``", ``"symmetric``", ``"wrap``", ``"empty``", ``"<function>``"}
                The mode to pad data after zooming, default to ``self.padding_mode``.
                See also: https://numpy.org/doc/stable/reference/generated/numpy.pad.html
            align_corners: This only has an effect when mode is
                'linear', 'bilinear', 'bicubic' or 'trilinear'. Defaults to ``self.align_corners``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#interpolate

        """
        img_t: torch.Tensor
        img_t, orig_type, orig_device = convert_data_type(img, torch.Tensor)  # type: ignore

        _zoom = ensure_tuple_rep(self.zoom, img_t.ndim - 1)  # match the spatial image dim
        zoomed = torch.nn.functional.interpolate(  # type: ignore
            recompute_scale_factor=True,
            input=img_t.float().unsqueeze(0),
            scale_factor=list(_zoom),
            mode=look_up_option(self.mode if mode is None else mode, InterpolateMode).value,
            align_corners=self.align_corners if align_corners is None else align_corners,
        )
        zoomed = zoomed.squeeze(0)

        if self.keep_size and not np.allclose(img_t.shape, zoomed.shape):

            pad_vec = [(0, 0)] * len(img_t.shape)
            slice_vec = [slice(None)] * len(img_t.shape)
            for idx, (od, zd) in enumerate(zip(img_t.shape, zoomed.shape)):
                diff = od - zd
                half = abs(diff) // 2
                if diff > 0:  # need padding
                    pad_vec[idx] = (half, diff - half)
                elif diff < 0:  # need slicing
                    slice_vec[idx] = slice(half, half + od)

            padding_mode = look_up_option(padding_mode or self.padding_mode, NumpyPadMode)
            padder = Pad(pad_vec, padding_mode)
            zoomed = padder(zoomed)
            zoomed = zoomed[tuple(slice_vec)]

        out, *_ = convert_data_type(zoomed, orig_type, orig_device)
        return out


class Rotate90(TorchTransform, NumpyTransform):
    """
    Rotate an array by 90 degrees in the plane specified by `axes`.
    See np.rot90 for additional details:
    https://numpy.org/doc/stable/reference/generated/numpy.rot90.html.

    """

    def __init__(self, k: int = 1, spatial_axes: Tuple[int, int] = (0, 1)) -> None:
        """
        Args:
            k: number of times to rotate by 90 degrees.
            spatial_axes: 2 int numbers, defines the plane to rotate with 2 spatial axes.
                Default: (0, 1), this is the first two axis in spatial dimensions.
                If axis is negative it counts from the last to the first axis.
        """
        self.k = k
        spatial_axes_: Tuple[int, int] = ensure_tuple(spatial_axes)  # type: ignore
        if len(spatial_axes_) != 2:
            raise ValueError("spatial_axes must be 2 int numbers to indicate the axes to rotate 90 degrees.")
        self.spatial_axes = spatial_axes_

    def __call__(self, img: DataObjects.Images) -> DataObjects.Images:
        """
        Args:
            img: channel first array, must have shape: (num_channels, H[, W, ..., ]),
        """
        if isinstance(img, torch.Tensor):
            return torch.rot90(img, self.k, map_spatial_axes(img.ndim, self.spatial_axes)).to(img.dtype)
        return np.rot90(img, self.k, map_spatial_axes(img.ndim, self.spatial_axes)).astype(img.dtype)  # type: ignore


class RandRotate90(TorchTransform, NumpyTransform, RandomizableTransform):
    """
    With probability `prob`, input arrays are rotated by 90 degrees
    in the plane specified by `spatial_axes`.
    """

    def __init__(self, prob: float = 0.1, max_k: int = 3, spatial_axes: Tuple[int, int] = (0, 1)) -> None:
        """
        Args:
            prob: probability of rotating.
                (Default 0.1, with 10% probability it returns a rotated array)
            max_k: number of rotations will be sampled from `np.random.randint(max_k) + 1`, (Default 3).
            spatial_axes: 2 int numbers, defines the plane to rotate with 2 spatial axes.
                Default: (0, 1), this is the first two axis in spatial dimensions.
        """
        RandomizableTransform.__init__(self, prob)
        self.max_k = max_k
        self.spatial_axes = spatial_axes

        self._rand_k = 0

    def randomize(self, data: Optional[Any] = None) -> None:
        self._rand_k = self.R.randint(self.max_k) + 1
        super().randomize(None)

    def __call__(self, img: DataObjects.Images) -> DataObjects.Images:
        """
        Args:
            img: channel first array, must have shape: (num_channels, H[, W, ..., ]),
        """
        self.randomize()
        if not self._do_transform:
            return img
        rotator = Rotate90(self._rand_k, self.spatial_axes)
        return rotator(img)


class RandRotate(TorchTransform, RandomizableTransform):
    """
    Randomly rotate the input arrays.

    Args:
        range_x: Range of rotation angle in radians in the plane defined by the first and second axes.
            If single number, angle is uniformly sampled from (-range_x, range_x).
        range_y: Range of rotation angle in radians in the plane defined by the first and third axes.
            If single number, angle is uniformly sampled from (-range_y, range_y).
        range_z: Range of rotation angle in radians in the plane defined by the second and third axes.
            If single number, angle is uniformly sampled from (-range_z, range_z).
        prob: Probability of rotation.
        keep_size: If it is False, the output shape is adapted so that the
            input array is contained completely in the output.
            If it is True, the output shape is the same as the input. Default is True.
        mode: {``"bilinear"``, ``"nearest"``}
            Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
            See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
        padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
            Padding mode for outside grid values. Defaults to ``"border"``.
            See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
        align_corners: Defaults to False.
            See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
        dtype: data type for resampling computation. Defaults to ``np.float64`` for best precision.
            If None, use the data type of input data. To be compatible with other modules,
            the output data type is always ``np.float32``.
    """

    def __init__(
        self,
        range_x: Union[Tuple[float, float], float] = 0.0,
        range_y: Union[Tuple[float, float], float] = 0.0,
        range_z: Union[Tuple[float, float], float] = 0.0,
        prob: float = 0.1,
        keep_size: bool = True,
        mode: Union[GridSampleMode, str] = GridSampleMode.BILINEAR,
        padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.BORDER,
        align_corners: bool = False,
        dtype: DtypeLike = np.float64,
    ) -> None:
        RandomizableTransform.__init__(self, prob)
        self.range_x = ensure_tuple(range_x)
        if len(self.range_x) == 1:
            self.range_x = tuple(sorted([-self.range_x[0], self.range_x[0]]))
        self.range_y = ensure_tuple(range_y)
        if len(self.range_y) == 1:
            self.range_y = tuple(sorted([-self.range_y[0], self.range_y[0]]))
        self.range_z = ensure_tuple(range_z)
        if len(self.range_z) == 1:
            self.range_z = tuple(sorted([-self.range_z[0], self.range_z[0]]))

        self.keep_size = keep_size
        self.mode: GridSampleMode = look_up_option(mode, GridSampleMode)
        self.padding_mode: GridSamplePadMode = look_up_option(padding_mode, GridSamplePadMode)
        self.align_corners = align_corners
        self.dtype = dtype

        self.x = 0.0
        self.y = 0.0
        self.z = 0.0

    def randomize(self, data: Optional[Any] = None) -> None:
        super().randomize(None)
        self.x = self.R.uniform(low=self.range_x[0], high=self.range_x[1])
        self.y = self.R.uniform(low=self.range_y[0], high=self.range_y[1])
        self.z = self.R.uniform(low=self.range_z[0], high=self.range_z[1])

    def __call__(
        self,
        img: DataObjects.Images,
        mode: Optional[Union[GridSampleMode, str]] = None,
        padding_mode: Optional[Union[GridSamplePadMode, str]] = None,
        align_corners: Optional[bool] = None,
        dtype: DtypeLike = None,
    ) -> DataObjects.Images:
        """
        Args:
            img: channel first array, must have shape 2D: (nchannels, H, W), or 3D: (nchannels, H, W, D).
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``self.padding_mode``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            align_corners: Defaults to ``self.align_corners``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            dtype: data type for resampling computation. Defaults to ``self.dtype``.
                If None, use the data type of input data. To be compatible with other modules,
                the output data type is always ``np.float32``.
        """
        self.randomize()
        if not self._do_transform:
            return img
        rotator = Rotate(
            angle=self.x if img.ndim == 3 else (self.x, self.y, self.z),
            keep_size=self.keep_size,
            mode=look_up_option(mode or self.mode, GridSampleMode),
            padding_mode=look_up_option(padding_mode or self.padding_mode, GridSamplePadMode),
            align_corners=self.align_corners if align_corners is None else align_corners,
            dtype=dtype or self.dtype or img.dtype,  # type: ignore
        )
        return rotator(img)


class RandFlip(TorchTransform, RandomizableTransform):
    """
    Randomly flips the image along axes. Preserves shape.
    See numpy.flip for additional details.
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html

    Args:
        prob: Probability of flipping.
        spatial_axis: Spatial axes along which to flip over. Default is None.
    """

    def __init__(self, prob: float = 0.1, spatial_axis: Optional[Union[Sequence[int], int]] = None) -> None:
        RandomizableTransform.__init__(self, prob)
        self.flipper = Flip(spatial_axis=spatial_axis)

    def __call__(self, img: DataObjects.Images) -> DataObjects.Images:
        """
        Args:
            img: channel first array, must have shape: (num_channels, H[, W, ..., ]),
        """
        self.randomize(None)
        if not self._do_transform:
            return img
        return self.flipper(img)


class RandAxisFlip(TorchTransform, RandomizableTransform):
    """
    Randomly select a spatial axis and flip along it.
    See numpy.flip for additional details.
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html

    Args:
        prob: Probability of flipping.

    """

    def __init__(self, prob: float = 0.1) -> None:
        RandomizableTransform.__init__(self, prob)
        self._axis: Optional[int] = None

    def randomize(self, data: DataObjects.Images) -> None:
        super().randomize(None)
        self._axis = self.R.randint(data.ndim - 1)

    def __call__(self, img: DataObjects.Images) -> DataObjects.Images:
        """
        Args:
            img: channel first array, must have shape: (num_channels, H[, W, ..., ]),
        """
        self.randomize(data=img)
        if not self._do_transform:
            return img
        flipper = Flip(spatial_axis=self._axis)
        return flipper(img)


class RandZoom(TorchTransform, RandomizableTransform):
    """
    Randomly zooms input arrays with given probability within given zoom range.

    Args:
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
        mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
            The interpolation mode. Defaults to ``"area"``.
            See also: https://pytorch.org/docs/stable/nn.functional.html#interpolate
        padding_mode: {``"constant"``, ``"edge``", ``"linear_ramp``", ``"maximum``", ``"mean``", `"median``",
            ``"minimum``", `"reflect``", ``"symmetric``", ``"wrap``", ``"empty``", ``"<function>``"}
            The mode to pad data after zooming.
            See also: https://numpy.org/doc/stable/reference/generated/numpy.pad.html
        align_corners: This only has an effect when mode is
            'linear', 'bilinear', 'bicubic' or 'trilinear'. Default: None.
            See also: https://pytorch.org/docs/stable/nn.functional.html#interpolate
        keep_size: Should keep original size (pad if needed), default is True.
    """

    def __init__(
        self,
        prob: float = 0.1,
        min_zoom: Union[Sequence[float], float] = 0.9,
        max_zoom: Union[Sequence[float], float] = 1.1,
        mode: Union[InterpolateMode, str] = InterpolateMode.AREA,
        padding_mode: Union[NumpyPadMode, str] = NumpyPadMode.EDGE,
        align_corners: Optional[bool] = None,
        keep_size: bool = True,
    ) -> None:
        RandomizableTransform.__init__(self, prob)
        self.min_zoom = ensure_tuple(min_zoom)
        self.max_zoom = ensure_tuple(max_zoom)
        if len(self.min_zoom) != len(self.max_zoom):
            raise AssertionError("min_zoom and max_zoom must have same length.")
        self.mode: InterpolateMode = look_up_option(mode, InterpolateMode)
        self.padding_mode: NumpyPadMode = look_up_option(padding_mode, NumpyPadMode)
        self.align_corners = align_corners
        self.keep_size = keep_size

        self._zoom: Sequence[float] = [1.0]

    def randomize(self, data: Optional[Any] = None) -> None:
        super().randomize(None)
        self._zoom = [self.R.uniform(l, h) for l, h in zip(self.min_zoom, self.max_zoom)]

    def __call__(
        self,
        img: DataObjects.Images,
        mode: Optional[Union[InterpolateMode, str]] = None,
        padding_mode: Optional[Union[NumpyPadMode, str]] = None,
        align_corners: Optional[bool] = None,
    ) -> DataObjects.Images:
        """
        Args:
            img: channel first array, must have shape 2D: (nchannels, H, W), or 3D: (nchannels, H, W, D).
            mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
                The interpolation mode. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#interpolate
            padding_mode: {``"constant"``, ``"edge``", ``"linear_ramp``", ``"maximum``", ``"mean``", `"median``",
                ``"minimum``", `"reflect``", ``"symmetric``", ``"wrap``", ``"empty``", ``"<function>``"}
                The mode to pad data after zooming, default to ``self.padding_mode``.
                See also: https://numpy.org/doc/stable/reference/generated/numpy.pad.html
            align_corners: This only has an effect when mode is
                'linear', 'bilinear', 'bicubic' or 'trilinear'. Defaults to ``self.align_corners``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#interpolate
        """
        # match the spatial image dim
        self.randomize()

        if not self._do_transform:
            return img

        if len(self._zoom) == 1:
            # to keep the spatial shape ratio, use same random zoom factor for all dims
            self._zoom = ensure_tuple_rep(self._zoom[0], img.ndim - 1)
        elif len(self._zoom) == 2 and img.ndim > 3:
            # if 2 zoom factors provided for 3D data, use the first factor for H and W dims, second factor for D dim
            self._zoom = ensure_tuple_rep(self._zoom[0], img.ndim - 2) + ensure_tuple(self._zoom[-1])
        zoomer = Zoom(
            self._zoom,
            keep_size=self.keep_size,
            mode=look_up_option(mode or self.mode, InterpolateMode),
            padding_mode=look_up_option(padding_mode or self.padding_mode, NumpyPadMode),
            align_corners=align_corners or self.align_corners,
        )
        return zoomer(img)


class AffineGrid(TorchTransform):
    """
    Affine transforms on the coordinates.

    Args:
        rotate_params: angle range in radians. rotate_params[0] with be used to generate the 1st rotation
            parameter from `uniform[-rotate_params[0], rotate_params[0])`. Similarly, `rotate_params[1]` and
            `rotate_params[2]` are used in 3D affine for the range of 2nd and 3rd axes.
        shear_params: shear_params[0] with be used to generate the 1st shearing parameter from
            `uniform[-shear_params[0], shear_params[0])`. Similarly, `shear_params[1]` to
            `shear_params[N]` controls the range of the uniform distribution used to generate the 2nd to
            N-th parameter.
        translate_params : translate_params[0] with be used to generate the 1st shift parameter from
            `uniform[-translate_params[0], translate_params[0])`. Similarly, `translate_params[1]`
            to `translate_params[N]` controls the range of the uniform distribution used to generate
            the 2nd to N-th parameter.
        scale_params: scale_params[0] with be used to generate the 1st scaling factor from
            `uniform[-scale_params[0], scale_params[0]) + 1.0`. Similarly, `scale_params[1]` to
            `scale_params[N]` controls the range of the uniform distribution used to generate the 2nd to
            N-th parameter.
        as_tensor_output: whether to output tensor instead of numpy array.
            defaults to True.
        device: device to store the output grid data.
        affine: If applied, ignore the params (`rotate_params`, etc.) and use the
            supplied matrix. Should be square with each side = num of image spatial
            dimensions + 1.

    """

    def __init__(
        self,
        rotate_params: Optional[Union[Sequence[float], float]] = None,
        shear_params: Optional[Union[Sequence[float], float]] = None,
        translate_params: Optional[Union[Sequence[float], float]] = None,
        scale_params: Optional[Union[Sequence[float], float]] = None,
        device: Optional[torch.device] = None,
        affine: Optional[DataObjects.Images] = None,
    ) -> None:
        self.rotate_params = rotate_params
        self.shear_params = shear_params
        self.translate_params = translate_params
        self.scale_params = scale_params
        self.device = device
        self.affine = affine

    def __call__(
        self,
        spatial_size: Optional[Sequence[int]] = None,
        grid: Optional[DataObjects.Images] = None,
    ) -> Tuple[DataObjects.Images, DataObjects.Images]:
        """
        Args:
            spatial_size: output grid size.
            grid: grid to be transformed. Shape must be (3, H, W) for 2D or (4, H, W, D) for 3D.

        Raises:
            ValueError: When ``grid=None`` and ``spatial_size=None``. Incompatible values.

        """
        if grid is None:
            if spatial_size is not None:
                grid = create_grid(spatial_size)
            else:
                raise ValueError("Incompatible values: grid=None and spatial_size=None.")

        affine: DataObjects.Images
        if self.affine is None:
            spatial_dims = len(grid.shape) - 1
            affine = np.eye(spatial_dims + 1)
            if self.rotate_params:
                affine = affine @ create_rotate(spatial_dims, self.rotate_params)
            if self.shear_params:
                affine = affine @ create_shear(spatial_dims, self.shear_params)
            if self.translate_params:
                affine = affine @ create_translate(spatial_dims, self.translate_params)
            if self.scale_params:
                affine = affine @ create_scale(spatial_dims, self.scale_params)
        else:
            affine = self.affine

        grid, orig_type, orig_device = convert_data_type(grid, torch.Tensor, dtype=float, device=self.device)
        affine, *_ = convert_data_type(affine, torch.Tensor, dtype=float, device=grid.device)  # type: ignore

        grid = (affine @ grid.reshape((grid.shape[0], -1))).reshape([-1] + list(grid.shape[1:]))
        if grid is None or not isinstance(grid, torch.Tensor):
            raise ValueError("Unknown grid.")
        grid, *_ = convert_data_type(grid, orig_type, orig_device)
        return grid, affine


class RandAffineGrid(Randomizable, TorchTransform):
    """
    Generate randomised affine grid.
    """

    def __init__(
        self,
        rotate_range: RandRange = None,
        shear_range: RandRange = None,
        translate_range: RandRange = None,
        scale_range: RandRange = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Args:
            rotate_range: angle range in radians. If element `i` is iterable, then
                `uniform[-rotate_range[i][0], rotate_range[i][1])` will be used to generate the rotation parameter
                for the ith dimension. If not, `uniform[-rotate_range[i], rotate_range[i])` will be used. This can
                be altered on a per-dimension basis. E.g., `((0,3), 1, ...)`: for dim0, rotation will be in range
                `[0, 3]`, and for dim1 `[-1, 1]` will be used. Setting a single value will use `[-x, x]` for dim0
                and nothing for the remaining dimensions.
            shear_range: shear_range with format matching `rotate_range`.
            translate_range: translate_range with format matching `rotate_range`.
            scale_range: scaling_range with format matching `rotate_range`. A value of 1.0 is added to the result.
                This allows 0 to correspond to no change (i.e., a scaling of 1).
            device: device to store the output grid data.

        See also:
            - :py:meth:`monai.transforms.utils.create_rotate`
            - :py:meth:`monai.transforms.utils.create_shear`
            - :py:meth:`monai.transforms.utils.create_translate`
            - :py:meth:`monai.transforms.utils.create_scale`
        """
        self.rotate_range = ensure_tuple(rotate_range)
        self.shear_range = ensure_tuple(shear_range)
        self.translate_range = ensure_tuple(translate_range)
        self.scale_range = ensure_tuple(scale_range)

        self.rotate_params: Optional[List[float]] = None
        self.shear_params: Optional[List[float]] = None
        self.translate_params: Optional[List[float]] = None
        self.scale_params: Optional[List[float]] = None

        self.device = device
        self.affine: Optional[DataObjects.Images] = None

    def _get_rand_param(self, param_range, add_scalar: float = 0.0):
        out_param = []
        for f in param_range:
            if issequenceiterable(f):
                if len(f) != 2:
                    raise ValueError("If giving range as [min,max], should only have two elements per dim.")
                out_param.append(self.R.uniform(f[0], f[1]) + add_scalar)
            elif f is not None:
                out_param.append(self.R.uniform(-f, f) + add_scalar)
        return out_param

    def randomize(self, data: Optional[Any] = None) -> None:
        self.rotate_params = self._get_rand_param(self.rotate_range)
        self.shear_params = self._get_rand_param(self.shear_range)
        self.translate_params = self._get_rand_param(self.translate_range)
        self.scale_params = self._get_rand_param(self.scale_range, 1.0)

    def __call__(
        self,
        spatial_size: Optional[Sequence[int]] = None,
        grid: Optional[DataObjects.Images] = None,
    ) -> DataObjects.Images:
        """
        Args:
            spatial_size: output grid size.
            grid: grid to be transformed. Shape must be (3, H, W) for 2D or (4, H, W, D) for 3D.

        Returns:
            a 2D (3xHxW) or 3D (4xHxWxD) grid.
        """
        self.randomize()
        affine_grid = AffineGrid(
            rotate_params=self.rotate_params,
            shear_params=self.shear_params,
            translate_params=self.translate_params,
            scale_params=self.scale_params,
            device=self.device,
        )
        grid, self.affine = affine_grid(spatial_size, grid)
        return grid

    def get_transformation_matrix(self) -> Optional[DataObjects.Images]:
        """Get the most recently applied transformation matrix"""
        return self.affine


class RandDeformGrid(Randomizable, TorchTransform, NumpyTransform):
    """
    Generate random deformation grid.
    """

    def __init__(
        self,
        spacing: Union[Sequence[float], float],
        magnitude_range: Tuple[float, float],
        as_tensor_output: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Args:
            spacing: spacing of the grid in 2D or 3D.
                e.g., spacing=(1, 1) indicates pixel-wise deformation in 2D,
                spacing=(1, 1, 1) indicates voxel-wise deformation in 3D,
                spacing=(2, 2) indicates deformation field defined on every other pixel in 2D.
            magnitude_range: the random offsets will be generated from
                `uniform[magnitude[0], magnitude[1])`.
            as_tensor_output: whether to output tensor instead of numpy array.
                defaults to True.
            device: device to store the output grid data.
        """
        self.spacing = spacing
        self.magnitude = magnitude_range

        self.rand_mag = 1.0
        self.as_tensor_output = as_tensor_output
        self.random_offset: np.ndarray
        self.device = device

    def randomize(self, grid_size: Sequence[int]) -> None:
        self.random_offset = self.R.normal(size=([len(grid_size)] + list(grid_size))).astype(np.float32)
        self.rand_mag = self.R.uniform(self.magnitude[0], self.magnitude[1])

    def __call__(self, spatial_size: Sequence[int]):
        """
        Args:
            spatial_size: spatial size of the grid.
        """
        self.spacing = fall_back_tuple(self.spacing, (1.0,) * len(spatial_size))
        control_grid = create_control_grid(spatial_size, self.spacing)
        self.randomize(control_grid.shape[1:])
        control_grid[: len(spatial_size)] += self.rand_mag * self.random_offset
        if self.as_tensor_output:
            control_grid = torch.as_tensor(np.ascontiguousarray(control_grid), device=self.device)
        return control_grid


class Resample(TorchTransform):
    def __init__(
        self,
        mode: Union[GridSampleMode, str] = GridSampleMode.BILINEAR,
        padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.BORDER,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        computes output image using values from `img`, locations from `grid` using pytorch.
        supports spatially 2D or 3D (num_channels, H, W[, D]).

        Args:
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"border"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            device: device on which the tensor will be allocated.
        """
        self.mode: GridSampleMode = look_up_option(mode, GridSampleMode)
        self.padding_mode: GridSamplePadMode = look_up_option(padding_mode, GridSamplePadMode)
        self.as_tensor_output = as_tensor_output
        self.device = device

    def __call__(
        self,
        img: DataObjects.Images,
        grid: Optional[DataObjects.Images] = None,
        mode: Optional[Union[GridSampleMode, str]] = None,
        padding_mode: Optional[Union[GridSamplePadMode, str]] = None,
    ) -> DataObjects.Images:
        """
        Args:
            img: shape must be (num_channels, H, W[, D]).
            grid: shape must be (3, H, W) for 2D or (4, H, W, D) for 3D.
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``self.padding_mode``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
        """
        if grid is None:
            raise AssertionError("Error, grid argument must be supplied as an ndarray or tensor ")

        img_t: torch.Tensor
        img_t, orig_type, orig_device = convert_data_type(  # type: ignore
            img, torch.Tensor, device=self.device, dtype=torch.float32
        )
        grid, *_ = convert_data_type(deepcopy(grid), torch.Tensor, device=img_t.device, dtype=float)

        if USE_COMPILED:
            for i, dim in enumerate(img_t.shape[1:]):
                grid[i] += (dim - 1.0) / 2.0
            grid = grid[:-1] / grid[-1:]
            grid = grid.permute(list(range(grid.ndimension()))[1:] + [0])
            _padding_mode = look_up_option(
                self.padding_mode if padding_mode is None else padding_mode, GridSamplePadMode
            ).value
            if _padding_mode == "zeros":
                bound = 7
            elif _padding_mode == "border":
                bound = 0
            else:
                bound = 1
            _interp_mode = look_up_option(self.mode if mode is None else mode, GridSampleMode).value
            out = grid_pull(
                img_t.unsqueeze(0).float(),
                grid.unsqueeze(0).float(),
                bound=bound,
                extrapolate=True,
                interpolation=1 if _interp_mode == "bilinear" else _interp_mode,
            )[0]
        else:
            for i, dim in enumerate(img_t.shape[1:]):
                grid[i] = 2.0 * grid[i] / (dim - 1.0)
            grid = grid[:-1] / grid[-1:]
            index_ordering: List[int] = list(range(img_t.ndimension() - 2, -1, -1))
            grid = grid[index_ordering]
            grid = grid.permute(list(range(grid.ndimension()))[1:] + [0])
            out = torch.nn.functional.grid_sample(
                img_t.unsqueeze(0).float(),
                grid.unsqueeze(0).float(),
                mode=self.mode.value if mode is None else GridSampleMode(mode).value,
                padding_mode=self.padding_mode.value if padding_mode is None else GridSamplePadMode(padding_mode).value,
                align_corners=True,
            )[0]

        out, *_ = convert_data_type(out, orig_type, orig_device)
        return out  # type: ignore


class Affine(TorchTransform):
    """
    Transform ``img`` given the affine parameters.
    """

    def __init__(
        self,
        rotate_params: Optional[Union[Sequence[float], float]] = None,
        shear_params: Optional[Union[Sequence[float], float]] = None,
        translate_params: Optional[Union[Sequence[float], float]] = None,
        scale_params: Optional[Union[Sequence[float], float]] = None,
        spatial_size: Optional[Union[Sequence[int], int]] = None,
        mode: Union[GridSampleMode, str] = GridSampleMode.BILINEAR,
        padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.REFLECTION,
        device: Optional[torch.device] = None,
        image_only: bool = False,
    ) -> None:
        """
        The affine transformations are applied in rotate, shear, translate, scale order.

        Args:
            rotate_params: a rotation angle in radians, a scalar for 2D image, a tuple of 3 floats for 3D.
                Defaults to no rotation.
            shear_params: a tuple of 2 floats for 2D, a tuple of 6 floats for 3D. Defaults to no shearing.
            translate_params: a tuple of 2 floats for 2D, a tuple of 3 floats for 3D. Translation is in
                pixel/voxel relative to the center of the input image. Defaults to no translation.
            scale_params: a tuple of 2 floats for 2D, a tuple of 3 floats for 3D. Defaults to no scaling.
            spatial_size: output image spatial size.
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
                if the components of the `spatial_size` are non-positive values, the transform will use the
                corresponding components of img size. For example, `spatial_size=(32, -1)` will be adapted
                to `(32, 64)` if the second spatial dimension size of img is `64`.
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"reflection"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            device: device on which the tensor will be allocated.
            image_only: if True return only the image volume, otherwise return (image, affine).
        """
        self.affine_grid = AffineGrid(
            rotate_params=rotate_params,
            shear_params=shear_params,
            translate_params=translate_params,
            scale_params=scale_params,
            device=device,
        )
        self.image_only = image_only
        self.resampler = Resample(device=device)
        self.spatial_size = spatial_size
        self.mode: GridSampleMode = look_up_option(mode, GridSampleMode)
        self.padding_mode: GridSamplePadMode = look_up_option(padding_mode, GridSamplePadMode)

    def __call__(
        self,
        img: DataObjects.Images,
        spatial_size: Optional[Union[Sequence[int], int]] = None,
        mode: Optional[Union[GridSampleMode, str]] = None,
        padding_mode: Optional[Union[GridSamplePadMode, str]] = None,
    ) -> Union[DataObjects.Images, Tuple[DataObjects.Images, DataObjects.Images]]:
        """
        Args:
            img: shape must be (num_channels, H, W[, D]),
            spatial_size: output image spatial size.
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
                if `img` has two spatial dimensions, `spatial_size` should have 2 elements [h, w].
                if `img` has three spatial dimensions, `spatial_size` should have 3 elements [h, w, d].
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``self.padding_mode``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
        """
        sp_size = fall_back_tuple(spatial_size or self.spatial_size, img.shape[1:])
        grid, affine = self.affine_grid(spatial_size=sp_size)
        ret = self.resampler(img, grid=grid, mode=mode or self.mode, padding_mode=padding_mode or self.padding_mode)

        return ret if self.image_only else (ret, affine)


class RandAffine(RandomizableTransform, TorchTransform):
    """
    Random affine transform.
    """

    def __init__(
        self,
        prob: float = 0.1,
        rotate_range: RandRange = None,
        shear_range: RandRange = None,
        translate_range: RandRange = None,
        scale_range: RandRange = None,
        spatial_size: Optional[Union[Sequence[int], int]] = None,
        mode: Union[GridSampleMode, str] = GridSampleMode.BILINEAR,
        padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.REFLECTION,
        cache_grid: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Args:
            prob: probability of returning a randomized affine grid.
                defaults to 0.1, with 10% chance returns a randomized grid.
            rotate_range: angle range in radians. If element `i` is iterable, then
                `uniform[-rotate_range[i][0], rotate_range[i][1])` will be used to generate the rotation parameter
                for the ith dimension. If not, `uniform[-rotate_range[i], rotate_range[i])` will be used. This can
                be altered on a per-dimension basis. E.g., `((0,3), 1, ...)`: for dim0, rotation will be in range
                `[0, 3]`, and for dim1 `[-1, 1]` will be used. Setting a single value will use `[-x, x]` for dim0
                and nothing for the remaining dimensions.
            shear_range: shear_range with format matching `rotate_range`.
            translate_range: translate_range with format matching `rotate_range`.
            scale_range: scaling_range with format matching `rotate_range`. A value of 1.0 is added to the result.
                This allows 0 to correspond to no change (i.e., a scaling of 1).
            spatial_size: output image spatial size.
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
                if the components of the `spatial_size` are non-positive values, the transform will use the
                corresponding components of img size. For example, `spatial_size=(32, -1)` will be adapted
                to `(32, 64)` if the second spatial dimension size of img is `64`.
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"reflection"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            cache_grid: whether to cache the identity sampling grid.
                If the spatial size is not dynamically defined by input image, enabling this option could
                accelerate the transform.
            device: device on which the tensor will be allocated.

        See also:
            - :py:class:`RandAffineGrid` for the random affine parameters configurations.
            - :py:class:`Affine` for the affine transformation parameters configurations.
        """
        RandomizableTransform.__init__(self, prob)

        self.rand_affine_grid = RandAffineGrid(
            rotate_range=rotate_range,
            shear_range=shear_range,
            translate_range=translate_range,
            scale_range=scale_range,
            device=device,
        )
        self.resampler = Resample(device=device)

        self.spatial_size = spatial_size
        self.cache_grid = cache_grid
        self._cached_grid = self._init_identity_cache()
        self.mode: GridSampleMode = GridSampleMode(mode)
        self.padding_mode: GridSamplePadMode = GridSamplePadMode(padding_mode)

    def _init_identity_cache(self):
        """
        Create cache of the identity grid if cache_grid=True and spatial_size is known.
        """
        if self.spatial_size is None:
            if self.cache_grid:
                warnings.warn(
                    "cache_grid=True is not compatible with the dynamic spatial_size, please specify 'spatial_size'."
                )
            return None
        _sp_size = ensure_tuple(self.spatial_size)
        _ndim = len(_sp_size)
        if _sp_size != fall_back_tuple(_sp_size, [1] * _ndim) or _sp_size != fall_back_tuple(_sp_size, [2] * _ndim):
            # dynamic shape because it falls back to different outcomes
            if self.cache_grid:
                warnings.warn(
                    "cache_grid=True is not compatible with the dynamic spatial_size "
                    f"'spatial_size={self.spatial_size}', please specify 'spatial_size'."
                )
            return None
        return torch.tensor(create_grid(spatial_size=_sp_size)).to(self.rand_affine_grid.device)

    def get_identity_grid(self, spatial_size: Sequence[int]):
        """
        Return a cached or new identity grid depends on the availability.

        Args:
            spatial_size: non-dynamic spatial size
        """
        ndim = len(spatial_size)
        if spatial_size != fall_back_tuple(spatial_size, [1] * ndim) or spatial_size != fall_back_tuple(
            spatial_size, [2] * ndim
        ):
            raise RuntimeError(f"spatial_size should not be dynamic, got {spatial_size}.")
        return create_grid(spatial_size=spatial_size) if self._cached_grid is None else self._cached_grid

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "RandAffine":
        self.rand_affine_grid.set_random_state(seed, state)
        super().set_random_state(seed, state)
        return self

    def randomize(self, data: Optional[Any] = None) -> None:
        super().randomize(None)
        self.rand_affine_grid.randomize()

    def __call__(
        self,
        img: DataObjects.Images,
        spatial_size: Optional[Union[Sequence[int], int]] = None,
        mode: Optional[Union[GridSampleMode, str]] = None,
        padding_mode: Optional[Union[GridSamplePadMode, str]] = None,
    ) -> DataObjects.Images:
        """
        Args:
            img: shape must be (num_channels, H, W[, D]),
            spatial_size: output image spatial size.
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
                if `img` has two spatial dimensions, `spatial_size` should have 2 elements [h, w].
                if `img` has three spatial dimensions, `spatial_size` should have 3 elements [h, w, d].
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``self.padding_mode``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
        """
        self.randomize()
        # if not doing transform and spatial size doesn't change, nothing to do
        # except convert to float and convert numpy/torch
        sp_size = fall_back_tuple(spatial_size or self.spatial_size, img.shape[1:])
        do_resampling = self._do_transform or (sp_size != ensure_tuple(img.shape[1:]))
        if not do_resampling:
            img, *_ = convert_data_type(img, dtype=np.float32)
            return img
        grid = self.get_identity_grid(sp_size)
        if self._do_transform:
            grid = self.rand_affine_grid(grid=grid)
        return self.resampler(
            img=img, grid=grid, mode=mode or self.mode, padding_mode=padding_mode or self.padding_mode
        )


class Rand2DElastic(TorchTransform, RandomizableTransform):
    """
    Random elastic deformation and affine in 2D
    """

    def __init__(
        self,
        spacing: Union[Tuple[float, float], float],
        magnitude_range: Tuple[float, float],
        prob: float = 0.1,
        rotate_range: RandRange = None,
        shear_range: RandRange = None,
        translate_range: RandRange = None,
        scale_range: RandRange = None,
        spatial_size: Optional[Union[Tuple[int, int], int]] = None,
        mode: Union[GridSampleMode, str] = GridSampleMode.BILINEAR,
        padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.REFLECTION,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Args:
            spacing : distance in between the control points.
            magnitude_range: the random offsets will be generated from ``uniform[magnitude[0], magnitude[1])``.
            prob: probability of returning a randomized elastic transform.
                defaults to 0.1, with 10% chance returns a randomized elastic transform,
                otherwise returns a ``spatial_size`` centered area extracted from the input image.
            rotate_range: angle range in radians. If element `i` is iterable, then
                `uniform[-rotate_range[i][0], rotate_range[i][1])` will be used to generate the rotation parameter
                for the ith dimension. If not, `uniform[-rotate_range[i], rotate_range[i])` will be used. This can
                be altered on a per-dimension basis. E.g., `((0,3), 1, ...)`: for dim0, rotation will be in range
                `[0, 3]`, and for dim1 `[-1, 1]` will be used. Setting a single value will use `[-x, x]` for dim0
                and nothing for the remaining dimensions.
            shear_range: shear_range with format matching `rotate_range`.
            translate_range: translate_range with format matching `rotate_range`.
            scale_range: scaling_range with format matching `rotate_range`. A value of 1.0 is added to the result.
                This allows 0 to correspond to no change (i.e., a scaling of 1).
            spatial_size: specifying output image spatial size [h, w].
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
                if the components of the `spatial_size` are non-positive values, the transform will use the
                corresponding components of img size. For example, `spatial_size=(32, -1)` will be adapted
                to `(32, 64)` if the second spatial dimension size of img is `64`.
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"reflection"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            device: device on which the tensor will be allocated.

        See also:
            - :py:class:`RandAffineGrid` for the random affine parameters configurations.
            - :py:class:`Affine` for the affine transformation parameters configurations.
        """
        RandomizableTransform.__init__(self, prob)
        self.deform_grid = RandDeformGrid(
            spacing=spacing, magnitude_range=magnitude_range, as_tensor_output=True, device=device
        )
        self.rand_affine_grid = RandAffineGrid(
            rotate_range=rotate_range,
            shear_range=shear_range,
            translate_range=translate_range,
            scale_range=scale_range,
            device=device,
        )
        self.resampler = Resample(device=device)

        self.spatial_size = spatial_size
        self.mode: GridSampleMode = look_up_option(mode, GridSampleMode)
        self.padding_mode: GridSamplePadMode = look_up_option(padding_mode, GridSamplePadMode)

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "Rand2DElastic":
        self.deform_grid.set_random_state(seed, state)
        self.rand_affine_grid.set_random_state(seed, state)
        super().set_random_state(seed, state)
        return self

    def randomize(self, spatial_size: Sequence[int]) -> None:
        super().randomize(None)
        self.deform_grid.randomize(spatial_size)
        self.rand_affine_grid.randomize()

    def __call__(
        self,
        img: DataObjects.Images,
        spatial_size: Optional[Union[Tuple[int, int], int]] = None,
        mode: Optional[Union[GridSampleMode, str]] = None,
        padding_mode: Optional[Union[GridSamplePadMode, str]] = None,
    ) -> DataObjects.Images:
        """
        Args:
            img: shape must be (num_channels, H, W),
            spatial_size: specifying output image spatial size [h, w].
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``self.padding_mode``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
        """
        sp_size = fall_back_tuple(spatial_size or self.spatial_size, img.shape[1:])
        self.randomize(spatial_size=sp_size)
        if self._do_transform:
            grid = self.deform_grid(spatial_size=sp_size)
            grid = self.rand_affine_grid(grid=grid)
            grid = torch.nn.functional.interpolate(  # type: ignore
                recompute_scale_factor=True,
                input=torch.as_tensor(grid).unsqueeze(0),
                scale_factor=list(ensure_tuple(self.deform_grid.spacing)),
                mode=InterpolateMode.BICUBIC.value,
                align_corners=False,
            )
            grid = CenterSpatialCrop(roi_size=sp_size)(grid[0])
        else:
            grid = create_grid(spatial_size=sp_size)
        return self.resampler(img, grid, mode=mode or self.mode, padding_mode=padding_mode or self.padding_mode)


class Rand3DElastic(TorchTransform, RandomizableTransform):
    """
    Random elastic deformation and affine in 3D
    """

    def __init__(
        self,
        sigma_range: Tuple[float, float],
        magnitude_range: Tuple[float, float],
        prob: float = 0.1,
        rotate_range: RandRange = None,
        shear_range: RandRange = None,
        translate_range: RandRange = None,
        scale_range: RandRange = None,
        spatial_size: Optional[Union[Tuple[int, int, int], int]] = None,
        mode: Union[GridSampleMode, str] = GridSampleMode.BILINEAR,
        padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.REFLECTION,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Args:
            sigma_range: a Gaussian kernel with standard deviation sampled from
                ``uniform[sigma_range[0], sigma_range[1])`` will be used to smooth the random offset grid.
            magnitude_range: the random offsets on the grid will be generated from
                ``uniform[magnitude[0], magnitude[1])``.
            prob: probability of returning a randomized elastic transform.
                defaults to 0.1, with 10% chance returns a randomized elastic transform,
                otherwise returns a ``spatial_size`` centered area extracted from the input image.
            rotate_range: angle range in radians. If element `i` is iterable, then
                `uniform[-rotate_range[i][0], rotate_range[i][1])` will be used to generate the rotation parameter
                for the ith dimension. If not, `uniform[-rotate_range[i], rotate_range[i])` will be used. This can
                be altered on a per-dimension basis. E.g., `((0,3), 1, ...)`: for dim0, rotation will be in range
                `[0, 3]`, and for dim1 `[-1, 1]` will be used. Setting a single value will use `[-x, x]` for dim0
                and nothing for the remaining dimensions.
            shear_range: shear_range with format matching `rotate_range`.
            translate_range: translate_range with format matching `rotate_range`.
            scale_range: scaling_range with format matching `rotate_range`. A value of 1.0 is added to the result.
                This allows 0 to correspond to no change (i.e., a scaling of 1).
            spatial_size: specifying output image spatial size [h, w, d].
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
                if the components of the `spatial_size` are non-positive values, the transform will use the
                corresponding components of img size. For example, `spatial_size=(32, 32, -1)` will be adapted
                to `(32, 32, 64)` if the third spatial dimension size of img is `64`.
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"reflection"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            device: device on which the tensor will be allocated.

        See also:
            - :py:class:`RandAffineGrid` for the random affine parameters configurations.
            - :py:class:`Affine` for the affine transformation parameters configurations.
        """
        RandomizableTransform.__init__(self, prob)
        self.rand_affine_grid = RandAffineGrid(rotate_range, shear_range, translate_range, scale_range, device)
        self.resampler = Resample(device=device)

        self.sigma_range = sigma_range
        self.magnitude_range = magnitude_range
        self.spatial_size = spatial_size
        self.mode: GridSampleMode = look_up_option(mode, GridSampleMode)
        self.padding_mode: GridSamplePadMode = look_up_option(padding_mode, GridSamplePadMode)
        self.device = device

        self.rand_offset: np.ndarray
        self.magnitude = 1.0
        self.sigma = 1.0

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "Rand3DElastic":
        self.rand_affine_grid.set_random_state(seed, state)
        super().set_random_state(seed, state)
        return self

    def randomize(self, grid_size: Sequence[int]) -> None:
        super().randomize(None)
        if self._do_transform:
            self.rand_offset = self.R.uniform(-1.0, 1.0, [3] + list(grid_size)).astype(np.float32)
        self.magnitude = self.R.uniform(self.magnitude_range[0], self.magnitude_range[1])
        self.sigma = self.R.uniform(self.sigma_range[0], self.sigma_range[1])
        self.rand_affine_grid.randomize()

    def __call__(
        self,
        img: DataObjects.Images,
        spatial_size: Optional[Union[Tuple[int, int, int], int]] = None,
        mode: Optional[Union[GridSampleMode, str]] = None,
        padding_mode: Optional[Union[GridSamplePadMode, str]] = None,
    ) -> DataObjects.Images:
        """
        Args:
            img: shape must be (num_channels, H, W, D),
            spatial_size: specifying spatial 3D output image spatial size [h, w, d].
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``self.padding_mode``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
        """
        sp_size = fall_back_tuple(spatial_size or self.spatial_size, img.shape[1:])
        self.randomize(grid_size=sp_size)
        grid = create_grid(spatial_size=sp_size)
        if self._do_transform:
            if self.rand_offset is None:
                raise AssertionError
            grid, *_ = convert_data_type(grid, torch.Tensor, device=self.device)
            offset, *_ = convert_data_type(self.rand_offset, torch.Tensor, device=self.device)
            offset = offset.unsqueeze(0)  # type: ignore
            gaussian = GaussianFilter(3, self.sigma, 3.0).to(device=self.device)
            grid[:3] += gaussian(offset)[0] * self.magnitude
            grid = self.rand_affine_grid(grid=grid)
        return self.resampler(img, grid, mode=mode or self.mode, padding_mode=padding_mode or self.padding_mode)


class AddCoordinateChannels(TorchTransform, NumpyTransform):
    """
    Appends additional channels encoding coordinates of the input. Useful when e.g. training using patch-based sampling,
    to allow feeding of the patch's location into the network.

    This can be seen as a input-only version of CoordConv:

    Liu, R. et al. An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution, NeurIPS 2018.
    """

    def __init__(
        self,
        spatial_channels: Sequence[int],
    ) -> None:
        """
        Args:
            spatial_channels: the spatial dimensions that are to have their coordinates encoded in a channel and
                appended to the input. E.g., `(1,2,3)` will append three channels to the input, encoding the
                coordinates of the input's three spatial dimensions (0 is reserved for the channel dimension).
        """
        self.spatial_channels = spatial_channels

    def __call__(self, img: DataObjects.Images) -> DataObjects.Images:
        """
        Args:
            img: data to be transformed, assuming `img` is channel first.
        """
        if max(self.spatial_channels) > img.ndim - 1:
            raise ValueError(
                f"input has {img.ndim-1} spatial dimensions, cannot add AddCoordinateChannels channel for "
                f"dim {max(self.spatial_channels)}."
            )
        if 0 in self.spatial_channels:
            raise ValueError("cannot add AddCoordinateChannels channel for dimension 0, as 0 is channel dim.")

        spatial_dims = img.shape[1:]
        coord_channels = np.meshgrid(*tuple(np.linspace(-0.5, 0.5, s) for s in spatial_dims), indexing="ij")
        coord_channels, *_ = convert_data_type(
            coord_channels, type(img), device=img.device if isinstance(img, torch.Tensor) else None
        )
        # only keep required dimensions. need to subtract 1 since im will be 0-based
        # but user input is 1-based (because channel dim is 0)
        coord_channels = coord_channels[[s - 1 for s in self.spatial_channels]]
        if isinstance(img, torch.Tensor):
            out = torch.cat((img, coord_channels), dim=0)
        else:
            out = np.concatenate((img, coord_channels), axis=0)
        return out
