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
A collection of "vanilla" transforms for spatial operations
https://github.com/Project-MONAI/MONAI/wiki/MONAI_Design
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import torch

import monai
from monai.data.meta_obj import get_track_meta
from monai.data.meta_tensor import MetaTensor
from monai.data.utils import AFFINE_TOL, compute_shape_offset, to_affine_nd
from monai.networks.layers import AffineTransform
from monai.networks.utils import normalize_transform
from monai.transforms.inverse import TraceableTransform
from monai.transforms.utils import create_scale
from monai.transforms.utils_pytorch_numpy_unification import allclose
from monai.utils import (
    TraceKeys,
    convert_to_dst_type,
    convert_to_numpy,
    convert_to_tensor,
    ensure_tuple,
    fall_back_tuple,
    optional_import,
)

nib, has_nib = optional_import("nibabel")
cupy, _ = optional_import("cupy")
cupy_ndi, _ = optional_import("cupyx.scipy.ndimage")
np_ndi, _ = optional_import("scipy.ndimage")

__all__ = ["spatial_resample", "orientation", "flip"]


def spatial_resample(
    img, dst_affine, spatial_size, mode, padding_mode, align_corners, dtype_pt, transform_info
) -> torch.Tensor:
    """
    Functional implementation of resampling the input image to the specified ``dst_affine`` matrix and ``spatial_size``.
    This function operates eagerly or lazily according to
    ``transform_info[TraceKeys.LAZY_EVALUATION]`` (default ``False``).

    Args:
        img: data to be resampled, assuming `img` is channel-first.
        dst_affine: target affine matrix, if None, use the input affine matrix, effectively no resampling.
        spatial_size: output spatial size, if the component is ``-1``, use the corresponding input spatial size.
        mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
            Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
            and the value represents the order of the spline interpolation.
            See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
        padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
            Padding mode for outside grid values. Defaults to ``"border"``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            When `mode` is an integer, using numpy/cupy backends, this argument accepts
            {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
            See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
        align_corners: Geometrically, we consider the pixels of the input as squares rather than points.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            Defaults to ``None``, effectively using the value of `self.align_corners`.
        dtype_pt: data `dtype` for resampling computation.
        transform_info: a dictionary with the relevant information pertaining to an applied transform.
    """
    original_spatial_shape = img.peek_pending_shape() if isinstance(img, MetaTensor) else img.shape[1:]
    src_affine: torch.Tensor = img.peek_pending_affine() if isinstance(img, MetaTensor) else torch.eye(4)
    img = convert_to_tensor(data=img, track_meta=get_track_meta())
    # ensure spatial rank is <= 3
    spatial_rank = min(len(img.shape) - 1, src_affine.shape[0] - 1, 3)
    if (not isinstance(spatial_size, int) or spatial_size != -1) and spatial_size is not None:
        spatial_rank = min(len(ensure_tuple(spatial_size)), 3)  # infer spatial rank based on spatial_size
    src_affine = to_affine_nd(spatial_rank, src_affine).to(torch.float64)
    dst_affine = to_affine_nd(spatial_rank, dst_affine) if dst_affine is not None else src_affine
    dst_affine = convert_to_dst_type(dst_affine, src_affine)[0]
    if not isinstance(dst_affine, torch.Tensor):
        raise ValueError(f"dst_affine should be a torch.Tensor, got {type(dst_affine)}")

    in_spatial_size = torch.tensor(original_spatial_shape[:spatial_rank])
    if isinstance(spatial_size, int) and (spatial_size == -1):  # using the input spatial size
        spatial_size = in_spatial_size
    elif spatial_size is None and spatial_rank > 1:  # auto spatial size
        spatial_size, _ = compute_shape_offset(in_spatial_size, src_affine, dst_affine)  # type: ignore
    spatial_size = torch.tensor(
        fall_back_tuple(ensure_tuple(spatial_size)[:spatial_rank], in_spatial_size, lambda x: x >= 0)
    )
    extra_info = {
        "dtype": str(dtype_pt)[6:],  # remove "torch": torch.float32 -> float32
        "mode": mode.value if isinstance(mode, Enum) else mode,
        "padding_mode": padding_mode.value if isinstance(padding_mode, Enum) else padding_mode,
        "align_corners": align_corners if align_corners is not None else TraceKeys.NONE,
        "src_affine": src_affine,
    }
    try:
        _s = convert_to_numpy(src_affine)
        _d = convert_to_numpy(dst_affine)
        xform = np.eye(spatial_rank + 1) if spatial_rank < 2 else np.linalg.solve(_s, _d)
    except (np.linalg.LinAlgError, RuntimeError) as e:
        raise ValueError(f"src affine is not invertible {_s}, {_d}.") from e
    xform = convert_to_tensor(to_affine_nd(spatial_rank, xform)).to(device=img.device, dtype=torch.float64)
    affine_unchanged = (
        allclose(src_affine, dst_affine, atol=AFFINE_TOL) and allclose(spatial_size, in_spatial_size)
    ) or (allclose(xform, np.eye(len(xform)), atol=AFFINE_TOL) and allclose(spatial_size, in_spatial_size))
    lazy_evaluation = transform_info.get(TraceKeys.LAZY_EVALUATION, False)
    meta_info = TraceableTransform.track_transform_meta(
        img,
        sp_size=spatial_size,
        affine=None if affine_unchanged and not lazy_evaluation else xform,
        extra_info=extra_info,
        orig_size=original_spatial_shape,
        transform_info=transform_info,
        lazy_evaluation=lazy_evaluation,
    )
    # drop current meta first since line 102 is a shallow copy
    img = img.as_tensor() if isinstance(img, MetaTensor) else img
    if affine_unchanged or lazy_evaluation:
        # no significant change or lazy change, return original image
        out = convert_to_tensor(img, track_meta=get_track_meta())
        return out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else meta_info  # type: ignore
    im_size = list(img.shape)
    chns, in_sp_size, additional_dims = im_size[0], im_size[1 : spatial_rank + 1], im_size[spatial_rank + 1 :]

    if additional_dims:
        xform_shape = [-1] + in_sp_size
        img = img.reshape(xform_shape)
    img = img.to(dtype_pt)
    if isinstance(mode, int):
        dst_xform_1 = normalize_transform(spatial_size, "cpu", xform.dtype, True, True)[0].numpy()  # to (-1, 1)
        if not align_corners:
            norm = create_scale(spatial_rank, [(max(d, 2) - 1) / d for d in spatial_size])
            dst_xform_1 = norm.astype(float) @ dst_xform_1  # type: ignore # scaling (num_step - 1) / num_step
        dst_xform_d = normalize_transform(spatial_size, "cpu", xform.dtype, align_corners, False)[0].numpy()
        xform @= convert_to_dst_type(np.linalg.solve(dst_xform_d, dst_xform_1), xform)[0]
        affine_xform = monai.transforms.Affine(
            affine=xform, spatial_size=spatial_size, normalized=True, image_only=True, dtype=dtype_pt
        )
        with affine_xform.trace_transform(False):
            img = affine_xform(img, mode=mode, padding_mode=padding_mode)
    else:
        affine_xform = AffineTransform(  # type: ignore
            normalized=False, mode=mode, padding_mode=padding_mode, align_corners=align_corners, reverse_indexing=True
        )
        img = affine_xform(img.unsqueeze(0), theta=xform.to(img), spatial_size=spatial_size).squeeze(0)  # type: ignore
    if additional_dims:
        full_shape = (chns, *spatial_size, *additional_dims)
        img = img.reshape(full_shape)
    out = convert_to_tensor(img, track_meta=get_track_meta(), dtype=torch.float32)
    return out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else out  # type: ignore


def orientation(img, original_affine, spatial_ornt, transform_info):
    """
    Functional implementation of changing the input image's orientation into the specified based on `spatial_ornt`.
    This function operates eagerly or lazily according to
    ``transform_info[TraceKeys.LAZY_EVALUATION]`` (default ``False``).

    Args:
        img: data to be changed, assuming `img` is channel-first.
        original_affine: original affine of the input image.
        spatial_ornt: orientation.
        transform_info: a dictionary with the relevant information pertaining to an applied transform.
    """
    spatial_shape = img.peek_pending_shape() if isinstance(img, MetaTensor) else img.shape[1:]
    xform = nib.orientations.inv_ornt_aff(spatial_ornt, spatial_shape)
    img = convert_to_tensor(img, track_meta=get_track_meta())

    spatial_ornt[:, 0] += 1  # skip channel dim
    spatial_ornt = np.concatenate([np.array([[0, 1]]), spatial_ornt])
    axes = [ax for ax, flip in enumerate(spatial_ornt[:, 1]) if flip == -1]
    full_transpose = np.arange(len(spatial_shape) + 1)  # channel-first array
    full_transpose[: len(spatial_ornt)] = np.argsort(spatial_ornt[:, 0])
    extra_info = {"original_affine": original_affine}

    shape_np = convert_to_numpy(spatial_shape, wrap_sequence=True)
    shape_np = shape_np[[i - 1 for i in full_transpose if i > 0]]
    meta_info = TraceableTransform.track_transform_meta(
        img,
        sp_size=shape_np,
        affine=xform,
        extra_info=extra_info,
        orig_size=spatial_shape,
        transform_info=transform_info,
        lazy_evaluation=transform_info.get(TraceKeys.LAZY_EVALUATION, False),
    )
    out = convert_to_tensor(img.as_tensor() if isinstance(img, MetaTensor) else img, track_meta=get_track_meta())
    if transform_info.get(TraceKeys.LAZY_EVALUATION, False):
        return out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else meta_info
    if axes:
        out = torch.flip(out, dims=axes)
    if not np.all(full_transpose == np.arange(len(out.shape))):
        out = out.permute(full_transpose.tolist())
    return out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else out


def flip(img, sp_axes, transform_info):
    """
    Functional implementation of flip.
    This function operates eagerly or lazily according to
    ``transform_info[TraceKeys.LAZY_EVALUATION]`` (default ``False``).

    Args:
        img: data to be changed, assuming `img` is channel-first.
        sp_axes: spatial axes along which to flip over.
        transform_info: a dictionary with the relevant information pertaining to an applied transform.
    """
    sp_size = img.peek_pending_shape() if isinstance(img, MetaTensor) else img.shape[1:]
    sp_size = convert_to_numpy(sp_size, wrap_sequence=True).tolist()
    extra_info = {"axes": sp_axes}  # track the spatial axes
    axes = monai.transforms.utils.map_spatial_axes(img.ndim, sp_axes)  # use the axes with channel dim
    rank = img.peek_pending_rank() if isinstance(img, MetaTensor) else torch.tensor(3.0, dtype=torch.double)
    # axes include the channel dim
    xform = torch.eye(int(rank) + 1, dtype=torch.double)
    for axis in axes:
        sp = axis - 1
        xform[sp, sp], xform[sp, -1] = xform[sp, sp] * -1, sp_size[sp] - 1
    meta_info = TraceableTransform.track_transform_meta(
        img,
        sp_size=sp_size,
        affine=xform,
        extra_info=extra_info,
        transform_info=transform_info,
        lazy_evaluation=transform_info.get(TraceKeys.LAZY_EVALUATION, False),
    )
    out = convert_to_tensor(img.as_tensor() if isinstance(img, MetaTensor) else img, track_meta=get_track_meta())
    if transform_info.get(TraceKeys.LAZY_EVALUATION, False):
        return out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else meta_info
    out = torch.flip(out, axes)
    return out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else out
