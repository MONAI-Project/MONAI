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

import numpy as np
import torch

import monai
from monai.config import NdarrayOrTensor
from monai.utils import LazyAttr, convert_to_numpy, convert_to_tensor

__all__ = ["resample", "combine_transforms"]


class Affine:
    """A class to represent an affine transform matrix."""

    __slots__ = ("data",)

    def __init__(self, data):
        self.data = data

    @staticmethod
    def is_affine_shaped(data):
        """Check if the data is an affine matrix."""
        if isinstance(data, Affine):
            return True
        if isinstance(data, DisplacementField):
            return False
        if not hasattr(data, "shape") or len(data.shape) < 2:
            return False
        return data.shape[-1] in (3, 4) and data.shape[-1] == data.shape[-2]


class DisplacementField:
    """A class to represent a dense displacement field."""

    __slots__ = ("data",)

    def __init__(self, data):
        self.data = data

    @staticmethod
    def is_ddf_shaped(data):
        """Check if the data is a DDF."""
        if isinstance(data, DisplacementField):
            return True
        if isinstance(data, Affine):
            return False
        if not hasattr(data, "shape") or len(data.shape) < 3:
            return False
        return not Affine.is_affine_shaped(data)


def combine_transforms(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Given transforms A and B to be applied to x, return the combined transform (AB), so that A(B(x)) becomes AB(x)"""
    if Affine.is_affine_shaped(left) and Affine.is_affine_shaped(right):  # linear transforms
        left = convert_to_tensor(left.data if isinstance(left, Affine) else left, wrap_sequence=True)
        right = convert_to_tensor(right.data if isinstance(right, Affine) else right, wrap_sequence=True)
        return torch.matmul(left, right)
    if DisplacementField.is_ddf_shaped(left) and DisplacementField.is_ddf_shaped(
        right
    ):  # adds DDFs, do we need metadata if metatensor input?
        left = convert_to_tensor(left.data if isinstance(left, DisplacementField) else left, wrap_sequence=True)
        right = convert_to_tensor(right.data if isinstance(right, DisplacementField) else right, wrap_sequence=True)
        return left + right
    raise NotImplementedError


def affine_from_pending(pending_item):
    """Extract the affine matrix from a pending transform item."""
    if isinstance(pending_item, (torch.Tensor, np.ndarray)):
        return pending_item
    if isinstance(pending_item, dict):
        return pending_item[LazyAttr.AFFINE]
    return pending_item


def kwargs_from_pending(pending_item):
    """Extract kwargs from a pending transform item."""
    if not isinstance(pending_item, dict):
        return {}
    ret = {
        LazyAttr.INTERP_MODE: pending_item.get(LazyAttr.INTERP_MODE, None),  # interpolation mode
        LazyAttr.PADDING_MODE: pending_item.get(LazyAttr.PADDING_MODE, None),  # padding mode
    }
    if LazyAttr.SHAPE in pending_item:
        ret[LazyAttr.SHAPE] = pending_item[LazyAttr.SHAPE]
    if LazyAttr.DTYPE in pending_item:
        ret[LazyAttr.DTYPE] = pending_item[LazyAttr.DTYPE]
    return ret


def is_compatible_apply_kwargs(kwargs_1, kwargs_2):
    """Check if two sets of kwargs are compatible (to be combined in `apply`)."""
    return True


def require_interp(matrix, atol=1e-5):
    """
    returns None if the affine matrix suggests interpolation
    otherwise returns axes information about simple axes flipping/transposing/integer translation.
    if the affine matrices match these conditions, the resampling can be achieved by simple array operations
    such as flip/permute/pad_nd/slice
    """
    s = matrix[:, -1]
    if not np.allclose(s, np.round(s), atol=atol):
        return None

    ndim = len(matrix) - 1
    mat = convert_to_numpy(matrix)
    ox, oy = [], [0]
    for x, r in enumerate(mat[:ndim, :ndim]):
        for y, c in enumerate(r):
            if np.isclose(c, -1, atol=atol) or np.isclose(c, 1, atol=atol):
                y_channel = y + 1
                if x in ox or y_channel in oy:
                    return None
                else:
                    ox.append(x)
                    oy.append(y_channel)
            elif not np.isclose(c, 0.0, atol=atol):
                return None
    return oy


def resample(data: torch.Tensor, matrix: NdarrayOrTensor, spatial_size, kwargs: dict | None = None):
    """
    This is a minimal implementation of resample that always uses SpatialResample.
    `kwargs` supports "lazy_dtype", "lazy_padding_mode", "lazy_interpolation_mode", "lazy_dtype", "lazy_align_corners".

    See Also:
        :py:class:`monai.transforms.SpatialResample`
    """
    if not Affine.is_affine_shaped(matrix):
        raise NotImplementedError("calling dense grid resample API not implemented")
    kwargs = {} if kwargs is None else kwargs
    init_kwargs = {
        "dtype": kwargs.pop(LazyAttr.DTYPE, data.dtype),
        "align_corners": kwargs.pop(LazyAttr.ALIGN_CORNERS, None),
    }

    ndim = len(matrix) - 1
    img = convert_to_tensor(data=data, track_meta=monai.data.get_track_meta())
    init_affine = monai.data.to_affine_nd(ndim, img.affine)
    call_kwargs = {
        "spatial_size": img.peek_pending_shape() if spatial_size is None else spatial_size,
        "dst_affine": init_affine @ monai.utils.convert_to_dst_type(matrix, init_affine)[0],
        "mode": kwargs.pop(LazyAttr.INTERP_MODE, None),
        "padding_mode": kwargs.pop(LazyAttr.PADDING_MODE, None),
    }

    matrix_np = convert_to_numpy(matrix, wrap_sequence=True).copy()
    axes = require_interp(matrix_np)
    if axes is not None:
        # todo: if no change just return the array
        # todo: if on cpu, use the numpy array because flip is faster
        matrix_np = np.round(matrix_np)
        full_transpose = np.argsort(axes).tolist()
        if not np.all(full_transpose == np.arange(len(img.shape))):
            img = img.permute(full_transpose)
        matrix_np[:ndim] = matrix_np[[x - 1 for x in axes[1:]]]
        flip = [idx + 1 for idx, val in enumerate(matrix_np[:ndim]) if val[idx] == -1]
        if flip:
            img = torch.flip(img, dims=flip)
            for f in flip:
                ind_f = f - 1
                matrix_np[ind_f, ind_f] = 1
                matrix_np[ind_f, -1] = img.shape[f] - 1 - matrix_np[ind_f, -1]

        cc = np.asarray(np.meshgrid(*[[0.5, x - 0.5] for x in spatial_size], indexing="ij"))
        cc = cc.reshape((len(spatial_size), -1))
        src_cc = np.floor(matrix_np @ np.concatenate((cc, np.ones_like(cc[:1]))))
        src_start, src_end = src_cc.min(axis=1), src_cc.max(axis=1)
        to_pad, to_crop, do_pad, do_crop = [(0, 0)], [slice(None)], False, False
        for s, e, sp in zip(src_start, src_end, img.shape[1:]):
            do_pad, do_crop = do_pad or s < 0 or e > sp - 1, do_crop or s > 0 or e < sp - 1
            to_pad += [(0 if s >= 0 else int(-s), 0 if e < sp - 1 else int(e - sp + 1))]
            to_crop += [slice(int(max(s, 0)), int(e + 1 + to_pad[-1][0]))]
        if do_pad:
            p_mode = kwargs.pop(LazyAttr.PADDING_MODE, None)
            if p_mode is None or p_mode in ("zeros", "constant"):
                _mode = "constant"
            elif p_mode in ("reflection", "reflect", "grid_mirror", "mirror"):
                _mode = "reflect"
            elif p_mode in ("nearest", "border"):
                _mode = "replicate"
            else:
                _mode = "circular"
            img = monai.transforms.croppad.functional.pad_nd(img, to_pad, mode=_mode)  # todo set padding mode
        if do_crop:
            img = img[to_crop]
        img.affine = call_kwargs["dst_affine"]
        return img

    resampler = monai.transforms.SpatialResample(**init_kwargs)
    resampler.lazy_evaluation = False  # resampler is a lazytransform
    with resampler.trace_transform(False):  # don't track this transform in `img`
        new_img = resampler(img=img, **call_kwargs)
    return new_img
