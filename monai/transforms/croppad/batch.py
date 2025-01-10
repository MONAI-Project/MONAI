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
A collection of "vanilla" transforms for crop and pad operations acting on batches of data.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import Any

import numpy as np
import torch

from monai.data.meta_tensor import MetaTensor
from monai.data.utils import list_data_collate
from monai.transforms.croppad.array import CenterSpatialCrop, SpatialPad
from monai.transforms.inverse import InvertibleTransform
from monai.utils.enums import Method, PytorchPadMode, TraceKeys, DataCollateMode

__all__ = ["PadListDataCollate", "PadOrCropListDataCollate"]


def replace_element(to_replace, batch, idx, key_or_idx):
    # since tuple is immutable we'll have to recreate
    if isinstance(batch[idx], tuple):
        batch_idx_list = list(batch[idx])
        batch_idx_list[key_or_idx] = to_replace
        batch[idx] = tuple(batch_idx_list)
    # else, replace
    else:
        batch[idx][key_or_idx] = to_replace
    return batch


class PadListDataCollate(InvertibleTransform):
    """
    Same as MONAI's ``list_data_collate``, except any tensors are centrally padded to match the shape of the biggest
    tensor in each dimension. This transform is useful if some of the applied transforms generate batch data of
    different sizes.

    This can be used on both list and dictionary data.
    Note that in the case of the dictionary data, it may add the transform information to the list of invertible transforms
    if input batch have different spatial shape, so need to call static method: `inverse` before inverting other transforms.

    Note that normally, a user won't explicitly use the `__call__` method. Rather this would be passed to the `DataLoader`.
    This means that `__call__` handles data as it comes out of a `DataLoader`, containing batch dimension. However, the
    `inverse` operates on dictionaries containing images of shape `C,H,W,[D]`. This asymmetry is necessary so that we can
    pass the inverse through multiprocessing.

    Args:
        method: padding method (see :py:class:`monai.transforms.SpatialPad`)
        mode: padding mode (see :py:class:`monai.transforms.SpatialPad`)
        kwargs: other arguments for the `np.pad` or `torch.pad` function.
            note that `np.pad` treats channel dimension as the first dimension.

    """

    def __init__(self, method: str = Method.SYMMETRIC, mode: str = PytorchPadMode.CONSTANT, **kwargs) -> None:
        self.method = method
        self.mode = mode
        self.kwargs = kwargs

    def __call__(self, batch: Any):
        """
        Args:
            batch: batch of data to pad-collate
        """
        # data is either list of dicts or list of lists
        is_list_of_dicts = isinstance(batch[0], dict)
        # loop over items inside of each element in a batch
        batch_item = tuple(batch[0].keys()) if is_list_of_dicts else range(len(batch[0]))
        for key_or_idx in batch_item:
            # calculate max size of each dimension
            max_shapes = []
            for elem in batch:
                if not isinstance(elem[key_or_idx], (torch.Tensor, np.ndarray)):
                    break
                max_shapes.append(elem[key_or_idx].shape[1:])
            # len > 0 if objects were arrays, else skip as no padding to be done
            if not max_shapes:
                continue
            max_shape = np.array(max_shapes).max(axis=0)
            # If all same size, skip
            if np.all(np.array(max_shapes).min(axis=0) == max_shape):
                continue

            # Use `SpatialPad` to match sizes, Default params are central padding, padding with 0's
            padder = SpatialPad(spatial_size=max_shape, method=self.method, mode=self.mode, **self.kwargs)
            for idx, batch_i in enumerate(batch):
                orig_size = batch_i[key_or_idx].shape[1:]
                padded = padder(batch_i[key_or_idx])
                batch = replace_element(padded, batch, idx, key_or_idx)

                # If we have a dictionary of data, append to list
                # padder transform info is re-added with self.push_transform to ensure one info dict per transform.
                if is_list_of_dicts:
                    self.push_transform(
                        batch[idx],
                        key_or_idx,
                        orig_size=orig_size,
                        extra_info=self.pop_transform(batch[idx], key_or_idx, check=False),
                    )

        # After padding, use default list collator
        return list_data_collate(batch)

    @staticmethod
    def inverse(data: dict) -> dict[Hashable, np.ndarray]:
        if not isinstance(data, Mapping):
            raise RuntimeError(f"Inverse can only currently be applied on dictionaries, got type {type(data)}.")

        d = dict(data)
        for key in d:
            transforms = None
            if isinstance(d[key], MetaTensor):
                transforms = d[key].applied_operations
            else:
                transform_key = InvertibleTransform.trace_key(key)
                if transform_key in d:
                    transforms = d[transform_key]
            if not transforms or not isinstance(transforms[-1], dict):
                continue
            if transforms[-1].get(TraceKeys.CLASS_NAME) == PadListDataCollate.__name__:
                xform = transforms.pop()
                cropping = CenterSpatialCrop(xform.get(TraceKeys.ORIG_SIZE, -1))
                with cropping.trace_transform(False):
                    d[key] = cropping(d[key])  # fallback to image size
        return d

class PadOrCropListDataCollate(InvertibleTransform):
    """
    This class enhances `PadListDataCollate` via supporting pad (to maximal sizes), crop (to minimal sizes) and
    resize (by pad or crop) to specified sizes.
    This transform is useful if some of the applied transforms generate batch data of
    different sizes.

    This can be used on both list and dictionary data.
    Note that in the case of the dictionary data, it may add the transform information to the list of invertible transforms
    if input batch have different spatial shape, so need to call static method: `inverse` before inverting other transforms.

    Note that normally, a user won't explicitly use the `__call__` method. Rather this would be passed to the `DataLoader`.
    This means that `__call__` handles data as it comes out of a `DataLoader`, containing batch dimension. However, the
    `inverse` operates on dictionaries containing images of shape `C,H,W,[D]`. This asymmetry is necessary so that we can
    pass the inverse through multiprocessing.

    Args:
        mode: available modes: {``"pad"``, ``"crop"``, ``"resize"``}.
        spatial_size: the spatial size of output data after padding or crop.
            If has non-positive values, the corresponding size of input image will be used (no padding).
        pad_method: padding method (see :py:class:`monai.transforms.SpatialPad`)
        pad_mode: padding mode (see :py:class:`monai.transforms.SpatialPad`)
        pad_kwargs: other arguments for the `np.pad` or `torch.pad` function.
            note that `np.pad` treats channel dimension as the first dimension.

    """

    def __init__(
        self,
        mode: str = DataCollateMode.PAD,
        spatial_size: Sequence[int] | int = -1,
        pad_method: str = Method.SYMMETRIC,
        pad_mode: str = PytorchPadMode.CONSTANT,
        **pad_kwargs,
    ) -> None:
        self.mode = mode
        if self.mode == DataCollateMode.RESIZE:
            self.resizer = ResizeWithPadOrCrop(spatial_size=spatial_size, method=pad_method, mode=pad_mode, **pad_kwargs)
        elif self.mode == DataCollateMode.PAD:
            self.pad_method = pad_method
            self.pad_mode = pad_mode
            self.pad_kwargs = pad_kwargs
        elif self.mode != DataCollateMode.CROP:
            raise ValueError(f"mode should be 'pad', 'crop' or 'reize', got {self.mode}.")

    def __call__(self, batch: Any):
        """
        Args:
            batch: batch of data to pad-collate
        """
        # data is either list of dicts or list of lists
        is_list_of_dicts = isinstance(batch[0], dict)
        # loop over items inside of each element in a batch
        batch_item = tuple(batch[0].keys()) if is_list_of_dicts else range(len(batch[0]))
        for key_or_idx in batch_item:
            shapes = []
            for elem in batch:
                if not isinstance(elem[key_or_idx], (torch.Tensor, np.ndarray)):
                    break
                shapes.append(elem[key_or_idx].shape[1:])
            # len > 0 if objects were arrays, else skip as no padding to be done
            if not shapes:
                continue
            if self.mode == DataCollateMode.RESIZE:
                transform = self.resizer
            else:
                # calculate max and min size of each dimension
                max_shape, min_shape = np.array(shapes).max(axis=0), np.array(shapes).min(axis=0)
                # If all same size, skip
                if np.all(min_shape == max_shape):
                    continue
                if self.mode == DataCollateMode.PAD:
                    transform = SpatialPad(spatial_size=max_shape, method=self.pad_method, mode=self.pad_mode, **self.pad_kwargs)
                else:
                    transform = CenterSpatialCrop(roi_size=min_shape)
            for idx, batch_i in enumerate(batch):
                orig_size = batch_i[key_or_idx].shape[1:]
                to_replace = transform(batch_i[key_or_idx])
                batch = replace_element(to_replace, batch, idx, key_or_idx)

                # If we have a dictionary of data, append to list
                # transform info is re-added with self.push_transform to ensure one info dict per transform.
                if is_list_of_dicts:
                    self.push_transform(
                        batch[idx],
                        key_or_idx,
                        orig_size=orig_size,
                        extra_info=self.pop_transform(batch[idx], key_or_idx, check=False),
                    )

        # After padding, use default list collator
        return list_data_collate(batch)

    @staticmethod
    def inverse(data: dict) -> dict[Hashable, np.ndarray]:
        if not isinstance(data, Mapping):
            raise RuntimeError("Inverse can only currently be applied on dictionaries.")

        d = dict(data)
        for key in d:
            transforms = None
            if isinstance(d[key], MetaTensor):
                transforms = d[key].applied_operations
            else:
                transform_key = InvertibleTransform.trace_key(key)
                if transform_key in d:
                    transforms = d[transform_key]
            if not transforms or not isinstance(transforms[-1], dict):
                continue
            if transforms[-1].get(TraceKeys.CLASS_NAME) == PadOrCropListDataCollate.__name__:
                xform = transforms.pop()
                cropping = CenterSpatialCrop(xform.get(TraceKeys.ORIG_SIZE, -1))
                with cropping.trace_transform(False):
                    d[key] = cropping(d[key])  # fallback to image size
        return d
