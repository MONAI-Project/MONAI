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
A collection of "vanilla" transforms for utility functions.
"""

from __future__ import annotations

import logging
import sys
import time
import warnings
from collections.abc import Hashable, Mapping, Sequence
from copy import deepcopy
from functools import partial
from typing import Any, Callable, Union

import numpy as np
import torch
import torch.nn as nn

from monai.config import DtypeLike
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.meta_obj import get_track_meta
from monai.data.meta_tensor import MetaTensor
from monai.data.utils import is_no_channel, no_collation, orientation_ras_lps
from monai.networks.layers.simplelayers import (
    ApplyFilter,
    EllipticalFilter,
    GaussianFilter,
    LaplaceFilter,
    MeanFilter,
    SavitzkyGolayFilter,
    SharpenFilter,
    median_filter,
)
from monai.transforms.inverse import InvertibleTransform, TraceableTransform
from monai.transforms.traits import MultiSampleTrait
from monai.transforms.transform import Randomizable, RandomizableTrait, RandomizableTransform, Transform
from monai.transforms.utils import (
    apply_affine_to_points,
    extreme_points_to_image,
    get_extreme_points,
    map_binary_to_indices,
    map_classes_to_indices,
)
from monai.transforms.utils_pytorch_numpy_unification import concatenate, in1d, linalg_inv, moveaxis, unravel_indices
from monai.utils import (
    MetaKeys,
    TraceKeys,
    convert_data_type,
    convert_to_cupy,
    convert_to_numpy,
    convert_to_tensor,
    ensure_tuple,
    look_up_option,
    min_version,
    optional_import,
)
from monai.utils.enums import TransformBackends
from monai.utils.misc import is_module_ver_at_least
from monai.utils.type_conversion import convert_to_dst_type, get_dtype_string, get_equivalent_dtype

PILImageImage, has_pil = optional_import("PIL.Image", name="Image")
pil_image_fromarray, _ = optional_import("PIL.Image", name="fromarray")
cp, has_cp = optional_import("cupy")

__all__ = [
    "Identity",
    "RandIdentity",
    "AsChannelLast",
    "AddCoordinateChannels",
    "EnsureChannelFirst",
    "EnsureType",
    "RepeatChannel",
    "RemoveRepeatedChannel",
    "SplitDim",
    "CastToType",
    "ToTensor",
    "ToNumpy",
    "ToPIL",
    "Transpose",
    "SqueezeDim",
    "DataStats",
    "SimulateDelay",
    "Lambda",
    "RandLambda",
    "LabelToMask",
    "FgBgToIndices",
    "ClassesToIndices",
    "ConvertToMultiChannelBasedOnBratsClasses",
    "ConvertToMultiChannelBasedOnBrats23Classes",
    "ConvertToMultiChannelBasedOnBrats23ClassesNoReg",
    "AddExtremePointsChannel",
    "TorchVision",
    "TorchIO",
    "MapLabelValue",
    "IntensityStats",
    "ToDevice",
    "CuCIM",
    "RandCuCIM",
    "RandTorchIO",
    "RandTorchVision",
    "ToCupy",
    "ImageFilter",
    "RandImageFilter",
    "ApplyTransformToPoints",
]


class Identity(Transform):
    """
    Do nothing to the data.
    As the output value is same as input, it can be used as a testing tool to verify the transform chain,
    Compose or transform adaptor, etc.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        """
        return img


class RandIdentity(RandomizableTrait):
    """
    Do nothing to the data. This transform is random, so can be used to stop the caching of any
    subsequent transforms.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, data: Any) -> Any:
        return data


class AsChannelLast(Transform):
    """
    Change the channel dimension of the image to the last dimension.

    Some of other 3rd party transforms assume the input image is in the channel-last format with shape
    (spatial_dim_1[, spatial_dim_2, ...], num_channels).

    This transform could be used to convert, for example, a channel-first image array in shape
    (num_channels, spatial_dim_1[, spatial_dim_2, ...]) into the channel-last format,
    so that MONAI transforms can construct a chain with other 3rd party transforms together.

    Args:
        channel_dim: which dimension of input image is the channel, default is the first dimension.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, channel_dim: int = 0) -> None:
        if not (isinstance(channel_dim, int) and channel_dim >= -1):
            raise ValueError(f"invalid channel dimension ({channel_dim}).")
        self.channel_dim = channel_dim

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        """
        out: NdarrayOrTensor = convert_to_tensor(moveaxis(img, self.channel_dim, -1), track_meta=get_track_meta())
        return out


class EnsureChannelFirst(Transform):
    """
    Adjust or add the channel dimension of input data to ensure `channel_first` shape.

    This extracts the `original_channel_dim` info from provided meta_data dictionary or MetaTensor input. This value
    should state which dimension is the channel dimension so that it can be moved forward, or contain "no_channel" to
    state no dimension is the channel and so a 1-size first dimension is to be added.

    Args:
        strict_check: whether to raise an error when the meta information is insufficient.
        channel_dim: This argument can be used to specify the original channel dimension (integer) of the input array.
            It overrides the `original_channel_dim` from provided MetaTensor input.
            If the input array doesn't have a channel dim, this value should be ``'no_channel'``.
            If this is set to `None`, this class relies on `img` or `meta_dict` to provide the channel dimension.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, strict_check: bool = True, channel_dim: None | str | int = None):
        self.strict_check = strict_check
        self.input_channel_dim = channel_dim

    def __call__(self, img: torch.Tensor, meta_dict: Mapping | None = None) -> torch.Tensor:
        """
        Apply the transform to `img`.
        """
        if not isinstance(img, MetaTensor) and not isinstance(meta_dict, Mapping):
            if self.input_channel_dim is None:
                msg = "Metadata not available and channel_dim=None, EnsureChannelFirst is not in use."
                if self.strict_check:
                    raise ValueError(msg)
                warnings.warn(msg)
                return img
            else:
                img = MetaTensor(img)

        if isinstance(img, MetaTensor):
            meta_dict = img.meta

        channel_dim = meta_dict.get(MetaKeys.ORIGINAL_CHANNEL_DIM, None) if isinstance(meta_dict, Mapping) else None
        if self.input_channel_dim is not None:
            channel_dim = float("nan") if self.input_channel_dim == "no_channel" else self.input_channel_dim

        if channel_dim is None:
            msg = "Unknown original_channel_dim in the MetaTensor meta dict or `meta_dict` or `channel_dim`."
            if self.strict_check:
                raise ValueError(msg)
            warnings.warn(msg)
            return img

        # track the original channel dim
        if isinstance(meta_dict, dict):
            meta_dict[MetaKeys.ORIGINAL_CHANNEL_DIM] = channel_dim

        if is_no_channel(channel_dim):
            result = img[None]
        else:
            result = moveaxis(img, int(channel_dim), 0)  # type: ignore

        return convert_to_tensor(result, track_meta=get_track_meta())  # type: ignore


class RepeatChannel(Transform):
    """
    Repeat channel data to construct expected input shape for models.
    The `repeats` count includes the origin data, for example:
    ``RepeatChannel(repeats=2)([[1, 2], [3, 4]])`` generates: ``[[1, 2], [1, 2], [3, 4], [3, 4]]``

    Args:
        repeats: the number of repetitions for each element.
    """

    backend = [TransformBackends.TORCH]

    def __init__(self, repeats: int) -> None:
        if repeats <= 0:
            raise ValueError(f"repeats count must be greater than 0, got {repeats}.")
        self.repeats = repeats

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the transform to `img`, assuming `img` is a "channel-first" array.
        """
        repeat_fn = torch.repeat_interleave if isinstance(img, torch.Tensor) else np.repeat
        return convert_to_tensor(repeat_fn(img, self.repeats, 0), track_meta=get_track_meta())  # type: ignore


class RemoveRepeatedChannel(Transform):
    """
    RemoveRepeatedChannel data to undo RepeatChannel
    The `repeats` count specifies the deletion of the origin data, for example:
    ``RemoveRepeatedChannel(repeats=2)([[1, 2], [1, 2], [3, 4], [3, 4]])`` generates: ``[[1, 2], [3, 4]]``

    Args:
        repeats: the number of repetitions to be deleted for each element.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, repeats: int) -> None:
        if repeats <= 0:
            raise ValueError(f"repeats count must be greater than 0, got {repeats}.")

        self.repeats = repeats

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the transform to `img`, assuming `img` is a "channel-first" array.
        """
        if img.shape[0] < 2:
            raise ValueError(f"Image must have more than one channel, got {img.shape[0]} channels.")

        out: NdarrayOrTensor = convert_to_tensor(img[:: self.repeats, :], track_meta=get_track_meta())
        return out


class SplitDim(Transform, MultiSampleTrait):
    """
    Given an image of size X along a certain dimension, return a list of length X containing
    images. Useful for converting 3D images into a stack of 2D images, splitting multichannel inputs into
    single channels, for example.

    Note: `torch.split`/`np.split` is used, so the outputs are views of the input (shallow copy).

    Args:
        dim: dimension on which to split
        keepdim: if `True`, output will have singleton in the split dimension. If `False`, this
            dimension will be squeezed.
        update_meta: whether to update the MetaObj in each split result.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, dim: int = -1, keepdim: bool = True, update_meta=True) -> None:
        self.dim = dim
        self.keepdim = keepdim
        self.update_meta = update_meta

    def __call__(self, img: torch.Tensor) -> list[torch.Tensor]:
        """
        Apply the transform to `img`.
        """
        n_out = img.shape[self.dim]
        if isinstance(img, torch.Tensor):
            outputs = list(torch.split(img, 1, self.dim))
        else:
            outputs = np.split(img, n_out, self.dim)
        for idx, item in enumerate(outputs):
            if not self.keepdim:
                outputs[idx] = item.squeeze(self.dim)
            if self.update_meta and isinstance(img, MetaTensor):
                if not isinstance(item, MetaTensor):
                    item = MetaTensor(item, meta=img.meta)
                if self.dim == 0:  # don't update affine if channel dim
                    continue
                ndim = len(item.affine)
                shift = torch.eye(ndim, device=item.affine.device, dtype=item.affine.dtype)
                shift[self.dim - 1, -1] = idx
                item.affine = item.affine @ shift
        return outputs


class CastToType(Transform):
    """
    Cast the Numpy data to specified numpy data type, or cast the PyTorch Tensor to
    specified PyTorch data type.

    Example:
        >>> import numpy as np
        >>> import torch
        >>> transform = CastToType(dtype=np.float32)

        >>> # Example with a numpy array
        >>> img_np = np.array([0, 127, 255], dtype=np.uint8)
        >>> img_np_casted = transform(img_np)
        >>> img_np_casted
        array([  0. , 127. , 255. ], dtype=float32)

        >>> # Example with a PyTorch tensor
        >>> img_tensor = torch.tensor([0, 127, 255], dtype=torch.uint8)
        >>> img_tensor_casted = transform(img_tensor)
        >>> img_tensor_casted
        tensor([  0., 127., 255.])  # dtype is float32
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, dtype=np.float32) -> None:
        """
        Args:
            dtype: convert image to this data type, default is `np.float32`.
        """
        self.dtype = dtype

    def __call__(self, img: NdarrayOrTensor, dtype: DtypeLike | torch.dtype = None) -> NdarrayOrTensor:
        """
        Apply the transform to `img`, assuming `img` is a numpy array or PyTorch Tensor.

        Args:
            dtype: convert image to this data type, default is `self.dtype`.

        Raises:
            TypeError: When ``img`` type is not in ``Union[numpy.ndarray, torch.Tensor]``.

        """
        return convert_data_type(img, output_type=type(img), dtype=dtype or self.dtype)[0]


class ToTensor(Transform):
    """
    Converts the input image to a tensor without applying any other transformations.
    Input data can be PyTorch Tensor, numpy array, list, dictionary, int, float, bool, str, etc.
    Will convert Tensor, Numpy array, float, int, bool to Tensor, strings and objects keep the original.
    For dictionary, list or tuple, convert every item to a Tensor if applicable and `wrap_sequence=False`.

    Args:
        dtype: target data type to when converting to Tensor.
        device: target device to put the converted Tensor data.
        wrap_sequence: if `False`, then lists will recursively call this function, default to `True`.
            E.g., if `False`, `[1, 2]` -> `[tensor(1), tensor(2)]`, if `True`, then `[1, 2]` -> `tensor([1, 2])`.
        track_meta: whether to convert to `MetaTensor` or regular tensor, default to `None`,
            use the return value of ``get_track_meta``.

    """

    backend = [TransformBackends.TORCH]

    def __init__(
        self,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        wrap_sequence: bool = True,
        track_meta: bool | None = None,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.wrap_sequence = wrap_sequence
        self.track_meta = get_track_meta() if track_meta is None else bool(track_meta)

    def __call__(self, img: NdarrayOrTensor):
        """
        Apply the transform to `img` and make it contiguous.
        """
        if isinstance(img, MetaTensor):
            img.applied_operations = []  # drops tracking info
        return convert_to_tensor(
            img, dtype=self.dtype, device=self.device, wrap_sequence=self.wrap_sequence, track_meta=self.track_meta
        )


class EnsureType(Transform):
    """
    Ensure the input data to be a PyTorch Tensor or numpy array, support: `numpy array`, `PyTorch Tensor`,
    `float`, `int`, `bool`, `string` and `object` keep the original.
    If passing a dictionary, list or tuple, still return dictionary, list or tuple will recursively convert
    every item to the expected data type if `wrap_sequence=False`.

    Args:
        data_type: target data type to convert, should be "tensor" or "numpy".
        dtype: target data content type to convert, for example: np.float32, torch.float, etc.
        device: for Tensor data type, specify the target device.
        wrap_sequence: if `False`, then lists will recursively call this function, default to `True`.
        track_meta: if `True` convert to ``MetaTensor``, otherwise to Pytorch ``Tensor``,
            if ``None`` behave according to return value of py:func:`monai.data.meta_obj.get_track_meta`.

    Example with wrap_sequence=True:
        >>> import numpy as np
        >>> import torch
        >>> transform = EnsureType(data_type="tensor", wrap_sequence=True)
        >>> # Converting a list to a tensor
        >>> data_list = [1, 2., 3]
        >>> tensor_data = transform(data_list)
        >>> tensor_data
        tensor([1., 2., 3.])    # All elements have dtype float32

    Example with wrap_sequence=False:
        >>> transform = EnsureType(data_type="tensor", wrap_sequence=False)
        >>> # Converting each element in a list to individual tensors
        >>> data_list = [1, 2, 3]
        >>> tensors_list = transform(data_list)
        >>> tensors_list
        [tensor(1), tensor(2.), tensor(3)]  # Only second element is float32 rest are int64
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        data_type: str = "tensor",
        dtype: DtypeLike | torch.dtype = None,
        device: torch.device | None = None,
        wrap_sequence: bool = True,
        track_meta: bool | None = None,
    ) -> None:
        self.data_type = look_up_option(data_type.lower(), {"tensor", "numpy"})
        self.dtype = dtype
        self.device = device
        self.wrap_sequence = wrap_sequence
        self.track_meta = get_track_meta() if track_meta is None else bool(track_meta)

    def __call__(self, data: NdarrayOrTensor, dtype: DtypeLike | torch.dtype = None):
        """
        Args:
            data: input data can be PyTorch Tensor, numpy array, list, dictionary, int, float, bool, str, etc.
                will ensure Tensor, Numpy array, float, int, bool as Tensors or numpy arrays, strings and
                objects keep the original. for dictionary, list or tuple, ensure every item as expected type
                if applicable and `wrap_sequence=False`.
            dtype: target data content type to convert, for example: np.float32, torch.float, etc.

        """
        if self.data_type == "tensor":
            output_type = MetaTensor if self.track_meta else torch.Tensor
        else:
            output_type = np.ndarray  # type: ignore
        out: NdarrayOrTensor
        out, *_ = convert_data_type(
            data=data,
            output_type=output_type,  # type: ignore
            dtype=self.dtype if dtype is None else dtype,
            device=self.device,
            wrap_sequence=self.wrap_sequence,
        )
        return out


class ToNumpy(Transform):
    """
    Converts the input data to numpy array, can support list or tuple of numbers and PyTorch Tensor.

    Args:
        dtype: target data type when converting to numpy array.
        wrap_sequence: if `False`, then lists will recursively call this function, default to `True`.
            E.g., if `False`, `[1, 2]` -> `[array(1), array(2)]`, if `True`, then `[1, 2]` -> `array([1, 2])`.

    """

    backend = [TransformBackends.NUMPY]

    def __init__(self, dtype: DtypeLike = None, wrap_sequence: bool = True) -> None:
        super().__init__()
        self.dtype = dtype
        self.wrap_sequence = wrap_sequence

    def __call__(self, img: NdarrayOrTensor):
        """
        Apply the transform to `img` and make it contiguous.
        """
        return convert_to_numpy(img, dtype=self.dtype, wrap_sequence=self.wrap_sequence)


class ToCupy(Transform):
    """
    Converts the input data to CuPy array, can support list or tuple of numbers, NumPy and PyTorch Tensor.

    Args:
        dtype: data type specifier. It is inferred from the input by default.
            if not None, must be an argument of `numpy.dtype`, for more details:
            https://docs.cupy.dev/en/stable/reference/generated/cupy.array.html.
        wrap_sequence: if `False`, then lists will recursively call this function, default to `True`.
            E.g., if `False`, `[1, 2]` -> `[array(1), array(2)]`, if `True`, then `[1, 2]` -> `array([1, 2])`.

    """

    backend = [TransformBackends.CUPY]

    def __init__(self, dtype: np.dtype | None = None, wrap_sequence: bool = True) -> None:
        super().__init__()
        self.dtype = dtype
        self.wrap_sequence = wrap_sequence

    def __call__(self, data: NdarrayOrTensor):
        """
        Create a CuPy array from `data` and make it contiguous
        """
        return convert_to_cupy(data, dtype=self.dtype, wrap_sequence=self.wrap_sequence)


class ToPIL(Transform):
    """
    Converts the input image (in the form of NumPy array or PyTorch Tensor) to PIL image
    """

    backend = [TransformBackends.NUMPY]

    def __call__(self, img):
        """
        Apply the transform to `img`.
        """
        if isinstance(img, PILImageImage):
            return img
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        return pil_image_fromarray(img)


class Transpose(Transform):
    """
    Transposes the input image based on the given `indices` dimension ordering.
    """

    backend = [TransformBackends.TORCH]

    def __init__(self, indices: Sequence[int] | None) -> None:
        self.indices = None if indices is None else tuple(indices)

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        """
        img = convert_to_tensor(img, track_meta=get_track_meta())
        return img.permute(self.indices or tuple(range(img.ndim)[::-1]))  # type: ignore


class SqueezeDim(Transform):
    """
    Squeeze a unitary dimension.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, dim: int | None = 0, update_meta=True) -> None:
        """
        Args:
            dim: dimension to be squeezed. Default = 0
                "None" works when the input is numpy array.
            update_meta: whether to update the meta info if the input is a metatensor. Default is ``True``.

        Raises:
            TypeError: When ``dim`` is not an ``Optional[int]``.

        """
        if dim is not None and not isinstance(dim, int):
            raise TypeError(f"dim must be None or a int but is {type(dim).__name__}.")
        self.dim = dim
        self.update_meta = update_meta

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Args:
            img: numpy arrays with required dimension `dim` removed
        """
        img = convert_to_tensor(img, track_meta=get_track_meta())
        if self.dim is None:
            if self.update_meta:
                warnings.warn("update_meta=True is ignored when dim=None.")
            return img.squeeze()
        dim = (self.dim + len(img.shape)) if self.dim < 0 else self.dim
        # for pytorch/numpy unification
        if img.shape[dim] != 1:
            raise ValueError(f"Can only squeeze singleton dimension, got shape {img.shape[dim]} of {img.shape}.")
        img = img.squeeze(dim)
        if self.update_meta and isinstance(img, MetaTensor) and dim > 0 and len(img.affine.shape) == 2:
            h, w = img.affine.shape
            affine, device = img.affine, img.affine.device if isinstance(img.affine, torch.Tensor) else None
            if h > dim:
                affine = affine[torch.arange(0, h, device=device) != dim - 1]
            if w > dim:
                affine = affine[:, torch.arange(0, w, device=device) != dim - 1]
            if (affine.shape[0] == affine.shape[1]) and not np.linalg.det(convert_to_numpy(affine, wrap_sequence=True)):
                warnings.warn(f"After SqueezeDim, img.affine is ill-posed: \n{img.affine}.")
            img.affine = affine
        return img


class DataStats(Transform):
    """
    Utility transform to show the statistics of data for debug or analysis.
    It can be inserted into any place of a transform chain and check results of previous transforms.
    It support both `numpy.ndarray` and `torch.tensor` as input data,
    so it can be used in pre-processing and post-processing.

    It gets logger from `logging.getLogger(name)`, we can setup a logger outside first with the same `name`.
    If the log level of `logging.RootLogger` is higher than `INFO`, will add a separate `StreamHandler`
    log handler with `INFO` level and record to `stdout`.

    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        prefix: str = "Data",
        data_type: bool = True,
        data_shape: bool = True,
        value_range: bool = True,
        data_value: bool = False,
        meta_info: bool = False,
        additional_info: Callable | None = None,
        name: str = "DataStats",
    ) -> None:
        """
        Args:
            prefix: will be printed in format: "{prefix} statistics".
            data_type: whether to show the type of input data.
            data_shape: whether to show the shape of input data.
            value_range: whether to show the value range of input data.
            data_value: whether to show the raw value of input data.
                a typical example is to print some properties of Nifti image: affine, pixdim, etc.
            meta_info: whether to show the data of MetaTensor.
            additional_info: user can define callable function to extract additional info from input data.
            name: identifier of `logging.logger` to use, defaulting to "DataStats".

        Raises:
            TypeError: When ``additional_info`` is not an ``Optional[Callable]``.

        """
        if not isinstance(prefix, str):
            raise ValueError(f"prefix must be a string, got {type(prefix)}.")
        self.prefix = prefix
        self.data_type = data_type
        self.data_shape = data_shape
        self.value_range = value_range
        self.data_value = data_value
        self.meta_info = meta_info
        if additional_info is not None and not callable(additional_info):
            raise TypeError(f"additional_info must be None or callable but is {type(additional_info).__name__}.")
        self.additional_info = additional_info
        self._logger_name = name
        _logger = logging.getLogger(self._logger_name)
        _logger.setLevel(logging.INFO)
        if logging.root.getEffectiveLevel() > logging.INFO:
            # Avoid duplicate stream handlers to be added when multiple DataStats are used in a chain.
            has_console_handler = any(
                hasattr(h, "is_data_stats_handler") and h.is_data_stats_handler for h in _logger.handlers
            )
            if not has_console_handler:
                # if the root log level is higher than INFO, set a separate stream handler to record
                console = logging.StreamHandler(sys.stdout)
                console.setLevel(logging.INFO)
                console.is_data_stats_handler = True  # type:ignore[attr-defined]
                _logger.addHandler(console)

    def __call__(
        self,
        img: NdarrayOrTensor,
        prefix: str | None = None,
        data_type: bool | None = None,
        data_shape: bool | None = None,
        value_range: bool | None = None,
        data_value: bool | None = None,
        meta_info: bool | None = None,
        additional_info: Callable | None = None,
    ) -> NdarrayOrTensor:
        """
        Apply the transform to `img`, optionally take arguments similar to the class constructor.
        """
        lines = [f"{prefix or self.prefix} statistics:"]

        if self.data_type if data_type is None else data_type:
            lines.append(f"Type: {type(img)} {img.dtype if hasattr(img, 'dtype') else None}")
        if self.data_shape if data_shape is None else data_shape:
            lines.append(f"Shape: {img.shape if hasattr(img, 'shape') else None}")
        if self.value_range if value_range is None else value_range:
            if isinstance(img, np.ndarray):
                lines.append(f"Value range: ({np.min(img)}, {np.max(img)})")
            elif isinstance(img, torch.Tensor):
                lines.append(f"Value range: ({torch.min(img)}, {torch.max(img)})")
            else:
                lines.append(f"Value range: (not a PyTorch or Numpy array, type: {type(img)})")
        if self.data_value if data_value is None else data_value:
            lines.append(f"Value: {img}")
        if self.meta_info if meta_info is None else meta_info:
            metadata = getattr(img, "meta", "(input is not a MetaTensor)")
            lines.append(f"Meta info: {repr(metadata)}")
        additional_info = self.additional_info if additional_info is None else additional_info
        if additional_info is not None:
            lines.append(f"Additional info: {additional_info(img)}")
        separator = "\n"
        output = f"{separator.join(lines)}"
        logging.getLogger(self._logger_name).info(output)
        return img


class SimulateDelay(Transform):
    """
    This is a pass through transform to be used for testing purposes. It allows
    adding fake behaviors that are useful for testing purposes to simulate
    how large datasets behave without needing to test on large data sets.

    For example, simulating slow NFS data transfers, or slow network transfers
    in testing by adding explicit timing delays. Testing of small test data
    can lead to incomplete understanding of real world issues, and may lead
    to sub-optimal design choices.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, delay_time: float = 0.0) -> None:
        """
        Args:
            delay_time: The minimum amount of time, in fractions of seconds,
                to accomplish this delay task.
        """
        super().__init__()
        self.delay_time: float = delay_time

    def __call__(self, img: NdarrayOrTensor, delay_time: float | None = None) -> NdarrayOrTensor:
        """
        Args:
            img: data remain unchanged throughout this transform.
            delay_time: The minimum amount of time, in fractions of seconds,
                to accomplish this delay task.
        """
        time.sleep(self.delay_time if delay_time is None else delay_time)
        return img


class Lambda(InvertibleTransform):
    """
    Apply a user-defined lambda as a transform.

    For example:

    .. code-block:: python
        :emphasize-lines: 2

        image = np.ones((10, 2, 2))
        lambd = Lambda(func=lambda x: x[:4, :, :])
        print(lambd(image).shape)
        (4, 2, 2)

    Args:
        func: Lambda/function to be applied.
        inv_func: Lambda/function of inverse operation, default to `lambda x: x`.
        track_meta:  If `False`, then standard data objects will be returned (e.g., torch.Tensor` and `np.ndarray`)
            as opposed to MONAI's enhanced objects. By default, this is `True`.

    Raises:
        TypeError: When ``func`` is not an ``Optional[Callable]``.

    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self, func: Callable | None = None, inv_func: Callable = no_collation, track_meta: bool = True
    ) -> None:
        if func is not None and not callable(func):
            raise TypeError(f"func must be None or callable but is {type(func).__name__}.")
        self.func = func
        self.inv_func = inv_func
        self.track_meta = track_meta

    def __call__(self, img: NdarrayOrTensor, func: Callable | None = None):
        """
        Apply `self.func` to `img`.

        Args:
            func: Lambda/function to be applied. Defaults to `self.func`.

        Raises:
            TypeError: When ``func`` is not an ``Optional[Callable]``.

        """
        fn = func if func is not None else self.func
        if not callable(fn):
            raise TypeError(f"func must be None or callable but is {type(fn).__name__}.")
        out = fn(img)
        # convert to MetaTensor if necessary
        if isinstance(out, (np.ndarray, torch.Tensor)) and not isinstance(out, MetaTensor) and self.track_meta:
            out = MetaTensor(out)
        if isinstance(out, MetaTensor):
            self.push_transform(out)
        return out

    def inverse(self, data: torch.Tensor):
        if isinstance(data, MetaTensor):
            self.pop_transform(data)
        return self.inv_func(data)


class RandLambda(Lambda, RandomizableTransform):
    """
    Randomizable version :py:class:`monai.transforms.Lambda`, the input `func` may contain random logic,
    or randomly execute the function based on `prob`.

    Args:
        func: Lambda/function to be applied.
        prob: probability of executing the random function, default to 1.0, with 100% probability to execute.
        inv_func: Lambda/function of inverse operation, default to `lambda x: x`.
        track_meta:  If `False`, then standard data objects will be returned (e.g., torch.Tensor` and `np.ndarray`)
            as opposed to MONAI's enhanced objects. By default, this is `True`.

    For more details, please check :py:class:`monai.transforms.Lambda`.
    """

    backend = Lambda.backend

    def __init__(
        self,
        func: Callable | None = None,
        prob: float = 1.0,
        inv_func: Callable = no_collation,
        track_meta: bool = True,
    ) -> None:
        Lambda.__init__(self=self, func=func, inv_func=inv_func, track_meta=track_meta)
        RandomizableTransform.__init__(self=self, prob=prob)

    def __call__(self, img: NdarrayOrTensor, func: Callable | None = None):
        self.randomize(img)
        out = deepcopy(super().__call__(img, func) if self._do_transform else img)
        # convert to MetaTensor if necessary
        if not isinstance(out, MetaTensor) and self.track_meta:
            out = MetaTensor(out)
        if isinstance(out, MetaTensor):
            lambda_info = self.pop_transform(out) if self._do_transform else {}
            self.push_transform(out, extra_info=lambda_info)
        return out

    def inverse(self, data: torch.Tensor):
        do_transform = self.get_most_recent_transform(data).pop(TraceKeys.DO_TRANSFORM)
        if do_transform:
            data = super().inverse(data)
        else:
            self.pop_transform(data)
        return data


class LabelToMask(Transform):
    """
    Convert labels to mask for other tasks. A typical usage is to convert segmentation labels
    to mask data to pre-process images and then feed the images into classification network.
    It can support single channel labels or One-Hot labels with specified `select_labels`.
    For example, users can select `label value = [2, 3]` to construct mask data, or select the
    second and the third channels of labels to construct mask data.
    The output mask data can be a multiple channels binary data or a single channel binary
    data that merges all the channels.

    Args:
        select_labels: labels to generate mask from. for 1 channel label, the `select_labels`
            is the expected label values, like: [1, 2, 3]. for One-Hot format label, the
            `select_labels` is the expected channel indices.
        merge_channels: whether to use `np.any()` to merge the result on channel dim. if yes,
            will return a single channel mask with binary data.

    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(  # pytype: disable=annotation-type-mismatch
        self, select_labels: Sequence[int] | int, merge_channels: bool = False
    ) -> None:  # pytype: disable=annotation-type-mismatch
        self.select_labels = ensure_tuple(select_labels)
        self.merge_channels = merge_channels

    def __call__(
        self, img: NdarrayOrTensor, select_labels: Sequence[int] | int | None = None, merge_channels: bool = False
    ) -> NdarrayOrTensor:
        """
        Args:
            select_labels: labels to generate mask from. for 1 channel label, the `select_labels`
                is the expected label values, like: [1, 2, 3]. for One-Hot format label, the
                `select_labels` is the expected channel indices.
            merge_channels: whether to use `np.any()` to merge the result on channel dim. if yes,
                will return a single channel mask with binary data.
        """
        img = convert_to_tensor(img, track_meta=get_track_meta())
        if select_labels is None:
            select_labels = self.select_labels
        else:
            select_labels = ensure_tuple(select_labels)

        if img.shape[0] > 1:
            data = img[[*select_labels]]
        else:
            where: Callable = np.where if isinstance(img, np.ndarray) else torch.where  # type: ignore
            if isinstance(img, np.ndarray) or is_module_ver_at_least(torch, (1, 8, 0)):
                data = where(in1d(img, select_labels), True, False).reshape(img.shape)
            # pre pytorch 1.8.0, need to use 1/0 instead of True/False
            else:
                data = where(
                    in1d(img, select_labels), torch.tensor(1, device=img.device), torch.tensor(0, device=img.device)
                ).reshape(img.shape)

        if merge_channels or self.merge_channels:
            if isinstance(img, np.ndarray) or is_module_ver_at_least(torch, (1, 8, 0)):
                return data.any(0)[None]
            # pre pytorch 1.8.0 compatibility
            return data.to(torch.uint8).any(0)[None].to(bool)  # type: ignore

        return data


class FgBgToIndices(Transform, MultiSampleTrait):
    """
    Compute foreground and background of the input label data, return the indices.
    If no output_shape specified, output data will be 1 dim indices after flattening.
    This transform can help pre-compute foreground and background regions for other transforms.
    A typical usage is to randomly select foreground and background to crop.
    The main logic is based on :py:class:`monai.transforms.utils.map_binary_to_indices`.

    Args:
        image_threshold: if enabled `image` at runtime, use ``image > image_threshold`` to
            determine the valid image content area and select background only in this area.
        output_shape: expected shape of output indices. if not None, unravel indices to specified shape.

    """

    backend = [TransformBackends.NUMPY, TransformBackends.TORCH]

    def __init__(self, image_threshold: float = 0.0, output_shape: Sequence[int] | None = None) -> None:
        self.image_threshold = image_threshold
        self.output_shape = output_shape

    def __call__(
        self, label: NdarrayOrTensor, image: NdarrayOrTensor | None = None, output_shape: Sequence[int] | None = None
    ) -> tuple[NdarrayOrTensor, NdarrayOrTensor]:
        """
        Args:
            label: input data to compute foreground and background indices.
            image: if image is not None, use ``label = 0 & image > image_threshold``
                to define background. so the output items will not map to all the voxels in the label.
            output_shape: expected shape of output indices. if None, use `self.output_shape` instead.

        """
        if output_shape is None:
            output_shape = self.output_shape
        fg_indices, bg_indices = map_binary_to_indices(label, image, self.image_threshold)
        if output_shape is not None:
            fg_indices = unravel_indices(fg_indices, output_shape)
            bg_indices = unravel_indices(bg_indices, output_shape)
        return fg_indices, bg_indices


class ClassesToIndices(Transform, MultiSampleTrait):
    backend = [TransformBackends.NUMPY, TransformBackends.TORCH]

    def __init__(
        self,
        num_classes: int | None = None,
        image_threshold: float = 0.0,
        output_shape: Sequence[int] | None = None,
        max_samples_per_class: int | None = None,
    ) -> None:
        """
        Compute indices of every class of the input label data, return a list of indices.
        If no output_shape specified, output data will be 1 dim indices after flattening.
        This transform can help pre-compute indices of the class regions for other transforms.
        A typical usage is to randomly select indices of classes to crop.
        The main logic is based on :py:class:`monai.transforms.utils.map_classes_to_indices`.

        Args:
            num_classes: number of classes for argmax label, not necessary for One-Hot label.
            image_threshold: if enabled `image` at runtime, use ``image > image_threshold`` to
                determine the valid image content area and select only the indices of classes in this area.
            output_shape: expected shape of output indices. if not None, unravel indices to specified shape.
            max_samples_per_class: maximum length of indices to sample in each class to reduce memory consumption.
                Default is None, no subsampling.

        """
        self.num_classes = num_classes
        self.image_threshold = image_threshold
        self.output_shape = output_shape
        self.max_samples_per_class = max_samples_per_class

    def __call__(
        self, label: NdarrayOrTensor, image: NdarrayOrTensor | None = None, output_shape: Sequence[int] | None = None
    ) -> list[NdarrayOrTensor]:
        """
        Args:
            label: input data to compute the indices of every class.
            image: if image is not None, use ``image > image_threshold`` to define valid region, and only select
                the indices within the valid region.
            output_shape: expected shape of output indices. if None, use `self.output_shape` instead.

        """

        if output_shape is None:
            output_shape = self.output_shape
        indices: list[NdarrayOrTensor]
        indices = map_classes_to_indices(
            label, self.num_classes, image, self.image_threshold, self.max_samples_per_class
        )
        if output_shape is not None:
            indices = [unravel_indices(cls_indices, output_shape) for cls_indices in indices]

        return indices


class ConvertToMultiChannelBasedOnBratsClasses(Transform):
    """
    Convert labels to multi channels based on `brats18 <https://www.med.upenn.edu/sbia/brats2018/data.html>`_ classes,
    which include TC (Tumor core), WT (Whole tumor) and ET (Enhancing tumor):
    label 1 is the necrotic and non-enhancing tumor core, which should be counted under TC and WT subregion,
    label 2 is the peritumoral edema, which is counted only under WT subregion,
    label 4 is the GD-enhancing tumor, which should be counted under ET, TC, WT subregions.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        # if img has channel dim, squeeze it
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)

        result = [(img == 1) | (img == 4), (img == 1) | (img == 4) | (img == 2), img == 4]
        # merge labels 1 (tumor non-enh) and 4 (tumor enh) and 2 (large edema) to WT
        # label 4 is ET
        return torch.stack(result, dim=0) if isinstance(img, torch.Tensor) else np.stack(result, axis=0)


class ConvertToMultiChannelBasedOnBrats23Classes(Transform):
    """
    Convert labels to multi channels based on brats23 classes:
    label 1 is the necrotic and non-enhancing tumor core (NCR)
    label 2 is the peritumoral edema (ED)
    label 3 is the GD-enhancing tumor (ET)
    NOTE: REGION-BASED CONVERSION to TC, WT, ET
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        # if img has channel dim, squeeze it
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)

        result = [(img == 1) | (img == 3), (img == 1) | (img == 3) | (img == 2), img == 3]  # -> tc, wt, et
        # merge labels 1 (ncr) and 3 (et) and 2 (ed) to WT
        return torch.stack(result, dim=0) if isinstance(img, torch.Tensor) else np.stack(result, axis=0)


class ConvertToMultiChannelBasedOnBrats23ClassesNoReg(Transform):
    """
    Convert labels to multi channels based on brats23 classes:
    label 1 is the necrotic and non-enhancing tumor core (NCR)
    label 2 is the peritumoral edema (ED)
    label 3 is the GD-enhancing tumor (ET)
    NOTE: LABEL-BASED CONVERSION
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        # if img has channel dim, squeeze it
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)

        result = [(img == 1), (img == 2), (img == 3)]

        return torch.stack(result, dim=0) if isinstance(img, torch.Tensor) else np.stack(result, axis=0)


class AddExtremePointsChannel(Randomizable, Transform):
    """
    Add extreme points of label to the image as a new channel. This transform generates extreme
    point from label and applies a gaussian filter. The pixel values in points image are rescaled
    to range [rescale_min, rescale_max] and added as a new channel to input image. The algorithm is
    described in Roth et al., Going to Extremes: Weakly Supervised Medical Image Segmentation
    https://arxiv.org/abs/2009.11988.

    This transform only supports single channel labels (1, spatial_dim1, [spatial_dim2, ...]). The
    background ``index`` is ignored when calculating extreme points.

    Args:
        background: Class index of background label, defaults to 0.
        pert: Random perturbation amount to add to the points, defaults to 0.0.

    Raises:
        ValueError: When no label image provided.
        ValueError: When label image is not single channel.
    """

    backend = [TransformBackends.TORCH]

    def __init__(self, background: int = 0, pert: float = 0.0) -> None:
        self._background = background
        self._pert = pert
        self._points: list[tuple[int, ...]] = []

    def randomize(self, label: NdarrayOrTensor) -> None:
        self._points = get_extreme_points(label, rand_state=self.R, background=self._background, pert=self._pert)

    def __call__(
        self,
        img: NdarrayOrTensor,
        label: NdarrayOrTensor | None = None,
        sigma: Sequence[float] | float | Sequence[torch.Tensor] | torch.Tensor = 3.0,
        rescale_min: float = -1.0,
        rescale_max: float = 1.0,
    ) -> NdarrayOrTensor:
        """
        Args:
            img: the image that we want to add new channel to.
            label: label image to get extreme points from. Shape must be
                (1, spatial_dim1, [, spatial_dim2, ...]). Doesn't support one-hot labels.
            sigma: if a list of values, must match the count of spatial dimensions of input data,
                and apply every value in the list to 1 spatial dimension. if only 1 value provided,
                use it for all spatial dimensions.
            rescale_min: minimum value of output data.
            rescale_max: maximum value of output data.
        """
        if label is None:
            raise ValueError("This transform requires a label array!")
        if label.shape[0] != 1:
            raise ValueError("Only supports single channel labels!")

        # Generate extreme points
        self.randomize(label[0, :])

        points_image = extreme_points_to_image(
            points=self._points, label=label, sigma=sigma, rescale_min=rescale_min, rescale_max=rescale_max
        )
        points_image, *_ = convert_to_dst_type(points_image, img)  # type: ignore
        return concatenate((img, points_image), axis=0)


class TorchVision(Transform):
    """
    This is a wrapper transform for PyTorch TorchVision non-randomized transform based on the specified transform name and args.
    Data is converted to a torch.tensor before applying the transform and then converted back to the original data type.
    """

    backend = [TransformBackends.TORCH]

    def __init__(self, name: str, *args, **kwargs) -> None:
        """
        Args:
            name: The transform name in TorchVision package.
            args: parameters for the TorchVision transform.
            kwargs: parameters for the TorchVision transform.

        """
        super().__init__()
        self.name = name
        transform, _ = optional_import("torchvision.transforms", "0.8.0", min_version, name=name)
        self.trans = transform(*args, **kwargs)

    def __call__(self, img: NdarrayOrTensor):
        """
        Args:
            img: PyTorch Tensor data for the TorchVision transform.

        """
        img_t, *_ = convert_data_type(img, torch.Tensor)

        out = self.trans(img_t)
        out, *_ = convert_to_dst_type(src=out, dst=img)
        return out


class RandTorchVision(Transform, RandomizableTrait):
    """
    This is a wrapper transform for PyTorch TorchVision randomized transform based on the specified transform name and args.
    Data is converted to a torch.tensor before applying the transform and then converted back to the original data type.
    """

    backend = [TransformBackends.TORCH]

    def __init__(self, name: str, *args, **kwargs) -> None:
        """
        Args:
            name: The transform name in TorchVision package.
            args: parameters for the TorchVision transform.
            kwargs: parameters for the TorchVision transform.

        """
        super().__init__()
        self.name = name
        transform, _ = optional_import("torchvision.transforms", "0.8.0", min_version, name=name)
        self.trans = transform(*args, **kwargs)

    def __call__(self, img: NdarrayOrTensor):
        """
        Args:
            img: PyTorch Tensor data for the TorchVision transform.

        """
        img_t, *_ = convert_data_type(img, torch.Tensor)

        out = self.trans(img_t)
        out, *_ = convert_to_dst_type(src=out, dst=img)
        return out


class TorchIO(Transform):
    """
    This is a wrapper for TorchIO non-randomized transforms based on the specified transform name and args.
    See https://torchio.readthedocs.io/transforms/transforms.html for more details.
    """

    backend = [TransformBackends.TORCH]

    def __init__(self, name: str, *args, **kwargs) -> None:
        """
        Args:
            name: The transform name in TorchIO package.
            args: parameters for the TorchIO transform.
            kwargs: parameters for the TorchIO transform.
        """
        super().__init__()
        self.name = name
        transform, _ = optional_import("torchio.transforms", "0.18.0", min_version, name=name)
        self.trans = transform(*args, **kwargs)

    def __call__(self, img: Union[NdarrayOrTensor, Mapping[Hashable, NdarrayOrTensor]]):
        """
        Args:
            img: an instance of torchio.Subject, torchio.Image, numpy.ndarray, torch.Tensor, SimpleITK.Image,
                 or dict containing 4D tensors as values

        """
        return self.trans(img)


class RandTorchIO(Transform, RandomizableTrait):
    """
    This is a wrapper for TorchIO randomized transforms based on the specified transform name and args.
    See https://torchio.readthedocs.io/transforms/transforms.html for more details.
    Use this wrapper for all TorchIO transform inheriting from RandomTransform:
    https://torchio.readthedocs.io/transforms/augmentation.html#randomtransform
    """

    backend = [TransformBackends.TORCH]

    def __init__(self, name: str, *args, **kwargs) -> None:
        """
        Args:
            name: The transform name in TorchIO package.
            args: parameters for the TorchIO transform.
            kwargs: parameters for the TorchIO transform.
        """
        super().__init__()
        self.name = name
        transform, _ = optional_import("torchio.transforms", "0.18.0", min_version, name=name)
        self.trans = transform(*args, **kwargs)

    def __call__(self, img: Union[NdarrayOrTensor, Mapping[Hashable, NdarrayOrTensor]]):
        """
        Args:
            img: an instance of torchio.Subject, torchio.Image, numpy.ndarray, torch.Tensor, SimpleITK.Image,
                 or dict containing 4D tensors as values

        """
        return self.trans(img)


class MapLabelValue:
    """
    Utility to map label values to another set of values.
    For example, map [3, 2, 1] to [0, 1, 2], [1, 2, 3] -> [0.5, 1.5, 2.5], ["label3", "label2", "label1"] -> [0, 1, 2],
    [3.5, 2.5, 1.5] -> ["label0", "label1", "label2"], etc.
    The label data must be numpy array or array-like data and the output data will be numpy array.

    """

    backend = [TransformBackends.NUMPY, TransformBackends.TORCH]

    def __init__(self, orig_labels: Sequence, target_labels: Sequence, dtype: DtypeLike = np.float32) -> None:
        """
        Args:
            orig_labels: original labels that map to others.
            target_labels: expected label values, 1: 1 map to the `orig_labels`.
            dtype: convert the output data to dtype, default to float32.
                if dtype is from PyTorch, the transform will use the pytorch backend, else with numpy backend.

        """
        if len(orig_labels) != len(target_labels):
            raise ValueError("orig_labels and target_labels must have the same length.")

        self.orig_labels = orig_labels
        self.target_labels = target_labels
        self.pair = tuple((o, t) for o, t in zip(self.orig_labels, self.target_labels) if o != t)
        type_dtype = type(dtype)
        if getattr(type_dtype, "__module__", "") == "torch":
            self.use_numpy = False
            self.dtype = get_equivalent_dtype(dtype, data_type=torch.Tensor)
        else:
            self.use_numpy = True
            self.dtype = get_equivalent_dtype(dtype, data_type=np.ndarray)

    def __call__(self, img: NdarrayOrTensor):
        if self.use_numpy:
            img_np, *_ = convert_data_type(img, np.ndarray)
            _out_shape = img_np.shape
            img_flat = img_np.flatten()
            try:
                out_flat = img_flat.astype(self.dtype)
            except ValueError:
                # can't copy unchanged labels as the expected dtype is not supported, must map all the label values
                out_flat = np.zeros(shape=img_flat.shape, dtype=self.dtype)
            for o, t in self.pair:
                out_flat[img_flat == o] = t
            out_t = out_flat.reshape(_out_shape)
        else:
            img_t, *_ = convert_data_type(img, torch.Tensor)
            out_t = img_t.detach().clone().to(self.dtype)  # type: ignore
            for o, t in self.pair:
                out_t[img_t == o] = t
        out, *_ = convert_to_dst_type(src=out_t, dst=img, dtype=self.dtype)
        return out


class IntensityStats(Transform):
    """
    Compute statistics for the intensity values of input image and store into the metadata dictionary.
    For example: if `ops=[lambda x: np.mean(x), "max"]` and `key_prefix="orig"`, may generate below stats:
    `{"orig_custom_0": 1.5, "orig_max": 3.0}`.

    Args:
        ops: expected operations to compute statistics for the intensity.
            if a string, will map to the predefined operations, supported: ["mean", "median", "max", "min", "std"]
            mapping to `np.nanmean`, `np.nanmedian`, `np.nanmax`, `np.nanmin`, `np.nanstd`.
            if a callable function, will execute the function on input image.
        key_prefix: the prefix to combine with `ops` name to generate the key to store the results in the
            metadata dictionary. if some `ops` are callable functions, will use "{key_prefix}_custom_{index}"
            as the key, where index counts from 0.
        channel_wise: whether to compute statistics for every channel of input image separately.
            if True, return a list of values for every operation, default to False.

    """

    backend = [TransformBackends.NUMPY]

    def __init__(self, ops: Sequence[str | Callable], key_prefix: str, channel_wise: bool = False) -> None:
        self.ops = ensure_tuple(ops)
        self.key_prefix = key_prefix
        self.channel_wise = channel_wise

    def __call__(
        self, img: NdarrayOrTensor, meta_data: dict | None = None, mask: np.ndarray | None = None
    ) -> tuple[NdarrayOrTensor, dict]:
        """
        Compute statistics for the intensity of input image.

        Args:
            img: input image to compute intensity stats.
            meta_data: metadata dictionary to store the statistics data, if None, will create an empty dictionary.
            mask: if not None, mask the image to extract only the interested area to compute statistics.
                mask must have the same shape as input `img`.

        """
        img_np, *_ = convert_data_type(img, np.ndarray)
        if meta_data is None:
            meta_data = {}

        if mask is not None:
            if mask.shape != img_np.shape:
                raise ValueError(f"mask must have the same shape as input `img`, got {mask.shape} and {img_np.shape}.")
            if mask.dtype != bool:
                raise TypeError(f"mask must be bool array, got type {mask.dtype}.")
            img_np = img_np[mask]

        supported_ops = {
            "mean": np.nanmean,
            "median": np.nanmedian,
            "max": np.nanmax,
            "min": np.nanmin,
            "std": np.nanstd,
        }

        def _compute(op: Callable, data: np.ndarray):
            if self.channel_wise:
                return [op(c) for c in data]
            return op(data)

        custom_index = 0
        for o in self.ops:
            if isinstance(o, str):
                o = look_up_option(o, supported_ops.keys())
                meta_data[self.key_prefix + "_" + o] = _compute(supported_ops[o], img_np)  # type: ignore
            elif callable(o):
                meta_data[self.key_prefix + "_custom_" + str(custom_index)] = _compute(o, img_np)
                custom_index += 1
            else:
                raise ValueError("ops must be key string for predefined operations or callable function.")

        return img, meta_data


class ToDevice(Transform):
    """
    Move PyTorch Tensor to the specified device.
    It can help cache data into GPU and execute following logic on GPU directly.

    Note:
        If moving data to GPU device in the multi-processing workers of DataLoader, may got below CUDA error:
        "RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing,
        you must use the 'spawn' start method."
        So usually suggest to set `num_workers=0` in the `DataLoader` or `ThreadDataLoader`.

    """

    backend = [TransformBackends.TORCH]

    def __init__(self, device: torch.device | str, **kwargs) -> None:
        """
        Args:
            device: target device to move the Tensor, for example: "cuda:1".
            kwargs: other args for the PyTorch `Tensor.to()` API, for more details:
                https://pytorch.org/docs/stable/generated/torch.Tensor.to.html.

        """
        self.device = device
        self.kwargs = kwargs

    def __call__(self, img: torch.Tensor):
        if not isinstance(img, torch.Tensor):
            raise ValueError("img must be PyTorch Tensor, consider converting img by `EnsureType` transform first.")

        return img.to(self.device, **self.kwargs)


class CuCIM(Transform):
    """
    Wrap a non-randomized cuCIM transform, defined based on the transform name and args.
    For randomized transforms use :py:class:`monai.transforms.RandCuCIM`.

    Args:
        name: the transform name in CuCIM package
        args: parameters for the CuCIM transform
        kwargs: parameters for the CuCIM transform

    Note:
        CuCIM transform only work with CuPy arrays, so this transform expects input data to be `cupy.ndarray`.
        Users can call `ToCuPy` transform to convert a numpy array or torch tensor to cupy array.
    """

    def __init__(self, name: str, *args, **kwargs) -> None:
        super().__init__()
        self.name = name
        self.transform, _ = optional_import("cucim.core.operations.expose.transform", name=name)
        self.args = args
        self.kwargs = kwargs

    def __call__(self, data):
        """
        Args:
            data: a CuPy array (`cupy.ndarray`) for the cuCIM transform

        Returns:
            `cupy.ndarray`

        """
        return self.transform(data, *self.args, **self.kwargs)


class RandCuCIM(CuCIM, RandomizableTrait):
    """
    Wrap a randomized cuCIM transform, defined based on the transform name and args
    For deterministic non-randomized transforms use :py:class:`monai.transforms.CuCIM`.

    Args:
        name: the transform name in CuCIM package.
        args: parameters for the CuCIM transform.
        kwargs: parameters for the CuCIM transform.

    Note:
        - CuCIM transform only work with CuPy arrays, so this transform expects input data to be `cupy.ndarray`.
          Users can call `ToCuPy` transform to convert a numpy array or torch tensor to cupy array.
        - If the random factor of the underlying cuCIM transform is not derived from `self.R`,
          the results may not be deterministic. See Also: :py:class:`monai.transforms.Randomizable`.
    """

    def __init__(self, name: str, *args, **kwargs) -> None:
        CuCIM.__init__(self, name, *args, **kwargs)


class AddCoordinateChannels(Transform):
    """
    Appends additional channels encoding coordinates of the input. Useful when e.g. training using patch-based sampling,
    to allow feeding of the patch's location into the network.

    This can be seen as a input-only version of CoordConv:

    Liu, R. et al. An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution, NeurIPS 2018.

    Args:
        spatial_dims: the spatial dimensions that are to have their coordinates encoded in a channel and
            appended to the input image. E.g., `(0, 1, 2)` represents `H, W, D` dims and append three channels
            to the input image, encoding the coordinates of the input's three spatial dimensions.

    """

    backend = [TransformBackends.NUMPY]

    def __init__(self, spatial_dims: Sequence[int]) -> None:
        self.spatial_dims = spatial_dims

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Args:
            img: data to be transformed, assuming `img` is channel first.
        """
        if max(self.spatial_dims) > img.ndim - 2 or min(self.spatial_dims) < 0:
            raise ValueError(f"`spatial_dims` values must be within [0, {img.ndim - 2}]")

        spatial_size = img.shape[1:]
        coord_channels = np.array(np.meshgrid(*tuple(np.linspace(-0.5, 0.5, s) for s in spatial_size), indexing="ij"))
        coord_channels, *_ = convert_to_dst_type(coord_channels, img)  # type: ignore
        coord_channels = coord_channels[list(self.spatial_dims)]
        return concatenate((img, coord_channels), axis=0)


class ImageFilter(Transform):
    """
    Applies a convolution filter to the input image.

    Args:
        filter:
            A string specifying the filter, a custom filter as ``torch.Tenor`` or ``np.ndarray`` or a ``nn.Module``.
            Available options for string are: ``mean``, ``laplace``, ``elliptical``, ``sobel``, ``sharpen``, ``median``, ``gauss``
            See below for short explanations on every filter.
        filter_size:
            A single integer value specifying the size of the quadratic or cubic filter.
            Computational complexity scales to the power of 2 (2D filter) or 3 (3D filter), which
            should be considered when choosing filter size.
        kwargs:
            Additional arguments passed to filter function, required by ``sobel`` and ``gauss``.
            See below for details.

    Raises:
        ValueError: When ``filter_size`` is not an uneven integer
        ValueError: When ``filter`` is an array and ``ndim`` is not in [1,2,3]
        ValueError: When ``filter`` is an array and any dimension has an even shape
        NotImplementedError: When ``filter`` is a string and not in ``self.supported_filters``
        KeyError: When necessary ``kwargs`` are not passed to a filter that requires additional arguments.


    **Mean Filtering:** ``filter='mean'``

    Mean filtering can smooth edges and remove aliasing artifacts in an segmentation image.
    See also py:func:`monai.networks.layers.simplelayers.MeanFilter`
    Example 2D filter (5 x 5)::

        [[1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1]]

    If smoothing labels with this filter, ensure they are in one-hot format.

    **Outline Detection:** ``filter='laplace'``

    Laplacian filtering for outline detection in images. Can be used to transform labels to contours.
    See also py:func:`monai.networks.layers.simplelayers.LaplaceFilter`

    Example 2D filter (5x5)::

        [[-1., -1., -1., -1., -1.],
         [-1., -1., -1., -1., -1.],
         [-1., -1., 24., -1., -1.],
         [-1., -1., -1., -1., -1.],
         [-1., -1., -1., -1., -1.]]


    **Dilation:** ``filter='elliptical'``

    An elliptical filter can be used to dilate labels or label-contours.
    Example 2D filter (5x5)::

        [[0., 0., 1., 0., 0.],
         [1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1.],
         [0., 0., 1., 0., 0.]]


    **Edge Detection:** ``filter='sobel'``

    This filter allows for additional arguments passed as ``kwargs`` during initialization.
    See also py:func:`monai.transforms.post.SobelGradients`

    *kwargs*

    * ``spatial_axes``: the axes that define the direction of the gradient to be calculated.
      It calculates the gradient along each of the provide axis.
      By default it calculate the gradient for all spatial axes.
    * ``normalize_kernels``: if normalize the Sobel kernel to provide proper gradients. Defaults to True.
    * ``normalize_gradients``: if normalize the output gradient to 0 and 1. Defaults to False.
    * ``padding_mode``: the padding mode of the image when convolving with Sobel kernels. Defaults to ``"reflect"``.
      Acceptable values are ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
      See ``torch.nn.Conv1d()`` for more information.
    * ``dtype``: kernel data type (torch.dtype). Defaults to ``torch.float32``.


    **Sharpening:** ``filter='sharpen'``

    Sharpen an image with a 2D or 3D filter.
    Example 2D filter (5x5)::

        [[ 0.,  0., -1.,  0.,  0.],
         [-1., -1., -1., -1., -1.],
         [-1., -1., 17., -1., -1.],
         [-1., -1., -1., -1., -1.],
         [ 0.,  0., -1.,  0.,  0.]]


    **Gaussian Smooth:** ``filter='gauss'``

    Blur/smooth an image with 2D or 3D gaussian filter.
    This filter requires additional arguments passed as ``kwargs`` during initialization.
    See also py:func:`monai.networks.layers.simplelayers.GaussianFilter`

    *kwargs*

    * ``sigma``: std. could be a single value, or spatial_dims number of values.
    * ``truncated``: spreads how many stds.
    * ``approx``: discrete Gaussian kernel type, available options are "erf", "sampled", and "scalespace".


    **Median Filter:** ``filter='median'``

    Blur an image with 2D or 3D median filter to remove noise.
    Useful in image preprocessing to improve results of later processing.
    See also py:func:`monai.networks.layers.simplelayers.MedianFilter`


    **Savitzky Golay Filter:** ``filter = 'savitzky_golay'``

    Convolve a Tensor along a particular axis with a Savitzky-Golay kernel.
    This filter requires additional arguments passed as ``kwargs`` during initialization.
    See also py:func:`monai.networks.layers.simplelayers.SavitzkyGolayFilter`

    *kwargs*

    * ``order``: Order of the polynomial to fit to each window, must be less than ``window_length``.
    * ``axis``: (optional): Axis along which to apply the filter kernel. Default 2 (first spatial dimension).
    * ``mode``: (string, optional): padding mode passed to convolution class. ``'zeros'``, ``'reflect'``, ``'replicate'`` or
      ``'circular'``. Default: ``'zeros'``. See torch.nn.Conv1d() for more information.

    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]
    supported_filters = sorted(
        ["mean", "laplace", "elliptical", "sobel", "sharpen", "median", "gauss", "savitzky_golay"]
    )

    def __init__(self, filter: str | NdarrayOrTensor | nn.Module, filter_size: int | None = None, **kwargs) -> None:
        self._check_filter_format(filter, filter_size)
        self._check_kwargs_are_present(filter, **kwargs)
        self.filter = filter
        self.filter_size = filter_size
        self.additional_args_for_filter = kwargs

    def __call__(
        self, img: NdarrayOrTensor, meta_dict: dict | None = None, applied_operations: list | None = None
    ) -> NdarrayOrTensor:
        """
        Args:
            img: torch tensor data to apply filter to with shape: [channels, height, width[, depth]]
            meta_dict: An optional dictionary with metadata
            applied_operations: An optional list of operations that have been applied to the data

        Returns:
            A MetaTensor with the same shape as `img` and identical metadata
        """
        if isinstance(img, MetaTensor):
            meta_dict = img.meta
            applied_operations = img.applied_operations

        img_, prev_type, device = convert_data_type(img, torch.Tensor)
        ndim = img_.ndim - 1  # assumes channel first format

        if isinstance(self.filter, str):
            self.filter = self._get_filter_from_string(self.filter, self.filter_size, ndim)  # type: ignore
        elif isinstance(self.filter, (torch.Tensor, np.ndarray)):
            self.filter = ApplyFilter(self.filter)

        img_ = self._apply_filter(img_)
        if meta_dict is not None or applied_operations is not None:
            img_ = MetaTensor(img_, meta=meta_dict, applied_operations=applied_operations)
        else:
            img_, *_ = convert_data_type(img_, prev_type, device)
        return img_

    def _check_all_values_uneven(self, x: tuple) -> None:
        for value in x:
            if value % 2 == 0:
                raise ValueError(f"Only uneven filters are supported, but filter size is {x}")

    def _check_filter_format(self, filter: str | NdarrayOrTensor | nn.Module, filter_size: int | None = None) -> None:
        if isinstance(filter, str):
            if filter != "gauss" and not filter_size:  # Gauss is the only filter that does not require `filter_size`
                raise ValueError("`filter_size` must be specified when specifying filters by string.")
            if filter_size and filter_size % 2 == 0:
                raise ValueError("`filter_size` should be a single uneven integer.")
            if filter not in self.supported_filters:
                raise NotImplementedError(f"{filter}. Supported filters are {self.supported_filters}.")
        elif isinstance(filter, (torch.Tensor, np.ndarray)):
            if filter.ndim not in [1, 2, 3]:
                raise ValueError("Only 1D, 2D, and 3D filters are supported.")
            self._check_all_values_uneven(filter.shape)
        elif not isinstance(filter, (nn.Module, Transform)):
            raise TypeError(
                f"{type(filter)} is not supported."
                "Supported types are `class 'str'`, `class 'torch.Tensor'`, `class 'np.ndarray'`, "
                "`class 'torch.nn.modules.module.Module'`, `class 'monai.transforms.Transform'`"
            )

    def _check_kwargs_are_present(self, filter: str | NdarrayOrTensor | nn.Module, **kwargs: Any) -> None:
        """
        Perform sanity checks on the kwargs if the filter contains the required keys.
        If the filter is ``gauss``, kwargs should contain ``sigma``.
        If the filter is ``savitzky_golay``, kwargs should contain ``order``.

        Args:
            filter: A string specifying the filter, a custom filter as ``torch.Tenor`` or ``np.ndarray`` or a ``nn.Module``.
            kwargs: additional arguments defining the filter.

        Raises:
            KeyError if the filter doesn't contain the requirement key.
        """

        if not isinstance(filter, str):
            return
        if filter == "gauss" and "sigma" not in kwargs.keys():
            raise KeyError("`filter='gauss', requires the additional keyword argument `sigma`")
        if filter == "savitzky_golay" and "order" not in kwargs.keys():
            raise KeyError("`filter='savitzky_golay', requires the additional keyword argument `order`")

    def _get_filter_from_string(self, filter: str, size: int, ndim: int) -> nn.Module | Callable:
        if filter == "mean":
            return MeanFilter(ndim, size)
        elif filter == "laplace":
            return LaplaceFilter(ndim, size)
        elif filter == "elliptical":
            return EllipticalFilter(ndim, size)
        elif filter == "sobel":
            from monai.transforms.post.array import SobelGradients  # cannot import on top because of circular imports

            allowed_keys = SobelGradients.__init__.__annotations__.keys()
            kwargs = {k: v for k, v in self.additional_args_for_filter.items() if k in allowed_keys}
            return SobelGradients(size, **kwargs)
        elif filter == "sharpen":
            return SharpenFilter(ndim, size)
        elif filter == "gauss":
            allowed_keys = GaussianFilter.__init__.__annotations__.keys()
            kwargs = {k: v for k, v in self.additional_args_for_filter.items() if k in allowed_keys}
            return GaussianFilter(ndim, **kwargs)
        elif filter == "median":
            return partial(median_filter, kernel_size=size, spatial_dims=ndim)
        elif filter == "savitzky_golay":
            allowed_keys = SavitzkyGolayFilter.__init__.__annotations__.keys()
            kwargs = {k: v for k, v in self.additional_args_for_filter.items() if k in allowed_keys}
            return SavitzkyGolayFilter(size, **kwargs)
        else:
            raise NotImplementedError(f"Filter {filter} not implemented")

    def _apply_filter(self, img: torch.Tensor) -> torch.Tensor:
        if isinstance(self.filter, Transform):
            img = self.filter(img)
        else:
            img = self.filter(img.unsqueeze(0))  # type: ignore
            img = img[0]  # add and remove batch dim
        return img


class RandImageFilter(RandomizableTransform):
    """
    Randomly apply a convolutional filter to the input data.

    Args:
        filter:
            A string specifying the filter or a custom filter as `torch.Tenor` or `np.ndarray`.
            Available options are: `mean`, `laplace`, `elliptical`, `gaussian``
            See below for short explanations on every filter.
        filter_size:
            A single integer value specifying the size of the quadratic or cubic filter.
            Computational complexity scales to the power of 2 (2D filter) or 3 (3D filter), which
            should be considered when choosing filter size.
        prob:
            Probability the transform is applied to the data
    """

    backend = ImageFilter.backend

    def __init__(
        self, filter: str | NdarrayOrTensor, filter_size: int | None = None, prob: float = 0.1, **kwargs
    ) -> None:
        super().__init__(prob)
        self.filter = ImageFilter(filter, filter_size, **kwargs)

    def __call__(self, img: NdarrayOrTensor, meta_dict: Mapping | None = None) -> NdarrayOrTensor:
        """
        Args:
            img: torch tensor data to apply filter to with shape: [channels, height, width[, depth]]
            meta_dict: An optional dictionary with metadata
            kwargs: optional arguments required by specific filters. E.g. `sigma`if filter is `gauss`.
                see py:func:`monai.transforms.utility.array.ImageFilter` for more details

        Returns:
            A MetaTensor with the same shape as `img` and identical metadata
        """
        self.randomize(None)
        if self._do_transform:
            img = self.filter(img)
        return img


class ApplyTransformToPoints(InvertibleTransform, Transform):
    """
    Transform points between image coordinates and world coordinates.
    The input coordinates are assumed to be in the shape (C, N, 2 or 3), where C represents the number of channels
    and N denotes the number of points. It will return a tensor with the same shape as the input.

    Args:
        dtype: The desired data type for the output.
        affine: A 3x3 or 4x4 affine transformation matrix applied to points. This matrix typically originates
            from the image. For 2D points, a 3x3 matrix can be provided, avoiding the need to add an unnecessary
            Z dimension. While a 4x4 matrix is required for 3D transformations, it's important to note that when
            applying a 4x4 matrix to 2D points, the additional dimensions are handled accordingly.
            The matrix is always converted to float64 for computation, which can be computationally
            expensive when applied to a large number of points.
            If None, will try to use the affine matrix from the input data.
        invert_affine: Whether to invert the affine transformation matrix applied to the points. Defaults to ``True``.
            Typically, the affine matrix is derived from an image and represents its location in world space,
            while the points are in world coordinates. A value of ``True`` represents transforming these
            world space coordinates to the image's coordinate space, and ``False`` the inverse of this operation.
        affine_lps_to_ras: Defaults to ``False``. Set to `True` if your point data is in the RAS coordinate system
            or you're using `ITKReader` with `affine_lps_to_ras=True`.
            This ensures the correct application of the affine transformation between LPS (left-posterior-superior)
            and RAS (right-anterior-superior) coordinate systems. This argument ensures the points and the affine
            matrix are in the same coordinate system.

    Use Cases:
        - Transforming points between world space and image space, and vice versa.
        - Automatically handling inverse transformations between image space and world space.
        - If points have an existing affine transformation, the class computes and
          applies the required delta affine transformation.

    """

    def __init__(
        self,
        dtype: DtypeLike | torch.dtype | None = None,
        affine: torch.Tensor | None = None,
        invert_affine: bool = True,
        affine_lps_to_ras: bool = False,
    ) -> None:
        self.dtype = dtype
        self.affine = affine
        self.invert_affine = invert_affine
        self.affine_lps_to_ras = affine_lps_to_ras

    def _compute_final_affine(self, affine: torch.Tensor, applied_affine: torch.Tensor | None = None) -> torch.Tensor:
        """
        Compute the final affine transformation matrix to apply to the point data.

        Args:
            data: Input coordinates assumed to be in the shape (C, N, 2 or 3).
            affine: 3x3 or 4x4 affine transformation matrix.

        Returns:
            Final affine transformation matrix.
        """

        affine = convert_data_type(affine, dtype=torch.float64)[0]

        if self.affine_lps_to_ras:
            affine = orientation_ras_lps(affine)

        if self.invert_affine:
            affine = linalg_inv(affine)
            if applied_affine is not None:
                affine = affine @ applied_affine

        return affine

    def transform_coordinates(
        self, data: torch.Tensor, affine: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict]:
        """
        Transform coordinates using an affine transformation matrix.

        Args:
            data: The input coordinates are assumed to be in the shape (C, N, 2 or 3),
                where C represents the number of channels and N denotes the number of points.
            affine: 3x3 or 4x4 affine transformation matrix. The matrix is always converted to float64 for computation,
                which can be computationally expensive when applied to a large number of points.

        Returns:
            Transformed coordinates.
        """
        data = convert_to_tensor(data, track_meta=get_track_meta())
        if affine is None and self.invert_affine:
            raise ValueError("affine must be provided when invert_affine is True.")
        # applied_affine is the affine transformation matrix that has already been applied to the point data
        applied_affine: torch.Tensor | None = getattr(data, "affine", None)
        affine = applied_affine if affine is None else affine
        if affine is None:
            raise ValueError("affine must be provided if data does not have an affine matrix.")

        final_affine = self._compute_final_affine(affine, applied_affine)
        out = apply_affine_to_points(data, final_affine, dtype=self.dtype)

        extra_info = {
            "invert_affine": self.invert_affine,
            "dtype": get_dtype_string(self.dtype),
            "image_affine": affine,
            "affine_lps_to_ras": self.affine_lps_to_ras,
        }

        xform = orientation_ras_lps(linalg_inv(final_affine)) if self.affine_lps_to_ras else linalg_inv(final_affine)
        meta_info = TraceableTransform.track_transform_meta(
            data, affine=xform, extra_info=extra_info, transform_info=self.get_transform_info()
        )

        return out, meta_info

    def __call__(self, data: torch.Tensor, affine: torch.Tensor | None = None):
        """
        Args:
            data: The input coordinates are assumed to be in the shape (C, N, 2 or 3),
                where C represents the number of channels and N denotes the number of points.
            affine: A 3x3 or 4x4 affine transformation matrix, this argument will take precedence over ``self.affine``.
        """
        if data.ndim != 3 or data.shape[-1] not in (2, 3):
            raise ValueError(f"data should be in shape (C, N, 2 or 3), got {data.shape}.")
        affine = self.affine if affine is None else affine
        if affine is not None and affine.shape not in ((3, 3), (4, 4)):
            raise ValueError(f"affine should be in shape (3, 3) or (4, 4), got {affine.shape}.")

        out, meta_info = self.transform_coordinates(data, affine)

        return out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else out

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        transform = self.pop_transform(data)
        inverse_transform = ApplyTransformToPoints(
            dtype=transform[TraceKeys.EXTRA_INFO]["dtype"],
            invert_affine=not transform[TraceKeys.EXTRA_INFO]["invert_affine"],
            affine_lps_to_ras=transform[TraceKeys.EXTRA_INFO]["affine_lps_to_ras"],
        )
        with inverse_transform.trace_transform(False):
            data = inverse_transform(data, transform[TraceKeys.EXTRA_INFO]["image_affine"])

        return data
