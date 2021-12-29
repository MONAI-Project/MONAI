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

"""Transforms using a smooth spatial field generated by interpolating from smaller randomized fields."""

from typing import Any, Optional, Sequence, Union

import numpy as np
import torch
from torch.nn.functional import grid_sample, interpolate

import monai
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.transform import Randomizable, RandomizableTransform
from monai.utils import GridSampleMode, GridSamplePadMode, InterpolateMode
from monai.utils.enums import TransformBackends
from monai.utils.module import look_up_option, pytorch_after
from monai.utils.type_conversion import convert_to_dst_type, convert_to_tensor

__all__ = ["SmoothField", "RandSmoothFieldAdjustContrast", "RandSmoothFieldAdjustIntensity", "RandSmoothDeform"]


class SmoothField(Randomizable):
    """
    Generate a smooth field array by defining a smaller randomized field and then reinterpolating to the desired size.

    This exploits interpolation to create a smoothly varying field used for other applications. An initial randomized
    field is defined with `rand_size` dimensions with `pad` number of values padding it along each dimension using
    `pad_val` as the value. If `spatial_size` is given this is interpolated to that size, otherwise if None the random
    array is produced uninterpolated. The output is always a Pytorch tensor allocated on the specified device.

    Args:
        rand_size: size of the randomized field to start from
        pad: number of pixels/voxels along the edges of the field to pad with `pad_val`
        pad_val: value with which to pad field edges
        low: low value for randomized field
        high: high value for randomized field
        channels: number of channels of final output
        spatial_size: final output size of the array, None to produce original uninterpolated field
        mode: interpolation mode for resizing the field
        align_corners: if True align the corners when upsampling field
        device: Pytorch device to define field on
    """

    def __init__(
        self,
        rand_size: Sequence[int],
        pad: int = 0,
        pad_val: float = 0,
        low: float = -1.0,
        high: float = 1.0,
        channels: int = 1,
        spatial_size: Optional[Sequence[int]] = None,
        mode: Union[InterpolateMode, str] = InterpolateMode.AREA,
        align_corners: Optional[bool] = None,
        device: Optional[torch.device] = None,
    ):
        self.rand_size = tuple(rand_size)
        self.pad = pad
        self.low = low
        self.high = high
        self.channels = channels
        self.mode = mode
        self.align_corners = align_corners
        self.device = device

        self.spatial_size: Optional[Sequence[int]] = None
        self.spatial_zoom: Optional[Sequence[float]] = None

        if low >= high:
            raise ValueError("Value for `low` must be less than `high` otherwise field will be zeros")

        self.total_rand_size = tuple(rs + self.pad * 2 for rs in self.rand_size)

        self.field = torch.ones((1, self.channels) + self.total_rand_size, device=self.device) * pad_val

        self.crand_size = (self.channels,) + self.rand_size

        pad_slice = slice(None) if self.pad == 0 else slice(self.pad, -self.pad)
        self.rand_slices = (0, slice(None)) + (pad_slice,) * len(self.rand_size)

        self.set_spatial_size(spatial_size)

    def randomize(self, data: Optional[Any] = None) -> None:
        self.field[self.rand_slices] = torch.from_numpy(self.R.uniform(self.low, self.high, self.crand_size))

    def set_spatial_size(self, spatial_size: Optional[Sequence[int]]) -> None:
        """
        Set the `spatial_size` and `spatial_zoom` attributes used for interpolating the field to the given
        dimension, or not interpolate at all if None.

        Args:
            spatial_size: new size to interpolate to, or None to not interpolate
        """
        if spatial_size is None:
            self.spatial_size = None
            self.spatial_zoom = None
        else:
            self.spatial_size = tuple(spatial_size)
            self.spatial_zoom = tuple(s / f for s, f in zip(self.spatial_size, self.total_rand_size))

    def set_mode(self, mode: Union[monai.utils.InterpolateMode, str]) -> None:
        self.mode = mode

    def __call__(self, randomize=False) -> torch.Tensor:
        if randomize:
            self.randomize()

        field = self.field.to(self.device).clone()

        if self.spatial_zoom is not None:
            resized_field = interpolate(  # type: ignore
                input=field,  # type: ignore
                scale_factor=self.spatial_zoom,
                mode=look_up_option(self.mode, InterpolateMode).value,
                align_corners=self.align_corners,
                recompute_scale_factor=False,
            )

            mina = resized_field.min()
            maxa = resized_field.max()
            minv = self.field.min()
            maxv = self.field.max()

            # faster than rescale_array (?)
            norm_field = (resized_field.squeeze(0) - mina).div_(maxa - mina)
            field = norm_field.mul_(maxv - minv).add_(minv)

        return field


class RandSmoothFieldAdjustContrast(RandomizableTransform):
    """
    Randomly adjust the contrast of input images by calculating a randomized smooth field for each invocation.

    This uses SmoothField internally to define the adjustment over the image. If `pad` is greater than 0 the
    edges of the input volume of that width will be mostly unchanged. Contrast is changed by raising input
    values by the power of the smooth field so the range of values given by `gamma` should be chosen with this
    in mind. For example, a minimum value of 0 in `gamma` will produce white areas so this should be avoided.
    Afte the contrast is adjusted the values of the result are rescaled to the range of the original input.

    Args:
        spatial_size: size of input array's spatial dimensions
        rand_size: size of the randomized field to start from
        pad: number of pixels/voxels along the edges of the field to pad with 1
        mode: interpolation mode to use when upsampling
        align_corners: if True align the corners when upsampling field
        prob: probability transform is applied
        gamma: (min, max) range for exponential field
        device: Pytorch device to define field on
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        spatial_size: Sequence[int],
        rand_size: Sequence[int],
        pad: int = 0,
        mode: Union[InterpolateMode, str] = InterpolateMode.AREA,
        align_corners: Optional[bool] = None,
        prob: float = 0.1,
        gamma: Union[Sequence[float], float] = (0.5, 4.5),
        device: Optional[torch.device] = None,
    ):
        super().__init__(prob)

        if isinstance(gamma, (int, float)):
            self.gamma = (0.5, gamma)
        else:
            if len(gamma) != 2:
                raise ValueError("Argument `gamma` should be a number or pair of numbers.")

            self.gamma = (min(gamma), max(gamma))

        self.sfield = SmoothField(
            rand_size=rand_size,
            pad=pad,
            pad_val=1,
            low=self.gamma[0],
            high=self.gamma[1],
            channels=1,
            spatial_size=spatial_size,
            mode=mode,
            align_corners=align_corners,
            device=device,
        )

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "RandSmoothFieldAdjustContrast":
        super().set_random_state(seed, state)
        self.sfield.set_random_state(seed, state)
        return self

    def randomize(self, data: Optional[Any] = None) -> None:
        super().randomize(None)

        if self._do_transform:
            self.sfield.randomize()

    def set_mode(self, mode: Union[monai.utils.InterpolateMode, str]) -> None:
        self.sfield.set_mode(mode)

    def __call__(self, img: NdarrayOrTensor, randomize: bool = True) -> NdarrayOrTensor:
        """
        Apply the transform to `img`, if `randomize` randomizing the smooth field otherwise reusing the previous.
        """
        if randomize:
            self.randomize()

        if not self._do_transform:
            return img

        img_min = img.min()
        img_max = img.max()
        img_rng = img_max - img_min

        field = self.sfield()
        rfield, *_ = convert_to_dst_type(field, img)

        # everything below here is to be computed using the destination type (numpy, tensor, etc.)

        img = (img - img_min) / (img_rng + 1e-10)  # rescale to unit values
        img = img ** rfield  # contrast is changed by raising image data to a power, in this case the field

        out = (img * img_rng) + img_min  # rescale back to the original image value range

        return out


class RandSmoothFieldAdjustIntensity(RandomizableTransform):
    """
    Randomly adjust the intensity of input images by calculating a randomized smooth field for each invocation.

    This uses SmoothField internally to define the adjustment over the image. If `pad` is greater than 0 the
    edges of the input volume of that width will be mostly unchanged. Intensity is changed by multiplying the
    inputs by the smooth field, so the values of `gamma` should be chosen with this in mind. The default values
    of `(0.1, 1.0)` are sensible in that values will not be zeroed out by the field nor multiplied greater than
    the original value range.

    Args:
        spatial_size: size of input array
        rand_size: size of the randomized field to start from
        pad: number of pixels/voxels along the edges of the field to pad with 1
        mode: interpolation mode to use when upsampling
        align_corners: if True align the corners when upsampling field
        prob: probability transform is applied
        gamma: (min, max) range of intensity multipliers
        device: Pytorch device to define field on
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        spatial_size: Sequence[int],
        rand_size: Sequence[int],
        pad: int = 0,
        mode: Union[InterpolateMode, str] = InterpolateMode.AREA,
        align_corners: Optional[bool] = None,
        prob: float = 0.1,
        gamma: Union[Sequence[float], float] = (0.1, 1.0),
        device: Optional[torch.device] = None,
    ):
        super().__init__(prob)

        if isinstance(gamma, (int, float)):
            self.gamma = (0.5, gamma)
        else:
            if len(gamma) != 2:
                raise ValueError("Argument `gamma` should be a number or pair of numbers.")

            self.gamma = (min(gamma), max(gamma))

        self.sfield = SmoothField(
            rand_size=rand_size,
            pad=pad,
            pad_val=1,
            low=self.gamma[0],
            high=self.gamma[1],
            channels=1,
            spatial_size=spatial_size,
            mode=mode,
            align_corners=align_corners,
            device=device,
        )

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "RandSmoothFieldAdjustIntensity":
        super().set_random_state(seed, state)
        self.sfield.set_random_state(seed, state)
        return self

    def randomize(self, data: Optional[Any] = None) -> None:
        super().randomize(None)

        if self._do_transform:
            self.sfield.randomize()

    def set_mode(self, mode: Union[InterpolateMode, str]) -> None:
        self.sfield.set_mode(mode)

    def __call__(self, img: NdarrayOrTensor, randomize: bool = True) -> NdarrayOrTensor:
        """
        Apply the transform to `img`, if `randomize` randomizing the smooth field otherwise reusing the previous.
        """

        if randomize:
            self.randomize()

        if not self._do_transform:
            return img

        field = self.sfield()
        rfield, *_ = convert_to_dst_type(field, img)

        # everything below here is to be computed using the destination type (numpy, tensor, etc.)

        out = img * rfield

        return out


class RandSmoothDeform(RandomizableTransform):
    """
    Deform an image using a random smooth field and Pytorch's grid_sample.

    The amount of deformation is given by `def_range` in fractions of the size of the image. The size of each dimension
    of the input image is always defined as 2 regardless of actual image voxel dimensions, that is the coordinates in
    every dimension range from -1 to 1. A value of 0.1 means pixels/voxels can be moved by up to 5% of the image's size.

    Args:
        spatial_size: input array size to which deformation grid is interpolated
        rand_size: size of the randomized field to start from
        pad: number of pixels/voxels along the edges of the field to pad with 0
        field_mode: interpolation mode to use when upsampling the deformation field
        align_corners: if True align the corners when upsampling field
        prob: probability transform is applied
        def_range: value of the deformation range in image size fractions, single min/max value  or min/max pair
        grid_dtype: type for the deformation grid calculated from the field
        grid_mode: interpolation mode used for sampling input using deformation grid
        grid_padding_mode: padding mode used for sampling input using deformation grid
        grid_align_corners: if True align the corners when sampling the deformation grid
        device: Pytorch device to define field on
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        spatial_size: Sequence[int],
        rand_size: Sequence[int],
        pad: int = 0,
        field_mode: Union[InterpolateMode, str] = InterpolateMode.AREA,
        align_corners: Optional[bool] = None,
        prob: float = 0.1,
        def_range: Union[Sequence[float], float] = 1.0,
        grid_dtype=torch.float32,
        grid_mode: Union[GridSampleMode, str] = GridSampleMode.NEAREST,
        grid_padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.BORDER,
        grid_align_corners: Optional[bool] = False,
        device: Optional[torch.device] = None,
    ):
        super().__init__(prob)

        self.grid_dtype = grid_dtype
        self.grid_mode = grid_mode
        self.def_range = def_range
        self.device = device
        self.grid_align_corners = grid_align_corners
        self.grid_padding_mode = grid_padding_mode

        if isinstance(def_range, (int, float)):
            self.def_range = (-def_range, def_range)
        else:
            if len(def_range) != 2:
                raise ValueError("Argument `def_range` should be a number or pair of numbers.")

            self.def_range = (min(def_range), max(def_range))

        self.sfield = SmoothField(
            spatial_size=spatial_size,
            rand_size=rand_size,
            pad=pad,
            low=self.def_range[0],
            high=self.def_range[1],
            channels=len(rand_size),
            mode=field_mode,
            align_corners=align_corners,
            device=device,
        )

        grid_space = spatial_size if spatial_size is not None else self.sfield.field.shape[2:]
        grid_ranges = [torch.linspace(-1, 1, d) for d in grid_space]

        if pytorch_after(1, 10):
            grid = torch.meshgrid(*grid_ranges, indexing="ij")
        else:
            grid = torch.meshgrid(*grid_ranges)

        self.grid = torch.stack(grid).unsqueeze(0).to(self.device, self.grid_dtype)

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "Randomizable":
        super().set_random_state(seed, state)
        self.sfield.set_random_state(seed, state)
        return self

    def randomize(self, data: Optional[Any] = None) -> None:
        super().randomize(None)

        if self._do_transform:
            self.sfield.randomize()

    def set_field_mode(self, mode: Union[monai.utils.InterpolateMode, str]) -> None:
        self.sfield.set_mode(mode)

    def set_grid_mode(self, mode: Union[monai.utils.GridSampleMode, str]) -> None:
        self.grid_mode = mode

    def __call__(
        self, img: NdarrayOrTensor, randomize: bool = True, device: Optional[torch.device] = None
    ) -> NdarrayOrTensor:
        if randomize:
            self.randomize()

        if not self._do_transform:
            return img

        device = device if device is not None else self.device

        field = self.sfield()

        dgrid = self.grid + field.to(self.grid_dtype)
        dgrid = dgrid.moveaxis(1, -1)

        img_t = convert_to_tensor(img[None], torch.float32, device)

        out = grid_sample(
            input=img_t,
            grid=dgrid,
            mode=look_up_option(self.grid_mode, GridSampleMode).value,
            align_corners=self.grid_align_corners,
            padding_mode=look_up_option(self.grid_padding_mode, GridSamplePadMode).value,
        )

        out_t, *_ = convert_to_dst_type(out.squeeze(0), img)

        return out_t
