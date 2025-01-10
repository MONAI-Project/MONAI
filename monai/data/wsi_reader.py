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

import warnings
from abc import abstractmethod
from collections.abc import Sequence
from os.path import abspath
from typing import Any

import numpy as np
import torch

from monai.config import DtypeLike, NdarrayOrTensor, PathLike
from monai.data.image_reader import ImageReader, _stack_images
from monai.data.utils import is_supported_format
from monai.utils import (
    WSIPatchKeys,
    dtype_numpy_to_torch,
    dtype_torch_to_numpy,
    ensure_tuple,
    ensure_tuple_rep,
    optional_import,
    require_pkg,
)
from monai.utils.misc import ConvertUnits

OpenSlide, _ = optional_import("openslide", name="OpenSlide")
TiffFile, _ = optional_import("tifffile", name="TiffFile")

__all__ = ["BaseWSIReader", "WSIReader", "CuCIMWSIReader", "OpenSlideWSIReader", "TiffFileWSIReader"]


class BaseWSIReader(ImageReader):
    """
    An abstract class that defines APIs to load patches from whole slide image files.

    Args:
        level: the whole slide image level at which the patches are extracted.
        mpp: the resolution in micron per pixel at which the patches are extracted.
        mpp_rtol: the acceptable relative tolerance for resolution in micro per pixel.
        mpp_atol: the acceptable absolute tolerance for resolution in micro per pixel.
        power: the objective power at which the patches are extracted.
        power_rtol: the acceptable relative tolerance for objective power.
        power_atol: the acceptable absolute tolerance for objective power.
        channel_dim: the desired dimension for color channel.
        dtype: the data type of output image.
        device: target device to put the extracted patch. Note that if device is "cuda"",
            the output will be converted to torch tenor and sent to the gpu even if the dtype is numpy.
        mode: the output image color mode, e.g., "RGB" or "RGBA".
        kwargs: additional args for the reader

        Notes:
            Only one of resolution parameters, `level`, `mpp`, or `power`, should be provided.
            If such parameters are provided in `get_data` method, those will override the values provided here.
            If none of them are provided here or in `get_data`, `level=0` will be used.

    Typical usage of a concrete implementation of this class is:

    .. code-block:: python

        image_reader = MyWSIReader()
        wsi = image_reader.read(filepath, **kwargs)
        img_data, meta_data = image_reader.get_data(wsi)

    - The `read` call converts an image filename into whole slide image object,
    - The `get_data` call fetches the image data, as well as metadata.

    The following methods needs to be implemented for any concrete implementation of this class:

    - `read` reads a whole slide image object from a given file
    - `get_size` returns the size of the whole slide image of a given wsi object at a given level.
    - `get_level_count` returns the number of levels in the whole slide image
    - `_get_patch` extracts and returns a patch image form the whole slide image
    - `_get_metadata` extracts and returns metadata for a whole slide image and a specific patch.


    """

    supported_suffixes: list[str] = []
    backend = ""

    def __init__(
        self,
        level: int | None = None,
        mpp: float | tuple[float, float] | None = None,
        mpp_rtol: float = 0.05,
        mpp_atol: float = 0.0,
        power: int | None = None,
        power_rtol: float = 0.05,
        power_atol: float = 0.0,
        channel_dim: int = 0,
        dtype: DtypeLike | torch.dtype = np.uint8,
        device: torch.device | str | None = None,
        mode: str = "RGB",
        **kwargs,
    ):
        super().__init__()
        self.level = level
        self.channel_dim = channel_dim
        self.set_dtype(dtype)
        self.set_device(device)
        self.mode = mode
        self.kwargs = kwargs
        self.mpp: tuple[float, float] | None = ensure_tuple_rep(mpp, 2) if mpp is not None else None
        self.power = power
        self.mpp_rtol = mpp_rtol
        self.mpp_atol = mpp_atol
        self.power_rtol = power_rtol
        self.power_atol = power_atol
        self.metadata: dict[Any, Any] = {}

    def set_dtype(self, dtype):
        self.dtype: torch.dtype | np.dtype
        if isinstance(dtype, torch.dtype):
            self.dtype = dtype
        else:
            self.dtype = np.dtype(dtype)

    def set_device(self, device):
        if device is None or isinstance(device, (torch.device, str)):
            self.device = device
        else:
            raise ValueError(f"`device` must be `torch.device`, `str` or `None` but {type(device)} is given.")

    @abstractmethod
    def get_size(self, wsi, level: int) -> tuple[int, int]:
        """
        Returns the size (height, width) of the whole slide image at a given level.

        Args:
            wsi: a whole slide image object loaded from a file.
            level: the level number where the size is calculated.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    def _find_closest_level(
        self, name: str, value: tuple, value_at_levels: Sequence[tuple], atol: float, rtol: float
    ) -> int:
        """Find the level corresponding to the value of the quantity in the list of values at each level.
        Args:
            name: the name of the requested quantity
            value: the value of requested quantity
            value_at_levels: list of value of the quantity at each level
            atol: the tolerance for the value
            rtol: relative tolerance for the value
        """
        if value in value_at_levels:
            return value_at_levels.index(value)

        closest_value = min(value_at_levels, key=lambda a_value: sum([abs(x - y) for x, y in zip(a_value, value)]))
        for i in range(len(value)):
            if abs(closest_value[i] - value[i]) > atol + rtol * abs(value[i]):
                raise ValueError(
                    f"The requested {name} < {value} > does not exist in this whole slide image "
                    f"(with {name}_rtol={rtol} and {name}_atol={atol}). "
                    f"Here is the list of available {name}: {value_at_levels}. "
                    f"The closest matching available {name} is {closest_value}."
                    f"Please consider changing the tolerances or use another {name}."
                )
        return value_at_levels.index(closest_value)

    def get_valid_level(
        self, wsi, level: int | None, mpp: float | tuple[float, float] | None, power: int | None
    ) -> int:
        """
        Returns the level associated to the resolution parameters in the whole slide image.

        Args:
            wsi: a whole slide image object loaded from a file.
            level: the level number.
            mpp: the micron-per-pixel resolution.
            power: the objective power.

        """

        # Try instance parameters if no resolution is provided
        if mpp is None and power is None and level is None:
            mpp = self.mpp
            power = self.power
            level = self.level

        # Ensure that at most one resolution parameter is provided.
        resolution = [val[0] for val in [("level", level), ("mpp", mpp), ("power", power)] if val[1] is not None]
        if len(resolution) > 1:
            raise ValueError(f"Only one of `level`, `mpp`, or `power` should be provided. {resolution} are provided.")

        n_levels = self.get_level_count(wsi)

        if mpp is not None:
            mpp_ = ensure_tuple_rep(mpp, 2)
            available_mpps = [self.get_mpp(wsi, level) for level in range(n_levels)]
            level = self._find_closest_level("mpp", mpp_, available_mpps, self.mpp_atol, self.mpp_rtol)
        elif power is not None:
            power_ = ensure_tuple(power)
            available_powers = [(self.get_power(wsi, level),) for level in range(n_levels)]
            level = self._find_closest_level("power", power_, available_powers, self.power_atol, self.power_rtol)
        else:
            if level is None:
                # Set the default value if no resolution parameter is provided.
                level = 0
            if level >= n_levels:
                raise ValueError(f"The maximum level of this image is {n_levels-1} while level={level} is requested)!")

        return level

    @abstractmethod
    def get_level_count(self, wsi) -> int:
        """
        Returns the number of levels in the whole slide image.

        Args:
            wsi: a whole slide image object loaded from a file.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    def get_downsample_ratio(self, wsi, level: int) -> float:
        """
        Returns the down-sampling ratio of the whole slide image at a given level.

        Args:
            wsi: a whole slide image object loaded from a file.
            level: the level number where the downsample ratio is calculated.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    def get_file_path(self, wsi) -> str:
        """Return the file path for the WSI object"""
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    def get_mpp(self, wsi, level: int) -> tuple[float, float]:
        """
        Returns the micro-per-pixel resolution of the whole slide image at a given level.

        Args:
            wsi: a whole slide image object loaded from a file.
            level: the level number where the mpp is calculated.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    def get_power(self, wsi, level: int) -> float:
        """
        Returns the objective power of the whole slide image at a given level.

        Args:
            wsi: a whole slide image object loaded from a file.
            level: the level number where the objective power is calculated.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    def _get_patch(
        self, wsi, location: tuple[int, int], size: tuple[int, int], level: int, dtype: DtypeLike, mode: str
    ) -> np.ndarray:
        """
        Extracts and returns a patch image form the whole slide image.

        Args:
            wsi: a whole slide image object loaded from a file or a lis of such objects
            location: (top, left) tuple giving the top left pixel in the level 0 reference frame. Defaults to (0, 0).
            size: (height, width) tuple giving the patch size at the given level (`level`).
                If None, it is set to the full image size at the given level.
            level: the level number.
            dtype: the data type of output image.
            mode: the output image mode, 'RGB' or 'RGBA'.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    def _get_metadata(
        self, wsi, patch: NdarrayOrTensor, location: tuple[int, int], size: tuple[int, int], level: int
    ) -> dict:
        """
        Returns metadata of the extracted patch from the whole slide image.

        Args:
            wsi: the whole slide image object, from which the patch is loaded.
            patch: extracted patch from whole slide image.
            location: (top, left) tuple giving the top left pixel in the level 0 reference frame. Defaults to (0, 0).
            size: (height, width) tuple giving the patch size at the given level (`level`).
                If None, it is set to the full image size at the given level.
            level: the level number.

        """
        if self.channel_dim >= len(patch.shape) or self.channel_dim < -len(patch.shape):
            raise ValueError(
                f"The desired channel_dim ({self.channel_dim}) is out of bound for image shape: {patch.shape}"
            )
        channel_dim: int = self.channel_dim + (len(patch.shape) if self.channel_dim < 0 else 0)
        metadata: dict = {
            "backend": self.backend,
            "original_channel_dim": channel_dim,
            "spatial_shape": np.array(patch.shape[:channel_dim] + patch.shape[channel_dim + 1 :]),
            WSIPatchKeys.COUNT.value: 1,
            WSIPatchKeys.PATH.value: self.get_file_path(wsi),
            WSIPatchKeys.LOCATION.value: np.asarray(location),
            WSIPatchKeys.SIZE.value: np.asarray(size),
            WSIPatchKeys.LEVEL.value: level,
        }
        return metadata

    def get_data(
        self,
        wsi,
        location: tuple[int, int] = (0, 0),
        size: tuple[int, int] | None = None,
        level: int | None = None,
        mpp: float | tuple[float, float] | None = None,
        power: int | None = None,
        mode: str | None = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Verifies inputs, extracts patches from WSI image and generates metadata.

        Args:
            wsi: a whole slide image object loaded from a file or a list of such objects.
            location: (top, left) tuple giving the top left pixel in the level 0 reference frame. Defaults to (0, 0).
            size: (height, width) tuple giving the patch size at the given level (`level`).
                If not provided or None, it is set to the full image size at the given level.
            level: the whole slide image level at which the patches are extracted.
            mpp: the resolution in micron per pixel at which the patches are extracted.
            power: the objective power at which the patches are extracted.
            dtype: the data type of output image.
            mode: the output image mode, 'RGB' or 'RGBA'.

        Returns:
            a tuples, where the first element is an image patch [CxHxW] or stack of patches,
                and second element is a dictionary of metadata.

        Notes:
            Only one of resolution parameters, `level`, `mpp`, or `power`, should be provided.
            If none of them are provided, it uses the defaults that are set during class instantiation.
            If none of them are set here or during class instantiation, `level=0` will be used.
        """
        if mode is None:
            mode = self.mode
        patch_list: list = []
        metadata_list: list = []

        # CuImage object is iterable, so ensure_tuple won't work on single object
        if not isinstance(wsi, (list, tuple)):
            wsi = (wsi,)
        for each_wsi in ensure_tuple(wsi):
            # get the valid level based on resolution info
            level = self.get_valid_level(each_wsi, level, mpp, power)

            # Verify location
            if location is None:
                location = (0, 0)
            wsi_size = self.get_size(each_wsi, 0)
            if location[0] > wsi_size[0] or location[1] > wsi_size[1]:
                raise ValueError(f"Location is outside of the image: location={location}, image size={wsi_size}")

            # Verify size
            if size is None:
                if location != (0, 0):
                    raise ValueError("Patch size should be defined to extract patches.")
                size = self.get_size(each_wsi, level)
            else:
                if size[0] <= 0 or size[1] <= 0:
                    raise ValueError(f"Patch size should be greater than zero, provided: patch size = {size}")

            # Get numpy dtype if it is not already.
            dtype_np = dtype_torch_to_numpy(self.dtype) if isinstance(self.dtype, torch.dtype) else self.dtype
            # Extract a patch or the entire image
            patch: NdarrayOrTensor
            patch = self._get_patch(each_wsi, location=location, size=size, level=level, dtype=dtype_np, mode=mode)

            # Convert the patch to torch.Tensor if dtype is torch
            if isinstance(self.dtype, torch.dtype) or (
                self.device is not None and torch.device(self.device).type == "cuda"
            ):
                # Ensure dtype is torch.dtype if the device is not "cpu"
                dtype_torch = (
                    dtype_numpy_to_torch(self.dtype) if not isinstance(self.dtype, torch.dtype) else self.dtype
                )
                # Copy the numpy array if it is not writable
                if patch.flags["WRITEABLE"]:
                    patch = torch.as_tensor(patch, dtype=dtype_torch, device=self.device)
                else:
                    patch = torch.tensor(patch, dtype=dtype_torch, device=self.device)

            # check if the image has three dimensions (2D + color)
            if patch.ndim != 3:
                raise ValueError(
                    f"The image dimension should be 3 but has {patch.ndim}. "
                    "`WSIReader` is designed to work only with 2D images with color channel."
                )
            # Check if there are four color channels for RGBA
            if mode == "RGBA":
                if patch.shape[self.channel_dim] != 4:
                    raise ValueError(
                        f"The image is expected to have four color channels in '{mode}' mode but has "
                        f"{patch.shape[self.channel_dim]}."
                    )
            # Check if there are three color channels for RGB
            elif mode in "RGB" and patch.shape[self.channel_dim] != 3:
                raise ValueError(
                    f"The image is expected to have three color channels in '{mode}' mode but has "
                    f"{patch.shape[self.channel_dim]}. "
                )
            # Get patch-related metadata
            metadata: dict = self._get_metadata(wsi=each_wsi, patch=patch, location=location, size=size, level=level)
            # Create a list of patches and metadata
            patch_list.append(patch)
            metadata_list.append(metadata)
        if len(wsi) > 1:
            if len({m["original_channel_dim"] for m in metadata_list}) > 1:
                raise ValueError("original_channel_dim is not consistent across wsi objects.")
            if len({tuple(m["spatial_shape"]) for m in metadata_list}) > 1:
                raise ValueError("spatial_shape is not consistent across wsi objects.")
            for key in WSIPatchKeys:
                metadata[key] = [m[key] for m in metadata_list]
        return _stack_images(patch_list, metadata), metadata

    def _compute_mpp_target_res(self, closest_lvl, closest_lvl_dim, mpp_list, mpp: tuple):
        """
        Computes the target dimensions for resizing a whole slide image
        to match a user-specified resolution in microns per pixel (MPP).

        Args:
            closest_lvl: Whole slide image level closest to user-provided MPP resolution.
            closest_lvl_dim: Dimensions (height, width) of the image at the closest level.
            mpp_list: List of MPP values for all levels of the whole slide image.
            mpp: The MPP resolution at which the whole slide image representation should be extracted.

        """
        mpp_closest_lvl = mpp_list[closest_lvl]
        mpp_closest_lvl_x, mpp_closest_lvl_y = mpp_closest_lvl

        ds_factor_x = mpp_closest_lvl_x / mpp[0]
        ds_factor_y = mpp_closest_lvl_y / mpp[1]

        target_res_x = int(np.round(closest_lvl_dim[1] * ds_factor_x))
        target_res_y = int(np.round(closest_lvl_dim[0] * ds_factor_y))

        return target_res_x, target_res_y

    def _compute_mpp_tolerances(self, closest_lvl, mpp_list, mpp, atol, rtol) -> bool:
        """
        Determines if user-provided MPP values are within a specified tolerance of the closest
        level's MPP and checks if the closest level has higher resolution than desired MPP.

        Args:
            closest_lvl: Whole slide image level closest to user-provided MPP resolution.
            mpp_list: List of MPP values for all levels of the whole slide image.
            mpp: The MPP resolution at which the whole slide image representation should be extracted.
            atol: Absolute tolerance for MPP comparison.
            rtol: Relative tolerance for MPP comparison.

        """
        user_mpp_x, user_mpp_y = mpp
        mpp_closest_lvl_x, mpp_closest_lvl_y = mpp_list[closest_lvl]

        # Define tolerance intervals for x and y of closest level
        lower_bound_x = mpp_closest_lvl_x * (1 - rtol) - atol
        upper_bound_x = mpp_closest_lvl_x * (1 + rtol) + atol
        lower_bound_y = mpp_closest_lvl_y * (1 - rtol) - atol
        upper_bound_y = mpp_closest_lvl_y * (1 + rtol) + atol

        # Check if user-provided mpp_x and mpp_y fall within the tolerance intervals for closest level
        is_within_tolerance_x = (user_mpp_x >= lower_bound_x) & (user_mpp_x <= upper_bound_x)
        is_within_tolerance_y = (user_mpp_y >= lower_bound_y) & (user_mpp_y <= upper_bound_y)
        is_within_tolerance = is_within_tolerance_x & is_within_tolerance_y

        # If mpp_closest_level < mpp -> closest_level has higher res than img at mpp => downscale from closest_level to mpp
        closest_level_is_bigger_x = mpp_closest_lvl_x < user_mpp_x
        closest_level_is_bigger_y = mpp_closest_lvl_y < user_mpp_y
        closest_level_is_bigger = closest_level_is_bigger_x & closest_level_is_bigger_y

        return is_within_tolerance, closest_level_is_bigger

    def verify_suffix(self, filename: Sequence[PathLike] | PathLike) -> bool:
        """
        Verify whether the specified file or files format is supported by WSI reader.

        The list of supported suffixes are read from `self.supported_suffixes`.

        Args:
            filename: filename or a list of filenames to read.

        """
        return is_supported_format(filename, self.supported_suffixes)


class WSIReader(BaseWSIReader):
    """
    Read whole slide images and extract patches using different backend libraries

    Args:
        backend: the name of backend whole slide image reader library, the default is cuCIM.
        level: the whole slide image level at which the patches are extracted.
        mpp: the resolution in micron per pixel at which the patches are extracted.
        mpp_rtol: the acceptable relative tolerance for resolution in micro per pixel.
        mpp_atol: the acceptable absolute tolerance for resolution in micro per pixel.
        power: the objective power at which the patches are extracted.
        power_rtol: the acceptable relative tolerance for objective power.
        power_atol: the acceptable absolute tolerance for objective power.
        channel_dim: the desired dimension for color channel. Default to 0 (channel first).
        dtype: the data type of output image. Defaults to `np.uint8`.
        device: target device to put the extracted patch. Note that if device is "cuda"",
            the output will be converted to torch tenor and sent to the gpu even if the dtype is numpy.
        mode: the output image color mode, "RGB" or "RGBA". Defaults to "RGB".
        num_workers: number of workers for multi-thread image loading (cucim backend only).
        kwargs: additional arguments to be passed to the backend library

        Notes:
            Only one of resolution parameters, `level`, `mpp`, or `power`, should be provided.
            If such parameters are provided in `get_data` method, those will override the values provided here.
            If none of them are provided here or in `get_data`, `level=0` will be used.

    """

    supported_backends = ["cucim", "openslide", "tifffile"]

    def __init__(
        self,
        backend="cucim",
        level: int | None = None,
        mpp: float | tuple[float, float] | None = None,
        mpp_rtol: float = 0.05,
        mpp_atol: float = 0.0,
        power: int | None = None,
        power_rtol: float = 0.05,
        power_atol: float = 0.0,
        channel_dim: int = 0,
        dtype: DtypeLike | torch.dtype = np.uint8,
        device: torch.device | str | None = None,
        mode: str = "RGB",
        **kwargs,
    ):
        self.backend = backend.lower()
        self.reader: CuCIMWSIReader | OpenSlideWSIReader | TiffFileWSIReader
        if self.backend == "cucim":
            self.reader = CuCIMWSIReader(
                level=level,
                mpp=mpp,
                mpp_rtol=mpp_rtol,
                mpp_atol=mpp_atol,
                power=power,
                power_rtol=power_rtol,
                power_atol=power_atol,
                channel_dim=channel_dim,
                dtype=dtype,
                device=device,
                mode=mode,
                **kwargs,
            )
        elif self.backend == "openslide":
            self.reader = OpenSlideWSIReader(
                level=level,
                mpp=mpp,
                mpp_rtol=mpp_rtol,
                mpp_atol=mpp_atol,
                power=power,
                power_rtol=power_rtol,
                power_atol=power_atol,
                channel_dim=channel_dim,
                dtype=dtype,
                device=device,
                mode=mode,
                **kwargs,
            )
        elif self.backend == "tifffile":
            self.reader = TiffFileWSIReader(
                level=level,
                mpp=mpp,
                mpp_rtol=mpp_rtol,
                mpp_atol=mpp_atol,
                power=power,
                power_rtol=power_rtol,
                power_atol=power_atol,
                channel_dim=channel_dim,
                dtype=dtype,
                device=device,
                mode=mode,
                **kwargs,
            )
        else:
            raise ValueError(
                f"The supported backends are cucim, openslide, and tifffile but '{self.backend}' was given."
            )
        self.supported_suffixes = self.reader.supported_suffixes
        self.level = self.reader.level
        self.mpp_rtol = self.reader.mpp_rtol
        self.mpp_atol = self.reader.mpp_atol
        self.power = self.reader.power
        self.power_rtol = self.reader.power_rtol
        self.power_atol = self.reader.power_atol
        self.channel_dim = self.reader.channel_dim
        self.dtype = self.reader.dtype
        self.device = self.reader.device
        self.mode = self.reader.mode
        self.kwargs = self.reader.kwargs
        self.metadata = self.reader.metadata
        self.mpp = self.reader.mpp

    def get_level_count(self, wsi) -> int:
        """
        Returns the number of levels in the whole slide image.

        Args:
            wsi: a whole slide image object loaded from a file.

        """
        return self.reader.get_level_count(wsi)

    def get_size(self, wsi, level: int) -> tuple[int, int]:
        """
        Returns the size (height, width) of the whole slide image at a given level.

        Args:
            wsi: a whole slide image object loaded from a file.
            level: the level number where the size is calculated.

        """
        return self.reader.get_size(wsi, level)

    def get_downsample_ratio(self, wsi, level: int) -> float:
        """
        Returns the down-sampling ratio of the whole slide image at a given level.

        Args:
            wsi: a whole slide image object loaded from a file.
            level: the level number where the downsample ratio is calculated.

        """
        return self.reader.get_downsample_ratio(wsi, level)

    def get_file_path(self, wsi) -> str:
        """Return the file path for the WSI object"""
        return self.reader.get_file_path(wsi)

    def get_mpp(self, wsi, level: int) -> tuple[float, float]:
        """
        Returns the micro-per-pixel resolution of the whole slide image at a given level.

        Args:
            wsi: a whole slide image object loaded from a file.
            level: the level number where the mpp is calculated.

        """
        return self.reader.get_mpp(wsi, level)

    def get_wsi_at_mpp(self, wsi, mpp: tuple, atol: float = 0.00, rtol: float = 0.05) -> np.ndarray:
        """
        Returns the representation of the whole slide image at a given micro-per-pixel (mpp) resolution.
        The optional tolerance parameters are considered at the level whose mpp value is closest to the one provided by the user.
        If the user-provided mpp is larger than the mpp of the closest level,
        the image is downscaled to a resolution that matches the user-provided mpp.
        Otherwise, if the closest level's resolution is not sufficient to meet the user's requested resolution,
        the next lower level (which has a higher resolution) is chosen.
        The image from this level is then down-scaled to achieve a resolution at the user-provided mpp value.

        Args:
            wsi: whole slide image object from WSIReader
            mpp: the resolution in micron per pixel at which the representation of the whole slide image should be extracted.
            atol: the acceptable absolute tolerance for resolution in micro per pixel.
            rtol: the acceptable relative tolerance for resolution in micro per pixel.

        """
        return self.reader.get_wsi_at_mpp(wsi, mpp, atol, rtol)

    def get_power(self, wsi, level: int) -> float:
        """
        Returns the micro-per-pixel resolution of the whole slide image at a given level.

        Args:
            wsi: a whole slide image object loaded from a file.
            level: the level number where the objective power is calculated.

        """
        return self.reader.get_power(wsi, level)

    def _get_patch(
        self, wsi, location: tuple[int, int], size: tuple[int, int], level: int, dtype: DtypeLike, mode: str
    ) -> np.ndarray:
        """
        Extracts and returns a patch image form the whole slide image.

        Args:
            wsi: a whole slide image object loaded from a file or a lis of such objects.
            location: (top, left) tuple giving the top left pixel in the level 0 reference frame. Defaults to (0, 0).
            size: (height, width) tuple giving the patch size at the given level (`level`).
                If None, it is set to the full image size at the given level.
            level: the level number.
            dtype: the data type of output image
            mode: the output image mode, 'RGB' or 'RGBA'.

        """
        return self.reader._get_patch(wsi=wsi, location=location, size=size, level=level, dtype=dtype, mode=mode)

    def read(self, data: Sequence[PathLike] | PathLike | np.ndarray, **kwargs):
        """
        Read whole slide image objects from given file or list of files.

        Args:
            data: file name or a list of file names to read.
            kwargs: additional args for the reader module (overrides `self.kwargs` for existing keys).

        Returns:
            whole slide image object or list of such objects.

        """
        return self.reader.read(data=data, **kwargs)


@require_pkg(pkg_name="cucim")
class CuCIMWSIReader(BaseWSIReader):
    """
    Read whole slide images and extract patches using cuCIM library.

    Args:
        level: the whole slide image level at which the patches are extracted.
        mpp: the resolution in micron per pixel at which the patches are extracted.
        mpp_rtol: the acceptable relative tolerance for resolution in micro per pixel.
        mpp_atol: the acceptable absolute tolerance for resolution in micro per pixel.
        power: the objective power at which the patches are extracted.
        power_rtol: the acceptable relative tolerance for objective power.
        power_atol: the acceptable absolute tolerance for objective power.
        channel_dim: the desired dimension for color channel. Default to 0 (channel first).
        dtype: the data type of output image. Defaults to `np.uint8`.
        device: target device to put the extracted patch. Note that if device is "cuda"",
            the output will be converted to torch tenor and sent to the gpu even if the dtype is numpy.
        mode: the output image color mode, "RGB" or "RGBA". Defaults to "RGB".
        num_workers: number of workers for multi-thread image loading.
        kwargs: additional args for `cucim.CuImage` module:
            https://github.com/rapidsai/cucim/blob/main/cpp/include/cucim/cuimage.h

        Notes:
            Only one of resolution parameters, `level`, `mpp`, or `power`, should be provided.
            If such parameters are provided in `get_data` method, those will override the values provided here.
            If none of them are provided here or in `get_data`, `level=0` will be used.

    """

    supported_suffixes = ["tif", "tiff", "svs"]
    backend = "cucim"

    def __init__(self, num_workers: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.num_workers = num_workers

    @staticmethod
    def get_level_count(wsi) -> int:
        """
        Returns the number of levels in the whole slide image.

        Args:
            wsi: a whole slide image object loaded from a file.

        """
        return wsi.resolutions["level_count"]  # type: ignore

    def get_size(self, wsi, level: int) -> tuple[int, int]:
        """
        Returns the size (height, width) of the whole slide image at a given level.

        Args:
            wsi: a whole slide image object loaded from a file.
            level: the level number where the size is calculated.

        """
        return (wsi.resolutions["level_dimensions"][level][1], wsi.resolutions["level_dimensions"][level][0])

    def get_downsample_ratio(self, wsi, level: int) -> float:
        """
        Returns the down-sampling ratio of the whole slide image at a given level.

        Args:
            wsi: a whole slide image object loaded from a file.
            level: the level number where the downsample ratio is calculated.

        """
        return float(wsi.resolutions["level_downsamples"][level])

    @staticmethod
    def get_file_path(wsi) -> str:
        """Return the file path for the WSI object"""
        return str(abspath(wsi.path))

    def get_mpp(self, wsi, level: int) -> tuple[float, float]:
        """
        Returns the micro-per-pixel resolution of the whole slide image at a given level.

        Args:
            wsi: a whole slide image object loaded from a file.
            level: the level number where the mpp is calculated.

        """
        downsample_ratio = self.get_downsample_ratio(wsi, level)

        if "aperio" in wsi.metadata:
            mpp_ = wsi.metadata["aperio"].get("MPP")
            if mpp_:
                return (downsample_ratio * float(mpp_),) * 2
        if "cucim" in wsi.metadata:
            mpp_ = wsi.metadata["cucim"].get("spacing")
            if mpp_ and isinstance(mpp_, Sequence) and len(mpp_) >= 2:
                if mpp_[0] and mpp_[1]:
                    return (downsample_ratio * mpp_[1], downsample_ratio * mpp_[0])

        raise ValueError("`mpp` cannot be obtained for this file. Please use `level` instead.")

    def get_wsi_at_mpp(self, wsi, mpp: tuple, atol: float = 0.00, rtol: float = 0.05) -> Any:
        """
        Returns the representation of the whole slide image at a given micro-per-pixel (mpp) resolution.
        The optional tolerance parameters are considered at the level whose mpp value is closest to the one provided by the user.
        If the user-provided mpp is larger than the mpp of the closest level,
        the image is downscaled to a resolution that matches the user-provided mpp.
        Otherwise, if the closest level's resolution is not sufficient to meet the user's requested resolution,
        the next lower level (which has a higher resolution) is chosen.
        The image from this level is then down-scaled to achieve a resolution at the user-provided mpp value.

        Args:
            wsi: whole slide image object from WSIReader
            mpp: the resolution in micron per pixel at which the representation of the whole slide image should be extracted.
            atol: the acceptable absolute tolerance for resolution in micro per pixel.
            rtol: the acceptable relative tolerance for resolution in micro per pixel.

        """
        cp, _ = optional_import("cupy")

        mpp_list = [self.get_mpp(wsi, lvl) for lvl in range(wsi.resolutions["level_count"])]
        closest_lvl = self._find_closest_level("mpp", mpp, mpp_list, 0, 5)

        within_tolerance, closest_level_is_bigger = super()._compute_mpp_tolerances(closest_lvl, mpp_list, mpp, atol, rtol)

        if within_tolerance:
            # If the image at the desired mpp resolution is within tolerances, return the image at closest_level.
            closest_lvl_wsi = wsi.read_region(
                (0, 0), level=closest_lvl, size=wsi.resolutions["level_dimensions"][closest_lvl], num_workers=self.num_workers
            )

        elif closest_level_is_bigger:
            # Otherwise, select the level closest to the desired mpp with a higher resolution and downsample it.
            closest_lvl_wsi = self._resize_to_mpp_res(wsi, closest_lvl, mpp_list, mpp)

        else:
            # If both checks fail, increase resolution (i.e., decrement level) and then downsample it.
            closest_lvl = closest_lvl - 1
            closest_lvl_wsi = self._resize_to_mpp_res(wsi, closest_lvl, mpp_list, mpp)

        wsi_arr = cp.asnumpy(closest_lvl_wsi)
        return wsi_arr

    def get_power(self, wsi, level: int) -> float:
        """
        Returns the objective power of the whole slide image at a given level.

        Args:
            wsi: a whole slide image object loaded from a file.
            level: the level number where the objective power is calculated.

        """
        if "aperio" in wsi.metadata:
            objective_power = wsi.metadata["aperio"].get("AppMag")
            if objective_power:
                downsample_ratio = self.get_downsample_ratio(wsi, level)
                return float(objective_power) / downsample_ratio

        raise ValueError(
            "Currently, cuCIM backend can obtain the objective power only for Aperio images. "
            "Please use `level` (or `mpp`) instead, or try OpenSlide backend."
        )

    def read(self, data: Sequence[PathLike] | PathLike | np.ndarray, **kwargs):
        """
        Read whole slide image objects from given file or list of files.

        Args:
            data: file name or a list of file names to read.
            kwargs: additional args that overrides `self.kwargs` for existing keys.
                For more details look at https://github.com/rapidsai/cucim/blob/main/cpp/include/cucim/cuimage.h

        Returns:
            whole slide image object or list of such objects.

        """
        cuimage_cls, _ = optional_import("cucim", name="CuImage")
        wsi_list: list = []

        filenames: Sequence[PathLike] = ensure_tuple(data)
        kwargs_ = self.kwargs.copy()
        kwargs_.update(kwargs)
        for filename in filenames:
            wsi = cuimage_cls(filename, **kwargs_)
            wsi_list.append(wsi)

        return wsi_list if len(filenames) > 1 else wsi_list[0]

    def _get_patch(
        self, wsi, location: tuple[int, int], size: tuple[int, int], level: int, dtype: DtypeLike, mode: str
    ) -> np.ndarray:
        """
        Extracts and returns a patch image form the whole slide image.

        Args:
            wsi: a whole slide image object loaded from a file or a lis of such objects.
            location: (top, left) tuple giving the top left pixel in the level 0 reference frame. Defaults to (0, 0).
            size: (height, width) tuple giving the patch size at the given level (`level`).
                If None, it is set to the full image size at the given level.
            level: the level number.
            dtype: the data type of output image.
            mode: the output image mode, 'RGB' or 'RGBA'.

        """
        # Extract a patch or the entire image
        # (reverse the order of location and size to become WxH for cuCIM)
        patch: np.ndarray = wsi.read_region(
            location=location[::-1], size=size[::-1], level=level, num_workers=self.num_workers
        )

        # Convert to numpy
        patch = np.asarray(patch, dtype=dtype)

        # Make the channel to desired dimensions
        patch = np.moveaxis(patch, -1, self.channel_dim)

        # Check if the color channel is 3 (RGB) or 4 (RGBA)
        if mode in "RGB":
            if patch.shape[self.channel_dim] not in [3, 4]:
                raise ValueError(
                    f"The image is expected to have three or four color channels in '{mode}' mode but has "
                    f"{patch.shape[self.channel_dim]}. "
                )
            patch = np.take(patch, [0, 1, 2], self.channel_dim)

        return patch

    def _resize_to_mpp_res(self, wsi, closest_lvl, mpp_list, user_mpp: tuple):
        """
        Resizes the whole slide image to the specified resolution in microns per pixel (mpp).

        Args:
            wsi: whole slide image object from WSIReader
            user_mpp: the resolution in microns per pixel at which the whole slide image representation should be extracted.
            closest_lvl: the wsi level that is closest to the user-provided mpp resolution.
            mpp_list: list of mpp values for all levels of a whole slide image.

        """
        cucim_resize, _ = optional_import("cucim.skimage.transform", name="resize")
        cp, _ = optional_import("cupy")

        closest_lvl_dim = wsi.resolutions["level_dimensions"][closest_lvl]

        target_res_x, target_res_y = super()._compute_mpp_target_res(closest_lvl, closest_lvl_dim, mpp_list, user_mpp)

        wsi_arr = cp.array(wsi.read_region((0, 0), level=closest_lvl, size=closest_lvl_dim, num_workers=self.num_workers))
        closest_lvl_wsi = cucim_resize(wsi_arr, (target_res_x, target_res_y), order=1)

        return closest_lvl_wsi


@require_pkg(pkg_name="openslide")
class OpenSlideWSIReader(BaseWSIReader):
    """
    Read whole slide images and extract patches using OpenSlide library.

    Args:
        level: the whole slide image level at which the patches are extracted.
        mpp: the resolution in micron per pixel at which the patches are extracted.
        mpp_rtol: the acceptable relative tolerance for resolution in micro per pixel.
        mpp_atol: the acceptable absolute tolerance for resolution in micro per pixel.
        power: the objective power at which the patches are extracted.
        power_rtol: the acceptable relative tolerance for objective power.
        power_atol: the acceptable absolute tolerance for objective power.
        channel_dim: the desired dimension for color channel. Default to 0 (channel first).
        dtype: the data type of output image. Defaults to `np.uint8`.
        device: target device to put the extracted patch. Note that if device is "cuda"",
            the output will be converted to torch tenor and sent to the gpu even if the dtype is numpy.
        mode: the output image color mode, "RGB" or "RGBA". Defaults to "RGB".
        kwargs: additional args for `openslide.OpenSlide` module.

        Notes:
            Only one of resolution parameters, `level`, `mpp`, or `power`, should be provided.
            If such parameters are provided in `get_data` method, those will override the values provided here.
            If none of them are provided here or in `get_data`, `level=0` will be used.

    """

    supported_suffixes = ["tif", "tiff", "svs"]
    backend = "openslide"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def get_level_count(wsi) -> int:
        """
        Returns the number of levels in the whole slide image.

        Args:
            wsi: a whole slide image object loaded from a file.

        """
        return wsi.level_count  # type: ignore

    def get_size(self, wsi, level: int) -> tuple[int, int]:
        """
        Returns the size (height, width) of the whole slide image at a given level.

        Args:
            wsi: a whole slide image object loaded from a file.
            level: the level number where the size is calculated.

        """
        return (wsi.level_dimensions[level][1], wsi.level_dimensions[level][0])

    def get_downsample_ratio(self, wsi, level: int) -> float:
        """
        Returns the down-sampling ratio of the whole slide image at a given level.

        Args:
            wsi: a whole slide image object loaded from a file.
            level: the level number where the downsample ratio is calculated.

        """
        return wsi.level_downsamples[level]  # type: ignore

    @staticmethod
    def get_file_path(wsi) -> str:
        """Return the file path for the WSI object"""
        return str(abspath(wsi._filename))

    def get_mpp(self, wsi, level: int) -> tuple[float, float]:
        """
        Returns the micro-per-pixel resolution of the whole slide image at a given level.

        Args:
            wsi: a whole slide image object loaded from a file.
            level: the level number where the mpp is calculated.

        """
        downsample_ratio = self.get_downsample_ratio(wsi, level)
        if (
            "openslide.mpp-x" in wsi.properties
            and "openslide.mpp-y" in wsi.properties
            and wsi.properties["openslide.mpp-y"]
            and wsi.properties["openslide.mpp-x"]
        ):
            return (
                downsample_ratio * float(wsi.properties["openslide.mpp-y"]),
                downsample_ratio * float(wsi.properties["openslide.mpp-x"]),
            )

        if (
            "tiff.XResolution" in wsi.properties
            and "tiff.YResolution" in wsi.properties
            and wsi.properties["tiff.YResolution"]
            and wsi.properties["tiff.XResolution"]
        ):
            unit = wsi.properties.get("tiff.ResolutionUnit")
            if unit is None:
                warnings.warn("The resolution unit is missing, `micrometer` will be used as default.")
                unit = "micrometer"

            convert_to_micron = ConvertUnits(unit, "micrometer")
            return (
                convert_to_micron(downsample_ratio / float(wsi.properties["tiff.YResolution"])),
                convert_to_micron(downsample_ratio / float(wsi.properties["tiff.XResolution"])),
            )

        raise ValueError("`mpp` cannot be obtained for this file. Please use `level` instead.")

    def get_wsi_at_mpp(self, wsi, mpp: tuple, atol: float = 0.00, rtol: float = 0.05) -> np.ndarray:
        """
        Returns the representation of the whole slide image at a given micro-per-pixel (mpp) resolution.
        The optional tolerance parameters are considered at the level whose mpp value is closest to the one provided by the user.
        If the user-provided mpp is larger than the mpp of the closest level,
        the image is downscaled to a resolution that matches the user-provided mpp.
        Otherwise, if the closest level's resolution is not sufficient to meet the user's requested resolution,
        the next lower level (which has a higher resolution) is chosen.
        The image from this level is then down-scaled to achieve a resolution at the user-provided mpp value.

        Args:
            wsi: whole slide image object from WSIReader
            mpp: the resolution in micron per pixel at which the representation of the whole slide image should be extracted.
            atol: the acceptable absolute tolerance for resolution in micro per pixel.
            rtol: the acceptable relative tolerance for resolution in micro per pixel.

        """

        mpp_list = [self.get_mpp(wsi, lvl) for lvl in range(wsi.level_count)]
        closest_lvl = self._find_closest_level("mpp", mpp, mpp_list, 0, 5)

        within_tolerance, closest_level_is_bigger = super()._compute_mpp_tolerances(closest_lvl, mpp_list, mpp, atol, rtol)

        if within_tolerance:
            # If the image at the desired mpp resolution is within tolerances, return the image at closest_level.
            closest_lvl_wsi = wsi.read_region(
                (0, 0), level=closest_lvl, size=wsi.level_dimensions[closest_lvl]
            )

        elif closest_level_is_bigger:
            # Otherwise, select the level closest to the desired mpp with a higher resolution and downsample it.
            closest_lvl_wsi = self._resize_to_mpp_res(wsi, closest_lvl, mpp_list, mpp)

        else:
            # If both checks fail, increase resolution (i.e., decrement level) and then downsample it.
            closest_lvl = closest_lvl - 1
            closest_lvl_wsi = self._resize_to_mpp_res(wsi, closest_lvl, mpp_list, mpp)

        wsi_arr = np.array(closest_lvl_wsi)
        return wsi_arr

    def get_power(self, wsi, level: int) -> float:
        """
        Returns the objective power of the whole slide image at a given level.

        Args:
            wsi: a whole slide image object loaded from a file.
            level: the level number where the objective power is calculated.

        """
        objective_power = wsi.properties.get("openslide.objective-power")
        if objective_power:
            downsample_ratio = self.get_downsample_ratio(wsi, level)
            return float(objective_power) / downsample_ratio

        raise ValueError("Objective `power` cannot be obtained for this file. Please use `level` (or `mpp`) instead.")

    def read(self, data: Sequence[PathLike] | PathLike | np.ndarray, **kwargs):
        """
        Read whole slide image objects from given file or list of files.

        Args:
            data: file name or a list of file names to read.
            kwargs: additional args that overrides `self.kwargs` for existing keys.

        Returns:
            whole slide image object or list of such objects.

        """
        wsi_list: list = []

        filenames: Sequence[PathLike] = ensure_tuple(data)
        kwargs_ = self.kwargs.copy()
        kwargs_.update(kwargs)
        for filename in filenames:
            wsi = OpenSlide(filename, **kwargs_)
            wsi_list.append(wsi)

        return wsi_list if len(filenames) > 1 else wsi_list[0]

    def _get_patch(
        self, wsi, location: tuple[int, int], size: tuple[int, int], level: int, dtype: DtypeLike, mode: str
    ) -> np.ndarray:
        """
        Extracts and returns a patch image form the whole slide image.

        Args:
            wsi: a whole slide image object loaded from a file or a lis of such objects
            location: (top, left) tuple giving the top left pixel in the level 0 reference frame. Defaults to (0, 0).
            size: (height, width) tuple giving the patch size at the given level (`level`).
                If None, it is set to the full image size at the given level.
            level: the level number.
            dtype: the data type of output image.
            mode: the output image mode, 'RGB' or 'RGBA'.

        """
        # Extract a patch or the entire image
        # (reverse the order of location and size to become WxH for OpenSlide)
        pil_patch = wsi.read_region(location=location[::-1], size=size[::-1], level=level)

        # convert to RGB/RGBA
        pil_patch = pil_patch.convert(mode)

        # Convert to numpy
        patch = np.asarray(pil_patch, dtype=dtype)

        # Make the channel to desired dimensions
        patch = np.moveaxis(patch, -1, self.channel_dim)

        return patch

    def _resize_to_mpp_res(self, wsi, closest_lvl, mpp_list, user_mpp: tuple):
        """
        Resizes the whole slide image to the specified resolution in microns per pixel (mpp).

        Args:
            wsi: whole slide image object from WSIReader
            user_mpp: the resolution in microns per pixel at which the whole slide image representation should be extracted.
            closest_lvl: the wsi level that is closest to the user-provided mpp resolution.
            mpp_list: list of mpp values for all levels of a whole slide image.

        """
        pil_image, _ = optional_import("PIL", name="Image")

        closest_lvl_dim = wsi.level_dimensions[closest_lvl]

        target_res_x, target_res_y = super()._compute_mpp_target_res(closest_lvl, closest_lvl_dim, mpp_list, user_mpp)

        closest_lvl_wsi = wsi.read_region((0, 0), level=closest_lvl, size=closest_lvl_dim)
        closest_lvl_wsi = closest_lvl_wsi.resize((target_res_x, target_res_y), pil_image.BILINEAR)

        return closest_lvl_wsi


@require_pkg(pkg_name="tifffile")
class TiffFileWSIReader(BaseWSIReader):
    """
    Read whole slide images and extract patches using TiffFile library.

    Args:
        level: the whole slide image level at which the patches are extracted.
        mpp: the resolution in micron per pixel at which the patches are extracted.
        mpp_rtol: the acceptable relative tolerance for resolution in micro per pixel.
        mpp_atol: the acceptable absolute tolerance for resolution in micro per pixel.
        channel_dim: the desired dimension for color channel. Default to 0 (channel first).
        dtype: the data type of output image. Defaults to `np.uint8`.
        device: target device to put the extracted patch. Note that if device is "cuda"",
            the output will be converted to torch tenor and sent to the gpu even if the dtype is numpy.
        mode: the output image color mode, "RGB" or "RGBA". Defaults to "RGB".
        kwargs: additional args for `tifffile.TiffFile` module.

        Notes:
            - Objective power cannot be obtained via TiffFile backend.
            - Only one of resolution parameters, `level` or `mpp`, should be provided.
                If such parameters are provided in `get_data` method, those will override the values provided here.
                If none of them are provided here or in `get_data`, `level=0` will be used.

    """

    supported_suffixes = ["tif", "tiff", "svs"]
    backend = "tifffile"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def get_level_count(wsi) -> int:
        """
        Returns the number of levels in the whole slide image.

        Args:
            wsi: a whole slide image object loaded from a file.

        """
        return len(wsi.pages)

    def get_size(self, wsi, level: int) -> tuple[int, int]:
        """
        Returns the size (height, width) of the whole slide image at a given level.

        Args:
            wsi: a whole slide image object loaded from a file.
            level: the level number where the size is calculated.

        """
        return (wsi.pages[level].imagelength, wsi.pages[level].imagewidth)

    def get_downsample_ratio(self, wsi, level: int) -> float:
        """
        Returns the down-sampling ratio of the whole slide image at a given level.

        Args:
            wsi: a whole slide image object loaded from a file.
            level: the level number where the downsample ratio is calculated.

        """
        return float(wsi.pages[0].imagelength) / float(wsi.pages[level].imagelength)

    @staticmethod
    def get_file_path(wsi) -> str:
        """Return the file path for the WSI object"""
        return str(abspath(wsi.filehandle.path))

    def get_mpp(self, wsi, level: int) -> tuple[float, float]:
        """
        Returns the micro-per-pixel resolution of the whole slide image at a given level.

        Args:
            wsi: a whole slide image object loaded from a file.
            level: the level number where the mpp is calculated.

        """
        if (
            "XResolution" in wsi.pages[level].tags
            and "YResolution" in wsi.pages[level].tags
            and wsi.pages[level].tags["XResolution"].value
            and wsi.pages[level].tags["YResolution"].value
        ):
            unit = wsi.pages[level].tags.get("ResolutionUnit")
            if unit is not None:
                unit = str(unit.value.name)
            if unit is None or len(unit) == 0:
                warnings.warn("The resolution unit is missing. `micrometer` will be used as default.")
                unit = "micrometer"

            convert_to_micron = ConvertUnits(unit, "micrometer")

            # Here, x and y resolutions are rational numbers so each of them is represented by a tuple.
            yres = wsi.pages[level].tags["YResolution"].value
            xres = wsi.pages[level].tags["XResolution"].value
            if xres[0] & yres[0]:
                return convert_to_micron(yres[1] / yres[0]), convert_to_micron(xres[1] / xres[0])
            else:
                raise ValueError("The `XResolution` and/or `YResolution` property of the image is zero, "
                                 "which is needed to obtain `mpp` for this file. Please use `level` instead.")
        raise ValueError("`mpp` cannot be obtained for this file. Please use `level` instead.")

    def get_wsi_at_mpp(self, wsi, mpp: tuple, atol: float = 0.00, rtol: float = 0.05) -> np.ndarray:
        """
        Returns the representation of the whole slide image at a given micro-per-pixel (mpp) resolution.
        The optional tolerance parameters are considered at the level whose mpp value is closest to the one provided by the user.
        If the user-provided mpp is larger than the mpp of the closest level,
        the image is downscaled to a resolution that matches the user-provided mpp.
        Otherwise, if the closest level's resolution is not sufficient to meet the user's requested resolution,
        the next lower level (which has a higher resolution) is chosen.
        The image from this level is then down-scaled to achieve a resolution at the user-provided mpp value.

        Args:
            wsi: whole slide image object from WSIReader
            mpp: the resolution in micron per pixel at which the representation of the whole slide image should be extracted.
            atol: the acceptable absolute tolerance for resolution in micro per pixel.
            rtol: the acceptable relative tolerance for resolution in micro per pixel.

        """

        mpp_list = [self.get_mpp(wsi, lvl) for lvl in range(len(wsi.pages))]  # Fails for some Tifffiles
        closest_lvl = self._find_closest_level("mpp", mpp, mpp_list, 0, 5)

        within_tolerance, closest_level_is_bigger = super()._compute_mpp_tolerances(closest_lvl, mpp_list, mpp, atol, rtol)

        if within_tolerance:
            # If the image at the desired mpp resolution is within tolerances, return the image at closest_level.
            closest_lvl_wsi = wsi.read_region((0, 0), level=closest_lvl, size=self.get_size(wsi, closest_lvl))

        elif closest_level_is_bigger:
            # Otherwise, select the level closest to the desired mpp with a higher resolution and downsample it.
            closest_lvl_wsi = self._resize_to_mpp_res(wsi, closest_lvl, mpp_list, mpp)

        else:
            # If both checks fail, increase resolution (i.e., decrement level) and then downsample it.
            closest_lvl = closest_lvl - 1
            closest_lvl_wsi = self._resize_to_mpp_res(wsi, closest_lvl, mpp_list, mpp)

        wsi_arr = np.array(closest_lvl_wsi)
        return wsi_arr

    def get_power(self, wsi, level: int) -> float:
        """
        Returns the objective power of the whole slide image at a given level.

        Args:
            wsi: a whole slide image object loaded from a file.
            level: the level number where the objective power is calculated.

        """
        raise ValueError(
            "Currently, TiffFile does not provide a general API to obtain objective power."
            "Please use `level` (or `mpp`) instead, or try other backends."
        )

    def read(self, data: Sequence[PathLike] | PathLike | np.ndarray, **kwargs):
        """
        Read whole slide image objects from given file or list of files.

        Args:
            data: file name or a list of file names to read.
            kwargs: additional args that overrides `self.kwargs` for existing keys.

        Returns:
            whole slide image object or list of such objects.

        """
        wsi_list: list = []

        filenames: Sequence[PathLike] = ensure_tuple(data)
        kwargs_ = self.kwargs.copy()
        kwargs_.update(kwargs)
        for filename in filenames:
            wsi = TiffFile(filename, **kwargs_)
            wsi_list.append(wsi)

        return wsi_list if len(filenames) > 1 else wsi_list[0]

    def _get_patch(
        self, wsi, location: tuple[int, int], size: tuple[int, int], level: int, dtype: DtypeLike, mode: str
    ) -> np.ndarray:
        """
        Extracts and returns a patch image form the whole slide image.

        Args:
            wsi: a whole slide image object loaded from a file or a list of such objects
            location: (top, left) tuple giving the top left pixel in the level 0 reference frame. Defaults to (0, 0).
            size: (height, width) tuple giving the patch size at the given level (`level`).
                If None, it is set to the full image size at the given level.
            level: the level number.
            dtype: the data type of output image.
            mode: the output image mode, 'RGB' or 'RGBA'.

        """
        # Load the entire image
        wsi_image: np.ndarray = wsi.asarray(level=level).astype(dtype)
        if len(wsi_image.shape) < 3:
            wsi_image = wsi_image[..., None]

        # Extract patch
        downsampling_ratio = self.get_downsample_ratio(wsi=wsi, level=level)
        location_ = [round(location[i] / downsampling_ratio) for i in range(len(location))]
        patch = wsi_image[location_[0] : location_[0] + size[0], location_[1] : location_[1] + size[1], :]

        # Make the channel to desired dimensions
        patch = np.moveaxis(patch, -1, self.channel_dim)

        # Check if the color channel is 3 (RGB) or 4 (RGBA)
        if mode in "RGB":
            if patch.shape[self.channel_dim] not in [3, 4]:
                raise ValueError(
                    f"The image is expected to have three or four color channels in '{mode}' mode but has "
                    f"{patch.shape[self.channel_dim]}. "
                )
            patch = np.take(patch, [0, 1, 2], self.channel_dim)

        return patch

    def _resize_to_mpp_res(self, wsi, closest_lvl, mpp_list, user_mpp: tuple):
        """
        Resizes the whole slide image to the specified resolution in microns per pixel (mpp).

        Args:
            wsi: whole slide image object from WSIReader
            user_mpp: the resolution in microns per pixel at which the whole slide image representation should be extracted.
            closest_lvl: the wsi level that is closest to the user-provided mpp resolution.
            mpp_list: list of mpp values for all levels of a whole slide image.

        """
        pil_image, _ = optional_import("PIL", name="Image")

        closest_lvl_dim = self.get_size(wsi, closest_lvl)

        target_res_x, target_res_y = super()._compute_mpp_target_res(closest_lvl, closest_lvl_dim, mpp_list, user_mpp)

        closest_lvl_wsi = pil_image.fromarray(wsi.pages[closest_lvl].asarray())
        closest_lvl_wsi = closest_lvl_wsi.resize((target_res_x, target_res_y), pil_image.BILINEAR)

        return closest_lvl_wsi
