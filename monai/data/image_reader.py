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

import glob
import gzip
import io
import os
import re
import tempfile
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from torch.utils.data._utils.collate import np_str_obj_array_pattern

from monai.config import KeysCollection, PathLike
from monai.data.utils import (
    affine_to_spacing,
    correct_nifti_header_if_necessary,
    is_no_channel,
    is_supported_format,
    orientation_ras_lps,
)
from monai.utils import MetaKeys, SpaceKeys, TraceKeys, ensure_tuple, optional_import, require_pkg

if TYPE_CHECKING:
    import itk
    import nibabel as nib
    import nrrd
    import pydicom
    from nibabel.nifti1 import Nifti1Image
    from PIL import Image as PILImage

    has_nrrd = has_itk = has_nib = has_pil = has_pydicom = True
else:
    itk, has_itk = optional_import("itk", allow_namespace_pkg=True)
    nib, has_nib = optional_import("nibabel")
    Nifti1Image, _ = optional_import("nibabel.nifti1", name="Nifti1Image")
    PILImage, has_pil = optional_import("PIL.Image")
    pydicom, has_pydicom = optional_import("pydicom")
    nrrd, has_nrrd = optional_import("nrrd", allow_namespace_pkg=True)

cp, has_cp = optional_import("cupy")
kvikio, has_kvikio = optional_import("kvikio")

__all__ = ["ImageReader", "ITKReader", "NibabelReader", "NumpyReader", "PILReader", "PydicomReader", "NrrdReader"]


class ImageReader(ABC):
    """
    An abstract class defines APIs to load image files.

    Typical usage of an implementation of this class is:

    .. code-block:: python

        image_reader = MyImageReader()
        img_obj = image_reader.read(path_to_image)
        img_data, meta_data = image_reader.get_data(img_obj)

    - The `read` call converts image filenames into image objects,
    - The `get_data` call fetches the image data, as well as metadata.
    - A reader should implement `verify_suffix` with the logic of checking the input filename
      by the filename extensions.

    """

    @abstractmethod
    def verify_suffix(self, filename: Sequence[PathLike] | PathLike) -> bool:
        """
        Verify whether the specified `filename` is supported by the current reader.
        This method should return True if the reader is able to read the format suggested by the
        `filename`.

        Args:
            filename: file name or a list of file names to read.
                if a list of files, verify all the suffixes.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    def read(self, data: Sequence[PathLike] | PathLike, **kwargs) -> Sequence[Any] | Any:
        """
        Read image data from specified file or files.
        Note that it returns a data object or a sequence of data objects.

        Args:
            data: file name or a list of file names to read.
            kwargs: additional args for actual `read` API of 3rd party libs.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    def get_data(self, img) -> tuple[np.ndarray, dict]:
        """
        Extract data array and metadata from loaded image and return them.
        This function must return two objects, the first is a numpy array of image data,
        the second is a dictionary of metadata.

        Args:
            img: an image object loaded from an image file or a list of image objects.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")


def _copy_compatible_dict(from_dict: dict, to_dict: dict):
    if not isinstance(to_dict, dict):
        raise ValueError(f"to_dict must be a Dict, got {type(to_dict)}.")
    if not to_dict:
        for key in from_dict:
            datum = from_dict[key]
            if isinstance(datum, np.ndarray) and np_str_obj_array_pattern.search(datum.dtype.str) is not None:
                continue
            to_dict[key] = str(TraceKeys.NONE) if datum is None else datum  # NoneType to string for default_collate
    else:
        affine_key, shape_key = MetaKeys.AFFINE, MetaKeys.SPATIAL_SHAPE
        if affine_key in from_dict and not np.allclose(from_dict[affine_key], to_dict[affine_key]):
            raise RuntimeError(
                "affine matrix of all images should be the same for channel-wise concatenation. "
                f"Got {from_dict[affine_key]} and {to_dict[affine_key]}."
            )
        if shape_key in from_dict and not np.allclose(from_dict[shape_key], to_dict[shape_key]):
            raise RuntimeError(
                "spatial_shape of all images should be the same for channel-wise concatenation. "
                f"Got {from_dict[shape_key]} and {to_dict[shape_key]}."
            )


def _stack_images(image_list: list, meta_dict: dict, to_cupy: bool = False):
    if len(image_list) <= 1:
        return image_list[0]
    if not is_no_channel(meta_dict.get(MetaKeys.ORIGINAL_CHANNEL_DIM, None)):
        channel_dim = int(meta_dict[MetaKeys.ORIGINAL_CHANNEL_DIM])
        if to_cupy and has_cp:
            return cp.concatenate(image_list, axis=channel_dim)
        return np.concatenate(image_list, axis=channel_dim)
    # stack at a new first dim as the channel dim, if `'original_channel_dim'` is unspecified
    meta_dict[MetaKeys.ORIGINAL_CHANNEL_DIM] = 0
    if to_cupy and has_cp:
        return cp.stack(image_list, axis=0)
    return np.stack(image_list, axis=0)


@require_pkg(pkg_name="itk")
class ITKReader(ImageReader):
    """
    Load medical images based on ITK library.
    All the supported image formats can be found at:
    https://github.com/InsightSoftwareConsortium/ITK/tree/master/Modules/IO
    The loaded data array will be in C order, for example, a 3D image NumPy
    array index order will be `CDWH`.

    Args:
        channel_dim: the channel dimension of the input image, default is None.
            This is used to set original_channel_dim in the metadata, EnsureChannelFirstD reads this field.
            If None, `original_channel_dim` will be either `no_channel` or `-1`.

                - Nifti file is usually "channel last", so there is no need to specify this argument.
                - PNG file usually has `GetNumberOfComponentsPerPixel()==3`, so there is no need to specify this argument.

        series_name: the name of the DICOM series if there are multiple ones.
            used when loading DICOM series.
        reverse_indexing: whether to use a reversed spatial indexing convention for the returned data array.
            If ``False``, the spatial indexing convention is reversed to be compatible with ITK;
            otherwise, the spatial indexing follows the numpy convention. Default is ``False``.
            This option does not affect the metadata.
        series_meta: whether to load the metadata of the DICOM series (using the metadata from the first slice).
            This flag is checked only when loading DICOM series. Default is ``False``.
        affine_lps_to_ras: whether to convert the affine matrix from "LPS" to "RAS". Defaults to ``True``.
            Set to ``True`` to be consistent with ``NibabelReader``, otherwise the affine matrix remains in the ITK convention.
        kwargs: additional args for `itk.imread` API. more details about available args:
            https://github.com/InsightSoftwareConsortium/ITK/blob/master/Wrapping/Generators/Python/itk/support/extras.py

    """

    def __init__(
        self,
        channel_dim: str | int | None = None,
        series_name: str = "",
        reverse_indexing: bool = False,
        series_meta: bool = False,
        affine_lps_to_ras: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.kwargs = kwargs
        self.channel_dim = float("nan") if channel_dim == "no_channel" else channel_dim
        self.series_name = series_name
        self.reverse_indexing = reverse_indexing
        self.series_meta = series_meta
        self.affine_lps_to_ras = affine_lps_to_ras

    def verify_suffix(self, filename: Sequence[PathLike] | PathLike) -> bool:
        """
        Verify whether the specified file or files format is supported by ITK reader.

        Args:
            filename: file name or a list of file names to read.
                if a list of files, verify all the suffixes.

        """
        return has_itk

    def read(self, data: Sequence[PathLike] | PathLike, **kwargs):
        """
        Read image data from specified file or files, it can read a list of images
        and stack them together as multi-channel data in `get_data()`.
        If passing directory path instead of file path, will treat it as DICOM images series and read.
        Note that the returned object is ITK image object or list of ITK image objects.

        Args:
            data: file name or a list of file names to read,
            kwargs: additional args for `itk.imread` API, will override `self.kwargs` for existing keys.
                More details about available args:
                https://github.com/InsightSoftwareConsortium/ITK/blob/master/Wrapping/Generators/Python/itk/support/extras.py

        """
        img_ = []

        filenames: Sequence[PathLike] = ensure_tuple(data)
        kwargs_ = self.kwargs.copy()
        kwargs_.update(kwargs)
        for name in filenames:
            name = f"{name}"
            if Path(name).is_dir():
                # read DICOM series
                # https://examples.itk.org/src/io/gdcm/readdicomseriesandwrite3dimage/documentation
                names_generator = itk.GDCMSeriesFileNames.New()
                names_generator.SetUseSeriesDetails(True)
                names_generator.AddSeriesRestriction("0008|0021")  # Series Date
                names_generator.SetDirectory(name)
                series_uid = names_generator.GetSeriesUIDs()

                if len(series_uid) < 1:
                    raise FileNotFoundError(f"no DICOMs in: {name}.")
                if len(series_uid) > 1:
                    warnings.warn(f"the directory: {name} contains more than one DICOM series.")
                series_identifier = series_uid[0] if not self.series_name else self.series_name
                name = names_generator.GetFileNames(series_identifier)

                name = name[0] if len(name) == 1 else name  # type: ignore
                _obj = itk.imread(name, **kwargs_)
                if self.series_meta:
                    _reader = itk.ImageSeriesReader.New(FileNames=name)
                    _reader.Update()
                    _meta = _reader.GetMetaDataDictionaryArray()
                    if len(_meta) > 0:
                        # TODO: using the first slice's meta. this could be improved to filter unnecessary tags.
                        _obj.SetMetaDataDictionary(_meta[0])
                img_.append(_obj)
            else:
                img_.append(itk.imread(name, **kwargs_))
        return img_ if len(filenames) > 1 else img_[0]

    def get_data(self, img) -> tuple[np.ndarray, dict]:
        """
        Extract data array and metadata from loaded image and return them.
        This function returns two objects, first is numpy array of image data, second is dict of metadata.
        It constructs `affine`, `original_affine`, and `spatial_shape` and stores them in meta dict.
        When loading a list of files, they are stacked together at a new dimension as the first dimension,
        and the metadata of the first image is used to represent the output metadata.

        Args:
            img: an ITK image object loaded from an image file or a list of ITK image objects.

        """
        img_array: list[np.ndarray] = []
        compatible_meta: dict = {}

        for i in ensure_tuple(img):
            data = self._get_array_data(i)
            img_array.append(data)
            header = self._get_meta_dict(i)
            header[MetaKeys.ORIGINAL_AFFINE] = self._get_affine(i, self.affine_lps_to_ras)
            header[MetaKeys.SPACE] = SpaceKeys.RAS if self.affine_lps_to_ras else SpaceKeys.LPS
            header[MetaKeys.AFFINE] = header[MetaKeys.ORIGINAL_AFFINE].copy()
            header[MetaKeys.SPATIAL_SHAPE] = self._get_spatial_shape(i)
            if self.channel_dim is None:  # default to "no_channel" or -1
                header[MetaKeys.ORIGINAL_CHANNEL_DIM] = (
                    float("nan") if len(data.shape) == len(header[MetaKeys.SPATIAL_SHAPE]) else -1
                )
            else:
                header[MetaKeys.ORIGINAL_CHANNEL_DIM] = self.channel_dim
            _copy_compatible_dict(header, compatible_meta)

        return _stack_images(img_array, compatible_meta), compatible_meta

    def _get_meta_dict(self, img) -> dict:
        """
        Get all the metadata of the image and convert to dict type.

        Args:
            img: an ITK image object loaded from an image file.

        """
        img_meta_dict = img.GetMetaDataDictionary()
        meta_dict = {}
        for key in img_meta_dict.GetKeys():
            if key.startswith("ITK_"):
                continue
            val = img_meta_dict[key]
            meta_dict[key] = np.asarray(val) if type(val).__name__.startswith("itk") else val

        meta_dict["spacing"] = np.asarray(img.GetSpacing())
        return meta_dict

    def _get_affine(self, img, lps_to_ras: bool = True):
        """
        Get or construct the affine matrix of the image, it can be used to correct
        spacing, orientation or execute spatial transforms.

        Args:
            img: an ITK image object loaded from an image file.
            lps_to_ras: whether to convert the affine matrix from "LPS" to "RAS". Defaults to True.

        """
        direction = itk.array_from_matrix(img.GetDirection())
        spacing = np.asarray(img.GetSpacing())
        origin = np.asarray(img.GetOrigin())

        direction = np.asarray(direction)
        sr = min(max(direction.shape[0], 1), 3)
        affine: np.ndarray = np.eye(sr + 1)
        affine[:sr, :sr] = direction[:sr, :sr] @ np.diag(spacing[:sr])
        affine[:sr, -1] = origin[:sr]
        if lps_to_ras:
            affine = orientation_ras_lps(affine)
        return affine

    def _get_spatial_shape(self, img):
        """
        Get the spatial shape of `img`.

        Args:
            img: an ITK image object loaded from an image file.

        """
        sr = itk.array_from_matrix(img.GetDirection()).shape[0]
        sr = max(min(sr, 3), 1)
        _size = list(itk.size(img))
        if isinstance(self.channel_dim, int):
            _size.pop(self.channel_dim)
        return np.asarray(_size[:sr])

    def _get_array_data(self, img):
        """
        Get the raw array data of the image, converted to Numpy array.

        Following PyTorch conventions, the returned array data has contiguous channels,
        e.g. for an RGB image, all red channel image pixels are contiguous in memory.
        The last axis of the returned array is the channel axis.

        See also:

            - https://github.com/InsightSoftwareConsortium/ITK/blob/v5.2.1/Modules/Bridge/NumPy/wrapping/PyBuffer.i.in

        Args:
            img: an ITK image object loaded from an image file.

        """
        np_img = itk.array_view_from_image(img, keep_axes=False)
        if img.GetNumberOfComponentsPerPixel() == 1:  # handling spatial images
            return np_img if self.reverse_indexing else np_img.T
        # handling multi-channel images
        return np_img if self.reverse_indexing else np.moveaxis(np_img.T, 0, -1)


@require_pkg(pkg_name="pydicom")
class PydicomReader(ImageReader):
    """
    Load medical images based on Pydicom library.
    All the supported image formats can be found at:
    https://dicom.nema.org/medical/dicom/current/output/chtml/part10/chapter_7.html

    PydicomReader is also able to load segmentations, if a dicom file contains tag: `SegmentSequence`, the reader
    will consider it as segmentation data, and to load it successfully, `PerFrameFunctionalGroupsSequence` is required
    for dicom file, and for each frame of dicom file, `SegmentIdentificationSequence` is required.
    This method refers to the Highdicom library.

    This class refers to:
    https://nipy.org/nibabel/dicom/dicom_orientation.html#dicom-affine-formula
    https://github.com/pydicom/contrib-pydicom/blob/master/input-output/pydicom_series.py
    https://highdicom.readthedocs.io/en/latest/usage.html#parsing-segmentation-seg-images

    Args:
        channel_dim: the channel dimension of the input image, default is None.
            This is used to set original_channel_dim in the metadata, EnsureChannelFirstD reads this field.
            If None, `original_channel_dim` will be either `no_channel` or `-1`.
        affine_lps_to_ras: whether to convert the affine matrix from "LPS" to "RAS". Defaults to ``True``.
            Set to ``True`` to be consistent with ``NibabelReader``,
            otherwise the affine matrix remains in the Dicom convention.
        swap_ij: whether to swap the first two spatial axes. Default to ``True``, so that the outputs
            are consistent with the other readers.
        prune_metadata: whether to prune the saved information in metadata. This argument is used for
            `get_data` function. If True, only items that are related to the affine matrix will be saved.
            Default to ``True``.
        label_dict: label of the dicom data. If provided, it will be used when loading segmentation data.
            Keys of the dict are the classes, and values are the corresponding class number. For example:
            for TCIA collection "C4KC-KiTS", it can be: {"Kidney": 0, "Renal Tumor": 1}.
        fname_regex: a regular expression to match the file names when the input is a folder.
            If provided, only the matched files will be included. For example, to include the file name
            "image_0001.dcm", the regular expression could be `".*image_(\\d+).dcm"`. Default to `""`.
            Set it to `None` to use `pydicom.misc.is_dicom` to match valid files.
        kwargs: additional args for `pydicom.dcmread` API. more details about available args:
            https://pydicom.github.io/pydicom/stable/reference/generated/pydicom.filereader.dcmread.html
            If the `get_data` function will be called
            (for example, when using this reader with `monai.transforms.LoadImage`), please ensure that the argument
            `stop_before_pixels` is `True`, and `specific_tags` covers all necessary tags, such as `PixelSpacing`,
            `ImagePositionPatient`, `ImageOrientationPatient` and all `pixel_array` related tags.
    """

    def __init__(
        self,
        channel_dim: str | int | None = None,
        affine_lps_to_ras: bool = True,
        swap_ij: bool = True,
        prune_metadata: bool = True,
        label_dict: dict | None = None,
        fname_regex: str = "",
        **kwargs,
    ):
        super().__init__()
        self.kwargs = kwargs
        self.channel_dim = float("nan") if channel_dim == "no_channel" else channel_dim
        self.affine_lps_to_ras = affine_lps_to_ras
        self.swap_ij = swap_ij
        self.prune_metadata = prune_metadata
        self.label_dict = label_dict
        self.fname_regex = fname_regex

    def verify_suffix(self, filename: Sequence[PathLike] | PathLike) -> bool:
        """
        Verify whether the specified file or files format is supported by Pydicom reader.

        Args:
            filename: file name or a list of file names to read.
                if a list of files, verify all the suffixes.

        """
        return has_pydicom

    def read(self, data: Sequence[PathLike] | PathLike, **kwargs):
        """
        Read image data from specified file or files, it can read a list of images
        and stack them together as multi-channel data in `get_data()`.
        If passing directory path instead of file path, will treat it as DICOM images series and read.

        Args:
            data: file name or a list of file names to read,
            kwargs: additional args for `pydicom.dcmread` API, will override `self.kwargs` for existing keys.

        Returns:
            If `data` represents a filename: return a pydicom dataset object.
            If `data` represents a list of filenames or a directory: return a list of pydicom dataset object.
            If `data` represents a list of directories: return a list of list of pydicom dataset object.

        """
        img_ = []

        filenames: Sequence[PathLike] = ensure_tuple(data)
        kwargs_ = self.kwargs.copy()
        kwargs_.update(kwargs)

        self.has_series = False

        for name in filenames:
            name = f"{name}"
            if Path(name).is_dir():
                # read DICOM series
                if self.fname_regex is not None:
                    series_slcs = [slc for slc in glob.glob(os.path.join(name, "*")) if re.match(self.fname_regex, slc)]
                else:
                    series_slcs = [slc for slc in glob.glob(os.path.join(name, "*")) if pydicom.misc.is_dicom(slc)]
                slices = []
                for slc in series_slcs:
                    try:
                        slices.append(pydicom.dcmread(fp=slc, **kwargs_))
                    except pydicom.errors.InvalidDicomError as e:
                        warnings.warn(f"Failed to read {slc} with exception: \n{e}.", stacklevel=2)
                img_.append(slices if len(slices) > 1 else slices[0])
                if len(slices) > 1:
                    self.has_series = True
            else:
                ds = pydicom.dcmread(fp=name, **kwargs_)
                img_.append(ds)
        return img_ if len(filenames) > 1 else img_[0]

    def _combine_dicom_series(self, data: Iterable):
        """
        Combine dicom series (a list of pydicom dataset objects). Their data arrays will be stacked together at a new
        dimension as the last dimension.

        The stack order depends on Instance Number. The metadata will be based on the
        first slice's metadata, and some new items will be added:

        "spacing": the new spacing of the stacked slices.
        "lastImagePositionPatient": `ImagePositionPatient` for the last slice, it will be used to achieve the affine
            matrix.
        "spatial_shape": the spatial shape of the stacked slices.

        Args:
            data: a list of pydicom dataset objects.
        Returns:
            a tuple that consisted with data array and metadata.
        """
        slices: list = []
        # for a dicom series
        for slc_ds in data:
            if hasattr(slc_ds, "InstanceNumber"):
                slices.append(slc_ds)
            else:
                warnings.warn(f"slice: {slc_ds.filename} does not have InstanceNumber tag, skip it.")
        slices = sorted(slices, key=lambda s: s.InstanceNumber)

        if len(slices) == 0:
            raise ValueError("the input does not have valid slices.")

        first_slice = slices[0]
        average_distance = 0.0
        first_array = self._get_array_data(first_slice)
        shape = first_array.shape
        spacing = getattr(first_slice, "PixelSpacing", [1.0, 1.0, 1.0])
        prev_pos = getattr(first_slice, "ImagePositionPatient", (0.0, 0.0, 0.0))[2]
        stack_array = [first_array]
        for idx in range(1, len(slices)):
            slc_array = self._get_array_data(slices[idx])
            slc_shape = slc_array.shape
            slc_spacing = getattr(slices[idx], "PixelSpacing", (1.0, 1.0, 1.0))
            slc_pos = getattr(slices[idx], "ImagePositionPatient", (0.0, 0.0, float(idx)))[2]
            if not np.allclose(slc_spacing, spacing):
                warnings.warn(f"the list contains slices that have different spacings {spacing} and {slc_spacing}.")
            if shape != slc_shape:
                warnings.warn(f"the list contains slices that have different shapes {shape} and {slc_shape}.")
            average_distance += abs(prev_pos - slc_pos)
            prev_pos = slc_pos
            stack_array.append(slc_array)

        if len(slices) > 1:
            average_distance /= len(slices) - 1
            spacing.append(average_distance)
            stack_array = np.stack(stack_array, axis=-1)
            stack_metadata = self._get_meta_dict(first_slice)
            stack_metadata["spacing"] = np.asarray(spacing)
            if hasattr(slices[-1], "ImagePositionPatient"):
                stack_metadata["lastImagePositionPatient"] = np.asarray(slices[-1].ImagePositionPatient)
            stack_metadata[MetaKeys.SPATIAL_SHAPE] = shape + (len(slices),)
        else:
            stack_array = stack_array[0]
            stack_metadata = self._get_meta_dict(first_slice)
            stack_metadata["spacing"] = np.asarray(spacing)
            stack_metadata[MetaKeys.SPATIAL_SHAPE] = shape

        return stack_array, stack_metadata

    def get_data(self, data) -> tuple[np.ndarray, dict]:
        """
        Extract data array and metadata from loaded image and return them.
        This function returns two objects, first is numpy array of image data, second is dict of metadata.
        It constructs `affine`, `original_affine`, and `spatial_shape` and stores them in meta dict.
        For dicom series within the input, all slices will be stacked first,
        When loading a list of files (dicom file, or stacked dicom series), they are stacked together at a new
        dimension as the first dimension, and the metadata of the first image is used to represent the output metadata.

        To use this function, all pydicom dataset objects (if not segmentation data) should contain:
        `pixel_array`, `PixelSpacing`, `ImagePositionPatient` and `ImageOrientationPatient`.

        For segmentation data, we assume that the input is not a dicom series, and the object should contain
        `SegmentSequence` in order to identify it.
        In addition, tags (5200, 9229) and (5200, 9230) are required to achieve
        `PixelSpacing`, `ImageOrientationPatient` and `ImagePositionPatient`.

        Args:
            data: a pydicom dataset object, or a list of pydicom dataset objects, or a list of list of
                pydicom dataset objects.

        """

        dicom_data = []
        # combine dicom series if exists
        if self.has_series is True:
            # a list, all objects within a list belong to one dicom series
            if not isinstance(data[0], list):
                dicom_data.append(self._combine_dicom_series(data))
            # a list of list, each inner list represents a dicom series
            else:
                for series in data:
                    dicom_data.append(self._combine_dicom_series(series))
        else:
            # a single pydicom dataset object
            if not isinstance(data, list):
                data = [data]
            for d in data:
                if hasattr(d, "SegmentSequence"):
                    data_array, metadata = self._get_seg_data(d)
                else:
                    data_array = self._get_array_data(d)
                    metadata = self._get_meta_dict(d)
                    metadata[MetaKeys.SPATIAL_SHAPE] = data_array.shape
                dicom_data.append((data_array, metadata))

        img_array: list[np.ndarray] = []
        compatible_meta: dict = {}

        for data_array, metadata in ensure_tuple(dicom_data):
            img_array.append(np.ascontiguousarray(np.swapaxes(data_array, 0, 1) if self.swap_ij else data_array))
            affine = self._get_affine(metadata, self.affine_lps_to_ras)
            metadata[MetaKeys.SPACE] = SpaceKeys.RAS if self.affine_lps_to_ras else SpaceKeys.LPS
            if self.swap_ij:
                affine = affine @ np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
                sp_size = list(metadata[MetaKeys.SPATIAL_SHAPE])
                sp_size[0], sp_size[1] = sp_size[1], sp_size[0]
                metadata[MetaKeys.SPATIAL_SHAPE] = ensure_tuple(sp_size)
            metadata[MetaKeys.ORIGINAL_AFFINE] = affine
            metadata[MetaKeys.AFFINE] = affine.copy()
            if self.channel_dim is None:  # default to "no_channel" or -1
                metadata[MetaKeys.ORIGINAL_CHANNEL_DIM] = (
                    float("nan") if len(data_array.shape) == len(metadata[MetaKeys.SPATIAL_SHAPE]) else -1
                )
            else:
                metadata[MetaKeys.ORIGINAL_CHANNEL_DIM] = self.channel_dim
            metadata["spacing"] = affine_to_spacing(
                metadata[MetaKeys.ORIGINAL_AFFINE], r=len(metadata[MetaKeys.SPATIAL_SHAPE])
            )

            _copy_compatible_dict(metadata, compatible_meta)

        return _stack_images(img_array, compatible_meta), compatible_meta

    def _get_meta_dict(self, img) -> dict:
        """
        Get all the metadata of the image and convert to dict type.

        Args:
            img: a Pydicom dataset object.

        """

        metadata = img.to_json_dict(suppress_invalid_tags=True)

        if self.prune_metadata:
            prune_metadata = {}
            for key in ["00200037", "00200032", "00280030", "52009229", "52009230"]:
                if key in metadata.keys():
                    prune_metadata[key] = metadata[key]
            return prune_metadata

        # always remove Pixel Data "7FE00008" or "7FE00009" or "7FE00010"
        # always remove Data Set Trailing Padding "FFFCFFFC"
        for key in ["7FE00008", "7FE00009", "7FE00010", "FFFCFFFC"]:
            if key in metadata.keys():
                metadata.pop(key)

        return metadata  # type: ignore

    def _get_affine(self, metadata: dict, lps_to_ras: bool = True):
        """
        Get or construct the affine matrix of the image, it can be used to correct
        spacing, orientation or execute spatial transforms.

        Args:
            metadata: metadata with dict type.
            lps_to_ras: whether to convert the affine matrix from "LPS" to "RAS". Defaults to True.

        """
        affine: np.ndarray = np.eye(4)
        if not ("00200037" in metadata and "00200032" in metadata):
            return affine
        # "00200037" is the tag of `ImageOrientationPatient`
        rx, ry, rz, cx, cy, cz = metadata["00200037"]["Value"]
        # "00200032" is the tag of `ImagePositionPatient`
        sx, sy, sz = metadata["00200032"]["Value"]
        # "00280030" is the tag of `PixelSpacing`
        spacing = metadata["00280030"]["Value"] if "00280030" in metadata else (1.0, 1.0)
        dr, dc = metadata.get("spacing", spacing)[:2]
        affine[0, 0] = cx * dr
        affine[0, 1] = rx * dc
        affine[0, 3] = sx
        affine[1, 0] = cy * dr
        affine[1, 1] = ry * dc
        affine[1, 3] = sy
        affine[2, 0] = cz * dr
        affine[2, 1] = rz * dc
        affine[2, 2] = 1.0
        affine[2, 3] = sz

        # 3d
        if "lastImagePositionPatient" in metadata:
            t1n, t2n, t3n = metadata["lastImagePositionPatient"]
            n = metadata[MetaKeys.SPATIAL_SHAPE][-1]
            k1, k2, k3 = (t1n - sx) / (n - 1), (t2n - sy) / (n - 1), (t3n - sz) / (n - 1)
            affine[0, 2] = k1
            affine[1, 2] = k2
            affine[2, 2] = k3

        if lps_to_ras:
            affine = orientation_ras_lps(affine)
        return affine

    def _get_frame_data(self, img) -> Iterator:
        """
        yield frames and description from the segmentation image.
        This function is adapted from Highdicom:
        https://github.com/herrmannlab/highdicom/blob/v0.18.2/src/highdicom/seg/utils.py

        which has the following license...

        # =========================================================================
        # https://github.com/herrmannlab/highdicom/blob/v0.18.2/LICENSE
        #
        # Copyright 2020 MGH Computational Pathology
        # Permission is hereby granted, free of charge, to any person obtaining a
        # copy of this software and associated documentation files (the
        # "Software"), to deal in the Software without restriction, including
        # without limitation the rights to use, copy, modify, merge, publish,
        # distribute, sublicense, and/or sell copies of the Software, and to
        # permit persons to whom the Software is furnished to do so, subject to
        # the following conditions:
        # The above copyright notice and this permission notice shall be included
        # in all copies or substantial portions of the Software.
        # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
        # OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
        # MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
        # IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
        # CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
        # TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
        # SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
        # =========================================================================

        (https://github.com/herrmannlab/highdicom/issues/188)

        Args:
            img: a Pydicom dataset object that has attribute "SegmentSequence".

        """

        if not hasattr(img, "PerFrameFunctionalGroupsSequence"):
            raise NotImplementedError(
                f"To read dicom seg: {img.filename}, 'PerFrameFunctionalGroupsSequence' is required."
            )

        frame_seg_nums = []
        for f in img.PerFrameFunctionalGroupsSequence:
            if not hasattr(f, "SegmentIdentificationSequence"):
                raise NotImplementedError(
                    f"To read dicom seg: {img.filename}, 'SegmentIdentificationSequence' is required for each frame."
                )
            frame_seg_nums.append(int(f.SegmentIdentificationSequence[0].ReferencedSegmentNumber))

        frame_seg_nums_arr = np.array(frame_seg_nums)

        seg_descriptions = {int(f.SegmentNumber): f for f in img.SegmentSequence}

        for i in np.unique(frame_seg_nums_arr):
            indices = np.where(frame_seg_nums_arr == i)[0]
            yield (img.pixel_array[indices, ...], seg_descriptions[i])

    def _get_seg_data(self, img):
        """
        Get the array data and metadata of the segmentation image.

        Aegs:
            img: a Pydicom dataset object that has attribute "SegmentSequence".

        """

        metadata = self._get_meta_dict(img)
        n_classes = len(img.SegmentSequence)
        spatial_shape = list(img.pixel_array.shape)
        spatial_shape[0] = spatial_shape[0] // n_classes

        if self.label_dict is not None:
            metadata["labels"] = self.label_dict
            all_segs = np.zeros([*spatial_shape, len(self.label_dict)])
        else:
            metadata["labels"] = {}
            all_segs = np.zeros([*spatial_shape, n_classes])

        for i, (frames, description) in enumerate(self._get_frame_data(img)):
            segment_label = getattr(description, "SegmentLabel", f"label_{i}")
            class_name = getattr(description, "SegmentDescription", segment_label)
            if class_name not in metadata["labels"].keys():
                metadata["labels"][class_name] = i
            class_num = metadata["labels"][class_name]
            all_segs[..., class_num] = frames

        all_segs = all_segs.transpose([1, 2, 0, 3])
        metadata[MetaKeys.SPATIAL_SHAPE] = all_segs.shape[:-1]

        if "52009229" in metadata.keys():
            shared_func_group_seq = metadata["52009229"]["Value"][0]

            # get `ImageOrientationPatient`
            if "00209116" in shared_func_group_seq.keys():
                plane_orient_seq = shared_func_group_seq["00209116"]["Value"][0]
                if "00200037" in plane_orient_seq.keys():
                    metadata["00200037"] = plane_orient_seq["00200037"]

            # get `PixelSpacing`
            if "00289110" in shared_func_group_seq.keys():
                pixel_measure_seq = shared_func_group_seq["00289110"]["Value"][0]

                if "00280030" in pixel_measure_seq.keys():
                    pixel_spacing = pixel_measure_seq["00280030"]["Value"]
                    metadata["spacing"] = pixel_spacing
                    if "00180050" in pixel_measure_seq.keys():
                        metadata["spacing"] += pixel_measure_seq["00180050"]["Value"]

            if self.prune_metadata:
                metadata.pop("52009229")

        # get `ImagePositionPatient`
        if "52009230" in metadata.keys():
            first_frame_func_group_seq = metadata["52009230"]["Value"][0]
            if "00209113" in first_frame_func_group_seq.keys():
                plane_position_seq = first_frame_func_group_seq["00209113"]["Value"][0]
                if "00200032" in plane_position_seq.keys():
                    metadata["00200032"] = plane_position_seq["00200032"]
                    metadata["lastImagePositionPatient"] = metadata["52009230"]["Value"][-1]["00209113"]["Value"][0][
                        "00200032"
                    ]["Value"]
            if self.prune_metadata:
                metadata.pop("52009230")

        return all_segs, metadata

    def _get_array_data(self, img):
        """
        Get the array data of the image. If `RescaleSlope` and `RescaleIntercept` are available, the raw array data
        will be rescaled. The output data has the dtype np.float32 if the rescaling is applied.

        Args:
            img: a Pydicom dataset object.

        """
        # process Dicom series
        if not hasattr(img, "pixel_array"):
            raise ValueError(f"dicom data: {img.filename} does not have pixel_array.")
        data = img.pixel_array

        slope, offset = 1.0, 0.0
        rescale_flag = False
        if hasattr(img, "RescaleSlope"):
            slope = img.RescaleSlope
            rescale_flag = True
        if hasattr(img, "RescaleIntercept"):
            offset = img.RescaleIntercept
            rescale_flag = True
        if rescale_flag:
            data = data.astype(np.float32) * slope + offset

        return data


@require_pkg(pkg_name="nibabel")
class NibabelReader(ImageReader):
    """
    Load NIfTI format images based on Nibabel library.

    Args:
        channel_dim: the channel dimension of the input image, default is None.
            this is used to set original_channel_dim in the metadata, EnsureChannelFirstD reads this field.
            if None, `original_channel_dim` will be either `no_channel` or `-1`.
            most Nifti files are usually "channel last", no need to specify this argument for them.
        as_closest_canonical: if True, load the image as closest to canonical axis format.
        squeeze_non_spatial_dims: if True, non-spatial singletons will be squeezed, e.g. (256,256,1,3) -> (256,256,3)
        to_gpu: If True, load the image into GPU memory using CuPy and Kvikio. This can accelerate data loading.
            Default is False. CuPy and Kvikio are required for this option.
            Note: For compressed NIfTI files, some operations may still be performed on CPU memory,
            and the acceleration may not be significant. In some cases, it may be slower than loading on CPU.
            In practical use, it's recommended to add a warm up call before the actual loading.
            A related tutorial will be prepared in the future, and the document will be updated accordingly.
        kwargs: additional args for `nibabel.load` API. more details about available args:
            https://github.com/nipy/nibabel/blob/master/nibabel/loadsave.py

    """

    def __init__(
        self,
        channel_dim: str | int | None = None,
        as_closest_canonical: bool = False,
        squeeze_non_spatial_dims: bool = False,
        to_gpu: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.channel_dim = float("nan") if channel_dim == "no_channel" else channel_dim
        self.as_closest_canonical = as_closest_canonical
        self.squeeze_non_spatial_dims = squeeze_non_spatial_dims
        if to_gpu and (not has_cp or not has_kvikio):
            warnings.warn(
                "NibabelReader: CuPy and/or Kvikio not installed for GPU loading, falling back to CPU loading."
            )
            to_gpu = False

        if to_gpu:
            self.warmup_kvikio()

        self.to_gpu = to_gpu
        self.kwargs = kwargs

    def warmup_kvikio(self):
        """
        Warm up the Kvikio library to initialize the internal buffers, cuFile, GDS, etc.
        This can accelerate the data loading process when `to_gpu` is set to True.
        """
        if has_cp and has_kvikio:
            a = cp.arange(100)
            with tempfile.NamedTemporaryFile() as tmp_file:
                tmp_file_name = tmp_file.name
                f = kvikio.CuFile(tmp_file_name, "w")
                f.write(a)
                f.close()

                b = cp.empty_like(a)
                f = kvikio.CuFile(tmp_file_name, "r")
                f.read(b)

    def verify_suffix(self, filename: Sequence[PathLike] | PathLike) -> bool:
        """
        Verify whether the specified file or files format is supported by Nibabel reader.

        Args:
            filename: file name or a list of file names to read.
                if a list of files, verify all the suffixes.

        """
        suffixes: Sequence[str] = ["nii", "nii.gz"]
        return has_nib and is_supported_format(filename, suffixes)

    def read(self, data: Sequence[PathLike] | PathLike, **kwargs):
        """
        Read image data from specified file or files, it can read a list of images
        and stack them together as multi-channel data in `get_data()`.
        Note that the returned object is Nibabel image object or list of Nibabel image objects.

        Args:
            data: file name or a list of file names to read.
            kwargs: additional args for `nibabel.load` API, will override `self.kwargs` for existing keys.
                More details about available args:
                https://github.com/nipy/nibabel/blob/master/nibabel/loadsave.py

        """
        img_: list[Nifti1Image] = []

        filenames: Sequence[PathLike] = ensure_tuple(data)
        self.filenames = filenames
        kwargs_ = self.kwargs.copy()
        kwargs_.update(kwargs)
        for name in filenames:
            img = nib.load(name, **kwargs_)
            img = correct_nifti_header_if_necessary(img)
            img_.append(img)  # type: ignore
        return img_ if len(filenames) > 1 else img_[0]

    def get_data(self, img) -> tuple[np.ndarray, dict]:
        """
        Extract data array and metadata from loaded image and return them.
        This function returns two objects, first is numpy array of image data, second is dict of metadata.
        It constructs `affine`, `original_affine`, and `spatial_shape` and stores them in meta dict.
        When loading a list of files, they are stacked together at a new dimension as the first dimension,
        and the metadata of the first image is used to present the output metadata.

        Args:
            img: a Nibabel image object loaded from an image file or a list of Nibabel image objects.

        """
        # TODO: the actual type is list[np.ndarray | cp.ndarray]
        # should figure out how to define correct types without having cupy not found error
        # https://github.com/Project-MONAI/MONAI/pull/8188#discussion_r1886645918
        img_array: list[np.ndarray] = []
        compatible_meta: dict = {}

        for i, filename in zip(ensure_tuple(img), self.filenames):
            header = self._get_meta_dict(i)
            if MetaKeys.PIXDIM in header:
                header[MetaKeys.ORIGINAL_PIXDIM] = np.array(header[MetaKeys.PIXDIM], copy=True)
            header[MetaKeys.AFFINE] = self._get_affine(i)
            header[MetaKeys.ORIGINAL_AFFINE] = self._get_affine(i)
            header["as_closest_canonical"] = self.as_closest_canonical
            if self.as_closest_canonical:
                i = nib.as_closest_canonical(i)
                header[MetaKeys.AFFINE] = self._get_affine(i)
            header[MetaKeys.SPATIAL_SHAPE] = self._get_spatial_shape(i)
            header[MetaKeys.SPACE] = SpaceKeys.RAS
            data = self._get_array_data(i, filename)
            if self.squeeze_non_spatial_dims:
                for d in range(len(data.shape), len(header[MetaKeys.SPATIAL_SHAPE]), -1):
                    if data.shape[d - 1] == 1:
                        data = data.squeeze(axis=d - 1)
            img_array.append(data)
            if self.channel_dim is None:  # default to "no_channel" or -1
                header[MetaKeys.ORIGINAL_CHANNEL_DIM] = (
                    float("nan") if len(data.shape) == len(header[MetaKeys.SPATIAL_SHAPE]) else -1
                )
            else:
                header[MetaKeys.ORIGINAL_CHANNEL_DIM] = self.channel_dim
            _copy_compatible_dict(header, compatible_meta)

        return _stack_images(img_array, compatible_meta, to_cupy=self.to_gpu), compatible_meta

    def _get_meta_dict(self, img) -> dict:
        """
        Get the all the metadata of the image and convert to dict type.

        Args:
            img: a Nibabel image object loaded from an image file.

        """
        # swap to little endian as PyTorch doesn't support big endian
        try:
            header = img.header.as_byteswapped("<")
        except ValueError:
            header = img.header
        return dict(header)

    def _get_affine(self, img):
        """
        Get the affine matrix of the image, it can be used to correct
        spacing, orientation or execute spatial transforms.

        Args:
            img: a Nibabel image object loaded from an image file.

        """
        return np.array(img.affine, copy=True)

    def _get_spatial_shape(self, img):
        """
        Get the spatial shape of image data, it doesn't contain the channel dim.

        Args:
            img: a Nibabel image object loaded from an image file.

        """
        # swap to little endian as PyTorch doesn't support big endian
        try:
            header = img.header.as_byteswapped("<")
        except ValueError:
            header = img.header
        dim = header.get("dim", None)
        if dim is None:
            dim = header.get("dims")  # mgh format?
            dim = np.insert(dim, 0, 3)
        ndim = dim[0]
        size = list(dim[1:])
        if not is_no_channel(self.channel_dim):
            size.pop(int(self.channel_dim))  # type: ignore
        spatial_rank = max(min(ndim, 3), 1)
        return np.asarray(size[:spatial_rank])

    def _get_array_data(self, img, filename):
        """
        Get the raw array data of the image, converted to Numpy array.

        Args:
            img: a Nibabel image object loaded from an image file.
            filename: file name of the image.

        """
        if self.to_gpu:
            file_size = os.path.getsize(filename)
            image = cp.empty(file_size, dtype=cp.uint8)
            with kvikio.CuFile(filename, "r") as f:
                f.read(image)
            if filename.endswith(".nii.gz"):
                # for compressed data, have to tansfer to CPU to decompress
                # and then transfer back to GPU. It is not efficient compared to .nii file
                # and may be slower than CPU loading in some cases.
                warnings.warn("Loading compressed NIfTI file into GPU may not be efficient.")
                compressed_data = cp.asnumpy(image)
                with gzip.GzipFile(fileobj=io.BytesIO(compressed_data)) as gz_file:
                    decompressed_data = gz_file.read()

                image = cp.frombuffer(decompressed_data, dtype=cp.uint8)
            data_shape = img.shape
            data_offset = img.dataobj.offset
            data_dtype = img.dataobj.dtype
            return image[data_offset:].view(data_dtype).reshape(data_shape, order="F")
        return np.asanyarray(img.dataobj, order="C")


class NumpyReader(ImageReader):
    """
    Load NPY or NPZ format data based on Numpy library, they can be arrays or pickled objects.
    A typical usage is to load the `mask` data for classification task.
    It can load part of the npz file with specified `npz_keys`.

    Args:
        npz_keys: if loading npz file, only load the specified keys, if None, load all the items.
            stack the loaded items together to construct a new first dimension.
        channel_dim: if not None, explicitly specify the channel dim, otherwise, treat the array as no channel.
        kwargs: additional args for `numpy.load` API except `allow_pickle`. more details about available args:
            https://numpy.org/doc/stable/reference/generated/numpy.load.html

    """

    def __init__(self, npz_keys: KeysCollection | None = None, channel_dim: str | int | None = None, **kwargs):
        super().__init__()
        if npz_keys is not None:
            npz_keys = ensure_tuple(npz_keys)
        self.npz_keys = npz_keys
        self.channel_dim = float("nan") if channel_dim == "no_channel" else channel_dim
        self.kwargs = kwargs

    def verify_suffix(self, filename: Sequence[PathLike] | PathLike) -> bool:
        """
        Verify whether the specified file or files format is supported by Numpy reader.

        Args:
            filename: file name or a list of file names to read.
                if a list of files, verify all the suffixes.
        """
        suffixes: Sequence[str] = ["npz", "npy"]
        return is_supported_format(filename, suffixes)

    def read(self, data: Sequence[PathLike] | PathLike, **kwargs):
        """
        Read image data from specified file or files, it can read a list of data files
        and stack them together as multi-channel data in `get_data()`.
        Note that the returned object is Numpy array or list of Numpy arrays.

        Args:
            data: file name or a list of file names to read.
            kwargs: additional args for `numpy.load` API except `allow_pickle`, will override `self.kwargs` for existing keys.
                More details about available args:
                https://numpy.org/doc/stable/reference/generated/numpy.load.html

        """
        img_: list[Nifti1Image] = []

        filenames: Sequence[PathLike] = ensure_tuple(data)
        kwargs_ = self.kwargs.copy()
        kwargs_.update(kwargs)
        for name in filenames:
            img = np.load(name, allow_pickle=True, **kwargs_)
            if Path(name).name.endswith(".npz"):
                # load expected items from NPZ file
                npz_keys = list(img.keys()) if self.npz_keys is None else self.npz_keys
                for k in npz_keys:
                    img_.append(img[k])
            else:
                img_.append(img)

        return img_ if len(img_) > 1 else img_[0]

    def get_data(self, img) -> tuple[np.ndarray, dict]:
        """
        Extract data array and metadata from loaded image and return them.
        This function returns two objects, first is numpy array of image data, second is dict of metadata.
        It constructs `affine`, `original_affine`, and `spatial_shape` and stores them in meta dict.
        When loading a list of files, they are stacked together at a new dimension as the first dimension,
        and the metadata of the first image is used to represent the output metadata.

        Args:
            img: a Numpy array loaded from a file or a list of Numpy arrays.

        """
        img_array: list[np.ndarray] = []
        compatible_meta: dict = {}
        if isinstance(img, np.ndarray):
            img = (img,)

        for i in ensure_tuple(img):
            header: dict[MetaKeys, Any] = {}
            if isinstance(i, np.ndarray):
                # if `channel_dim` is None, can not detect the channel dim, use all the dims as spatial_shape
                spatial_shape = np.asarray(i.shape)
                if isinstance(self.channel_dim, int):
                    spatial_shape = np.delete(spatial_shape, self.channel_dim)
                header[MetaKeys.SPATIAL_SHAPE] = spatial_shape
                header[MetaKeys.SPACE] = SpaceKeys.RAS
            img_array.append(i)
            header[MetaKeys.ORIGINAL_CHANNEL_DIM] = (
                self.channel_dim if isinstance(self.channel_dim, int) else float("nan")
            )
            _copy_compatible_dict(header, compatible_meta)

        return _stack_images(img_array, compatible_meta), compatible_meta


@require_pkg(pkg_name="PIL")
class PILReader(ImageReader):
    """
    Load common 2D image format (supports PNG, JPG, BMP) file or files from provided path.

    Args:
        converter: additional function to convert the image data after `read()`.
            for example, use `converter=lambda image: image.convert("LA")` to convert image format.
        reverse_indexing: whether to swap axis 0 and 1 after loading the array, this is enabled by default,
            so that output of the reader is consistent with the other readers. Set this option to ``False`` to use
            the PIL backend's original spatial axes convention.
        kwargs: additional args for `Image.open` API in `read()`, mode details about available args:
            https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.open
    """

    def __init__(self, converter: Callable | None = None, reverse_indexing: bool = True, **kwargs):
        super().__init__()
        self.converter = converter
        self.reverse_indexing = reverse_indexing
        self.kwargs = kwargs

    def verify_suffix(self, filename: Sequence[PathLike] | PathLike) -> bool:
        """
        Verify whether the specified file or files format is supported by PIL reader.

        Args:
            filename: file name or a list of file names to read.
                if a list of files, verify all the suffixes.
        """
        suffixes: Sequence[str] = ["png", "jpg", "jpeg", "bmp"]
        return has_pil and is_supported_format(filename, suffixes)

    def read(self, data: Sequence[PathLike] | PathLike | np.ndarray, **kwargs):
        """
        Read image data from specified file or files, it can read a list of images
        and stack them together as multi-channel data in `get_data()`.
        Note that the returned object is PIL image or list of PIL image.

        Args:
            data: file name or a list of file names to read.
            kwargs: additional args for `Image.open` API in `read()`, will override `self.kwargs` for existing keys.
                Mode details about available args:
                https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.open

        """
        img_: list[PILImage.Image] = []

        filenames: Sequence[PathLike] = ensure_tuple(data)
        kwargs_ = self.kwargs.copy()
        kwargs_.update(kwargs)
        for name in filenames:
            img = PILImage.open(name, **kwargs_)
            if callable(self.converter):
                img = self.converter(img)
            img_.append(img)

        return img_ if len(filenames) > 1 else img_[0]

    def get_data(self, img) -> tuple[np.ndarray, dict]:
        """
        Extract data array and metadata from loaded image and return them.
        This function returns two objects, first is numpy array of image data, second is dict of metadata.
        It computes `spatial_shape` and stores it in meta dict.
        When loading a list of files, they are stacked together at a new dimension as the first dimension,
        and the metadata of the first image is used to represent the output metadata.
        Note that by default `self.reverse_indexing` is set to ``True``, which swaps axis 0 and 1 after loading
        the array because the spatial axes definition in PIL is different from other common medical packages.

        Args:
            img: a PIL Image object loaded from a file or a list of PIL Image objects.

        """
        img_array: list[np.ndarray] = []
        compatible_meta: dict = {}

        for i in ensure_tuple(img):
            header = self._get_meta_dict(i)
            header[MetaKeys.SPATIAL_SHAPE] = self._get_spatial_shape(i)
            data = np.moveaxis(np.asarray(i), 0, 1) if self.reverse_indexing else np.asarray(i)
            img_array.append(data)
            header[MetaKeys.ORIGINAL_CHANNEL_DIM] = (
                float("nan") if len(data.shape) == len(header[MetaKeys.SPATIAL_SHAPE]) else -1
            )
            _copy_compatible_dict(header, compatible_meta)

        return _stack_images(img_array, compatible_meta), compatible_meta

    def _get_meta_dict(self, img) -> dict:
        """
        Get the all the metadata of the image and convert to dict type.
        Args:
            img: a PIL Image object loaded from an image file.

        """
        return {"format": img.format, "mode": img.mode, "width": img.width, "height": img.height}

    def _get_spatial_shape(self, img):
        """
        Get the spatial shape of image data, it doesn't contain the channel dim.
        Args:
            img: a PIL Image object loaded from an image file.
        """
        return np.asarray((img.width, img.height))


@dataclass
class NrrdImage:
    """Class to wrap nrrd image array and metadata header"""

    array: np.ndarray
    header: dict


@require_pkg(pkg_name="nrrd")
class NrrdReader(ImageReader):
    """
    Load NRRD format images based on pynrrd library.

    Args:
        channel_dim: the channel dimension of the input image, default is None.
            This is used to set original_channel_dim in the metadata, EnsureChannelFirstD reads this field.
            If None, `original_channel_dim` will be either `no_channel` or `0`.
            NRRD files are usually "channel first".
        dtype: dtype of the data array when loading image.
        index_order: Specify whether the returned data array should be in C-order (‘C’) or Fortran-order (‘F’).
            Numpy is usually in C-order, but default on the NRRD header is F
        affine_lps_to_ras: whether to convert the affine matrix from "LPS" to "RAS". Defaults to ``True``.
            Set to ``True`` to be consistent with ``NibabelReader``, otherwise the affine matrix is unmodified.

        kwargs: additional args for `nrrd.read` API. more details about available args:
            https://github.com/mhe/pynrrd/blob/master/nrrd/reader.py

    """

    def __init__(
        self,
        channel_dim: str | int | None = None,
        dtype: np.dtype | type | str | None = np.float32,
        index_order: str = "F",
        affine_lps_to_ras: bool = True,
        **kwargs,
    ):
        self.channel_dim = float("nan") if channel_dim == "no_channel" else channel_dim
        self.dtype = dtype
        self.index_order = index_order
        self.affine_lps_to_ras = affine_lps_to_ras
        self.kwargs = kwargs

    def verify_suffix(self, filename: Sequence[PathLike] | PathLike) -> bool:
        """
        Verify whether the specified `filename` is supported by pynrrd reader.

        Args:
            filename: file name or a list of file names to read.
                if a list of files, verify all the suffixes.

        """
        suffixes: Sequence[str] = ["nrrd", "seg.nrrd"]
        return has_nrrd and is_supported_format(filename, suffixes)

    def read(self, data: Sequence[PathLike] | PathLike, **kwargs) -> Sequence[Any] | Any:
        """
        Read image data from specified file or files.
        Note that it returns a data object or a sequence of data objects.

        Args:
            data: file name or a list of file names to read.
            kwargs: additional args for actual `read` API of 3rd party libs.

        """
        img_: list = []
        filenames: Sequence[PathLike] = ensure_tuple(data)
        kwargs_ = self.kwargs.copy()
        kwargs_.update(kwargs)
        for name in filenames:
            nrrd_image = NrrdImage(*nrrd.read(name, index_order=self.index_order, **kwargs_))
            img_.append(nrrd_image)
        return img_ if len(filenames) > 1 else img_[0]

    def get_data(self, img: NrrdImage | list[NrrdImage]) -> tuple[np.ndarray, dict]:
        """
        Extract data array and metadata from loaded image and return them.
        This function must return two objects, the first is a numpy array of image data,
        the second is a dictionary of metadata.

        Args:
            img: a `NrrdImage` loaded from an image file or a list of image objects.

        """
        img_array: list[np.ndarray] = []
        compatible_meta: dict = {}

        for i in ensure_tuple(img):
            data = i.array.astype(self.dtype)
            img_array.append(data)
            header = dict(i.header)
            if self.index_order == "C":
                header = self._convert_f_to_c_order(header)
            header[MetaKeys.ORIGINAL_AFFINE] = self._get_affine(header)

            if self.affine_lps_to_ras:
                header = self._switch_lps_ras(header)
            if header.get(MetaKeys.SPACE, "left-posterior-superior") == "left-posterior-superior":
                header[MetaKeys.SPACE] = SpaceKeys.LPS  # assuming LPS if not specified

            header[MetaKeys.AFFINE] = header[MetaKeys.ORIGINAL_AFFINE].copy()
            header[MetaKeys.SPATIAL_SHAPE] = header["sizes"].copy()
            [header.pop(k) for k in ("sizes", "space origin", "space directions")]  # rm duplicated data in header

            if self.channel_dim is None:  # default to "no_channel" or -1
                header[MetaKeys.ORIGINAL_CHANNEL_DIM] = (
                    float("nan") if len(data.shape) == len(header[MetaKeys.SPATIAL_SHAPE]) else 0
                )
            else:
                header[MetaKeys.ORIGINAL_CHANNEL_DIM] = self.channel_dim
            _copy_compatible_dict(header, compatible_meta)

        return _stack_images(img_array, compatible_meta), compatible_meta

    def _get_affine(self, header: dict) -> np.ndarray:
        """
        Get the affine matrix of the image, it can be used to correct
        spacing, orientation or execute spatial transforms.

        Args:
            img: A `NrrdImage` loaded from image file

        """
        direction = header["space directions"]
        origin = header["space origin"]

        x, y = direction.shape
        affine_diam = min(x, y) + 1
        affine: np.ndarray = np.eye(affine_diam)
        affine[:x, :y] = direction.T
        affine[: (affine_diam - 1), -1] = origin  # len origin is always affine_diam - 1
        return affine

    def _switch_lps_ras(self, header: dict) -> dict:
        """
        For compatibility with nibabel, switch from LPS to RAS. Adapt affine matrix and
        `space` argument in header accordingly. If no information of space is given in the header,
        LPS is assumed and thus converted to RAS. If information about space is given,
        but is not LPS, the unchanged header is returned.

        Args:
            header: The image metadata as dict

        """
        if "space" not in header or header["space"] == "left-posterior-superior":
            header[MetaKeys.ORIGINAL_AFFINE] = orientation_ras_lps(header[MetaKeys.ORIGINAL_AFFINE])
            header[MetaKeys.SPACE] = SpaceKeys.RAS
        return header

    def _convert_f_to_c_order(self, header: dict) -> dict:
        """
        All header fields of a NRRD are specified in `F` (Fortran) order, even if the image was read as C-ordered array.
        1D arrays of header['space origin'] and header['sizes'] become inverted, e.g, [1,2,3] -> [3,2,1]
        The 2D Array for header['space directions'] is transposed: [[1,0,0],[0,2,0],[0,0,3]] -> [[3,0,0],[0,2,0],[0,0,1]]
        For more details refer to: https://pynrrd.readthedocs.io/en/latest/user-guide.html#index-ordering

        Args:
            header: The image metadata as dict

        """

        header["space directions"] = np.rot90(np.flip(header["space directions"], 0))
        header["space origin"] = header["space origin"][::-1]
        header["sizes"] = header["sizes"][::-1]
        return header
