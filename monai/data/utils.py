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

import hashlib
import json
import logging
import math
import os
import pickle
import sys
from collections import abc, defaultdict
from collections.abc import Generator, Iterable, Mapping, Sequence, Sized
from copy import deepcopy
from functools import reduce
from itertools import product, starmap, zip_longest
from pathlib import PurePath
from typing import Any

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

from monai import config
from monai.config.type_definitions import NdarrayOrTensor, NdarrayTensor, PathLike
from monai.data.meta_obj import MetaObj
from monai.utils import (
    MAX_SEED,
    BlendMode,
    Method,
    NumpyPadMode,
    TraceKeys,
    convert_data_type,
    convert_to_dst_type,
    ensure_tuple,
    ensure_tuple_rep,
    ensure_tuple_size,
    fall_back_tuple,
    first,
    get_equivalent_dtype,
    issequenceiterable,
    look_up_option,
    optional_import,
)

pd, _ = optional_import("pandas")
DataFrame, _ = optional_import("pandas", name="DataFrame")
nib, _ = optional_import("nibabel")

__all__ = [
    "AFFINE_TOL",
    "SUPPORTED_PICKLE_MOD",
    "affine_to_spacing",
    "compute_importance_map",
    "compute_shape_offset",
    "convert_tables_to_dicts",
    "correct_nifti_header_if_necessary",
    "create_file_basename",
    "decollate_batch",
    "dense_patch_slices",
    "get_random_patch",
    "get_valid_patch_size",
    "is_supported_format",
    "iter_patch",
    "iter_patch_position",
    "iter_patch_slices",
    "json_hashing",
    "list_data_collate",
    "no_collation",
    "orientation_ras_lps",
    "pad_list_data_collate",
    "partition_dataset",
    "partition_dataset_classes",
    "pickle_hashing",
    "rectify_header_sform_qform",
    "reorient_spatial_axes",
    "resample_datalist",
    "select_cross_validation_folds",
    "set_rnd",
    "sorted_dict",
    "to_affine_nd",
    "worker_init_fn",
    "zoom_affine",
    "remove_keys",
    "remove_extra_metadata",
    "get_extra_metadata_keys",
    "PICKLE_KEY_SUFFIX",
    "is_no_channel",
]

# module to be used by `torch.save`
SUPPORTED_PICKLE_MOD = {"pickle": pickle}

# tolerance for affine matrix computation
AFFINE_TOL = 1e-3


def get_random_patch(
    dims: Sequence[int], patch_size: Sequence[int], rand_state: np.random.RandomState | None = None
) -> tuple[slice, ...]:
    """
    Returns a tuple of slices to define a random patch in an array of shape `dims` with size `patch_size` or the as
    close to it as possible within the given dimension. It is expected that `patch_size` is a valid patch for a source
    of shape `dims` as returned by `get_valid_patch_size`.

    Args:
        dims: shape of source array
        patch_size: shape of patch size to generate
        rand_state: a random state object to generate random numbers from

    Returns:
        (tuple of slice): a tuple of slice objects defining the patch
    """

    # choose the minimal corner of the patch
    rand_int = np.random.randint if rand_state is None else rand_state.randint
    min_corner = tuple(rand_int(0, ms - ps + 1) if ms > ps else 0 for ms, ps in zip(dims, patch_size))

    # create the slices for each dimension which define the patch in the source array
    return tuple(slice(mc, mc + ps) for mc, ps in zip(min_corner, patch_size))


def iter_patch_slices(
    image_size: Sequence[int],
    patch_size: Sequence[int] | int,
    start_pos: Sequence[int] = (),
    overlap: Sequence[float] | float = 0.0,
    padded: bool = True,
) -> Generator[tuple[slice, ...], None, None]:
    """
    Yield successive tuples of slices defining patches of size `patch_size` from an array of dimensions `image_size`.
    The iteration starts from position `start_pos` in the array, or starting at the origin if this isn't provided. Each
    patch is chosen in a contiguous grid using a rwo-major ordering.

    Args:
        image_size: dimensions of array to iterate over
        patch_size: size of patches to generate slices for, 0 or None selects whole dimension
        start_pos: starting position in the array, default is 0 for each dimension
        overlap: the amount of overlap of neighboring patches in each dimension (a value between 0.0 and 1.0).
            If only one float number is given, it will be applied to all dimensions. Defaults to 0.0.
        padded: if the image is padded so the patches can go beyond the borders. Defaults to False.

    Yields:
        Tuples of slice objects defining each patch
    """

    # ensure patch_size has the right length
    patch_size_ = get_valid_patch_size(image_size, patch_size)

    # create slices based on start position of each patch
    for position in iter_patch_position(
        image_size=image_size, patch_size=patch_size_, start_pos=start_pos, overlap=overlap, padded=padded
    ):
        yield tuple(slice(s, s + p) for s, p in zip(position, patch_size_))


def dense_patch_slices(
    image_size: Sequence[int], patch_size: Sequence[int], scan_interval: Sequence[int], return_slice: bool = True
) -> list[tuple[slice, ...]]:
    """
    Enumerate all slices defining ND patches of size `patch_size` from an `image_size` input image.

    Args:
        image_size: dimensions of image to iterate over
        patch_size: size of patches to generate slices
        scan_interval: dense patch sampling interval
        return_slice: whether to return a list of slices (or tuples of indices), defaults to True

    Returns:
        a list of slice objects defining each patch

    """
    num_spatial_dims = len(image_size)
    patch_size = get_valid_patch_size(image_size, patch_size)
    scan_interval = ensure_tuple_size(scan_interval, num_spatial_dims)

    scan_num = []
    for i in range(num_spatial_dims):
        if scan_interval[i] == 0:
            scan_num.append(1)
        else:
            num = int(math.ceil(float(image_size[i]) / scan_interval[i]))
            scan_dim = first(d for d in range(num) if d * scan_interval[i] + patch_size[i] >= image_size[i])
            scan_num.append(scan_dim + 1 if scan_dim is not None else 1)

    starts = []
    for dim in range(num_spatial_dims):
        dim_starts = []
        for idx in range(scan_num[dim]):
            start_idx = idx * scan_interval[dim]
            start_idx -= max(start_idx + patch_size[dim] - image_size[dim], 0)
            dim_starts.append(start_idx)
        starts.append(dim_starts)
    out = np.asarray([x.flatten() for x in np.meshgrid(*starts, indexing="ij")]).T
    if return_slice:
        return [tuple(slice(s, s + patch_size[d]) for d, s in enumerate(x)) for x in out]
    return [tuple((s, s + patch_size[d]) for d, s in enumerate(x)) for x in out]  # type: ignore


def iter_patch_position(
    image_size: Sequence[int],
    patch_size: Sequence[int] | int | np.ndarray,
    start_pos: Sequence[int] = (),
    overlap: Sequence[float] | float | Sequence[int] | int = 0.0,
    padded: bool = False,
):
    """
    Yield successive tuples of upper left corner of patches of size `patch_size` from an array of dimensions `image_size`.
    The iteration starts from position `start_pos` in the array, or starting at the origin if this isn't provided. Each
    patch is chosen in a contiguous grid using a rwo-major ordering.

    Args:
        image_size: dimensions of array to iterate over
        patch_size: size of patches to generate slices for, 0 or None selects whole dimension
        start_pos: starting position in the array, default is 0 for each dimension
        overlap: the amount of overlap of neighboring patches in each dimension.
            Either a float or list of floats between 0.0 and 1.0 to define relative overlap to patch size, or
            an int or list of ints to define number of pixels for overlap.
            If only one float/int number is given, it will be applied to all dimensions. Defaults to 0.0.
        padded: if the image is padded so the patches can go beyond the borders. Defaults to False.

    Yields:
        Tuples of positions defining the upper left corner of each patch
    """

    # ensure patchSize and startPos are the right length
    ndim = len(image_size)
    patch_size_ = get_valid_patch_size(image_size, patch_size)
    start_pos = ensure_tuple_size(start_pos, ndim)
    overlap = ensure_tuple_rep(overlap, ndim)

    # calculate steps, which depends on the amount of overlap
    if isinstance(overlap[0], float):
        steps = tuple(round(p * (1.0 - o)) for p, o in zip(patch_size_, overlap))
    else:
        steps = tuple(p - o for p, o in zip(patch_size_, overlap))

    # calculate the last starting location (depending on the padding)
    end_pos = image_size if padded else tuple(s - round(p) + 1 for s, p in zip(image_size, patch_size_))

    # collect the ranges to step over each dimension
    ranges = starmap(range, zip(start_pos, end_pos, steps))

    # choose patches by applying product to the ranges
    return product(*ranges)


def iter_patch(
    arr: NdarrayOrTensor,
    patch_size: Sequence[int] | int = 0,
    start_pos: Sequence[int] = (),
    overlap: Sequence[float] | float = 0.0,
    copy_back: bool = True,
    mode: str | None = NumpyPadMode.WRAP,
    **pad_opts: dict,
) -> Generator[tuple[NdarrayOrTensor, np.ndarray], None, None]:
    """
    Yield successive patches from `arr` of size `patch_size`. The iteration can start from position `start_pos` in `arr`
    but drawing from a padded array extended by the `patch_size` in each dimension (so these coordinates can be negative
    to start in the padded region). If `copy_back` is True the values from each patch are written back to `arr`.

    Args:
        arr: array to iterate over
        patch_size: size of patches to generate slices for, 0 or None selects whole dimension.
            For 0 or None, padding and overlap ratio of the corresponding dimension will be 0.
        start_pos: starting position in the array, default is 0 for each dimension
        overlap: the amount of overlap of neighboring patches in each dimension (a value between 0.0 and 1.0).
            If only one float number is given, it will be applied to all dimensions. Defaults to 0.0.
        copy_back: if True data from the yielded patches is copied back to `arr` once the generator completes
        mode: available modes: (Numpy) {``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
            ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
            (PyTorch) {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
            One of the listed string values or a user supplied function.
            If None, no wrapping is performed. Defaults to ``"wrap"``.
            See also: https://numpy.org/doc/stable/reference/generated/numpy.pad.html
            https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
            requires pytorch >= 1.10 for best compatibility.
        pad_opts: other arguments for the `np.pad` or `torch.pad` function.
            note that `np.pad` treats channel dimension as the first dimension.

    Yields:
        Patches of array data from `arr` which are views into a padded array which can be modified, if `copy_back` is
        True these changes will be reflected in `arr` once the iteration completes.

    Note:
        coordinate format is:

            [1st_dim_start, 1st_dim_end,
             2nd_dim_start, 2nd_dim_end,
             ...,
             Nth_dim_start, Nth_dim_end]]

    """

    from monai.transforms.croppad.functional import pad_nd  # needs to be here to avoid circular import

    # ensure patchSize and startPos are the right length
    patch_size_ = get_valid_patch_size(arr.shape, patch_size)
    start_pos = ensure_tuple_size(start_pos, arr.ndim)

    # set padded flag to false if pad mode is None
    padded = bool(mode)
    is_v = [bool(p) for p in ensure_tuple_size(patch_size, arr.ndim)]  # whether a valid patch size provided
    _pad_size = tuple(p if v and padded else 0 for p, v in zip(patch_size_, is_v))  # pad p if v else 0
    _overlap = [op if v else 0.0 for op, v in zip(ensure_tuple_rep(overlap, arr.ndim), is_v)]  # overlap if v else 0.0
    # pad image by maximum values needed to ensure patches are taken from inside an image
    if padded:
        arrpad = pad_nd(arr, to_pad=[(p, p) for p in _pad_size], mode=mode, **pad_opts)  # type: ignore
        # choose a start position in the padded image
        start_pos_padded = tuple(s + p for s, p in zip(start_pos, _pad_size))

        # choose a size to iterate over which is smaller than the actual padded image to prevent producing
        # patches which are only in the padded regions
        iter_size = tuple(s + p for s, p in zip(arr.shape, _pad_size))
    else:
        arrpad = arr
        start_pos_padded = start_pos
        iter_size = arr.shape

    for slices in iter_patch_slices(iter_size, patch_size_, start_pos_padded, _overlap, padded=padded):
        # compensate original image padding
        if padded:
            coords_no_pad = tuple((coord.start - p, coord.stop - p) for coord, p in zip(slices, _pad_size))
        else:
            coords_no_pad = tuple((coord.start, coord.stop) for coord in slices)
        yield arrpad[slices], np.asarray(coords_no_pad)  # data and coords (in numpy; works with torch loader)

    # copy back data from the padded image if required
    if copy_back:
        slices = tuple(slice(p, p + s) for p, s in zip(_pad_size, arr.shape))
        arr[...] = arrpad[slices]  # type: ignore


def get_valid_patch_size(image_size: Sequence[int], patch_size: Sequence[int] | int | np.ndarray) -> tuple[int, ...]:
    """
    Given an image of dimensions `image_size`, return a patch size tuple taking the dimension from `patch_size` if this is
    not 0/None. Otherwise, or if `patch_size` is shorter than `image_size`, the dimension from `image_size` is taken. This ensures
    the returned patch size is within the bounds of `image_size`. If `patch_size` is a single number this is interpreted as a
    patch of the same dimensionality of `image_size` with that size in each dimension.
    """
    ndim = len(image_size)
    patch_size_ = ensure_tuple_size(patch_size, ndim)

    # ensure patch size dimensions are not larger than image dimension, if a dimension is None or 0 use whole dimension
    return tuple(min(ms, ps or ms) for ms, ps in zip(image_size, patch_size_))


def dev_collate(batch, level: int = 1, logger_name: str = "dev_collate"):
    """
    Recursively run collate logic and provide detailed loggings for debugging purposes.
    It reports results at the 'critical' level, is therefore suitable in the context of exception handling.

    Args:
        batch: batch input to collate
        level: current level of recursion for logging purposes
        logger_name: name of logger to use for logging

    See also: https://pytorch.org/docs/stable/data.html#working-with-collate-fn
    """
    elem = batch[0]
    elem_type = type(elem)
    l_str = ">" * level
    batch_str = f"{batch[:10]}{' ... ' if len(batch) > 10 else ''}"
    if isinstance(elem, torch.Tensor):
        try:
            logging.getLogger(logger_name).critical(f"{l_str} collate/stack a list of tensors")
            return torch.stack(batch, 0)
        except TypeError as e:
            logging.getLogger(logger_name).critical(
                f"{l_str} E: {e}, type {[type(elem).__name__ for elem in batch]} in collate({batch_str})"
            )
            return
        except RuntimeError as e:
            logging.getLogger(logger_name).critical(
                f"{l_str} E: {e}, shape {[elem.shape for elem in batch]} in collate({batch_str})"
            )
            return
    elif elem_type.__module__ == "numpy" and elem_type.__name__ != "str_" and elem_type.__name__ != "string_":
        if elem_type.__name__ in ["ndarray", "memmap"]:
            logging.getLogger(logger_name).critical(f"{l_str} collate/stack a list of numpy arrays")
            return dev_collate([torch.as_tensor(b) for b in batch], level=level, logger_name=logger_name)
        elif elem.shape == ():  # scalars
            return batch
    elif isinstance(elem, (float, int, str, bytes)):
        return batch
    elif isinstance(elem, abc.Mapping):
        out = {}
        for key in elem:
            logging.getLogger(logger_name).critical(f'{l_str} collate dict key "{key}" out of {len(elem)} keys')
            out[key] = dev_collate([d[key] for d in batch], level=level + 1, logger_name=logger_name)
        return out
    elif isinstance(elem, abc.Sequence):
        it = iter(batch)
        els = list(it)
        try:
            sizes = [len(elem) for elem in els]  # may not have `len`
        except TypeError:
            types = [type(elem).__name__ for elem in els]
            logging.getLogger(logger_name).critical(f"{l_str} E: type {types} in collate({batch_str})")
            return
        logging.getLogger(logger_name).critical(f"{l_str} collate list of sizes: {sizes}.")
        if any(s != sizes[0] for s in sizes):
            logging.getLogger(logger_name).critical(
                f"{l_str} collate list inconsistent sizes, got size: {sizes}, in collate({batch_str})"
            )
        transposed = zip(*batch)
        return [dev_collate(samples, level=level + 1, logger_name=logger_name) for samples in transposed]
    logging.getLogger(logger_name).critical(f"{l_str} E: unsupported type in collate {batch_str}.")
    return


PICKLE_KEY_SUFFIX = TraceKeys.KEY_SUFFIX


def pickle_operations(data, key=PICKLE_KEY_SUFFIX, is_encode: bool = True):
    """
    Applied_operations are dictionaries with varying sizes, this method converts them to bytes so that we can (de-)collate.

    Args:
        data: a list or dictionary with substructures to be pickled/unpickled.
        key: the key suffix for the target substructures, defaults to "_transforms" (`data.utils.PICKLE_KEY_SUFFIX`).
        is_encode: whether it's encoding using pickle.dumps (True) or decoding using pickle.loads (False).
    """
    if isinstance(data, Mapping):
        data = dict(data)
        for k in data:
            if f"{k}".endswith(key):
                if is_encode and not isinstance(data[k], bytes):
                    data[k] = pickle.dumps(data[k], 0)
                if not is_encode and isinstance(data[k], bytes):
                    data[k] = pickle.loads(data[k])
        return {k: pickle_operations(v, key=key, is_encode=is_encode) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [pickle_operations(item, key=key, is_encode=is_encode) for item in data]
    return data


def collate_meta_tensor_fn(batch, *, collate_fn_map=None):
    """
    Collate a sequence of meta tensor into a single batched metatensor. This is called by `collage_meta_tensor`
    and so should not be used as a collate function directly in dataloaders.
    """
    from torch.utils.data._utils.collate import collate_tensor_fn  # imported here for pylint/mypy issues

    collated = collate_tensor_fn(batch)

    meta_dicts = [i.meta or TraceKeys.NONE for i in batch]
    common_ = set.intersection(*[set(d.keys()) for d in meta_dicts if isinstance(d, dict)])
    if common_:
        meta_dicts = [{k: d[k] for k in common_} if isinstance(d, dict) else TraceKeys.NONE for d in meta_dicts]
    collated.meta = default_collate(meta_dicts)
    collated.applied_operations = [i.applied_operations or TraceKeys.NONE for i in batch]
    collated.is_batch = True
    return collated


def collate_meta_tensor(batch):
    """collate a sequence of meta tensor sequences/dictionaries into
    a single batched metatensor or a dictionary of batched metatensor"""
    if not isinstance(batch, Sequence):
        raise NotImplementedError()
    elem_0 = first(batch)
    if isinstance(elem_0, MetaObj):
        return collate_meta_tensor_fn(batch)
    if isinstance(elem_0, Mapping):
        return {k: collate_meta_tensor([d[k] for d in batch]) for k in elem_0}
    if isinstance(elem_0, (tuple, list)):
        return [collate_meta_tensor([d[i] for d in batch]) for i in range(len(elem_0))]

    # no more recursive search for MetaTensor
    return default_collate(batch)


def list_data_collate(batch: Sequence):
    """
    Enhancement for PyTorch DataLoader default collate.
    If dataset already returns a list of batch data that generated in transforms, need to merge all data to 1 list.
    Then it's same as the default collate behavior.

    Note:
        Need to use this collate if apply some transforms that can generate batch data.

    """
    from torch.utils.data._utils.collate import default_collate_fn_map

    from monai.data.meta_tensor import MetaTensor

    default_collate_fn_map.update({MetaTensor: collate_meta_tensor_fn})
    elem = batch[0]
    data = [i for k in batch for i in k] if isinstance(elem, list) else batch
    key = None
    collate_fn = default_collate
    try:
        if config.USE_META_DICT:
            data = pickle_operations(data)  # bc 0.9.0
        if isinstance(elem, Mapping):
            ret = {}
            for k in elem:
                key = k
                data_for_batch = [d[key] for d in data]
                ret[key] = collate_fn(data_for_batch)
        else:
            ret = collate_fn(data)
        return ret
    except RuntimeError as re:
        re_str = str(re)
        if "equal size" in re_str:
            if key is not None:
                re_str += f"\nCollate error on the key '{key}' of dictionary data."
            re_str += (
                "\n\nMONAI hint: if your transforms intentionally create images of different shapes, creating your "
                + "`DataLoader` with `collate_fn=pad_list_data_collate` might solve this problem (check its "
                + "documentation)."
            )
        _ = dev_collate(data)
        raise RuntimeError(re_str) from re
    except TypeError as re:
        re_str = str(re)
        if "numpy" in re_str and "Tensor" in re_str:
            if key is not None:
                re_str += f"\nCollate error on the key '{key}' of dictionary data."
            re_str += (
                "\n\nMONAI hint: if your transforms intentionally create mixtures of torch Tensor and numpy ndarray, "
                + "creating your `DataLoader` with `collate_fn=pad_list_data_collate` might solve this problem "
                + "(check its documentation)."
            )
        _ = dev_collate(data)
        raise TypeError(re_str) from re


def _non_zipping_check(batch_data: Mapping | Iterable, detach: bool, pad: bool, fill_value):
    """
    Utility function based on `decollate_batch`, to identify the largest batch size from the collated data.
    returns batch_size, the list of non-iterable items, and the dictionary or list with their items decollated.

    See `decollate_batch` for more details.
    """
    _deco: Mapping | Sequence
    if isinstance(batch_data, Mapping):
        _deco = {key: decollate_batch(batch_data[key], detach, pad=pad, fill_value=fill_value) for key in batch_data}
    elif isinstance(batch_data, Iterable):
        _deco = [decollate_batch(b, detach, pad=pad, fill_value=fill_value) for b in batch_data]
    else:
        raise NotImplementedError(f"Unable to de-collate: {batch_data}, type: {type(batch_data)}.")
    batch_size, non_iterable = 0, []
    for k, v in _deco.items() if isinstance(_deco, Mapping) else enumerate(_deco):
        if not isinstance(v, Iterable) or isinstance(v, (str, bytes)) or (isinstance(v, torch.Tensor) and v.ndim == 0):
            # Not running the usual list decollate here:
            # don't decollate ['test', 'test'] into [['t', 't'], ['e', 'e'], ['s', 's'], ['t', 't']]
            # torch.tensor(0) is iterable but iter(torch.tensor(0)) raises TypeError: iteration over a 0-d tensor
            non_iterable.append(k)
        elif isinstance(v, Sized):
            batch_size = max(batch_size, len(v))
    return batch_size, non_iterable, _deco


def decollate_batch(batch, detach: bool = True, pad=True, fill_value=None):
    """De-collate a batch of data (for example, as produced by a `DataLoader`).

    Returns a list of structures with the original tensor's 0-th dimension sliced into elements using `torch.unbind`.

    Images originally stored as (B,C,H,W,[D]) will be returned as (C,H,W,[D]). Other information,
    such as metadata, may have been stored in a list (or a list inside nested dictionaries). In
    this case we return the element of the list corresponding to the batch idx.

    Return types aren't guaranteed to be the same as the original, since numpy arrays will have been
    converted to torch.Tensor, sequences may be converted to lists of tensors,
    mappings may be converted into dictionaries.

    For example:

    .. code-block:: python

        batch_data = {
            "image": torch.rand((2,1,10,10)),
            DictPostFix.meta("image"): {"scl_slope": torch.Tensor([0.0, 0.0])}
        }
        out = decollate_batch(batch_data)
        print(len(out))
        >>> 2

        print(out[0])
        >>> {'image': tensor([[[4.3549e-01...43e-01]]]), DictPostFix.meta("image"): {'scl_slope': 0.0}}

        batch_data = [torch.rand((2,1,10,10)), torch.rand((2,3,5,5))]
        out = decollate_batch(batch_data)
        print(out[0])
        >>> [tensor([[[4.3549e-01...43e-01]]], tensor([[[5.3435e-01...45e-01]]])]

        batch_data = torch.rand((2,1,10,10))
        out = decollate_batch(batch_data)
        print(out[0])
        >>> tensor([[[4.3549e-01...43e-01]]])

        batch_data = {
            "image": [1, 2, 3], "meta": [4, 5],  # undetermined batch size
        }
        out = decollate_batch(batch_data, pad=True, fill_value=0)
        print(out)
        >>> [{'image': 1, 'meta': 4}, {'image': 2, 'meta': 5}, {'image': 3, 'meta': 0}]
        out = decollate_batch(batch_data, pad=False)
        print(out)
        >>> [{'image': 1, 'meta': 4}, {'image': 2, 'meta': 5}]

    Args:
        batch: data to be de-collated.
        detach: whether to detach the tensors. Scalars tensors will be detached into number types
            instead of torch tensors.
        pad: when the items in a batch indicate different batch size, whether to pad all the sequences to the longest.
            If False, the batch size will be the length of the shortest sequence.
        fill_value: when `pad` is True, the `fillvalue` to use when padding, defaults to `None`.
    """
    if batch is None:
        return batch
    if isinstance(batch, (float, int, str, bytes)) or (
        type(batch).__module__ == "numpy" and not isinstance(batch, Iterable)
    ):
        return batch
    if isinstance(batch, torch.Tensor):
        if detach:
            batch = batch.detach()
        if batch.ndim == 0:
            return batch.item() if detach else batch
        out_list = torch.unbind(batch, dim=0)
        # if of type MetaObj, decollate the metadata
        if isinstance(batch, MetaObj):
            for t, m in zip(out_list, decollate_batch(batch.meta)):
                if isinstance(t, MetaObj):
                    t.meta = m
                    t.is_batch = False
            for t, m in zip(out_list, batch.applied_operations):
                if isinstance(t, MetaObj):
                    t.applied_operations = m
                    t.is_batch = False
        if out_list[0].ndim == 0 and detach:
            return [t.item() for t in out_list]
        return list(out_list)

    b, non_iterable, deco = _non_zipping_check(batch, detach, pad, fill_value)
    if b <= 0:  # all non-iterable, single item "batch"? {"image": 1, "label": 1}
        return deco
    if pad:  # duplicate non-iterable items to the longest batch
        for k in non_iterable:
            deco[k] = [deepcopy(deco[k]) for _ in range(b)]
    if isinstance(deco, Mapping):
        _gen = zip_longest(*deco.values(), fillvalue=fill_value) if pad else zip(*deco.values())
        ret = [dict(zip(deco, item)) for item in _gen]
        if not config.USE_META_DICT:
            return ret
        return pickle_operations(ret, is_encode=False)  # bc 0.9.0
    if isinstance(deco, Iterable):
        _gen = zip_longest(*deco, fillvalue=fill_value) if pad else zip(*deco)
        ret_list = [list(item) for item in _gen]
        if not config.USE_META_DICT:
            return ret_list
        return pickle_operations(ret_list, is_encode=False)  # bc 0.9.0
    raise NotImplementedError(f"Unable to de-collate: {batch}, type: {type(batch)}.")


def pad_list_data_collate(batch: Sequence, method: str = Method.SYMMETRIC, mode: str = NumpyPadMode.CONSTANT, **kwargs):
    """
    Function version of :py:class:`monai.transforms.croppad.batch.PadListDataCollate`.

    Same as MONAI's ``list_data_collate``, except any tensors are centrally padded to match the shape of the biggest
    tensor in each dimension. This transform is useful if some of the applied transforms generate batch data of
    different sizes.

    This can be used on both list and dictionary data.
    Note that in the case of the dictionary data, this decollate function may add the transform information of
    `PadListDataCollate` to the list of invertible transforms if input batch have different spatial shape, so need to
    call static method: `monai.transforms.croppad.batch.PadListDataCollate.inverse` before inverting other transforms.

    Args:
        batch: batch of data to pad-collate
        method: padding method (see :py:class:`monai.transforms.SpatialPad`)
        mode: padding mode (see :py:class:`monai.transforms.SpatialPad`)
        kwargs: other arguments for the `np.pad` or `torch.pad` function.
            note that `np.pad` treats channel dimension as the first dimension.

    """
    from monai.transforms.croppad.batch import PadListDataCollate  # needs to be here to avoid circular import

    return PadListDataCollate(method=method, mode=mode, **kwargs)(batch)


def no_collation(x):
    """
    No any collation operation.
    """
    return x


def worker_init_fn(worker_id: int) -> None:
    """
    Callback function for PyTorch DataLoader `worker_init_fn`.
    It can set different random seed for the transforms in different workers.

    """
    worker_info = torch.utils.data.get_worker_info()
    set_rnd(worker_info.dataset, seed=worker_info.seed)  # type: ignore[union-attr]


def set_rnd(obj, seed: int) -> int:
    """
    Set seed or random state for all randomizable properties of obj.

    Args:
        obj: object to set seed or random state for.
        seed: set the random state with an integer seed.
    """
    if isinstance(obj, (tuple, list)):  # ZipDataset.data is a list
        _seed = seed
        for item in obj:
            _seed = set_rnd(item, seed=seed)
        return seed if _seed == seed else seed + 1  # return a different seed if there are randomizable items
    if not hasattr(obj, "__dict__"):
        return seed  # no attribute
    if hasattr(obj, "set_random_state"):
        obj.set_random_state(seed=seed % MAX_SEED)
        return seed + 1  # a different seed for the next component
    for key in obj.__dict__:
        if key.startswith("__"):  # skip the private methods
            continue
        seed = set_rnd(obj.__dict__[key], seed=seed)
    return seed


def affine_to_spacing(affine: NdarrayTensor, r: int = 3, dtype=float, suppress_zeros: bool = True) -> NdarrayTensor:
    """
    Computing the current spacing from the affine matrix.

    Args:
        affine: a d x d affine matrix.
        r: indexing based on the spatial rank, spacing is computed from `affine[:r, :r]`.
        dtype: data type of the output.
        suppress_zeros: whether to suppress the zeros with ones.

    Returns:
        an `r` dimensional vector of spacing.
    """
    if len(affine.shape) != 2 or affine.shape[0] != affine.shape[1]:
        raise ValueError(f"affine must be a square matrix, got {affine.shape}.")
    _affine, *_ = convert_to_dst_type(affine[:r, :r], dst=affine, dtype=dtype)
    if isinstance(_affine, torch.Tensor):
        spacing = torch.sqrt(torch.sum(_affine * _affine, dim=0))
    else:
        spacing = np.sqrt(np.sum(_affine * _affine, axis=0))
    if suppress_zeros:
        spacing[spacing == 0] = 1.0
    spacing_, *_ = convert_to_dst_type(spacing, dst=affine, dtype=dtype)
    return spacing_


def correct_nifti_header_if_necessary(img_nii):
    """
    Check nifti object header's format, update the header if needed.
    In the updated image pixdim matches the affine.

    Args:
        img_nii: nifti image object
    """
    if img_nii.header.get("dim") is None:
        return img_nii  # not nifti?
    dim = img_nii.header["dim"][0]
    if dim >= 5:
        return img_nii  # do nothing for high-dimensional array
    # check that affine matches zooms
    pixdim = np.asarray(img_nii.header.get_zooms())[:dim]
    norm_affine = affine_to_spacing(img_nii.affine, r=dim)
    if np.allclose(pixdim, norm_affine):
        return img_nii
    if hasattr(img_nii, "get_sform"):
        return rectify_header_sform_qform(img_nii)
    return img_nii


def rectify_header_sform_qform(img_nii):
    """
    Look at the sform and qform of the nifti object and correct it if any
    incompatibilities with pixel dimensions

    Adapted from https://github.com/NifTK/NiftyNet/blob/v0.6.0/niftynet/io/misc_io.py

    Args:
        img_nii: nifti image object
    """
    d = img_nii.header["dim"][0]
    pixdim = np.asarray(img_nii.header.get_zooms())[:d]
    sform, qform = img_nii.get_sform(), img_nii.get_qform()
    norm_sform = affine_to_spacing(sform, r=d)
    norm_qform = affine_to_spacing(qform, r=d)
    sform_mismatch = not np.allclose(norm_sform, pixdim)
    qform_mismatch = not np.allclose(norm_qform, pixdim)

    if img_nii.header["sform_code"] != 0:
        if not sform_mismatch:
            return img_nii
        if not qform_mismatch:
            img_nii.set_sform(img_nii.get_qform())
            return img_nii
    if img_nii.header["qform_code"] != 0:
        if not qform_mismatch:
            return img_nii
        if not sform_mismatch:
            img_nii.set_qform(img_nii.get_sform())
            return img_nii

    norm = affine_to_spacing(img_nii.affine, r=d)

    img_nii.header.set_zooms(norm)
    return img_nii


def zoom_affine(affine: np.ndarray, scale: np.ndarray | Sequence[float], diagonal: bool = True):
    """
    To make column norm of `affine` the same as `scale`.  If diagonal is False,
    returns an affine that combines orthogonal rotation and the new scale.
    This is done by first decomposing `affine`, then setting the zoom factors to
    `scale`, and composing a new affine; the shearing factors are removed.  If
    diagonal is True, returns a diagonal matrix, the scaling factors are set
    to the diagonal elements.  This function always return an affine with zero
    translations.

    Args:
        affine (nxn matrix): a square matrix.
        scale: new scaling factor along each dimension. if the components of the `scale` are non-positive values,
            will use the corresponding components of the original pixdim, which is computed from the `affine`.
        diagonal: whether to return a diagonal scaling matrix.
            Defaults to True.

    Raises:
        ValueError: When ``affine`` is not a square matrix.
        ValueError: When ``scale`` contains a nonpositive scalar.

    Returns:
        the updated `n x n` affine.

    """

    affine = np.array(affine, dtype=float, copy=True)
    if len(affine) != len(affine[0]):
        raise ValueError(f"affine must be n x n, got {len(affine)} x {len(affine[0])}.")
    scale_np = np.array(scale, dtype=float, copy=True)

    d = len(affine) - 1
    # compute original pixdim
    norm = affine_to_spacing(affine, r=d)
    if len(scale_np) < d:  # defaults based on affine
        scale_np = np.append(scale_np, norm[len(scale_np) :])
    scale_np = scale_np[:d]
    scale_np = np.asarray(fall_back_tuple(scale_np, norm))

    scale_np[scale_np == 0] = 1.0
    if diagonal:
        return np.diag(np.append(scale_np, [1.0]))
    rzs = affine[:-1, :-1]  # rotation zoom scale
    zs = np.linalg.cholesky(rzs.T @ rzs).T
    rotation = rzs @ np.linalg.inv(zs)
    s = np.sign(np.diag(zs)) * np.abs(scale_np)
    # construct new affine with rotation and zoom
    new_affine = np.eye(len(affine))
    new_affine[:-1, :-1] = rotation @ np.diag(s)
    return new_affine


def compute_shape_offset(
    spatial_shape: np.ndarray | Sequence[int],
    in_affine: NdarrayOrTensor,
    out_affine: NdarrayOrTensor,
    scale_extent: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Given input and output affine, compute appropriate shapes
    in the output space based on the input array's shape.
    This function also returns the offset to put the shape
    in a good position with respect to the world coordinate system.

    Args:
        spatial_shape: input array's shape
        in_affine (matrix): 2D affine matrix
        out_affine (matrix): 2D affine matrix
        scale_extent: whether the scale is computed based on the spacing or the full extent of voxels, for example, for
            a factor of 0.5 scaling:

            option 1, "o" represents a voxel, scaling the distance between voxels::

                o--o--o
                o-----o

            option 2, each voxel has a physical extent, scaling the full voxel extent::

                | voxel 1 | voxel 2 | voxel 3 | voxel 4 |
                |      voxel 1      |      voxel 2      |

            Option 1 may reduce the number of locations that requiring interpolation. Option 2 is more resolution
            agnostic, that is, resampling coordinates depend on the scaling factor, not on the number of voxels.
            Default is False, using option 1 to compute the shape and offset.

    """
    shape = np.array(spatial_shape, copy=True, dtype=float)
    sr = len(shape)
    in_affine_ = convert_data_type(to_affine_nd(sr, in_affine), np.ndarray)[0]
    out_affine_ = convert_data_type(to_affine_nd(sr, out_affine), np.ndarray)[0]
    in_coords = [(-0.5, dim - 0.5) if scale_extent else (0.0, dim - 1.0) for dim in shape]
    corners: np.ndarray = np.asarray(np.meshgrid(*in_coords, indexing="ij")).reshape((len(shape), -1))
    corners = np.concatenate((corners, np.ones_like(corners[:1])))
    try:
        corners_out = np.linalg.solve(out_affine_, in_affine_) @ corners
    except np.linalg.LinAlgError as e:
        raise ValueError(f"Affine {out_affine_} is not invertible") from e
    corners = in_affine_ @ corners
    all_dist = corners_out[:-1].copy()
    corners_out = corners_out[:-1] / corners_out[-1]
    out_shape = np.round(np.ptp(corners_out, axis=1)) if scale_extent else np.round(np.ptp(corners_out, axis=1) + 1.0)
    offset = None
    for i in range(corners.shape[1]):
        min_corner = np.min(all_dist - all_dist[:, i : i + 1], 1)
        if np.allclose(min_corner, 0.0, rtol=AFFINE_TOL):
            offset = corners[:-1, i]  # corner is the smallest, shift the corner to origin
            break
    if offset is None:  # otherwise make output image center aligned with the input image center
        offset = in_affine_[:-1, :-1] @ (shape / 2.0) + in_affine_[:-1, -1] - out_affine_[:-1, :-1] @ (out_shape / 2.0)
    if scale_extent:
        in_offset = np.append(0.5 * (shape / out_shape - 1.0), 1.0)
        offset = np.abs((in_affine_ @ in_offset / in_offset[-1])[:-1]) * np.sign(offset)
    return out_shape.astype(int, copy=False), offset  # type: ignore


def to_affine_nd(r: np.ndarray | int, affine: NdarrayTensor, dtype=np.float64) -> NdarrayTensor:
    """
    Using elements from affine, to create a new affine matrix by
    assigning the rotation/zoom/scaling matrix and the translation vector.

    When ``r`` is an integer, output is an (r+1)x(r+1) matrix,
    where the top left kxk elements are copied from ``affine``,
    the last column of the output affine is copied from ``affine``'s last column.
    `k` is determined by `min(r, len(affine) - 1)`.

    When ``r`` is an affine matrix, the output has the same shape as ``r``,
    and the top left kxk elements are copied from ``affine``,
    the last column of the output affine is copied from ``affine``'s last column.
    `k` is determined by `min(len(r) - 1, len(affine) - 1)`.

    Args:
        r (int or matrix): number of spatial dimensions or an output affine to be filled.
        affine (matrix): 2D affine matrix
        dtype: data type of the output array.

    Raises:
        ValueError: When ``affine`` dimensions is not 2.
        ValueError: When ``r`` is nonpositive.

    Returns:
        an (r+1) x (r+1) matrix (tensor or ndarray depends on the input ``affine`` data type)

    """
    dtype = get_equivalent_dtype(dtype, np.ndarray)
    affine_np = convert_data_type(affine, output_type=np.ndarray, dtype=dtype, wrap_sequence=True)[0]
    affine_np = affine_np.copy()
    if affine_np.ndim != 2:
        raise ValueError(f"affine must have 2 dimensions, got {affine_np.ndim}.")
    new_affine = np.array(r, dtype=dtype, copy=True)
    if new_affine.ndim == 0:
        sr: int = int(new_affine.astype(np.uint))
        if not np.isfinite(sr) or sr < 0:
            raise ValueError(f"r must be positive, got {sr}.")
        new_affine = np.eye(sr + 1, dtype=dtype)
    d = max(min(len(new_affine) - 1, len(affine_np) - 1), 1)
    new_affine[:d, :d] = affine_np[:d, :d]
    if d > 1:
        new_affine[:d, -1] = affine_np[:d, -1]
    output, *_ = convert_to_dst_type(new_affine, affine, dtype=dtype)
    return output


def reorient_spatial_axes(
    data_shape: Sequence[int], init_affine: NdarrayOrTensor, target_affine: NdarrayOrTensor
) -> tuple[np.ndarray, NdarrayOrTensor]:
    """
    Given the input ``init_affine``, compute the orientation transform between
    it and ``target_affine`` by rearranging/flipping the axes.

    Returns the orientation transform and the updated affine (tensor or ndarray
    depends on the input ``affine`` data type).
    Note that this function requires external module ``nibabel.orientations``.
    """
    init_affine_, *_ = convert_data_type(init_affine, np.ndarray)
    target_affine_, *_ = convert_data_type(target_affine, np.ndarray)
    start_ornt = nib.orientations.io_orientation(init_affine_)
    target_ornt = nib.orientations.io_orientation(target_affine_)
    try:
        ornt_transform = nib.orientations.ornt_transform(start_ornt, target_ornt)
    except ValueError as e:
        raise ValueError(f"The input affine {init_affine} and target affine {target_affine} are not compatible.") from e
    new_affine = init_affine_ @ nib.orientations.inv_ornt_aff(ornt_transform, data_shape)
    new_affine, *_ = convert_to_dst_type(new_affine, init_affine)
    return ornt_transform, new_affine


def create_file_basename(
    postfix: str,
    input_file_name: PathLike,
    folder_path: PathLike,
    data_root_dir: PathLike = "",
    separate_folder: bool = True,
    patch_index=None,
    makedirs: bool = True,
) -> str:
    """
    Utility function to create the path to the output file based on the input
    filename (file name extension is not added by this function).
    When ``data_root_dir`` is not specified, the output file name is:

        `folder_path/input_file_name (no ext.) /input_file_name (no ext.)[_postfix][_patch_index]`

    otherwise the relative path with respect to ``data_root_dir`` will be inserted, for example:

    .. code-block:: python

        from monai.data import create_file_basename
        create_file_basename(
            postfix="seg",
            input_file_name="/foo/bar/test1/image.png",
            folder_path="/output",
            data_root_dir="/foo/bar",
            separate_folder=True,
            makedirs=False)
        # output: /output/test1/image/image_seg

    Args:
        postfix: output name's postfix
        input_file_name: path to the input image file.
        folder_path: path for the output file
        data_root_dir: if not empty, it specifies the beginning parts of the input file's
            absolute path. This is used to compute `input_file_rel_path`, the relative path to the file from
            `data_root_dir` to preserve folder structure when saving in case there are files in different
            folders with the same file names.
        separate_folder: whether to save every file in a separate folder, for example: if input filename is
            `image.nii`, postfix is `seg` and folder_path is `output`, if `True`, save as:
            `output/image/image_seg.nii`, if `False`, save as `output/image_seg.nii`. default to `True`.
        patch_index: if not None, append the patch index to filename.
        makedirs: whether to create the folder if it does not exist.
    """

    # get the filename and directory
    filedir, filename = os.path.split(input_file_name)
    # remove extension
    filename, ext = os.path.splitext(filename)
    if ext == ".gz":
        filename, ext = os.path.splitext(filename)
    # use data_root_dir to find relative path to file
    filedir_rel_path = ""
    if data_root_dir and filedir:
        filedir_rel_path = os.path.relpath(filedir, data_root_dir)

    # output folder path will be original name without the extension
    output = os.path.join(folder_path, filedir_rel_path)

    if separate_folder:
        output = os.path.join(output, filename)

    if makedirs:
        # create target folder if no existing
        os.makedirs(output, exist_ok=True)

    # add the sub-folder plus the postfix name to become the file basename in the output path
    output = os.path.join(output, filename + "_" + postfix if postfix != "" else filename)

    if patch_index is not None:
        output += f"_{patch_index}"

    return os.path.normpath(output)


def compute_importance_map(
    patch_size: tuple[int, ...],
    mode: BlendMode | str = BlendMode.CONSTANT,
    sigma_scale: Sequence[float] | float = 0.125,
    device: torch.device | int | str = "cpu",
    dtype: torch.dtype | str | None = torch.float32,
) -> torch.Tensor:
    """Get importance map for different weight modes.

    Args:
        patch_size: Size of the required importance map. This should be either H, W [,D].
        mode: {``"constant"``, ``"gaussian"``}
            How to blend output of overlapping windows. Defaults to ``"constant"``.

            - ``"constant``": gives equal weight to all predictions.
            - ``"gaussian``": gives less weight to predictions on edges of windows.

        sigma_scale: Sigma_scale to calculate sigma for each dimension
            (sigma = sigma_scale * dim_size). Used for gaussian mode only.
        device: Device to put importance map on.
        dtype: Data type of the output importance map.

    Raises:
        ValueError: When ``mode`` is not one of ["constant", "gaussian"].

    Returns:
        Tensor of size patch_size.

    """
    mode = look_up_option(mode, BlendMode)
    device = torch.device(device)
    if mode == BlendMode.CONSTANT:
        importance_map = torch.ones(patch_size, device=device, dtype=torch.float)
    elif mode == BlendMode.GAUSSIAN:
        sigma_scale = ensure_tuple_rep(sigma_scale, len(patch_size))
        sigmas = [i * sigma_s for i, sigma_s in zip(patch_size, sigma_scale)]

        for i in range(len(patch_size)):
            x = torch.arange(
                start=-(patch_size[i] - 1) / 2.0, end=(patch_size[i] - 1) / 2.0 + 1, dtype=torch.float, device=device
            )
            x = torch.exp(x**2 / (-2 * sigmas[i] ** 2))  # 1D gaussian
            importance_map = importance_map.unsqueeze(-1) * x[(None,) * i] if i > 0 else x
    else:
        raise ValueError(
            f"Unsupported mode: {mode}, available options are [{BlendMode.CONSTANT}, {BlendMode.CONSTANT}]."
        )
    # handle non-positive weights
    min_non_zero = max(torch.min(importance_map).item(), 1e-3)
    importance_map = torch.clamp_(importance_map.to(torch.float), min=min_non_zero).to(dtype)
    return importance_map


def is_supported_format(filename: Sequence[PathLike] | PathLike, suffixes: Sequence[str]) -> bool:
    """
    Verify whether the specified file or files format match supported suffixes.
    If supported suffixes is None, skip the verification and return True.

    Args:
        filename: file name or a list of file names to read.
            if a list of files, verify all the suffixes.
        suffixes: all the supported image suffixes of current reader, must be a list of lower case suffixes.

    """
    filenames: Sequence[PathLike] = ensure_tuple(filename)
    for name in filenames:
        full_suffix = "".join(map(str.lower, PurePath(name).suffixes))
        if all(f".{s.lower()}" not in full_suffix for s in suffixes):
            return False

    return True


def partition_dataset(
    data: Sequence,
    ratios: Sequence[float] | None = None,
    num_partitions: int | None = None,
    shuffle: bool = False,
    seed: int = 0,
    drop_last: bool = False,
    even_divisible: bool = False,
):
    """
    Split the dataset into N partitions. It can support shuffle based on specified random seed.
    Will return a set of datasets, every dataset contains 1 partition of original dataset.
    And it can split the dataset based on specified ratios or evenly split into `num_partitions`.
    Refer to: https://pytorch.org/docs/stable/distributed.html#module-torch.distributed.launch.

    Note:
        It also can be used to partition dataset for ranks in distributed training.
        For example, partition dataset before training and use `CacheDataset`, every rank trains with its own data.
        It can avoid duplicated caching content in each rank, but will not do global shuffle before every epoch:

        .. code-block:: python

            data_partition = partition_dataset(
                data=train_files,
                num_partitions=dist.get_world_size(),
                shuffle=True,
                even_divisible=True,
            )[dist.get_rank()]

            train_ds = SmartCacheDataset(
                data=data_partition,
                transform=train_transforms,
                replace_rate=0.2,
                cache_num=15,
            )

    Args:
        data: input dataset to split, expect a list of data.
        ratios: a list of ratio number to split the dataset, like [8, 1, 1].
        num_partitions: expected number of the partitions to evenly split, only works when `ratios` not specified.
        shuffle: whether to shuffle the original dataset before splitting.
        seed: random seed to shuffle the dataset, only works when `shuffle` is True.
        drop_last: only works when `even_divisible` is False and no ratios specified.
            if True, will drop the tail of the data to make it evenly divisible across partitions.
            if False, will add extra indices to make the data evenly divisible across partitions.
        even_divisible: if True, guarantee every partition has same length.

    Examples::

        >>> data = [1, 2, 3, 4, 5]
        >>> partition_dataset(data, ratios=[0.6, 0.2, 0.2], shuffle=False)
        [[1, 2, 3], [4], [5]]
        >>> partition_dataset(data, num_partitions=2, shuffle=False)
        [[1, 3, 5], [2, 4]]
        >>> partition_dataset(data, num_partitions=2, shuffle=False, even_divisible=True, drop_last=True)
        [[1, 3], [2, 4]]
        >>> partition_dataset(data, num_partitions=2, shuffle=False, even_divisible=True, drop_last=False)
        [[1, 3, 5], [2, 4, 1]]
        >>> partition_dataset(data, num_partitions=2, shuffle=False, even_divisible=False, drop_last=False)
        [[1, 3, 5], [2, 4]]

    """
    data_len = len(data)
    datasets = []

    indices = list(range(data_len))
    if shuffle:
        # deterministically shuffle based on fixed seed for every process
        rs = np.random.RandomState(seed)
        rs.shuffle(indices)

    if ratios:
        next_idx = 0
        rsum = sum(ratios)
        for r in ratios:
            start_idx = next_idx
            next_idx = min(start_idx + int(r / rsum * data_len + 0.5), data_len)
            datasets.append([data[i] for i in indices[start_idx:next_idx]])
        return datasets

    if not num_partitions:
        raise ValueError("must specify number of partitions or ratios.")
    # evenly split the data without ratios
    if not even_divisible and drop_last:
        raise RuntimeError("drop_last only works when even_divisible is True.")
    if data_len < num_partitions:
        raise RuntimeError(f"there is no enough data to be split into {num_partitions} partitions.")

    if drop_last and data_len % num_partitions != 0:
        # split to nearest available length that is evenly divisible
        num_samples = math.ceil((data_len - num_partitions) / num_partitions)
    else:
        num_samples = math.ceil(data_len / num_partitions)
    # use original data length if not even divisible
    total_size = num_samples * num_partitions if even_divisible else data_len

    if not drop_last and total_size - data_len > 0:
        # add extra samples to make it evenly divisible
        indices += indices[: (total_size - data_len)]
    else:
        # remove tail of data to make it evenly divisible
        indices = indices[:total_size]

    for i in range(num_partitions):
        _indices = indices[i:total_size:num_partitions]
        datasets.append([data[j] for j in _indices])

    return datasets


def partition_dataset_classes(
    data: Sequence,
    classes: Sequence[int],
    ratios: Sequence[float] | None = None,
    num_partitions: int | None = None,
    shuffle: bool = False,
    seed: int = 0,
    drop_last: bool = False,
    even_divisible: bool = False,
):
    """
    Split the dataset into N partitions based on the given class labels.
    It can make sure the same ratio of classes in every partition.
    Others are same as :py:class:`monai.data.partition_dataset`.

    Args:
        data: input dataset to split, expect a list of data.
        classes: a list of labels to help split the data, the length must match the length of data.
        ratios: a list of ratio number to split the dataset, like [8, 1, 1].
        num_partitions: expected number of the partitions to evenly split, only works when no `ratios`.
        shuffle: whether to shuffle the original dataset before splitting.
        seed: random seed to shuffle the dataset, only works when `shuffle` is True.
        drop_last: only works when `even_divisible` is False and no ratios specified.
            if True, will drop the tail of the data to make it evenly divisible across partitions.
            if False, will add extra indices to make the data evenly divisible across partitions.
        even_divisible: if True, guarantee every partition has same length.

    Examples::

        >>> data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        >>> classes = [2, 0, 2, 1, 3, 2, 2, 0, 2, 0, 3, 3, 1, 3]
        >>> partition_dataset_classes(data, classes, shuffle=False, ratios=[2, 1])
        [[2, 8, 4, 1, 3, 6, 5, 11, 12], [10, 13, 7, 9, 14]]

    """
    if not issequenceiterable(classes) or len(classes) != len(data):
        raise ValueError(f"length of classes {classes} must match the dataset length {len(data)}.")
    datasets = []
    class_indices = defaultdict(list)
    for i, c in enumerate(classes):
        class_indices[c].append(i)

    class_partition_indices: list[Sequence] = []
    for _, per_class_indices in sorted(class_indices.items()):
        per_class_partition_indices = partition_dataset(
            data=per_class_indices,
            ratios=ratios,
            num_partitions=num_partitions,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
            even_divisible=even_divisible,
        )
        if not class_partition_indices:
            class_partition_indices = per_class_partition_indices
        else:
            for part, data_indices in zip(class_partition_indices, per_class_partition_indices):
                part += data_indices

    rs = np.random.RandomState(seed)
    for indices in class_partition_indices:
        if shuffle:
            rs.shuffle(indices)
        datasets.append([data[j] for j in indices])

    return datasets


def resample_datalist(data: Sequence, factor: float, random_pick: bool = False, seed: int = 0):
    """
    Utility function to resample the loaded datalist for training, for example:
    If factor < 1.0, randomly pick part of the datalist and set to Dataset, useful to quickly test the program.
    If factor > 1.0, repeat the datalist to enhance the Dataset.

    Args:
        data: original datalist to scale.
        factor: scale factor for the datalist, for example, factor=4.5, repeat the datalist 4 times and plus
            50% of the original datalist.
        random_pick: whether to randomly pick data if scale factor has decimal part.
        seed: random seed to randomly pick data.

    """
    scale, repeats = math.modf(factor)
    ret: list = list()

    for _ in range(int(repeats)):
        ret.extend(list(deepcopy(data)))
    if scale > 1e-6:
        ret.extend(partition_dataset(data=data, ratios=[scale, 1 - scale], shuffle=random_pick, seed=seed)[0])

    return ret


def select_cross_validation_folds(partitions: Sequence[Iterable], folds: Sequence[int] | int) -> list:
    """
    Select cross validation data based on data partitions and specified fold index.
    if a list of fold indices is provided, concatenate the partitions of these folds.

    Args:
        partitions: a sequence of datasets, each item is a iterable
        folds: the indices of the partitions to be combined.

    Returns:
        A list of combined datasets.

    Example::

        >>> partitions = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
        >>> select_cross_validation_folds(partitions, 2)
        [5, 6]
        >>> select_cross_validation_folds(partitions, [1, 2])
        [3, 4, 5, 6]
        >>> select_cross_validation_folds(partitions, [-1, 2])
        [9, 10, 5, 6]
    """
    return [data_item for fold_id in ensure_tuple(folds) for data_item in partitions[fold_id]]


def json_hashing(item) -> bytes:
    """

    Args:
        item: data item to be hashed

    Returns: the corresponding hash key

    """
    # TODO: Find way to hash transforms content as part of the cache
    cache_key = ""
    if sys.version_info.minor < 9:
        cache_key = hashlib.md5(json.dumps(item, sort_keys=True).encode("utf-8")).hexdigest()
    else:
        cache_key = hashlib.md5(
            json.dumps(item, sort_keys=True).encode("utf-8"), usedforsecurity=False  # type: ignore
        ).hexdigest()
    return f"{cache_key}".encode()


def pickle_hashing(item, protocol=pickle.HIGHEST_PROTOCOL) -> bytes:
    """

    Args:
        item: data item to be hashed
        protocol: protocol version used for pickling,
            defaults to `pickle.HIGHEST_PROTOCOL`.

    Returns: the corresponding hash key

    """
    cache_key = ""
    if sys.version_info.minor < 9:
        cache_key = hashlib.md5(pickle.dumps(sorted_dict(item), protocol=protocol)).hexdigest()
    else:
        cache_key = hashlib.md5(
            pickle.dumps(sorted_dict(item), protocol=protocol), usedforsecurity=False  # type: ignore
        ).hexdigest()
    return f"{cache_key}".encode()


def sorted_dict(item, key=None, reverse=False):
    """Return a new sorted dictionary from the `item`."""
    if not isinstance(item, dict):
        return item
    return {k: sorted_dict(v) if isinstance(v, dict) else v for k, v in sorted(item.items(), key=key, reverse=reverse)}


def convert_tables_to_dicts(
    dfs,
    row_indices: Sequence[int | str] | None = None,
    col_names: Sequence[str] | None = None,
    col_types: dict[str, dict[str, Any] | None] | None = None,
    col_groups: dict[str, Sequence[str]] | None = None,
    **kwargs,
) -> list[dict[str, Any]]:
    """
    Utility to join pandas tables, select rows, columns and generate groups.
    Will return a list of dictionaries, every dictionary maps to a row of data in tables.

    Args:
        dfs: data table in pandas Dataframe format. if providing a list of tables, will join them.
        row_indices: indices of the expected rows to load. it should be a list,
            every item can be a int number or a range `[start, end)` for the indices.
            for example: `row_indices=[[0, 100], 200, 201, 202, 300]`. if None,
            load all the rows in the file.
        col_names: names of the expected columns to load. if None, load all the columns.
        col_types: `type` and `default value` to convert the loaded columns, if None, use original data.
            it should be a dictionary, every item maps to an expected column, the `key` is the column
            name and the `value` is None or a dictionary to define the default value and data type.
            the supported keys in dictionary are: ["type", "default"], and note that the value of `default`
            should not be `None`. for example::

                col_types = {
                    "subject_id": {"type": str},
                    "label": {"type": int, "default": 0},
                    "ehr_0": {"type": float, "default": 0.0},
                    "ehr_1": {"type": float, "default": 0.0},
                }

        col_groups: args to group the loaded columns to generate a new column,
            it should be a dictionary, every item maps to a group, the `key` will
            be the new column name, the `value` is the names of columns to combine. for example:
            `col_groups={"ehr": [f"ehr_{i}" for i in range(10)], "meta": ["meta_1", "meta_2"]}`
        kwargs: additional arguments for `pandas.merge()` API to join tables.

    """
    df = reduce(lambda l, r: pd.merge(l, r, **kwargs), ensure_tuple(dfs))
    # parse row indices
    rows: list[int | str] = []
    if row_indices is None:
        rows = slice(df.shape[0])  # type: ignore
    else:
        for i in row_indices:
            if isinstance(i, (tuple, list)):
                if len(i) != 2:
                    raise ValueError("range of row indices must contain 2 values: start and end.")
                rows.extend(list(range(i[0], i[1])))
            else:
                rows.append(i)

    # convert to a list of dictionaries corresponding to every row
    data_ = df.loc[rows] if col_names is None else df.loc[rows, col_names]
    if isinstance(col_types, dict):
        # fill default values for NaN
        defaults = {k: v["default"] for k, v in col_types.items() if v is not None and v.get("default") is not None}
        if defaults:
            data_ = data_.fillna(value=defaults)
        # convert data types
        types = {k: v["type"] for k, v in col_types.items() if v is not None and "type" in v}
        if types:
            data_ = data_.astype(dtype=types, copy=False)
    data: list[dict] = data_.to_dict(orient="records")

    # group columns to generate new column
    if col_groups is not None:
        groups: dict[str, list] = {}
        for name, cols in col_groups.items():
            groups[name] = df.loc[rows, cols].values
        # invert items of groups to every row of data
        data = [dict(d, **{k: v[i] for k, v in groups.items()}) for i, d in enumerate(data)]

    return data


def orientation_ras_lps(affine: NdarrayTensor) -> NdarrayTensor:
    """
    Convert the ``affine`` between the `RAS` and `LPS` orientation
    by flipping the first two spatial dimensions.

    Args:
        affine: a 2D affine matrix.
    """
    sr = max(affine.shape[0] - 1, 1)  # spatial rank is at least 1
    flip_d = [[-1, 1], [-1, -1, 1], [-1, -1, 1, 1]]
    flip_diag = flip_d[min(sr - 1, 2)] + [1] * (sr - 3)
    if isinstance(affine, torch.Tensor):
        return torch.diag(torch.as_tensor(flip_diag).to(affine)) @ affine  # type: ignore
    return np.diag(flip_diag).astype(affine.dtype) @ affine  # type: ignore


def remove_keys(data: dict, keys: list[str]) -> None:
    """
    Remove keys from a dictionary. Operates in-place so nothing is returned.

    Args:
        data: dictionary to be modified.
        keys: keys to be deleted from dictionary.

    Returns:
        `None`
    """
    for k in keys:
        _ = data.pop(k, None)


def remove_extra_metadata(meta: dict) -> None:
    """
    Remove extra metadata from the dictionary. Operates in-place so nothing is returned.

    Args:
        meta: dictionary containing metadata to be modified.

    Returns:
        `None`
    """
    keys = get_extra_metadata_keys()
    remove_keys(data=meta, keys=keys)


def get_extra_metadata_keys() -> list[str]:
    """
    Get a list of unnecessary keys for metadata that can be removed.

    Returns:
        List of keys to be removed.
    """
    keys = [
        "srow_x",
        "srow_y",
        "srow_z",
        "quatern_b",
        "quatern_c",
        "quatern_d",
        "qoffset_x",
        "qoffset_y",
        "qoffset_z",
        "dim",
        "pixdim",
        *[f"dim[{i}]" for i in range(8)],
        *[f"pixdim[{i}]" for i in range(8)],
    ]

    # TODO: it would be good to remove these, but they are currently being used in the
    # codebase.
    # keys += [
    #     "original_affine",
    #     "spatial_shape",
    #     "spacing",
    # ]

    return keys


def is_no_channel(val) -> bool:
    """Returns whether `val` indicates "no_channel", for MetaKeys.ORIGINAL_CHANNEL_DIM."""
    if isinstance(val, torch.Tensor):
        return bool(torch.isnan(val))
    if isinstance(val, str):
        return val == "no_channel"
    if np.isscalar(val):
        return bool(np.isnan(val))
    return val is None
