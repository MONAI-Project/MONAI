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

import functools
import warnings
from collections.abc import Sequence
from copy import deepcopy
from typing import Any

import numpy as np
import torch

import monai
from monai.config.type_definitions import NdarrayTensor
from monai.data.meta_obj import MetaObj, get_track_meta
from monai.data.utils import affine_to_spacing, decollate_batch, list_data_collate, remove_extra_metadata
from monai.utils import look_up_option
from monai.utils.enums import LazyAttr, MetaKeys, PostFix, SpaceKeys
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type, convert_to_numpy, convert_to_tensor

__all__ = ["MetaTensor"]


@functools.lru_cache(None)
def _get_named_tuple_like_type(func):
    if (
        hasattr(torch, "return_types")
        and hasattr(func, "__name__")
        and hasattr(torch.return_types, func.__name__)
        and isinstance(getattr(torch.return_types, func.__name__), type)
    ):
        return getattr(torch.return_types, func.__name__)
    return None


def _not_requiring_metadata(ret):
    return isinstance(ret, (int, str, bytes, torch.Size, torch.dtype, torch.device, np.ndarray)) or not (
        isinstance(ret, MetaTensor) or (isinstance(ret, Sequence) and any(isinstance(x, MetaTensor) for x in ret))
    )


class MetaTensor(MetaObj, torch.Tensor):
    """
    Class that inherits from both `torch.Tensor` and `MetaObj`, adding support for metadata.

    Metadata is stored in the form of a dictionary. Nested, an affine matrix will be
    stored. This should be in the form of `torch.Tensor`.

    Behavior should be the same as `torch.Tensor` aside from the extended
    meta functionality.

    Copying of information:

        * For `c = a + b`, then auxiliary data (e.g., metadata) will be copied from the
          first instance of `MetaTensor` if `a.is_batch` is False
          (For batched data, the metadata will be shallow copied for efficiency purposes).

    Example:
        .. code-block:: python

            import torch
            from monai.data import MetaTensor

            t = torch.tensor([1,2,3])
            affine = torch.as_tensor([[2,0,0,0],
                                      [0,2,0,0],
                                      [0,0,2,0],
                                      [0,0,0,1]], dtype=torch.float64)
            meta = {"some": "info"}
            m = MetaTensor(t, affine=affine, meta=meta)
            m2 = m + m
            assert isinstance(m2, MetaTensor)
            assert m2.meta["some"] == "info"
            assert torch.all(m2.affine == affine)

    Notes:
        - Requires pytorch 1.9 or newer for full compatibility.
        - Older versions of pytorch (<=1.8), `torch.jit.trace(net, im)` may
          not work if `im` is of type `MetaTensor`. This can be resolved with
          `torch.jit.trace(net, im.as_tensor())`.
        - For pytorch < 1.8, sharing `MetaTensor` instances across processes may not be supported.
        - For pytorch < 1.9, next(iter(meta_tensor)) returns a torch.Tensor.
          see: https://github.com/pytorch/pytorch/issues/54457
        - A warning will be raised if in the constructor `affine` is not `None` and
          `meta` already contains the key `affine`.
        - You can query whether the `MetaTensor` is a batch with the `is_batch` attribute.
        - With a batch of data, `batch[0]` will return the 0th image
          with the 0th metadata. When the batch dimension is non-singleton, e.g.,
          `batch[:, 0]`, `batch[..., -1]` and `batch[1:3]`, then all (or a subset in the
          last example) of the metadata will be returned, and `is_batch` will return `True`.
        - When creating a batch with this class, use `monai.data.DataLoader` as opposed
          to `torch.utils.data.DataLoader`, as this will take care of collating the
          metadata properly.
    """

    @staticmethod
    def __new__(
        cls,
        x,
        affine: torch.Tensor | None = None,
        meta: dict | None = None,
        applied_operations: list | None = None,
        *args,
        **kwargs,
    ) -> MetaTensor:
        _kwargs = {"device": kwargs.pop("device", None), "dtype": kwargs.pop("dtype", None)} if kwargs else {}
        return torch.as_tensor(x, *args, **_kwargs).as_subclass(cls)

    def __init__(
        self,
        x,
        affine: torch.Tensor | None = None,
        meta: dict | None = None,
        applied_operations: list | None = None,
        *_args,
        **_kwargs,
    ) -> None:
        """
        Args:
            x: initial array for the MetaTensor. Can be a list, tuple, NumPy ndarray, scalar, and other types.
            affine: optional 4x4 array.
            meta: dictionary of metadata.
            applied_operations: list of previously applied operations on the MetaTensor,
                the list is typically maintained by `monai.transforms.TraceableTransform`.
                See also: :py:class:`monai.transforms.TraceableTransform`
            _args: additional args (currently not in use in this constructor).
            _kwargs: additional kwargs (currently not in use in this constructor).

        Note:
            If a `meta` dictionary is given, use it. Else, if `meta` exists in the input tensor `x`, use it.
            Else, use the default value. Similar for the affine, except this could come from
            four places, priority: `affine`, `meta["affine"]`, `x.affine`, `get_default_affine`.

        """
        super().__init__()
        # set meta
        if meta is not None:
            self.meta = meta
        elif isinstance(x, MetaObj):
            self.__dict__ = deepcopy(x.__dict__)
        # set the affine
        if affine is not None:
            if MetaKeys.AFFINE in self.meta:
                warnings.warn("Setting affine, but the applied meta contains an affine. This will be overwritten.")
            self.affine = affine
        elif MetaKeys.AFFINE in self.meta:
            # by using the setter function, we ensure it is converted to torch.Tensor if not already
            self.affine = self.meta[MetaKeys.AFFINE]
        else:
            self.affine = self.get_default_affine()
        # applied_operations
        if applied_operations is not None:
            self.applied_operations = applied_operations
        else:
            self.applied_operations = MetaObj.get_default_applied_operations()

        # if we are creating a new MetaTensor, then deep copy attributes
        if isinstance(x, torch.Tensor) and not isinstance(x, MetaTensor):
            self.copy_meta_from(self)

        if MetaKeys.SPACE not in self.meta:
            self.meta[MetaKeys.SPACE] = SpaceKeys.RAS  # defaulting to the right-anterior-superior space

    @staticmethod
    def update_meta(rets: Sequence, func, args, kwargs) -> Sequence:
        """
        Update the metadata from the output of `MetaTensor.__torch_function__`.

        The output of `torch.Tensor.__torch_function__` could be a single object or a
        sequence of them. Hence, in `MetaTensor.__torch_function__` we convert them to a
        list of not already, and then we loop across each element, processing metadata
        as necessary. For each element, if not of type `MetaTensor`, then nothing to do.

        Args:
            rets: the output from `torch.Tensor.__torch_function__`, which has been
                converted to a list in `MetaTensor.__torch_function__` if it wasn't
                already a `Sequence`.
            func: the torch function that was applied. Examples might be `torch.squeeze`
                or `torch.Tensor.__add__`. We need this since the metadata need to be
                treated differently if a batch of data is considered. For example,
                slicing (`torch.Tensor.__getitem__`) the ith element of the 0th
                dimension of a batch of data should return a ith tensor with the ith
                metadata.
            args: positional arguments that were passed to `func`.
            kwargs: keyword arguments that were passed to `func`.

        Returns:
            A sequence with the same number of elements as `rets`. For each element, if
            the input type was not `MetaTensor`, then no modifications will have been
            made. If global parameters have been set to false (e.g.,
            `not get_track_meta()`), then any `MetaTensor` will be converted to
            `torch.Tensor`. Else, metadata will be propagated as necessary (see
            :py:func:`MetaTensor._copy_meta`).
        """
        out = []
        metas = None  # optional output metadicts for each of the return value in `rets`
        is_batch = any(x.is_batch for x in MetaObj.flatten_meta_objs(args, kwargs.values()) if hasattr(x, "is_batch"))
        for idx, ret in enumerate(rets):
            # if not `MetaTensor`, nothing to do.
            if not isinstance(ret, MetaTensor):
                pass
            # if not tracking, convert to `torch.Tensor`.
            elif not get_track_meta():
                ret = ret.as_tensor()
            # else, handle the `MetaTensor` metadata.
            else:
                meta_args = MetaObj.flatten_meta_objs(args, kwargs.values())
                ret.is_batch = is_batch
                ret.copy_meta_from(meta_args, copy_attr=not is_batch)
                # the following is not implemented but the network arch may run into this case:
                # if func == torch.cat and any(m.is_batch if hasattr(m, "is_batch") else False for m in meta_args):
                #     raise NotImplementedError("torch.cat is not implemented for batch of MetaTensors.")
                if is_batch:
                    ret = MetaTensor._handle_batched(ret, idx, metas, func, args, kwargs)
            out.append(ret)
        # if the input was a tuple, then return it as a tuple
        return tuple(out) if isinstance(rets, tuple) else out

    @classmethod
    def _handle_batched(cls, ret, idx, metas, func, args, kwargs):
        """utility function to handle batched MetaTensors."""
        # If we have a batch of data, then we need to be careful if a slice of
        # the data is returned. Depending on how the data are indexed, we return
        # some or all of the metadata, and the return object may or may not be a
        # batch of data (e.g., `batch[:,-1]` versus `batch[0]`).
        # if indexing e.g., `batch[0]`
        if func == torch.Tensor.__getitem__:
            if idx > 0 or len(args) < 2 or len(args[0]) < 1:
                return ret
            batch_idx = args[1][0] if isinstance(args[1], Sequence) else args[1]
            # if using e.g., `batch[:, -1]` or `batch[..., -1]`, then the
            # first element will be `slice(None, None, None)` and `Ellipsis`,
            # respectively. Don't need to do anything with the metadata.
            if batch_idx in (slice(None, None, None), Ellipsis, None) or isinstance(batch_idx, torch.Tensor):
                return ret
            dec_batch = decollate_batch(args[0], detach=False)
            ret_meta = dec_batch[batch_idx]
            if isinstance(ret_meta, list) and ret_meta:  # e.g. batch[0:2], re-collate
                try:
                    ret_meta = list_data_collate(ret_meta)
                except (TypeError, ValueError, RuntimeError, IndexError) as e:
                    raise ValueError(
                        "Inconsistent batched metadata dicts when slicing a batch of MetaTensors, "
                        "please consider converting it into a torch Tensor using `x.as_tensor()` or "
                        "a numpy array using `x.array`."
                    ) from e
            elif isinstance(ret_meta, MetaObj):  # e.g. `batch[0]` or `batch[0, 1]`, batch_idx is int
                ret_meta.is_batch = False
            if hasattr(ret_meta, "__dict__"):
                ret.__dict__ = ret_meta.__dict__.copy()
        # `unbind` is used for `next(iter(batch))`. Also for `decollate_batch`.
        # But we only want to split the batch if the `unbind` is along the 0th dimension.
        elif func == torch.Tensor.unbind:
            if len(args) > 1:
                dim = args[1]
            elif "dim" in kwargs:
                dim = kwargs["dim"]
            else:
                dim = 0
            if dim == 0:
                if metas is None:
                    metas = decollate_batch(args[0], detach=False)
                if hasattr(metas[idx], "__dict__"):
                    ret.__dict__ = metas[idx].__dict__.copy()
                ret.is_batch = False
        return ret

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None) -> Any:
        """Wraps all torch functions."""
        if kwargs is None:
            kwargs = {}
        ret = super().__torch_function__(func, types, args, kwargs)
        # if `out` has been used as argument, metadata is not copied, nothing to do.
        # if "out" in kwargs:
        #     return ret
        if _not_requiring_metadata(ret):
            return ret
        if _get_named_tuple_like_type(func) is not None and isinstance(ret, _get_named_tuple_like_type(func)):
            # for torch.max(torch.tensor(1.0), dim=0), the return type is named-tuple like
            out_items = MetaTensor.update_meta(ret, func, args, kwargs)
            for idx in range(ret.n_fields):
                ret[idx].meta = out_items[idx].meta
                ret[idx].applied_operations = out_items[idx].applied_operations
            return ret
        # we might have 1 or multiple outputs. Might be MetaTensor, might be something
        # else (e.g., `__repr__` returns a string).
        # Convert to list (if necessary), process, and at end remove list if one was added.
        if not isinstance(ret, Sequence):
            ret = [ret]
            unpack = True
        else:
            unpack = False
        ret = MetaTensor.update_meta(ret, func, args, kwargs)
        return ret[0] if unpack else ret

    @staticmethod
    def _convert(x):
        if isinstance(x, (MetaTensor, torch.Tensor, tuple, list)):
            return convert_data_type(x, output_type=np.ndarray, wrap_sequence=False)[0]
        return x

    def __array_function__(self, func, types, args, kwargs):
        """for numpy Interoperability, so that we can compute ``np.sum(MetaTensor([1.0]))``."""
        try:
            if not func.__module__.startswith("numpy"):
                return NotImplemented
        except AttributeError:
            return NotImplemented
        _args = list(map(MetaTensor._convert, args))
        _kwargs = {k: MetaTensor._convert(v) for k, v in kwargs.items()}
        return func(*_args, **_kwargs)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        For numpy interoperability, so that we can compute ``MetaTensor([1.0]) >= np.asarray([1.0])``.
        This is for pytorch > 1.8.
        """
        try:
            if not type(ufunc).__module__.startswith("numpy"):
                return NotImplemented
        except AttributeError:
            return NotImplemented
        if method != "__call__":
            return NotImplemented
        _inputs = map(MetaTensor._convert, inputs)
        _kwargs = {k: MetaTensor._convert(v) for k, v in kwargs.items()}
        if "out" in _kwargs:
            return NotImplemented  # not supported
        try:
            return getattr(ufunc, method)(*_inputs, **_kwargs)
        except AttributeError:
            return NotImplemented

    @staticmethod
    def get_default_affine(dtype=torch.float64) -> torch.Tensor:
        return torch.eye(4, device=torch.device("cpu"), dtype=dtype)

    def as_tensor(self) -> torch.Tensor:
        """
        Return the `MetaTensor` as a `torch.Tensor`.
        It is OS dependent as to whether this will be a deep copy or not.
        """
        return self.as_subclass(torch.Tensor)

    def get_array(self, output_type=np.ndarray, dtype=None, device=None, *_args, **_kwargs):
        """
        Returns a new array in `output_type`, the array shares the same underlying storage when the output is a
        numpy array. Changes to self tensor will be reflected in the ndarray and vice versa.

        Args:
            output_type: output type, see also: :py:func:`monai.utils.convert_data_type`.
            dtype: dtype of output data. Converted to correct library type (e.g.,
                `np.float32` is converted to `torch.float32` if output type is `torch.Tensor`).
                If left blank, it remains unchanged.
            device: if the output is a `torch.Tensor`, select device (if `None`, unchanged).
            _args: currently unused parameters.
            _kwargs: currently unused parameters.
        """
        return convert_data_type(self, output_type=output_type, dtype=dtype, device=device, wrap_sequence=True)[0]

    def set_array(self, src, non_blocking: bool = False, *_args, **_kwargs):
        """
        Copies the elements from src into self tensor and returns self.
        The src tensor must be broadcastable with the self tensor.
        It may be of a different data type or reside on a different device.

        See also: `https://pytorch.org/docs/stable/generated/torch.Tensor.copy_.html`

        Args:
            src: the source tensor to copy from.
            non_blocking: if True and this copy is between CPU and GPU, the copy may occur
                asynchronously with respect to the host. For other cases, this argument has no effect.
            _args: currently unused parameters.
            _kwargs:  currently unused parameters.
        """
        converted: torch.Tensor = convert_to_tensor(src, track_meta=False, wrap_sequence=True)
        try:
            return self.copy_(converted, non_blocking=non_blocking)
        except RuntimeError:  # skip the shape checking
            self.data = converted
            return self

    @property
    def array(self):
        """
        Returns a numpy array of ``self``. The array and ``self`` shares the same underlying storage if self is on cpu.
        Changes to ``self`` (it's a subclass of torch.Tensor) will be reflected in the ndarray and vice versa.
        If ``self`` is not on cpu, the call will move the array to cpu and then the storage is not shared.

        :getter: see also: :py:func:`MetaTensor.get_array()`
        :setter: see also: :py:func:`MetaTensor.set_array()`
        """
        return self.get_array()

    @array.setter
    def array(self, src) -> None:
        """A default setter using ``self.set_array()``"""
        self.set_array(src)

    def as_dict(self, key: str, output_type=torch.Tensor, dtype=None) -> dict:
        """
        Get the object as a dictionary for backwards compatibility.
        This method does not make a deep copy of the objects.

        Args:
            key: Base key to store main data. The key for the metadata will be determined using `PostFix`.
            output_type: `torch.Tensor` or `np.ndarray` for the main data.
            dtype: dtype of output data. Converted to correct library type (e.g.,
                `np.float32` is converted to `torch.float32` if output type is `torch.Tensor`).
                If left blank, it remains unchanged.

        Return:
            A dictionary consisting of three keys, the main data (stored under `key`) and the metadata.
        """
        if output_type not in (torch.Tensor, np.ndarray):
            raise ValueError(f"output_type must be torch.Tensor or np.ndarray, got {output_type}.")
        return {
            key: self.get_array(output_type=output_type, dtype=dtype),
            PostFix.meta(key): self.meta,
            PostFix.transforms(key): self.applied_operations,
        }

    def astype(self, dtype, device=None, *_args, **_kwargs):
        """
        Cast to ``dtype``, sharing data whenever possible.

        Args:
            dtype: dtypes such as np.float32, torch.float, "np.float32", float.
            device: the device if `dtype` is a torch data type.
            _args: additional args (currently unused).
            _kwargs: additional kwargs (currently unused).

        Returns:
            data array instance
        """
        if isinstance(dtype, str):
            mod_str, *dtype = dtype.split(".", 1)
            dtype = mod_str if not dtype else dtype[0]
        else:
            mod_str = getattr(dtype, "__module__", "torch")
        mod_str = look_up_option(mod_str, {"torch", "numpy", "np"}, default="numpy")

        out_type: type[torch.Tensor] | type[np.ndarray] | None
        if mod_str == "torch":
            out_type = torch.Tensor
        elif mod_str in ("numpy", "np"):
            out_type = np.ndarray
        else:
            out_type = None
        return self.get_array(output_type=out_type, dtype=dtype, device=device)

    @property
    def affine(self) -> torch.Tensor:
        """Get the affine. Defaults to ``torch.eye(4, dtype=torch.float64)``"""
        return self.meta.get(MetaKeys.AFFINE, self.get_default_affine())  # type: ignore

    @affine.setter
    def affine(self, d: NdarrayTensor) -> None:
        """Set the affine."""
        self.meta[MetaKeys.AFFINE] = torch.as_tensor(d, device=torch.device("cpu"), dtype=torch.float64)

    @property
    def pixdim(self):
        """Get the spacing"""
        if self.is_batch:
            return [affine_to_spacing(a) for a in self.affine]
        return affine_to_spacing(self.affine)

    def set_pixdim(self) -> None:
        """Update pixdim based on current affine."""
        self.meta[MetaKeys.PIXDIM][1 : 1 + len(self.pixdim)] = affine_to_spacing(self.affine)

    def peek_pending_shape(self):
        """
        Get the currently expected spatial shape as if all the pending operations are executed.
        For tensors that have more than 3 spatial dimensions, only the shapes of the top 3 dimensions will be returned.
        """
        res = None
        if self.pending_operations:
            res = self.pending_operations[-1].get(LazyAttr.SHAPE, None)
        # default to spatial shape (assuming channel-first input)
        return tuple(convert_to_numpy(self.shape, wrap_sequence=True).tolist()[1:]) if res is None else res

    def peek_pending_affine(self):
        res = self.affine
        r = len(res) - 1
        if r not in (2, 3):
            warnings.warn(f"Only 2d and 3d affine are supported, got {r}d input.")
        for p in self.pending_operations:
            next_matrix = convert_to_tensor(p.get(LazyAttr.AFFINE), dtype=torch.float64)
            if next_matrix is None:
                continue
            res = convert_to_dst_type(res, next_matrix)[0]
            next_matrix = monai.data.utils.to_affine_nd(r, next_matrix)
            res = monai.transforms.lazy.utils.combine_transforms(res, next_matrix)
        return res

    def peek_pending_rank(self):
        a = self.pending_operations[-1].get(LazyAttr.AFFINE, None) if self.pending_operations else self.affine
        return 1 if a is None else int(max(1, len(a) - 1))

    def new_empty(self, size, dtype=None, device=None, requires_grad=False):  # type: ignore[override]
        """
        must be defined for deepcopy to work

        See:
            - https://pytorch.org/docs/stable/generated/torch.Tensor.new_empty.html#torch-tensor-new-empty
        """
        return type(self)(
            self.as_tensor().new_empty(size=size, dtype=dtype, device=device, requires_grad=requires_grad)
        )

    def clone(self, **kwargs):
        """
        Returns a copy of the MetaTensor instance.

        Args:
            kwargs: additional keyword arguments to `torch.clone`.

        See also: https://pytorch.org/docs/stable/generated/torch.clone.html
        """
        new_inst = MetaTensor(self.as_tensor().clone(**kwargs))
        new_inst.__dict__ = deepcopy(self.__dict__)
        return new_inst

    @staticmethod
    def ensure_torch_and_prune_meta(
        im: NdarrayTensor, meta: dict | None, simple_keys: bool = False, pattern: str | None = None, sep: str = "."
    ):
        """
        Convert the image to MetaTensor (when meta is not None). If `affine` is in the `meta` dictionary,
        convert that to `torch.Tensor`, too. Remove any superfluous metadata.

        Args:
            im: Input image (`np.ndarray` or `torch.Tensor`)
            meta: Metadata dictionary. When it's None, the metadata is not tracked, this method returns a torch.Tensor.
            simple_keys: whether to keep only a simple subset of metadata keys.
            pattern: combined with `sep`, a regular expression used to match and prune keys
                in the metadata (nested dictionary), default to None, no key deletion.
            sep: combined with `pattern`, used to match and delete keys in the metadata (nested dictionary).
                default is ".", see also :py:class:`monai.transforms.DeleteItemsd`.
                e.g. ``pattern=".*_code$", sep=" "`` removes any meta keys that ends with ``"_code"``.

        Returns:
            By default, a `MetaTensor` is returned.
            However, if `get_track_meta()` is `False` or meta=None, a `torch.Tensor` is returned.
        """
        img = convert_to_tensor(im, track_meta=get_track_meta() and meta is not None)  # potentially ascontiguousarray
        # if not tracking metadata, return `torch.Tensor`
        if not isinstance(img, MetaTensor):
            return img

        if meta is None:
            meta = {}

        # remove any superfluous metadata.
        if simple_keys:
            # ensure affine is of type `torch.Tensor`
            if MetaKeys.AFFINE in meta:
                meta[MetaKeys.AFFINE] = convert_to_tensor(meta[MetaKeys.AFFINE])  # bc-breaking
            remove_extra_metadata(meta)  # bc-breaking

        if pattern is not None:
            meta = monai.transforms.DeleteItemsd(keys=pattern, sep=sep, use_re=True)(meta)

        # return the `MetaTensor`
        if meta is None:
            meta = {}
        img.meta = meta
        if MetaKeys.AFFINE in meta:
            img.affine = meta[MetaKeys.AFFINE]  # this uses the affine property setter
        else:
            img.affine = MetaTensor.get_default_affine()
        return img

    def __repr__(self):  # type: ignore[override]
        """
        Prints a representation of the tensor.
        Prepends "meta" to ``torch.Tensor.__repr__``.
        Use ``print_verbose`` for associated metadata.
        """
        return f"meta{self.as_tensor().__repr__()}"

    def __str__(self):
        """
        Prints a representation of the tensor.
        Prepends "meta" to ``torch.Tensor.__str__``.
        Use ``print_verbose`` for associated metadata.
        """
        return f"meta{str(self.as_tensor())}"

    def __format__(self, format_spec):
        """
        returns the output of pytorch tensor's ``__format__`` method.
        """
        return self.as_tensor().__format__(format_spec)

    def print_verbose(self) -> None:
        """Verbose print with meta data."""
        print(self)
        if self.meta is not None:
            print(self.meta.__repr__())
