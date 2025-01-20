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
A collection of generic interfaces for MONAI transforms.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Hashable, Iterable, Mapping
from typing import Any, TypeVar

import numpy as np
import torch

from monai import config, transforms
from monai.config import KeysCollection
from monai.data.meta_tensor import MetaTensor
from monai.transforms.traits import LazyTrait, RandomizableTrait, ThreadUnsafe
from monai.utils import MAX_SEED, ensure_tuple, first
from monai.utils.enums import TransformBackends
from monai.utils.misc import MONAIEnvVars

__all__ = [
    "ThreadUnsafe",
    "apply_transform",
    "Randomizable",
    "LazyTransform",
    "RandomizableTransform",
    "Transform",
    "MapTransform",
]

ReturnType = TypeVar("ReturnType")


def _apply_transform(
    transform: Callable[..., ReturnType],
    data: Any,
    unpack_parameters: bool = False,
    lazy: bool | None = False,
    overrides: dict | None = None,
    logger_name: bool | str = False,
) -> ReturnType:
    """
    Perform a transform 'transform' on 'data', according to the other parameters specified.

    If `data` is a tuple and `unpack_parameters` is True, each parameter of `data` is unpacked
    as arguments to `transform`. Otherwise `data` is considered as single argument to `transform`.

    If 'lazy' is True, this method first checks whether it can execute this method lazily. If it
    can't, it will ensure that all pending lazy transforms on 'data' are applied before applying
    this 'transform' to it. If 'lazy' is True, and 'overrides' are provided, those overrides will
    be applied to the pending operations on 'data'. See ``Compose`` for more details on lazy
    resampling, which is an experimental feature for 1.2.

    Please note, this class is function is designed to be called by ``apply_transform``.
    In general, you should not need to make specific use of it unless you are implementing
    pipeline execution mechanisms.

    Args:
        transform: a callable to be used to transform `data`.
        data: the tensorlike or dictionary of tensorlikes to be executed on
        unpack_parameters: whether to unpack parameters for `transform`. Defaults to False.
        lazy: whether to enable lazy evaluation for lazy transforms. If False, transforms will be
            carried out on a transform by transform basis. If True, all lazy transforms will
            be executed by accumulating changes and resampling as few times as possible.
            See the :ref:`Lazy Resampling topic<lazy_resampling> for more information about lazy resampling.
        overrides: this optional parameter allows you to specify a dictionary of parameters that should be overridden
            when executing a pipeline. These each parameter that is compatible with a given transform is then applied
            to that transform before it is executed. Note that overrides are currently only applied when
            :ref:`Lazy Resampling<lazy_resampling>` is enabled for the pipeline or a given transform. If lazy is False
            they are ignored. Currently supported args are:
            {``"mode"``, ``"padding_mode"``, ``"dtype"``, ``"align_corners"``, ``"resample_mode"``, ``device``}.
        logger_name: this optional parameter allows you to specify a logger by name for logging of pipeline execution.
            Setting this to False disables logging. Setting it to True enables logging to the default loggers.
            Setting a string overrides the logger name to which logging is performed.

    Returns:
        ReturnType: The return type of `transform`.
    """
    from monai.transforms.lazy.functional import apply_pending_transforms_in_order

    data = apply_pending_transforms_in_order(transform, data, lazy, overrides, logger_name)

    if isinstance(data, tuple) and unpack_parameters:
        return transform(*data, lazy=lazy) if isinstance(transform, LazyTrait) else transform(*data)

    return transform(data, lazy=lazy) if isinstance(transform, LazyTrait) else transform(data)


def apply_transform(
    transform: Callable[..., ReturnType],
    data: Any,
    map_items: bool | int = True,
    unpack_items: bool = False,
    log_stats: bool | str = False,
    lazy: bool | None = None,
    overrides: dict | None = None,
) -> list[Any] | ReturnType:
    """
    Transform `data` with `transform`.

    If `data` is a list or tuple and `map_data` is True, each item of `data` will be transformed
    and this method returns a list of outcomes.
    otherwise transform will be applied once with `data` as the argument.

    Args:
        transform: a callable to be used to transform `data`.
        data: an object to be transformed.
        map_items: whether to apply transform to each item in `data`,
            if `data` is a list or tuple. Defaults to True.
            it can also be an int, if so, recursively map the items in `data` `map_items` times.
        unpack_items: whether to unpack parameters using `*`. Defaults to False.
        log_stats: log errors when they occur in the processing pipeline. By default, this is set to False, which
            disables the logger for processing pipeline errors. Setting it to None or True will enable logging to the
            default logger name. Setting it to a string specifies the logger to which errors should be logged.
        lazy: whether to execute in lazy mode or not. See the :ref:`Lazy Resampling topic<lazy_resampling> for more
            information about lazy resampling. Defaults to None.
        overrides: optional overrides to apply to transform parameters. This parameter is ignored unless transforms
            are being executed lazily. See the :ref:`Lazy Resampling topic<lazy_resampling> for more details and
            examples of its usage.

    Raises:
        Exception: When ``transform`` raises an exception.

    Returns:
        Union[List[ReturnType], ReturnType]: The return type of `transform` or a list thereof.
    """
    try:
        if isinstance(data, (list, tuple)) and (map_items or type(map_items) is int):
            # if map_items is an int, apply the transform to each item in the list `map_items` times
            if type(map_items) is int and map_items > 0:
                return [
                    apply_transform(transform, item, map_items - 2, unpack_items, log_stats, lazy, overrides)
                    for item in data
                ]
            else:
                return [_apply_transform(transform, item, unpack_items, lazy, overrides, log_stats) for item in data]
        return _apply_transform(transform, data, unpack_items, lazy, overrides, log_stats)
    except Exception as e:
        # if in debug mode, don't swallow exception so that the breakpoint
        # appears where the exception was raised.
        if MONAIEnvVars.debug():
            raise
        if log_stats is not False and not isinstance(transform, transforms.compose.Compose):
            # log the input data information of exact transform in the transform chain
            if isinstance(log_stats, str):
                datastats = transforms.utility.array.DataStats(data_shape=False, value_range=False, name=log_stats)
            else:
                datastats = transforms.utility.array.DataStats(data_shape=False, value_range=False)
            logger = logging.getLogger(datastats._logger_name)
            logger.error(f"\n=== Transform input info -- {type(transform).__name__} ===")
            if isinstance(data, (list, tuple)):
                data = data[0]

            def _log_stats(data, prefix: str | None = "Data"):
                if isinstance(data, (np.ndarray, torch.Tensor)):
                    # log data type, shape, range for array
                    datastats(img=data, data_shape=True, value_range=True, prefix=prefix)
                else:
                    # log data type and value for other metadata
                    datastats(img=data, data_value=True, prefix=prefix)

            if isinstance(data, dict):
                for k, v in data.items():
                    _log_stats(data=v, prefix=k)
            else:
                _log_stats(data=data)
        raise RuntimeError(f"applying transform {transform}") from e


class Randomizable(ThreadUnsafe, RandomizableTrait):
    """
    An interface for handling random state locally, currently based on a class
    variable `R`, which is an instance of `np.random.RandomState`.  This
    provides the flexibility of component-specific determinism without
    affecting the global states.  It is recommended to use this API with
    :py:class:`monai.data.DataLoader` for deterministic behaviour of the
    preprocessing pipelines. This API is not thread-safe. Additionally,
    deepcopying instance of this class often causes insufficient randomness as
    the random states will be duplicated.
    """

    R: np.random.RandomState = np.random.RandomState()

    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None) -> Randomizable:
        """
        Set the random state locally, to control the randomness, the derived
        classes should use :py:attr:`self.R` instead of `np.random` to introduce random
        factors.

        Args:
            seed: set the random state with an integer seed.
            state: set the random state with a `np.random.RandomState` object.

        Raises:
            TypeError: When ``state`` is not an ``Optional[np.random.RandomState]``.

        Returns:
            a Randomizable instance.

        """
        if seed is not None:
            _seed = np.int64(id(seed) if not isinstance(seed, (int, np.integer)) else seed)
            _seed = _seed % MAX_SEED  # need to account for Numpy2.0 which doesn't silently convert to int64
            self.R = np.random.RandomState(_seed)
            return self

        if state is not None:
            if not isinstance(state, np.random.RandomState):
                raise TypeError(f"state must be None or a np.random.RandomState but is {type(state).__name__}.")
            self.R = state
            return self

        self.R = np.random.RandomState()
        return self

    def randomize(self, data: Any) -> None:
        """
        Within this method, :py:attr:`self.R` should be used, instead of `np.random`, to introduce random factors.

        all :py:attr:`self.R` calls happen here so that we have a better chance to
        identify errors of sync the random state.

        This method can generate the random factors based on properties of the input data.

        Raises:
            NotImplementedError: When the subclass does not override this method.
        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")


class Transform(ABC):
    """
    An abstract class of a ``Transform``.
    A transform is callable that processes ``data``.

    It could be stateful and may modify ``data`` in place,
    the implementation should be aware of:

        #. thread safety when mutating its own states.
           When used from a multi-process context, transform's instance variables are read-only.
           thread-unsafe transforms should inherit :py:class:`monai.transforms.ThreadUnsafe`.
        #. ``data`` content unused by this transform may still be used in the
           subsequent transforms in a composed transform.
        #. storing too much information in ``data`` may cause some memory issue or IPC sync issue,
           especially in the multi-processing environment of PyTorch DataLoader.

    See Also

        :py:class:`monai.transforms.Compose`
    """

    # Transforms should add `monai.transforms.utils.TransformBackends` to this list if they are performing
    # the data processing using the corresponding backend APIs.
    # Most of MONAI transform's inputs and outputs will be converted into torch.Tensor or monai.data.MetaTensor.
    # This variable provides information about whether the input will be converted
    # to other data types during the transformation. Note that not all `dtype` (such as float32, uint8) are supported
    # by all the data types, the `dtype` during the conversion is determined automatically by each transform,
    # please refer to the transform's docstring.
    backend: list[TransformBackends] = []

    @abstractmethod
    def __call__(self, data: Any):
        """
        ``data`` is an element which often comes from an iteration over an
        iterable, such as :py:class:`torch.utils.data.Dataset`. This method should
        return an updated version of ``data``.
        To simplify the input validations, most of the transforms assume that

        - ``data`` is a Numpy ndarray, PyTorch Tensor or string,
        - the data shape can be:

          #. string data without shape, `LoadImage` transform expects file paths,
          #. most of the pre-/post-processing transforms expect: ``(num_channels, spatial_dim_1[, spatial_dim_2, ...])``,
             except for example: `AddChannel` expects (spatial_dim_1[, spatial_dim_2, ...])

        - the channel dimension is often not omitted even if number of channels is one.

        This method can optionally take additional arguments to help execute transformation operation.

        Raises:
            NotImplementedError: When the subclass does not override this method.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")


class LazyTransform(Transform, LazyTrait):
    """
    An implementation of functionality for lazy transforms that can be subclassed by array and
    dictionary transforms to simplify implementation of new lazy transforms.
    """

    def __init__(self, lazy: bool | None = False):
        if lazy is not None:
            if not isinstance(lazy, bool):
                raise TypeError(f"lazy must be a bool but is of type {type(lazy)}")
        self._lazy = lazy

    @property
    def lazy(self):
        return self._lazy

    @lazy.setter
    def lazy(self, lazy: bool | None):
        if lazy is not None:
            if not isinstance(lazy, bool):
                raise TypeError(f"lazy must be a bool but is of type {type(lazy)}")
        self._lazy = lazy

    @property
    def requires_current_data(self):
        return False


class RandomizableTransform(Randomizable, Transform):
    """
    An interface for handling random state locally, currently based on a class variable `R`,
    which is an instance of `np.random.RandomState`.
    This class introduces a randomized flag `_do_transform`, is mainly for randomized data augmentation transforms.
    For example:

    .. code-block:: python

        from monai.transforms import RandomizableTransform

        class RandShiftIntensity100(RandomizableTransform):
            def randomize(self):
                super().randomize(None)
                self._offset = self.R.uniform(low=0, high=100)

            def __call__(self, img):
                self.randomize()
                if not self._do_transform:
                    return img
                return img + self._offset

        transform = RandShiftIntensity()
        transform.set_random_state(seed=0)
        print(transform(10))

    """

    def __init__(self, prob: float = 1.0, do_transform: bool = True):
        self._do_transform = do_transform
        self.prob = min(max(prob, 0.0), 1.0)

    def randomize(self, data: Any) -> None:
        """
        Within this method, :py:attr:`self.R` should be used, instead of `np.random`, to introduce random factors.

        all :py:attr:`self.R` calls happen here so that we have a better chance to
        identify errors of sync the random state.

        This method can generate the random factors based on properties of the input data.
        """
        self._do_transform = self.R.rand() < self.prob


class MapTransform(Transform):
    """
    A subclass of :py:class:`monai.transforms.Transform` with an assumption
    that the ``data`` input of ``self.__call__`` is a MutableMapping such as ``dict``.

    The ``keys`` parameter will be used to get and set the actual data
    item to transform.  That is, the callable of this transform should
    follow the pattern:

        .. code-block:: python

            def __call__(self, data):
                for key in self.keys:
                    if key in data:
                        # update output data with some_transform_function(data[key]).
                    else:
                        # raise exception unless allow_missing_keys==True.
                return data

    Raises:
        ValueError: When ``keys`` is an empty iterable.
        TypeError: When ``keys`` type is not in ``Union[Hashable, Iterable[Hashable]]``.

    """

    def __new__(cls, *args, **kwargs):
        if config.USE_META_DICT:
            # call_update after MapTransform.__call__
            cls.__call__ = transforms.attach_hook(cls.__call__, MapTransform.call_update, "post")  # type: ignore

            if hasattr(cls, "inverse"):
                # inverse_update before InvertibleTransform.inverse
                cls.inverse: Any = transforms.attach_hook(cls.inverse, transforms.InvertibleTransform.inverse_update)
        return Transform.__new__(cls)

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
        super().__init__()
        self.keys: tuple[Hashable, ...] = ensure_tuple(keys)
        self.allow_missing_keys = allow_missing_keys
        if not self.keys:
            raise ValueError("keys must be non empty.")
        for key in self.keys:
            if not isinstance(key, Hashable):
                raise TypeError(f"keys must be one of (Hashable, Iterable[Hashable]) but is {type(keys).__name__}.")

    def call_update(self, data):
        """
        This function is to be called after every `self.__call__(data)`,
        update `data[key_transforms]` and `data[key_meta_dict]` using the content from MetaTensor `data[key]`,
        for MetaTensor backward compatibility 0.9.0.
        """
        if not isinstance(data, (list, tuple, Mapping)):
            return data
        is_dict = False
        if isinstance(data, Mapping):
            data, is_dict = [data], True
        if not data or not isinstance(data[0], Mapping):
            return data[0] if is_dict else data
        list_d = [dict(x) for x in data]  # list of dict for crop samples
        for idx, dict_i in enumerate(list_d):
            for k in dict_i:
                if not isinstance(dict_i[k], MetaTensor):
                    continue
                list_d[idx] = transforms.sync_meta_info(k, dict_i, t=not isinstance(self, transforms.InvertD))
        return list_d[0] if is_dict else list_d

    @abstractmethod
    def __call__(self, data):
        """
        ``data`` often comes from an iteration over an iterable,
        such as :py:class:`torch.utils.data.Dataset`.

        To simplify the input validations, this method assumes:

        - ``data`` is a Python dictionary,
        - ``data[key]`` is a Numpy ndarray, PyTorch Tensor or string, where ``key`` is an element
          of ``self.keys``, the data shape can be:

          #. string data without shape, `LoadImaged` transform expects file paths,
          #. most of the pre-/post-processing transforms expect: ``(num_channels, spatial_dim_1[, spatial_dim_2, ...])``,
             except for example: `AddChanneld` expects (spatial_dim_1[, spatial_dim_2, ...])

        - the channel dimension is often not omitted even if number of channels is one.

        Raises:
            NotImplementedError: When the subclass does not override this method.

        returns:
            An updated dictionary version of ``data`` by applying the transform.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    def key_iterator(self, data: Mapping[Hashable, Any], *extra_iterables: Iterable | None) -> Generator:
        """
        Iterate across keys and optionally extra iterables. If key is missing, exception is raised if
        `allow_missing_keys==False` (default). If `allow_missing_keys==True`, key is skipped.

        Args:
            data: data that the transform will be applied to
            extra_iterables: anything else to be iterated through
        """
        # if no extra iterables given, create a dummy list of Nones
        ex_iters = extra_iterables or [[None] * len(self.keys)]

        # loop over keys and any extra iterables
        _ex_iters: list[Any]
        for key, *_ex_iters in zip(self.keys, *ex_iters):
            # all normal, yield (what we yield depends on whether extra iterables were given)
            if key in data:
                yield (key,) + tuple(_ex_iters) if extra_iterables else key
            elif not self.allow_missing_keys:
                raise KeyError(
                    f"Key `{key}` of transform `{self.__class__.__name__}` was missing in the data"
                    " and allow_missing_keys==False."
                )

    def first_key(self, data: dict[Hashable, Any]):
        """
        Get the first available key of `self.keys` in the input `data` dictionary.
        If no available key, return an empty tuple `()`.

        Args:
            data: data that the transform will be applied to.

        """
        return first(self.key_iterator(data), ())
