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

import warnings
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Any

import numpy as np

import monai
from monai.apps.utils import get_logger
from monai.config import NdarrayOrTensor
from monai.transforms.inverse import InvertibleTransform

# For backwards compatibility (so this still works: from monai.transforms.compose import MapTransform)
from monai.transforms.lazy.functional import apply_pending_transforms
from monai.transforms.traits import ThreadUnsafe
from monai.transforms.transform import (  # noqa: F401
    LazyTransform,
    MapTransform,
    Randomizable,
    RandomizableTransform,
    Transform,
    apply_transform,
)
from monai.utils import MAX_SEED, TraceKeys, TraceStatusKeys, ensure_tuple, get_seed

logger = get_logger(__name__)

__all__ = ["Compose", "OneOf", "RandomOrder", "SomeOf", "execute_compose"]


def execute_compose(
    data: NdarrayOrTensor | Sequence[NdarrayOrTensor] | Mapping[Any, NdarrayOrTensor],
    transforms: Sequence[Any],
    map_items: bool | int = True,
    unpack_items: bool = False,
    start: int = 0,
    end: int | None = None,
    lazy: bool | None = False,
    overrides: dict | None = None,
    threading: bool = False,
    log_stats: bool | str = False,
) -> NdarrayOrTensor | Sequence[NdarrayOrTensor] | Mapping[Any, NdarrayOrTensor]:
    """
    ``execute_compose`` provides the implementation that the ``Compose`` class uses to execute a sequence
    of transforms. As well as being used by Compose, it can be used by subclasses of
    Compose and by code that doesn't have a Compose instance but needs to execute a
    sequence of transforms is if it were executed by Compose. It should only be used directly
    when it is not possible to use ``Compose.__call__`` to achieve the same goal.
    Args:
        data: a tensor-like object to be transformed
        transforms: a sequence of transforms to be carried out
        map_items: whether to apply transform to each item in the input `data` if `data` is a list or tuple.
            defaults to `True`. If set to an integer, recursively map the items in `data` `map_items` times.
        unpack_items: whether to unpack input `data` with `*` as parameters for the callable function of transform.
            defaults to `False`.
        start: the index of the first transform to be executed. If not set, this defaults to 0
        end: the index after the last transform to be executed. If set, the transform at index-1
            is the last transform that is executed. If this is not set, it defaults to len(transforms)
        lazy: whether to enable :ref:`lazy evaluation<lazy_resampling>` for lazy transforms. If False, transforms will be
            carried out on a transform by transform basis. If True, all lazy transforms will
            be executed by accumulating changes and resampling as few times as possible.
        overrides: this optional parameter allows you to specify a dictionary of parameters that should be overridden
            when executing a pipeline. These each parameter that is compatible with a given transform is then applied
            to that transform before it is executed. Note that overrides are currently only applied when
            :ref:`lazy evaluation<lazy_resampling>` is enabled for the pipeline or a given transform. If lazy is False
            they are ignored. Currently supported args are:
            {``"mode"``, ``"padding_mode"``, ``"dtype"``, ``"align_corners"``, ``"resample_mode"``, ``device``}.
        threading: whether executing is happening in a threaded environment. If set, copies are made
            of transforms that have the ``RandomizedTrait`` interface.
        log_stats: this optional parameter allows you to specify a logger by name for logging of pipeline execution.
            Setting this to False disables logging. Setting it to True enables logging to the default loggers.
            Setting a string overrides the logger name to which logging is performed.

    Returns:
        A tensorlike, sequence of tensorlikes or dict of tensorlists containing the result of running
        `data`` through the sequence of ``transforms``.
    """
    end_ = len(transforms) if end is None else end
    if start is None:
        raise ValueError(f"'start' ({start}) cannot be None")
    if start < 0:
        raise ValueError(f"'start' ({start}) cannot be less than 0")
    if start > end_:
        raise ValueError(f"'start' ({start}) must be less than 'end' ({end_})")
    if end_ > len(transforms):
        raise ValueError(f"'end' ({end_}) must be less than or equal to the transform count ({len(transforms)}")

    # no-op if the range is empty
    if start == end:
        return data

    for _transform in transforms[start:end]:
        if threading:
            _transform = deepcopy(_transform) if isinstance(_transform, ThreadUnsafe) else _transform
        data = apply_transform(
            _transform, data, map_items, unpack_items, lazy=lazy, overrides=overrides, log_stats=log_stats
        )
    data = apply_pending_transforms(data, None, overrides, logger_name=log_stats)
    return data


class Compose(Randomizable, InvertibleTransform, LazyTransform):
    """
    ``Compose`` provides the ability to chain a series of callables together in
    a sequential manner. Each transform in the sequence must take a single
    argument and return a single value.

    ``Compose`` can be used in two ways:

    #. With a series of transforms that accept and return a single
       ndarray / tensor / tensor-like parameter.
    #. With a series of transforms that accept and return a dictionary that
       contains one or more parameters. Such transforms must have pass-through
       semantics that unused values in the dictionary must be copied to the return
       dictionary. It is required that the dictionary is copied between input
       and output of each transform.

    If some transform takes a data item dictionary as input, and returns a
    sequence of data items in the transform chain, all following transforms
    will be applied to each item of this list if `map_items` is `True` (the
    default).  If `map_items` is `False`, the returned sequence is passed whole
    to the next callable in the chain.

    For example:

    A `Compose([transformA, transformB, transformC],
    map_items=True)(data_dict)` could achieve the following patch-based
    transformation on the `data_dict` input:

    #. transformA normalizes the intensity of 'img' field in the `data_dict`.
    #. transformB crops out image patches from the 'img' and 'seg' of
       `data_dict`, and return a list of three patch samples::

        {'img': 3x100x100 data, 'seg': 1x100x100 data, 'shape': (100, 100)}
                             applying transformB
                                 ---------->
        [{'img': 3x20x20 data, 'seg': 1x20x20 data, 'shape': (20, 20)},
         {'img': 3x20x20 data, 'seg': 1x20x20 data, 'shape': (20, 20)},
         {'img': 3x20x20 data, 'seg': 1x20x20 data, 'shape': (20, 20)},]

    #. transformC then randomly rotates or flips 'img' and 'seg' of
       each dictionary item in the list returned by transformB.

    The composed transforms will be set the same global random seed if user called
    `set_determinism()`.

    When using the pass-through dictionary operation, you can make use of
    :class:`monai.transforms.adaptors.adaptor` to wrap transforms that don't conform
    to the requirements. This approach allows you to use transforms from
    otherwise incompatible libraries with minimal additional work.

    Note:

        In many cases, Compose is not the best way to create pre-processing
        pipelines. Pre-processing is often not a strictly sequential series of
        operations, and much of the complexity arises when a not-sequential
        set of functions must be called as if it were a sequence.

        Example: images and labels
        Images typically require some kind of normalization that labels do not.
        Both are then typically augmented through the use of random rotations,
        flips, and deformations.
        Compose can be used with a series of transforms that take a dictionary
        that contains 'image' and 'label' entries. This might require wrapping
        `torchvision` transforms before passing them to compose.
        Alternatively, one can create a class with a `__call__` function that
        calls your pre-processing functions taking into account that not all of
        them are called on the labels.

    Lazy resampling:

        Lazy resampling is an experimental feature introduced in 1.2. Its purpose is
        to reduce the number of resample operations that must be carried out when executing
        a pipeline of transforms. This can provide significant performance improvements in
        terms of pipeline executing speed and memory usage, and can also significantly
        reduce the loss of information that occurs when performing a number of spatial
        resamples in succession.

        Lazy resampling can be enabled or disabled through the ``lazy`` parameter, either by
        specifying it at initialisation time or overriding it at call time.

        * False (default): Don't perform any lazy resampling
        * None: Perform lazy resampling based on the 'lazy' properties of the transform instances.
        * True: Always perform lazy resampling if possible. This will ignore the ``lazy`` properties
          of the transform instances

        Please see the :ref:`Lazy Resampling topic<lazy_resampling>` for more details of this feature
        and examples of its use.

    Args:
        transforms: sequence of callables.
        map_items: whether to apply transform to each item in the input `data` if `data` is a list or tuple.
            defaults to `True`. If set to an integer, recursively map the items in `data` `map_items` times.
        unpack_items: whether to unpack input `data` with `*` as parameters for the callable function of transform.
            defaults to `False`.
        log_stats: this optional parameter allows you to specify a logger by name for logging of pipeline execution.
            Setting this to False disables logging. Setting it to True enables logging to the default loggers.
            Setting a string overrides the logger name to which logging is performed.
        lazy: whether to enable :ref:`Lazy Resampling<lazy_resampling>` for lazy transforms. If False, transforms will
            be carried out on a transform by transform basis. If True, all lazy transforms will be executed by
            accumulating changes and resampling as few times as possible. If lazy is None, `Compose` will
            perform lazy execution on lazy transforms that have their `lazy` property set to True.
        overrides: this optional parameter allows you to specify a dictionary of parameters that should be overridden
            when executing a pipeline. These each parameter that is compatible with a given transform is then applied
            to that transform before it is executed. Note that overrides are currently only applied when
            :ref:`Lazy Resampling<lazy_resampling>` is enabled for the pipeline or a given transform. If lazy is False
            they are ignored. Currently supported args are:
            {``"mode"``, ``"padding_mode"``, ``"dtype"``, ``"align_corners"``, ``"resample_mode"``, ``device``}.
    """

    def __init__(
        self,
        transforms: Sequence[Callable] | Callable | None = None,
        map_items: bool | int = True,
        unpack_items: bool = False,
        log_stats: bool | str = False,
        lazy: bool | None = False,
        overrides: dict | None = None,
    ) -> None:
        LazyTransform.__init__(self, lazy=lazy)

        if transforms is None:
            transforms = []

        if not isinstance(map_items, (bool, int)):
            raise ValueError(
                f"Argument 'map_items' should be boolean or int. Got {type(map_items)}."
                "Check brackets when passing a sequence of callables."
            )

        self.transforms = ensure_tuple(transforms)
        self.map_items = map_items
        self.unpack_items = unpack_items
        self.log_stats = log_stats
        self.set_random_state(seed=get_seed())
        self.overrides = overrides

    @LazyTransform.lazy.setter  # type: ignore
    def lazy(self, val: bool):
        self._lazy = val

    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None) -> Compose:
        super().set_random_state(seed=seed, state=state)
        for _transform in self.transforms:
            if not isinstance(_transform, Randomizable):
                continue
            _transform.set_random_state(seed=self.R.randint(MAX_SEED, dtype="uint32"))
        return self

    def randomize(self, data: Any | None = None) -> None:
        for _transform in self.transforms:
            if not isinstance(_transform, Randomizable):
                continue
            try:
                _transform.randomize(data)
            except TypeError as type_error:
                tfm_name: str = type(_transform).__name__
                warnings.warn(
                    f"Transform '{tfm_name}' in Compose not randomized\n{tfm_name}.{type_error}.", RuntimeWarning
                )

    def get_index_of_first(self, predicate):
        """
        get_index_of_first takes a ``predicate`` and returns the index of the first transform that
        satisfies the predicate (ie. makes the predicate return True). If it is unable to find
        a transform that satisfies the ``predicate``, it returns None.

        Example:
            c = Compose([Flip(...), Rotate90(...), Zoom(...), RandRotate(...), Resize(...)])

            print(c.get_index_of_first(lambda t: isinstance(t, RandomTrait)))
            >>> 3
            print(c.get_index_of_first(lambda t: isinstance(t, Compose)))
            >>> None

        Note:
            This is only performed on the transforms directly held by this instance. If this
            instance has nested ``Compose`` transforms or other transforms that contain transforms,
            it does not iterate into them.


        Args:
            predicate: a callable that takes a single argument and returns a bool. When called
            it is passed a transform from the sequence of transforms contained by this compose
            instance.

        Returns:
            The index of the first transform in the sequence for which ``predicate`` returns
            True. None if no transform satisfies the ``predicate``

        """
        for i in range(len(self.transforms)):
            if predicate(self.transforms[i]):
                return i
        return None

    def flatten(self):
        """Return a Composition with a simple list of transforms, as opposed to any nested Compositions.

        e.g., `t1 = Compose([x, x, x, x, Compose([Compose([x, x]), x, x])]).flatten()`
        will result in the equivalent of `t1 = Compose([x, x, x, x, x, x, x, x])`.

        """
        new_transforms = []
        for t in self.transforms:
            if type(t) is Compose:  # nopep8
                new_transforms += t.flatten().transforms
            else:
                new_transforms.append(t)

        return Compose(new_transforms)

    def __len__(self):
        """Return number of transformations."""
        return len(self.flatten().transforms)

    def __call__(self, input_, start=0, end=None, threading=False, lazy: bool | None = None):
        _lazy = self._lazy if lazy is None else lazy
        result = execute_compose(
            input_,
            transforms=self.transforms,
            start=start,
            end=end,
            map_items=self.map_items,
            unpack_items=self.unpack_items,
            lazy=_lazy,
            overrides=self.overrides,
            threading=threading,
            log_stats=self.log_stats,
        )

        return result

    def inverse(self, data):
        self._raise_if_not_invertible(data)

        invertible_transforms = [t for t in self.flatten().transforms if isinstance(t, InvertibleTransform)]
        if not invertible_transforms:
            warnings.warn("inverse has been called but no invertible transforms have been supplied")

        if self._lazy is True:
            warnings.warn(
                f"'lazy' is set to {self._lazy} but lazy execution is not supported when inverting. "
                f"'lazy' has been overridden to False for the call to inverse"
            )
        # loop backwards over transforms
        for t in reversed(invertible_transforms):
            data = apply_transform(
                t.inverse, data, self.map_items, self.unpack_items, lazy=False, log_stats=self.log_stats
            )
        return data

    @staticmethod
    def _raise_if_not_invertible(data: Any):
        from monai.transforms.utils import has_status_keys

        invertible, reasons = has_status_keys(
            data, TraceStatusKeys.PENDING_DURING_APPLY, "Pending operations while applying an operation"
        )

        if invertible is False:
            if reasons is not None:
                reason_text = "\n".join(reasons)
                raise RuntimeError(f"Unable to run inverse on 'data' for the following reasons:\n{reason_text}")
            else:
                raise RuntimeError("Unable to run inverse on 'data'; no reason logged in trace data")


class OneOf(Compose):
    """
    ``OneOf`` provides the ability to randomly choose one transform out of a
    list of callables with pre-defined probabilities for each.

    Args:
        transforms: sequence of callables.
        weights: probabilities corresponding to each callable in transforms.
            Probabilities are normalized to sum to one.
        map_items: whether to apply transform to each item in the input `data` if `data` is a list or tuple.
            defaults to `True`. If set to an integer, recursively map the items in `data` `map_items` times.
        unpack_items: whether to unpack input `data` with `*` as parameters for the callable function of transform.
            defaults to `False`.
        log_stats: this optional parameter allows you to specify a logger by name for logging of pipeline execution.
            Setting this to False disables logging. Setting it to True enables logging to the default loggers.
            Setting a string overrides the logger name to which logging is performed.
        lazy: whether to enable :ref:`Lazy Resampling<lazy_resampling>` for lazy transforms. If False, transforms will
            be carried out on a transform by transform basis. If True, all lazy transforms will be executed by
            accumulating changes and resampling as few times as possible. If lazy is None, `Compose` will
            perform lazy execution on lazy transforms that have their `lazy` property set to True.
        overrides: this optional parameter allows you to specify a dictionary of parameters that should be overridden
            when executing a pipeline. These each parameter that is compatible with a given transform is then applied
            to that transform before it is executed. Note that overrides are currently only applied when
            :ref:`Lazy Resampling<lazy_resampling>` is enabled for the pipeline or a given transform. If lazy is False
            they are ignored. Currently supported args are:
            {``"mode"``, ``"padding_mode"``, ``"dtype"``, ``"align_corners"``, ``"resample_mode"``, ``device``}.
    """

    def __init__(
        self,
        transforms: Sequence[Callable] | Callable | None = None,
        weights: Sequence[float] | float | None = None,
        map_items: bool | int = True,
        unpack_items: bool = False,
        log_stats: bool | str = False,
        lazy: bool | None = False,
        overrides: dict | None = None,
    ) -> None:
        super().__init__(transforms, map_items, unpack_items, log_stats, lazy, overrides)
        if len(self.transforms) == 0:
            weights = []
        elif weights is None or isinstance(weights, float):
            weights = [1.0 / len(self.transforms)] * len(self.transforms)
        if len(weights) != len(self.transforms):
            raise ValueError(
                "transforms and weights should be same size if both specified as sequences, "
                f"got {len(weights)} and {len(self.transforms)}."
            )
        self.weights = ensure_tuple(self._normalize_probabilities(weights))
        self.log_stats = log_stats

    def _normalize_probabilities(self, weights):
        if len(weights) == 0:
            return weights
        weights = np.array(weights)
        if np.any(weights < 0):
            raise ValueError(f"Probabilities must be greater than or equal to zero, got {weights}.")
        if np.all(weights == 0):
            raise ValueError(f"At least one probability must be greater than zero, got {weights}.")
        weights = weights / weights.sum()
        return list(weights)

    def flatten(self):
        transforms = []
        weights = []
        for t, w in zip(self.transforms, self.weights):
            # if nested, probability is the current weight multiplied by the nested weights,
            # and so on recursively
            if isinstance(t, OneOf):
                tr = t.flatten()
                for t_, w_ in zip(tr.transforms, tr.weights):
                    transforms.append(t_)
                    weights.append(w_ * w)
            else:
                transforms.append(t)
                weights.append(w)
        return OneOf(transforms, weights, self.map_items, self.unpack_items)

    def __call__(self, data, start=0, end=None, threading=False, lazy: bool | None = None):
        if start != 0:
            raise ValueError(f"OneOf requires 'start' parameter to be 0 (start set to {start})")
        if end is not None:
            raise ValueError(f"OneOf requires 'end' parameter to be None (end set to {end}")

        if len(self.transforms) == 0:
            return data

        index = self.R.multinomial(1, self.weights).argmax()
        _transform = self.transforms[index]
        _lazy = self._lazy if lazy is None else lazy

        data = execute_compose(
            data,
            [_transform],
            start=start,
            end=end,
            map_items=self.map_items,
            unpack_items=self.unpack_items,
            lazy=_lazy,
            overrides=self.overrides,
            threading=threading,
            log_stats=self.log_stats,
        )

        # if the data is a mapping (dictionary), append the OneOf transform to the end
        if isinstance(data, monai.data.MetaTensor):
            self.push_transform(data, extra_info={"index": index})
        elif isinstance(data, Mapping):
            for key in data:  # dictionary not change size during iteration
                if isinstance(data[key], monai.data.MetaTensor):
                    self.push_transform(data[key], extra_info={"index": index})
        return data

    def inverse(self, data):
        if len(self.transforms) == 0:
            return data

        index = None
        if isinstance(data, monai.data.MetaTensor):
            index = self.pop_transform(data)[TraceKeys.EXTRA_INFO]["index"]
        elif isinstance(data, Mapping):
            for key in data:
                if isinstance(data[key], monai.data.MetaTensor):
                    index = self.pop_transform(data, key)[TraceKeys.EXTRA_INFO]["index"]
        else:
            raise RuntimeError(
                f"Inverse only implemented for Mapping (dictionary) or MetaTensor data, got type {type(data)}."
            )
        if index is None:
            # no invertible transforms have been applied
            return data

        _transform = self.transforms[index]
        # apply the inverse
        return _transform.inverse(data) if isinstance(_transform, InvertibleTransform) else data


class RandomOrder(Compose):
    """
    ``RandomOrder`` provides the ability to apply a list of transformations in random order.

    Args:
        transforms: sequence of callables.
        map_items: whether to apply transform to each item in the input `data` if `data` is a list or tuple.
            defaults to `True`.
        unpack_items: whether to unpack input `data` with `*` as parameters for the callable function of transform.
            defaults to `False`.
        log_stats: this optional parameter allows you to specify a logger by name for logging of pipeline execution.
            Setting this to False disables logging. Setting it to True enables logging to the default loggers.
            Setting a string overrides the logger name to which logging is performed.
        lazy: whether to enable :ref:`Lazy Resampling<lazy_resampling>` for lazy transforms. If False, transforms will
            be carried out on a transform by transform basis. If True, all lazy transforms will be executed by
            accumulating changes and resampling as few times as possible. If lazy is None, `Compose` will
            perform lazy execution on lazy transforms that have their `lazy` property set to True.
        overrides: this optional parameter allows you to specify a dictionary of parameters that should be overridden
            when executing a pipeline. These each parameter that is compatible with a given transform is then applied
            to that transform before it is executed. Note that overrides are currently only applied when
            :ref:`Lazy Resampling<lazy_resampling>` is enabled for the pipeline or a given transform. If lazy is False
            they are ignored. Currently supported args are:
            {``"mode"``, ``"padding_mode"``, ``"dtype"``, ``"align_corners"``, ``"resample_mode"``, ``device``}.
    """

    def __init__(
        self,
        transforms: Sequence[Callable] | Callable | None = None,
        map_items: bool = True,
        unpack_items: bool = False,
        log_stats: bool | str = False,
        lazy: bool | None = False,
        overrides: dict | None = None,
    ) -> None:
        super().__init__(transforms, map_items, unpack_items, log_stats, lazy, overrides)
        self.log_stats = log_stats

    def __call__(self, input_, start=0, end=None, threading=False, lazy: bool | None = None):
        if start != 0:
            raise ValueError(f"RandomOrder requires 'start' parameter to be 0 (start set to {start})")
        if end is not None:
            raise ValueError(f"RandomOrder requires 'end' parameter to be None (end set to {end}")

        if len(self.transforms) == 0:
            return input_

        num = len(self.transforms)
        applied_order = self.R.permutation(range(num))
        _lazy = self._lazy if lazy is None else lazy

        input_ = execute_compose(
            input_,
            [self.transforms[ind] for ind in applied_order],
            start=start,
            end=end,
            map_items=self.map_items,
            unpack_items=self.unpack_items,
            lazy=_lazy,
            threading=threading,
            log_stats=self.log_stats,
        )

        # if the data is a mapping (dictionary), append the RandomOrder transform to the end
        if isinstance(input_, monai.data.MetaTensor):
            self.push_transform(input_, extra_info={"applied_order": applied_order})
        elif isinstance(input_, Mapping):
            for key in input_:  # dictionary not change size during iteration
                if isinstance(input_[key], monai.data.MetaTensor):
                    self.push_transform(input_[key], extra_info={"applied_order": applied_order})
        return input_

    def inverse(self, data):
        if len(self.transforms) == 0:
            return data

        applied_order = None
        if isinstance(data, monai.data.MetaTensor):
            applied_order = self.pop_transform(data)[TraceKeys.EXTRA_INFO]["applied_order"]
        elif isinstance(data, Mapping):
            for key in data:
                if isinstance(data[key], monai.data.MetaTensor):
                    applied_order = self.pop_transform(data, key)[TraceKeys.EXTRA_INFO]["applied_order"]
        else:
            raise RuntimeError(
                f"Inverse only implemented for Mapping (dictionary) or MetaTensor data, got type {type(data)}."
            )
        if applied_order is None:
            # no invertible transforms have been applied
            return data

        # loop backwards over transforms
        for o in reversed(applied_order):
            if isinstance(self.transforms[o], InvertibleTransform):
                data = apply_transform(
                    self.transforms[o].inverse, data, self.map_items, self.unpack_items, log_stats=self.log_stats
                )
        return data


class SomeOf(Compose):
    """
    ``SomeOf`` samples a different sequence of transforms to apply each time it is called.

    It can be configured to sample a fixed or varying number of transforms each time its called. Samples are drawn
    uniformly, or from user supplied transform weights. When varying the number of transforms sampled per call,
    the number of transforms to sample that call is sampled uniformly from a range supplied by the user.

    Args:
        transforms: list of callables.
        map_items: whether to apply transform to each item in the input `data` if `data` is a list or tuple.
            Defaults to `True`.
        unpack_items: whether to unpack input `data` with `*` as parameters for the callable function of transform.
            Defaults to `False`.
        log_stats: this optional parameter allows you to specify a logger by name for logging of pipeline execution.
            Setting this to False disables logging. Setting it to True enables logging to the default loggers.
            Setting a string overrides the logger name to which logging is performed.
        num_transforms: a 2-tuple, int, or None. The 2-tuple specifies the minimum and maximum (inclusive) number of
            transforms to sample at each iteration. If an int is given, the lower and upper bounds are set equal.
            None sets it to `len(transforms)`. Default to `None`.
        replace: whether to sample with replacement. Defaults to `False`.
        weights: weights to use in for sampling transforms. Will be normalized to 1. Default: None (uniform).
        lazy: whether to enable :ref:`Lazy Resampling<lazy_resampling>` for lazy transforms. If False, transforms will
            be carried out on a transform by transform basis. If True, all lazy transforms will be executed by
            accumulating changes and resampling as few times as possible. If lazy is None, `Compose` will
            perform lazy execution on lazy transforms that have their `lazy` property set to True.
        overrides: this optional parameter allows you to specify a dictionary of parameters that should be overridden
            when executing a pipeline. These each parameter that is compatible with a given transform is then applied
            to that transform before it is executed. Note that overrides are currently only applied when
            :ref:`Lazy Resampling<lazy_resampling>` is enabled for the pipeline or a given transform. If lazy is False
            they are ignored. Currently supported args are:
            {``"mode"``, ``"padding_mode"``, ``"dtype"``, ``"align_corners"``, ``"resample_mode"``, ``device``}.
    """

    def __init__(
        self,
        transforms: Sequence[Callable] | Callable | None = None,
        map_items: bool = True,
        unpack_items: bool = False,
        log_stats: bool | str = False,
        num_transforms: int | tuple[int, int] | None = None,
        replace: bool = False,
        weights: list[int] | None = None,
        lazy: bool | None = False,
        overrides: dict | None = None,
    ) -> None:
        super().__init__(transforms, map_items, unpack_items, log_stats=log_stats, lazy=lazy, overrides=overrides)
        self.min_num_transforms, self.max_num_transforms = self._ensure_valid_num_transforms(num_transforms)
        self.replace = replace
        self.weights = self._normalize_probabilities(weights)
        self.log_stats = log_stats

    def _ensure_valid_num_transforms(self, num_transforms: int | tuple[int, int] | None) -> tuple:
        if (
            not isinstance(num_transforms, tuple)
            and not isinstance(num_transforms, list)
            and not isinstance(num_transforms, int)
            and num_transforms is not None
        ):
            raise ValueError(
                f"Expected num_transforms to be of type int, list, tuple or None, but it's {type(num_transforms)}"
            )

        if num_transforms is None:
            result = [len(self.transforms), len(self.transforms)]
        elif isinstance(num_transforms, int):
            n = min(len(self.transforms), num_transforms)
            result = [n, n]
        else:
            if len(num_transforms) != 2:
                raise ValueError(f"Expected len(num_transforms)=2, but it was {len(num_transforms)}")
            if not isinstance(num_transforms[0], int) or not isinstance(num_transforms[1], int):
                raise ValueError(
                    f"Expected (int,int), but received ({type(num_transforms[0])}, {type(num_transforms[1])})"
                )

            result = [num_transforms[0], num_transforms[1]]

        if result[0] < 0 or result[1] > len(self.transforms):
            raise ValueError(f"num_transforms={num_transforms} are out of the bounds [0, {len(self.transforms)}].")

        return ensure_tuple(result)

    # Modified from OneOf
    def _normalize_probabilities(self, weights):
        if weights is None or len(self.transforms) == 0:
            return None

        weights = np.array(weights)

        n_weights = len(weights)
        if n_weights != len(self.transforms):
            raise ValueError(f"Expected len(weights)={len(self.transforms)}, got: {n_weights}.")

        if np.any(weights < 0):
            raise ValueError(f"Probabilities must be greater than or equal to zero, got {weights}.")

        if np.all(weights == 0):
            raise ValueError(f"At least one probability must be greater than zero, got {weights}.")

        weights = weights / weights.sum()

        return ensure_tuple(list(weights))

    def __call__(self, data, start=0, end=None, threading=False, lazy: bool | None = None):
        if start != 0:
            raise ValueError(f"SomeOf requires 'start' parameter to be 0 (start set to {start})")
        if end is not None:
            raise ValueError(f"SomeOf requires 'end' parameter to be None (end set to {end}")

        if len(self.transforms) == 0:
            return data

        sample_size = self.R.randint(self.min_num_transforms, self.max_num_transforms + 1)
        applied_order = self.R.choice(len(self.transforms), sample_size, replace=self.replace, p=self.weights).tolist()
        _lazy = self._lazy if lazy is None else lazy

        data = execute_compose(
            data,
            [self.transforms[a] for a in applied_order],
            start=start,
            end=end,
            map_items=self.map_items,
            unpack_items=self.unpack_items,
            lazy=_lazy,
            overrides=self.overrides,
            threading=threading,
            log_stats=self.log_stats,
        )
        if isinstance(data, monai.data.MetaTensor):
            self.push_transform(data, extra_info={"applied_order": applied_order})
        elif isinstance(data, Mapping):
            for key in data:  # dictionary not change size during iteration
                if isinstance(data[key], monai.data.MetaTensor) or self.trace_key(key) in data:
                    self.push_transform(data, key, extra_info={"applied_order": applied_order})

        return data

    # From RandomOrder
    def inverse(self, data):
        if len(self.transforms) == 0:
            return data

        applied_order = None
        if isinstance(data, monai.data.MetaTensor):
            applied_order = self.pop_transform(data)[TraceKeys.EXTRA_INFO]["applied_order"]
        elif isinstance(data, Mapping):
            for key in data:
                if isinstance(data[key], monai.data.MetaTensor) or self.trace_key(key) in data:
                    applied_order = self.pop_transform(data, key)[TraceKeys.EXTRA_INFO]["applied_order"]
        else:
            raise RuntimeError(
                f"Inverse only implemented for Mapping (dictionary) or MetaTensor data, got type {type(data)}."
            )
        if applied_order is None:
            # no invertible transforms have been applied
            return data

        # loop backwards over transforms
        for o in reversed(applied_order):
            if isinstance(self.transforms[o], InvertibleTransform):
                data = apply_transform(
                    self.transforms[o].inverse, data, self.map_items, self.unpack_items, log_stats=self.log_stats
                )

        return data
