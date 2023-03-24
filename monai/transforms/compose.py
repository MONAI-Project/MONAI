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
from monai.config import NdarrayOrTensor
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.traits import ThreadUnsafe

# For backwards compatibility (so this still works: from monai.transforms.compose import MapTransform)
from monai.transforms.transform import (  # noqa: F401
    MapTransform,
    Randomizable,
    RandomizableTransform,
    Transform,
    apply_transform,
)
from monai.utils import MAX_SEED, ensure_tuple, get_seed
from monai.utils.enums import TraceKeys

__all__ = ["Compose", "OneOf", "RandomOrder"]


class Compose(Randomizable, InvertibleTransform):
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

    Args:
        transforms: sequence of callables.
        map_items: whether to apply transform to each item in the input `data` if `data` is a list or tuple.
            defaults to `True`.
        unpack_items: whether to unpack input `data` with `*` as parameters for the callable function of transform.
            defaults to `False`.
        log_stats: whether to log the detailed information of data and applied transform when error happened,
            for NumPy array and PyTorch Tensor, log the data shape and value range,
            for other metadata, log the values directly. default to `False`.

    """

    def __init__(
        self,
        transforms: Sequence[Callable] | Callable | None = None,
        map_items: bool = True,
        unpack_items: bool = False,
        log_stats: bool = False,
    ) -> None:
        if transforms is None:
            transforms = []
        self.transforms = ensure_tuple(transforms)
        self.map_items = map_items
        self.unpack_items = unpack_items
        self.log_stats = log_stats
        self.set_random_state(seed=get_seed())

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
                    f'Transform "{tfm_name}" in Compose not randomized\n{tfm_name}.{type_error}.', RuntimeWarning
                )

    def get_index_of_first(self, predicate):
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

    @classmethod
    def execute(
        cls,
        input_: NdarrayOrTensor,
        transforms: Sequence[Any],
        map_items: bool = True,
        unpack_items: bool = False,
        log_stats: bool = False,
        start: int = 0,
        end: int | None = None,
        threading: bool = False,
    ) -> NdarrayOrTensor:
        """
        ``execute`` provides the implementation that Compose uses to execute a sequence
        of transforms. As well as being used by Compose, it can be used by subclasses of
        Compose and by code that doesn't have a Compose instance but needs to execute a
        sequence of transforms is if it were executed by Compose. It should only be used directly
        when it is not possible to use ``Compose.__call__`` to achieve the same goal.
        Args:
            `input_`: a tensor-like object to be transformed
            transforms: a sequence of transforms to be carried out
            map_items: whether to apply the transform to each item in ``data``.
            Defaults to True if not set.
            unpack_items: whether to unpack parameters using '*'. Defaults to False if not set
            log_stats: whether to log detailed information about the application of ``transforms``
            to ``input_``. For NumPy ndarrays and PyTorch tensors, log only the data shape and
            value range. Defaults to False if not set.
            start: the index of the first transform to be executed. If not set, this defaults to 0
            end: the index after the last transform to be exectued. If set, the transform at index-1
            is the last transform that is executed. If this is not set, it defaults to len(transforms)
            threading: whether executing is happening in a threaded environment. If set, copies are made
            of transforms that have the ``RandomizedTrait`` interface.

        Returns:

        """
        end_ = len(transforms) if end is None else end
        if start is None:
            raise ValueError(f"'start' ({start}) cannot be None")
        if start > end_:
            raise ValueError(f"'start' ({start}) must be less than 'end' ({end_})")
        if end_ > len(transforms):
            raise ValueError(f"'end' ({end_}) must be less than or equal to the transform count ({len(transforms)}")

        # no-op if the range is empty
        if start == end:
            return input_

        for _transform in transforms[start:end]:
            if threading:
                _transform = deepcopy(_transform) if isinstance(_transform, ThreadUnsafe) else _transform
            input_ = apply_transform(_transform, input_, map_items, unpack_items, log_stats)  # type: ignore
        return input_

    def __call__(self, input_, start=0, end=None, threading=False):
        return Compose.execute(
            input_,
            self.transforms,
            map_items=self.map_items,
            unpack_items=self.unpack_items,
            start=start,
            end=end,
            threading=threading,
        )

    def inverse(self, data):
        invertible_transforms = [t for t in self.flatten().transforms if isinstance(t, InvertibleTransform)]
        if not invertible_transforms:
            warnings.warn("inverse has been called but no invertible transforms have been supplied")

        # loop backwards over transforms
        for t in reversed(invertible_transforms):
            data = apply_transform(t.inverse, data, self.map_items, self.unpack_items, self.log_stats)
        return data


class OneOf(Compose):
    """
    ``OneOf`` provides the ability to randomly choose one transform out of a
    list of callables with pre-defined probabilities for each.

    Args:
        transforms: sequence of callables.
        weights: probabilities corresponding to each callable in transforms.
            Probabilities are normalized to sum to one.
        map_items: whether to apply transform to each item in the input `data` if `data` is a list or tuple.
            defaults to `True`.
        unpack_items: whether to unpack input `data` with `*` as parameters for the callable function of transform.
            defaults to `False`.
        log_stats: whether to log the detailed information of data and applied transform when error happened,
            for NumPy array and PyTorch Tensor, log the data shape and value range,
            for other metadata, log the values directly. default to `False`.

    """

    def __init__(
        self,
        transforms: Sequence[Callable] | Callable | None = None,
        weights: Sequence[float] | float | None = None,
        map_items: bool = True,
        unpack_items: bool = False,
        log_stats: bool = False,
    ) -> None:
        super().__init__(transforms, map_items, unpack_items, log_stats)
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

    def __call__(self, data, start=0, end=None, threading=False):
        if len(self.transforms) == 0:
            return data

        index = self.R.multinomial(1, self.weights).argmax()
        _transform = self.transforms[index]

        data = Compose.execute(
            data,
            [_transform],
            map_items=self.map_items,
            unpack_items=self.unpack_items,
            start=start,
            end=end,
            threading=threading,
        )

        # if the data is a mapping (dictionary), append the OneOf transform to the end
        if isinstance(data, monai.data.MetaTensor):
            self.push_transform(data, extra_info={"index": index})
        elif isinstance(data, Mapping):
            for key in data:  # dictionary not change size during iteration
                if isinstance(data[key], monai.data.MetaTensor) or self.trace_key(key) in data:
                    self.push_transform(data, key, extra_info={"index": index})
        return data

    def inverse(self, data):
        if len(self.transforms) == 0:
            return data

        index = None
        if isinstance(data, monai.data.MetaTensor):
            index = self.pop_transform(data)[TraceKeys.EXTRA_INFO]["index"]
        elif isinstance(data, Mapping):
            for key in data:
                if isinstance(data[key], monai.data.MetaTensor) or self.trace_key(key) in data:
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
        log_stats: whether to log the detailed information of data and applied transform when error happened,
            for NumPy array and PyTorch Tensor, log the data shape and value range,
            for other metadata, log the values directly. default to `False`.

    """

    def __init__(
        self,
        transforms: Sequence[Callable] | Callable | None = None,
        map_items: bool = True,
        unpack_items: bool = False,
        log_stats: bool = False,
    ) -> None:
        super().__init__(transforms, map_items, unpack_items, log_stats)

    def __call__(self, input_, start=0, end=None, threading=False):
        if len(self.transforms) == 0:
            return input_
        num = len(self.transforms)
        applied_order = self.R.permutation(range(num))

        input_ = Compose.execute(
            input_,
            [self.transforms[ind] for ind in applied_order],
            map_items=self.map_items,
            unpack_items=self.unpack_items,
            start=start,
            end=end,
            threading=threading,
        )

        # if the data is a mapping (dictionary), append the RandomOrder transform to the end
        if isinstance(input_, monai.data.MetaTensor):
            self.push_transform(input_, extra_info={"applied_order": applied_order})
        elif isinstance(input_, Mapping):
            for key in input_:  # dictionary not change size during iteration
                if isinstance(input_[key], monai.data.MetaTensor) or self.trace_key(key) in input_:
                    self.push_transform(input_, key, extra_info={"applied_order": applied_order})
        return input_

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
            data = apply_transform(self.transforms[o].inverse, data, self.map_items, self.unpack_items, self.log_stats)
        return data
