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
A collection of "vanilla" transforms for the model output tensors.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from monai.config.type_definitions import NdarrayOrTensor
from monai.data.meta_obj import get_track_meta
from monai.data.meta_tensor import MetaTensor
from monai.networks import one_hot
from monai.networks.layers import GaussianFilter, apply_filter, separable_filtering
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.transform import Transform
from monai.transforms.utility.array import ToTensor
from monai.transforms.utils import (
    convert_applied_interp_mode,
    distance_transform_edt,
    fill_holes,
    get_largest_connected_component_mask,
    get_unique_labels,
    remove_small_objects,
)
from monai.transforms.utils_pytorch_numpy_unification import unravel_index
from monai.utils import TransformBackends, convert_data_type, convert_to_tensor, ensure_tuple, look_up_option
from monai.utils.type_conversion import convert_to_dst_type

__all__ = [
    "Activations",
    "AsDiscrete",
    "FillHoles",
    "KeepLargestConnectedComponent",
    "RemoveSmallObjects",
    "LabelFilter",
    "LabelToContour",
    "MeanEnsemble",
    "ProbNMS",
    "SobelGradients",
    "VoteEnsemble",
    "Invert",
    "DistanceTransformEDT",
]


class Activations(Transform):
    """
    Activation operations, typically `Sigmoid` or `Softmax`.

    Args:
        sigmoid: whether to execute sigmoid function on model output before transform.
            Defaults to ``False``.
        softmax: whether to execute softmax function on model output before transform.
            Defaults to ``False``.
        other: callable function to execute other activation layers, for example:
            `other = lambda x: torch.tanh(x)`. Defaults to ``None``.
        kwargs: additional parameters to `torch.softmax` (used when ``softmax=True``).
            Defaults to ``dim=0``, unrecognized parameters will be ignored.

    Raises:
        TypeError: When ``other`` is not an ``Optional[Callable]``.

    """

    backend = [TransformBackends.TORCH]

    def __init__(self, sigmoid: bool = False, softmax: bool = False, other: Callable | None = None, **kwargs) -> None:
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.kwargs = kwargs
        if other is not None and not callable(other):
            raise TypeError(f"other must be None or callable but is {type(other).__name__}.")
        self.other = other

    def __call__(
        self,
        img: NdarrayOrTensor,
        sigmoid: bool | None = None,
        softmax: bool | None = None,
        other: Callable | None = None,
    ) -> NdarrayOrTensor:
        """
        Args:
            sigmoid: whether to execute sigmoid function on model output before transform.
                Defaults to ``self.sigmoid``.
            softmax: whether to execute softmax function on model output before transform.
                Defaults to ``self.softmax``.
            other: callable function to execute other activation layers, for example:
                `other = torch.tanh`. Defaults to ``self.other``.

        Raises:
            ValueError: When ``sigmoid=True`` and ``softmax=True``. Incompatible values.
            TypeError: When ``other`` is not an ``Optional[Callable]``.
            ValueError: When ``self.other=None`` and ``other=None``. Incompatible values.

        """
        if sigmoid and softmax:
            raise ValueError("Incompatible values: sigmoid=True and softmax=True.")
        if other is not None and not callable(other):
            raise TypeError(f"other must be None or callable but is {type(other).__name__}.")

        # convert to float as activation must operate on float tensor
        img = convert_to_tensor(img, track_meta=get_track_meta())
        img_t, *_ = convert_data_type(img, torch.Tensor, dtype=torch.float)
        if sigmoid or self.sigmoid:
            img_t = torch.sigmoid(img_t)
        if softmax or self.softmax:
            img_t = torch.softmax(img_t, dim=self.kwargs.get("dim", 0))

        act_func = self.other if other is None else other
        if act_func is not None:
            img_t = act_func(img_t)
        out, *_ = convert_to_dst_type(img_t, img)
        return out


class AsDiscrete(Transform):
    """
    Convert the input tensor/array into discrete values, possible operations are:

        -  `argmax`.
        -  threshold input value to binary values.
        -  convert input value to One-Hot format (set ``to_one_hot=N``, `N` is the number of classes).
        -  round the value to the closest integer.

    Args:
        argmax: whether to execute argmax function on input data before transform.
            Defaults to ``False``.
        to_onehot: if not None, convert input data into the one-hot format with specified number of classes.
            Defaults to ``None``.
        threshold: if not None, threshold the float values to int number 0 or 1 with specified threshold.
            Defaults to ``None``.
        rounding: if not None, round the data according to the specified option,
            available options: ["torchrounding"].
        kwargs: additional parameters to `torch.argmax`, `monai.networks.one_hot`.
            currently ``dim``, ``keepdim``, ``dtype`` are supported, unrecognized parameters will be ignored.
            These default to ``0``, ``True``, ``torch.float`` respectively.

    Example:

        >>> transform = AsDiscrete(argmax=True)
        >>> print(transform(np.array([[[0.0, 1.0]], [[2.0, 3.0]]])))
        # [[[1.0, 1.0]]]

        >>> transform = AsDiscrete(threshold=0.6)
        >>> print(transform(np.array([[[0.0, 0.5], [0.8, 3.0]]])))
        # [[[0.0, 0.0], [1.0, 1.0]]]

        >>> transform = AsDiscrete(argmax=True, to_onehot=2, threshold=0.5)
        >>> print(transform(np.array([[[0.0, 1.0]], [[2.0, 3.0]]])))
        # [[[0.0, 0.0]], [[1.0, 1.0]]]

    """

    backend = [TransformBackends.TORCH]

    def __init__(
        self,
        argmax: bool = False,
        to_onehot: int | None = None,
        threshold: float | None = None,
        rounding: str | None = None,
        **kwargs,
    ) -> None:
        self.argmax = argmax
        if isinstance(to_onehot, bool):  # for backward compatibility
            raise ValueError("`to_onehot=True/False` is deprecated, please use `to_onehot=num_classes` instead.")
        self.to_onehot = to_onehot
        self.threshold = threshold
        self.rounding = rounding
        self.kwargs = kwargs

    def __call__(
        self,
        img: NdarrayOrTensor,
        argmax: bool | None = None,
        to_onehot: int | None = None,
        threshold: float | None = None,
        rounding: str | None = None,
    ) -> NdarrayOrTensor:
        """
        Args:
            img: the input tensor data to convert, if no channel dimension when converting to `One-Hot`,
                will automatically add it.
            argmax: whether to execute argmax function on input data before transform.
                Defaults to ``self.argmax``.
            to_onehot: if not None, convert input data into the one-hot format with specified number of classes.
                Defaults to ``self.to_onehot``.
            threshold: if not None, threshold the float values to int number 0 or 1 with specified threshold value.
                Defaults to ``self.threshold``.
            rounding: if not None, round the data according to the specified option,
                available options: ["torchrounding"].

        """
        if isinstance(to_onehot, bool):
            raise ValueError("`to_onehot=True/False` is deprecated, please use `to_onehot=num_classes` instead.")
        img = convert_to_tensor(img, track_meta=get_track_meta())
        img_t, *_ = convert_data_type(img, torch.Tensor)
        argmax = self.argmax if argmax is None else argmax
        if argmax:
            img_t = torch.argmax(img_t, dim=self.kwargs.get("dim", 0), keepdim=self.kwargs.get("keepdim", True))

        to_onehot = self.to_onehot if to_onehot is None else to_onehot
        if to_onehot is not None:
            if not isinstance(to_onehot, int):
                raise ValueError(f"the number of classes for One-Hot must be an integer, got {type(to_onehot)}.")
            img_t = one_hot(
                img_t, num_classes=to_onehot, dim=self.kwargs.get("dim", 0), dtype=self.kwargs.get("dtype", torch.float)
            )

        threshold = self.threshold if threshold is None else threshold
        if threshold is not None:
            img_t = img_t >= threshold

        rounding = self.rounding if rounding is None else rounding
        if rounding is not None:
            look_up_option(rounding, ["torchrounding"])
            img_t = torch.round(img_t)

        img, *_ = convert_to_dst_type(img_t, img, dtype=self.kwargs.get("dtype", torch.float))
        return img


class KeepLargestConnectedComponent(Transform):
    """
    Keeps only the largest connected component in the image.
    This transform can be used as a post-processing step to clean up over-segment areas in model output.

    The input is assumed to be a channel-first PyTorch Tensor:
      1) For not OneHot format data, the values correspond to expected labels,
      0 will be treated as background and the over-segment pixels will be set to 0.
      2) For OneHot format data, the values should be 0, 1 on each labels,
      the over-segment pixels will be set to 0 in its channel.

    For example:
    Use with applied_labels=[1], is_onehot=False, connectivity=1::

       [1, 0, 0]         [0, 0, 0]
       [0, 1, 1]    =>   [0, 1 ,1]
       [0, 1, 1]         [0, 1, 1]

    Use with applied_labels=[1, 2], is_onehot=False, independent=False, connectivity=1::

      [0, 0, 1, 0 ,0]           [0, 0, 1, 0 ,0]
      [0, 2, 1, 1 ,1]           [0, 2, 1, 1 ,1]
      [1, 2, 1, 0 ,0]    =>     [1, 2, 1, 0 ,0]
      [1, 2, 0, 1 ,0]           [1, 2, 0, 0 ,0]
      [2, 2, 0, 0 ,2]           [2, 2, 0, 0 ,0]

    Use with applied_labels=[1, 2], is_onehot=False, independent=True, connectivity=1::

      [0, 0, 1, 0 ,0]           [0, 0, 1, 0 ,0]
      [0, 2, 1, 1 ,1]           [0, 2, 1, 1 ,1]
      [1, 2, 1, 0 ,0]    =>     [0, 2, 1, 0 ,0]
      [1, 2, 0, 1 ,0]           [0, 2, 0, 0 ,0]
      [2, 2, 0, 0 ,2]           [2, 2, 0, 0 ,0]

    Use with applied_labels=[1, 2], is_onehot=False, independent=False, connectivity=2::

      [0, 0, 1, 0 ,0]           [0, 0, 1, 0 ,0]
      [0, 2, 1, 1 ,1]           [0, 2, 1, 1 ,1]
      [1, 2, 1, 0 ,0]    =>     [1, 2, 1, 0 ,0]
      [1, 2, 0, 1 ,0]           [1, 2, 0, 1 ,0]
      [2, 2, 0, 0 ,2]           [2, 2, 0, 0 ,2]

    """

    backend = [TransformBackends.NUMPY, TransformBackends.CUPY]

    def __init__(
        self,
        applied_labels: Sequence[int] | int | None = None,
        is_onehot: bool | None = None,
        independent: bool = True,
        connectivity: int | None = None,
        num_components: int = 1,
    ) -> None:
        """
        Args:
            applied_labels: Labels for applying the connected component analysis on.
                If given, voxels whose value is in this list will be analyzed.
                If `None`, all non-zero values will be analyzed.
            is_onehot: if `True`, treat the input data as OneHot format data, otherwise, not OneHot format data.
                default to None, which treats multi-channel data as OneHot and single channel data as not OneHot.
            independent: whether to treat ``applied_labels`` as a union of foreground labels.
                If ``True``, the connected component analysis will be performed on each foreground label independently
                and return the intersection of the largest components.
                If ``False``, the analysis will be performed on the union of foreground labels.
                default is `True`.
            connectivity: Maximum number of orthogonal hops to consider a pixel/voxel as a neighbor.
                Accepted values are ranging from  1 to input.ndim. If ``None``, a full
                connectivity of ``input.ndim`` is used. for more details:
                https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.label.
            num_components: The number of largest components to preserve.

        """
        super().__init__()
        self.applied_labels = ensure_tuple(applied_labels) if applied_labels is not None else None
        self.is_onehot = is_onehot
        self.independent = independent
        self.connectivity = connectivity
        self.num_components = num_components

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Args:
            img: shape must be (C, spatial_dim1[, spatial_dim2, ...]).

        Returns:
            An array with shape (C, spatial_dim1[, spatial_dim2, ...]).
        """
        is_onehot = img.shape[0] > 1 if self.is_onehot is None else self.is_onehot
        if self.applied_labels is not None:
            applied_labels = self.applied_labels
        else:
            applied_labels = tuple(get_unique_labels(img, is_onehot, discard=0))
        img = convert_to_tensor(img, track_meta=get_track_meta())
        img_: torch.Tensor = convert_to_tensor(img, track_meta=False)
        if self.independent:
            for i in applied_labels:
                foreground = img_[i] > 0 if is_onehot else img_[0] == i
                mask = get_largest_connected_component_mask(foreground, self.connectivity, self.num_components)
                if is_onehot:
                    img_[i][foreground != mask] = 0
                else:
                    img_[0][foreground != mask] = 0
            return convert_to_dst_type(img_, dst=img)[0]
        if not is_onehot:  # not one-hot, union of labels
            labels, *_ = convert_to_dst_type(applied_labels, dst=img_, wrap_sequence=True)
            foreground = (img_[..., None] == labels).any(-1)[0]
            mask = get_largest_connected_component_mask(foreground, self.connectivity, self.num_components)
            img_[0][foreground != mask] = 0
            return convert_to_dst_type(img_, dst=img)[0]
        # one-hot, union of labels
        foreground = (img_[applied_labels, ...] == 1).any(0)
        mask = get_largest_connected_component_mask(foreground, self.connectivity, self.num_components)
        for i in applied_labels:
            img_[i][foreground != mask] = 0
        return convert_to_dst_type(img_, dst=img)[0]


class RemoveSmallObjects(Transform):
    """
    Use `skimage.morphology.remove_small_objects` to remove small objects from images.
    See: https://scikit-image.org/docs/dev/api/skimage.morphology.html#remove-small-objects.

    Data should be one-hotted.

    Args:
        min_size: objects smaller than this size (in number of voxels; or surface area/volume value
            in whatever units your image is if by_measure is True) are removed.
        connectivity: Maximum number of orthogonal hops to consider a pixel/voxel as a neighbor.
            Accepted values are ranging from  1 to input.ndim. If ``None``, a full
            connectivity of ``input.ndim`` is used. For more details refer to linked scikit-image
            documentation.
        independent_channels: Whether or not to consider channels as independent. If true, then
            conjoining islands from different labels will be removed if they are below the threshold.
            If false, the overall size islands made from all non-background voxels will be used.
        by_measure: Whether the specified min_size is in number of voxels. if this is True then min_size
            represents a surface area or volume value of whatever units your image is in (mm^3, cm^2, etc.)
            default is False. e.g. if min_size is 3, by_measure is True and the units of your data is mm,
            objects smaller than 3mm^3 are removed.
        pixdim: the pixdim of the input image. if a single number, this is used for all axes.
            If a sequence of numbers, the length of the sequence must be equal to the image dimensions.

    Example::

        .. code-block:: python

            from monai.transforms import RemoveSmallObjects, Spacing, Compose
            from monai.data import MetaTensor

            data1 = torch.tensor([[[0, 0, 0, 0, 0], [0, 1, 1, 0, 1], [0, 0, 0, 1, 1]]])
            affine = torch.as_tensor([[2,0,0,0],
                                      [0,1,0,0],
                                      [0,0,1,0],
                                      [0,0,0,1]], dtype=torch.float64)
            data2 = MetaTensor(data1, affine=affine)

            # remove objects smaller than 3mm^3, input is MetaTensor
            trans = RemoveSmallObjects(min_size=3, by_measure=True)
            out = trans(data2)
            # remove objects smaller than 3mm^3, input is not MetaTensor
            trans = RemoveSmallObjects(min_size=3, by_measure=True, pixdim=(2, 1, 1))
            out = trans(data1)

            # remove objects smaller than 3 (in pixel)
            trans = RemoveSmallObjects(min_size=3)
            out = trans(data2)

            # If the affine of the data is not identity, you can also add Spacing before.
            trans = Compose([
                Spacing(pixdim=(1, 1, 1)),
                RemoveSmallObjects(min_size=3)
            ])

    """

    backend = [TransformBackends.NUMPY]

    def __init__(
        self,
        min_size: int = 64,
        connectivity: int = 1,
        independent_channels: bool = True,
        by_measure: bool = False,
        pixdim: Sequence[float] | float | np.ndarray | None = None,
    ) -> None:
        self.min_size = min_size
        self.connectivity = connectivity
        self.independent_channels = independent_channels
        self.by_measure = by_measure
        self.pixdim = pixdim

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Args:
            img: shape must be (C, spatial_dim1[, spatial_dim2, ...]). Data
                should be one-hotted.

        Returns:
            An array with shape (C, spatial_dim1[, spatial_dim2, ...]).
        """

        return remove_small_objects(
            img, self.min_size, self.connectivity, self.independent_channels, self.by_measure, self.pixdim
        )


class LabelFilter(Transform):
    """
    This transform filters out labels and can be used as a processing step to view only certain labels.

    The list of applied labels defines which labels will be kept.

    Note:
        All labels which do not match the `applied_labels` are set to the background label (0).

    For example:

    Use LabelFilter with applied_labels=[1, 5, 9]::

        [1, 2, 3]         [1, 0, 0]
        [4, 5, 6]    =>   [0, 5 ,0]
        [7, 8, 9]         [0, 0, 9]
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, applied_labels: Iterable[int] | int) -> None:
        """
        Initialize the LabelFilter class with the labels to filter on.

        Args:
            applied_labels: Label(s) to filter on.
        """
        self.applied_labels = ensure_tuple(applied_labels)

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Filter the image on the `applied_labels`.

        Args:
            img: Pytorch tensor or numpy array of any shape.

        Raises:
            NotImplementedError: The provided image was not a Pytorch Tensor or numpy array.

        Returns:
            Pytorch tensor or numpy array of the same shape as the input.
        """
        if not isinstance(img, (np.ndarray, torch.Tensor)):
            raise NotImplementedError(f"{self.__class__} can not handle data of type {type(img)}.")

        if isinstance(img, torch.Tensor):
            img = convert_to_tensor(img, track_meta=get_track_meta())
            img_ = convert_to_tensor(img, track_meta=False)
            if hasattr(torch, "isin"):  # `isin` is new in torch 1.10.0
                appl_lbls = torch.as_tensor(self.applied_labels, device=img_.device)
                out = torch.where(torch.isin(img_, appl_lbls), img_, torch.tensor(0.0).to(img_))
                return convert_to_dst_type(out, dst=img)[0]
            out: NdarrayOrTensor = self(img_.detach().cpu().numpy())  # type: ignore
            out = convert_to_dst_type(out, img)[0]  # type: ignore
            return out
        return np.asarray(np.where(np.isin(img, self.applied_labels), img, 0))


class FillHoles(Transform):
    r"""
    This transform fills holes in the image and can be used to remove artifacts inside segments.

    An enclosed hole is defined as a background pixel/voxel which is only enclosed by a single class.
    The definition of enclosed can be defined with the connectivity parameter::

        1-connectivity     2-connectivity     diagonal connection close-up

             [ ]           [ ]  [ ]  [ ]             [ ]
              |               \  |  /                 |  <- hop 2
        [ ]--[x]--[ ]      [ ]--[x]--[ ]        [x]--[ ]
              |               /  |  \             hop 1
             [ ]           [ ]  [ ]  [ ]

    It is possible to define for which labels the hole filling should be applied.
    The input image is assumed to be a PyTorch Tensor or numpy array with shape [C, spatial_dim1[, spatial_dim2, ...]].
    If C = 1, then the values correspond to expected labels.
    If C > 1, then a one-hot-encoding is expected where the index of C matches the label indexing.

    Note:

        The label 0 will be treated as background and the enclosed holes will be set to the neighboring class label.

        The performance of this method heavily depends on the number of labels.
        It is a bit faster if the list of `applied_labels` is provided.
        Limiting the number of `applied_labels` results in a big decrease in processing time.

    For example:

        Use FillHoles with default parameters::

            [1, 1, 1, 2, 2, 2, 3, 3]         [1, 1, 1, 2, 2, 2, 3, 3]
            [1, 0, 1, 2, 0, 0, 3, 0]    =>   [1, 1 ,1, 2, 0, 0, 3, 0]
            [1, 1, 1, 2, 2, 2, 3, 3]         [1, 1, 1, 2, 2, 2, 3, 3]

        The hole in label 1 is fully enclosed and therefore filled with label 1.
        The background label near label 2 and 3 is not fully enclosed and therefore not filled.
    """

    backend = [TransformBackends.NUMPY]

    def __init__(self, applied_labels: Iterable[int] | int | None = None, connectivity: int | None = None) -> None:
        """
        Initialize the connectivity and limit the labels for which holes are filled.

        Args:
            applied_labels: Labels for which to fill holes. Defaults to None, that is filling holes for all labels.
            connectivity: Maximum number of orthogonal hops to consider a pixel/voxel as a neighbor.
                Accepted values are ranging from  1 to input.ndim. Defaults to a full connectivity of ``input.ndim``.
        """
        super().__init__()
        self.applied_labels = ensure_tuple(applied_labels) if applied_labels else None
        self.connectivity = connectivity

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Fill the holes in the provided image.

        Note:
            The value 0 is assumed as background label.

        Args:
            img: Pytorch Tensor or numpy array of shape [C, spatial_dim1[, spatial_dim2, ...]].

        Raises:
            NotImplementedError: The provided image was not a Pytorch Tensor or numpy array.

        Returns:
            Pytorch Tensor or numpy array of shape [C, spatial_dim1[, spatial_dim2, ...]].
        """
        img = convert_to_tensor(img, track_meta=get_track_meta())
        img_np, *_ = convert_data_type(img, np.ndarray)
        out_np: np.ndarray = fill_holes(img_np, self.applied_labels, self.connectivity)
        out, *_ = convert_to_dst_type(out_np, img)
        return out


class LabelToContour(Transform):
    """
    Return the contour of binary input images that only compose of 0 and 1, with Laplacian kernel
    set as default for edge detection. Typical usage is to plot the edge of label or segmentation output.

    Args:
        kernel_type: the method applied to do edge detection, default is "Laplace".

    Raises:
        NotImplementedError: When ``kernel_type`` is not "Laplace".

    """

    backend = [TransformBackends.TORCH]

    def __init__(self, kernel_type: str = "Laplace") -> None:
        if kernel_type != "Laplace":
            raise NotImplementedError('Currently only kernel_type="Laplace" is supported.')
        self.kernel_type = kernel_type

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Args:
            img: torch tensor data to extract the contour, with shape: [channels, height, width[, depth]]

        Raises:
            ValueError: When ``image`` ndim is not one of [3, 4].

        Returns:
            A torch tensor with the same shape as img, note:
                1. it's the binary classification result of whether a pixel is edge or not.
                2. in order to keep the original shape of mask image, we use padding as default.
                3. the edge detection is just approximate because it defects inherent to Laplace kernel,
                   ideally the edge should be thin enough, but now it has a thickness.

        """
        img = convert_to_tensor(img, track_meta=get_track_meta())
        img_: torch.Tensor = convert_to_tensor(img, track_meta=False)
        spatial_dims = len(img_.shape) - 1
        img_ = img_.unsqueeze(0)  # adds a batch dim
        if spatial_dims == 2:
            kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32)
        elif spatial_dims == 3:
            kernel = -1.0 * torch.ones(3, 3, 3, dtype=torch.float32)
            kernel[1, 1, 1] = 26.0
        else:
            raise ValueError(f"{self.__class__} can only handle 2D or 3D images.")
        contour_img = apply_filter(img_, kernel)
        contour_img.clamp_(min=0.0, max=1.0)
        output, *_ = convert_to_dst_type(contour_img.squeeze(0), img)
        return output


class Ensemble:

    @staticmethod
    def get_stacked_torch(img: Sequence[NdarrayOrTensor] | NdarrayOrTensor) -> torch.Tensor:
        """Get either a sequence or single instance of np.ndarray/torch.Tensor. Return single torch.Tensor."""
        if isinstance(img, Sequence) and isinstance(img[0], np.ndarray):
            img = [torch.as_tensor(i) for i in img]
        elif isinstance(img, np.ndarray):
            img = torch.as_tensor(img)
        out: torch.Tensor = torch.stack(img) if isinstance(img, Sequence) else img  # type: ignore
        return out

    @staticmethod
    def post_convert(img: torch.Tensor, orig_img: Sequence[NdarrayOrTensor] | NdarrayOrTensor) -> NdarrayOrTensor:
        orig_img_ = orig_img[0] if isinstance(orig_img, Sequence) else orig_img
        out, *_ = convert_to_dst_type(img, orig_img_)
        return out


class MeanEnsemble(Ensemble, Transform):
    """
    Execute mean ensemble on the input data.
    The input data can be a list or tuple of PyTorch Tensor with shape: [C[, H, W, D]],
    Or a single PyTorch Tensor with shape: [E, C[, H, W, D]], the `E` dimension represents
    the output data from different models.
    Typically, the input data is model output of segmentation task or classification task.
    And it also can support to add `weights` for the input data.

    Args:
        weights: can be a list or tuple of numbers for input data with shape: [E, C, H, W[, D]].
            or a Numpy ndarray or a PyTorch Tensor data.
            the `weights` will be added to input data from highest dimension, for example:
            1. if the `weights` only has 1 dimension, it will be added to the `E` dimension of input data.
            2. if the `weights` has 2 dimensions, it will be added to `E` and `C` dimensions.
            it's a typical practice to add weights for different classes:
            to ensemble 3 segmentation model outputs, every output has 4 channels(classes),
            so the input data shape can be: [3, 4, H, W, D].
            and add different `weights` for different classes, so the `weights` shape can be: [3, 4].
            for example: `weights = [[1, 2, 3, 4], [4, 3, 2, 1], [1, 1, 1, 1]]`.

    """

    backend = [TransformBackends.TORCH]

    def __init__(self, weights: Sequence[float] | NdarrayOrTensor | None = None) -> None:
        self.weights = torch.as_tensor(weights, dtype=torch.float) if weights is not None else None

    def __call__(self, img: Sequence[NdarrayOrTensor] | NdarrayOrTensor) -> NdarrayOrTensor:
        out_pt = None
        total_weight = 0.0

        for i, pred in enumerate(img):
            pred = torch.as_tensor(pred)
            if out_pt is None:
                out_pt = torch.zeros_like(pred)

            if self.weights is not None:
                weight = self.weights[i].to(pred.device)

            out_pt += pred * weight
            total_weight += weight

        out_pt /= total_weight
        return self.post_convert(out_pt, img)


class VoteEnsemble(Ensemble, Transform):
    """
    Execute vote ensemble on the input data.
    The input data can be a list or tuple of PyTorch Tensor with shape: [C[, H, W, D]],
    Or a single PyTorch Tensor with shape: [E[, C, H, W, D]], the `E` dimension represents
    the output data from different models.
    Typically, the input data is model output of segmentation task or classification task.

    Note:
        This vote transform expects the input data is discrete values. It can be multiple channels
        data in One-Hot format or single channel data. It will vote to select the most common data
        between items.
        The output data has the same shape as every item of the input data.

    Args:
        num_classes: if the input is single channel data instead of One-Hot, we can't get class number
            from channel, need to explicitly specify the number of classes to vote.

    """

    backend = [TransformBackends.TORCH]

    def __init__(self, num_classes: int | None = None) -> None:
        self.num_classes = num_classes

    def __call__(self, img: Sequence[NdarrayOrTensor] | NdarrayOrTensor) -> NdarrayOrTensor:
        img_ = self.get_stacked_torch(img)

        if self.num_classes is not None:
            has_ch_dim = True
            if img_.ndimension() > 1 and img_.shape[1] > 1:
                warnings.warn("no need to specify num_classes for One-Hot format data.")
            else:
                if img_.ndimension() == 1:
                    # if no channel dim, need to remove channel dim after voting
                    has_ch_dim = False
                img_ = one_hot(img_, self.num_classes, dim=1)

        img_ = torch.mean(img_.float(), dim=0)

        if self.num_classes is not None:
            # if not One-Hot, use "argmax" to vote the most common class
            out_pt = torch.argmax(img_, dim=0, keepdim=has_ch_dim)
        else:
            # for One-Hot data, round the float number to 0 or 1
            out_pt = torch.round(img_)
        return self.post_convert(out_pt, img)


class ProbNMS(Transform):
    """
    Performs probability based non-maximum suppression (NMS) on the probabilities map via
    iteratively selecting the coordinate with highest probability and then move it as well
    as its surrounding values. The remove range is determined by the parameter `box_size`.
    If multiple coordinates have the same highest probability, only one of them will be
    selected.

    Args:
        spatial_dims: number of spatial dimensions of the input probabilities map.
            Defaults to 2.
        sigma: the standard deviation for gaussian filter.
            It could be a single value, or `spatial_dims` number of values. Defaults to 0.0.
        prob_threshold: the probability threshold, the function will stop searching if
            the highest probability is no larger than the threshold. The value should be
            no less than 0.0. Defaults to 0.5.
        box_size: the box size (in pixel) to be removed around the pixel with the maximum probability.
            It can be an integer that defines the size of a square or cube,
            or a list containing different values for each dimensions. Defaults to 48.

    Return:
        a list of selected lists, where inner lists contain probability and coordinates.
        For example, for 3D input, the inner lists are in the form of [probability, x, y, z].

    Raises:
        ValueError: When ``prob_threshold`` is less than 0.0.
        ValueError: When ``box_size`` is a list or tuple, and its length is not equal to `spatial_dims`.
        ValueError: When ``box_size`` has a less than 1 value.

    """

    backend = [TransformBackends.NUMPY]

    def __init__(
        self,
        spatial_dims: int = 2,
        sigma: Sequence[float] | float | Sequence[torch.Tensor] | torch.Tensor = 0.0,
        prob_threshold: float = 0.5,
        box_size: int | Sequence[int] = 48,
    ) -> None:
        self.sigma = sigma
        self.spatial_dims = spatial_dims
        if self.sigma != 0:
            self.filter = GaussianFilter(spatial_dims=spatial_dims, sigma=sigma)
        if prob_threshold < 0:
            raise ValueError("prob_threshold should be no less than 0.0.")
        self.prob_threshold = prob_threshold
        if isinstance(box_size, int):
            self.box_size = np.asarray([box_size] * spatial_dims)
        elif len(box_size) != spatial_dims:
            raise ValueError("the sequence length of box_size should be the same as spatial_dims.")
        else:
            self.box_size = np.asarray(box_size)
        if self.box_size.min() <= 0:
            raise ValueError("box_size should be larger than 0.")

        self.box_lower_bd = self.box_size // 2
        self.box_upper_bd = self.box_size - self.box_lower_bd

    def __call__(self, prob_map: NdarrayOrTensor):
        """
        prob_map: the input probabilities map, it must have shape (H[, W, ...]).
        """
        if self.sigma != 0:
            if not isinstance(prob_map, torch.Tensor):
                prob_map = torch.as_tensor(prob_map, dtype=torch.float)
            self.filter.to(prob_map.device)
            prob_map = self.filter(prob_map)

        prob_map_shape = prob_map.shape

        outputs = []
        while prob_map.max() > self.prob_threshold:
            max_idx = unravel_index(prob_map.argmax(), prob_map_shape)
            prob_max = prob_map[tuple(max_idx)]
            max_idx = max_idx.cpu().numpy() if isinstance(max_idx, torch.Tensor) else max_idx
            prob_max = prob_max.item() if isinstance(prob_max, torch.Tensor) else prob_max
            outputs.append([prob_max] + list(max_idx))

            idx_min_range = (max_idx - self.box_lower_bd).clip(0, None)
            idx_max_range = (max_idx + self.box_upper_bd).clip(None, prob_map_shape)
            # for each dimension, set values during index ranges to 0
            slices = tuple(slice(idx_min_range[i], idx_max_range[i]) for i in range(self.spatial_dims))
            prob_map[slices] = 0

        return outputs


class Invert(Transform):
    """
    Utility transform to automatically invert the previously applied transforms.
    """

    backend = [TransformBackends.TORCH]

    def __init__(
        self,
        transform: InvertibleTransform | None = None,
        nearest_interp: bool | Sequence[bool] = True,
        device: str | torch.device | None = None,
        post_func: Callable | None = None,
        to_tensor: bool | Sequence[bool] = True,
    ) -> None:
        """
        Args:
            transform: the previously applied transform.
            nearest_interp: whether to use `nearest` interpolation mode when inverting the spatial transforms,
                default to `True`. If `False`, use the same interpolation mode as the original transform.
            device: move the inverted results to a target device before `post_func`, default to `None`.
            post_func: postprocessing for the inverted result, should be a callable function.
            to_tensor: whether to convert the inverted data into PyTorch Tensor first, default to `True`.
        """
        if not isinstance(transform, InvertibleTransform):
            raise ValueError("transform is not invertible, can't invert transform for the data.")
        self.transform = transform
        self.nearest_interp = nearest_interp
        self.device = device
        self.post_func = post_func
        self.to_tensor = to_tensor
        self._totensor = ToTensor()

    def __call__(self, data):
        if not isinstance(data, MetaTensor):
            return data

        if self.nearest_interp:
            data.applied_operations = convert_applied_interp_mode(
                trans_info=data.applied_operations, mode="nearest", align_corners=None
            )

        data = data.detach()
        inverted = self.transform.inverse(data)
        if self.to_tensor and not isinstance(inverted, MetaTensor):
            inverted = self._totensor(inverted)
        if isinstance(inverted, torch.Tensor):
            inverted = inverted.to(device=self.device)
        if callable(self.post_func):
            inverted = self.post_func(inverted)
        return inverted


class SobelGradients(Transform):
    """Calculate Sobel gradients of a grayscale image with the shape of CxH[xWxDx...] or BxH[xWxDx...].

    Args:
        kernel_size: the size of the Sobel kernel. Defaults to 3.
        spatial_axes: the axes that define the direction of the gradient to be calculated. It calculate the gradient
            along each of the provide axis. By default it calculate the gradient for all spatial axes.
        normalize_kernels: if normalize the Sobel kernel to provide proper gradients. Defaults to True.
        normalize_gradients: if normalize the output gradient to 0 and 1. Defaults to False.
        padding_mode: the padding mode of the image when convolving with Sobel kernels. Defaults to `"reflect"`.
            Acceptable values are ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
            See ``torch.nn.Conv1d()`` for more information.
        dtype: kernel data type (torch.dtype). Defaults to `torch.float32`.

    """

    backend = [TransformBackends.TORCH]

    def __init__(
        self,
        kernel_size: int = 3,
        spatial_axes: Sequence[int] | int | None = None,
        normalize_kernels: bool = True,
        normalize_gradients: bool = False,
        padding_mode: str = "reflect",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.padding = padding_mode
        self.spatial_axes = spatial_axes
        self.normalize_kernels = normalize_kernels
        self.normalize_gradients = normalize_gradients
        self.kernel_diff, self.kernel_smooth = self._get_kernel(kernel_size, dtype)

    def _get_kernel(self, size, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        if size < 3:
            raise ValueError(f"Sobel kernel size should be at least three. {size} was given.")
        if size % 2 == 0:
            raise ValueError(f"Sobel kernel size should be an odd number. {size} was given.")

        kernel_diff = torch.tensor([[[-1, 0, 1]]], dtype=dtype)
        kernel_smooth = torch.tensor([[[1, 2, 1]]], dtype=dtype)
        kernel_expansion = torch.tensor([[[1, 2, 1]]], dtype=dtype)

        if self.normalize_kernels:
            if not dtype.is_floating_point:
                raise ValueError(
                    f"`dtype` for Sobel kernel should be floating point when `normalize_kernel==True`. {dtype} was given."
                )
            kernel_diff /= 2.0
            kernel_smooth /= 4.0
            kernel_expansion /= 4.0

        # Expand the kernel to larger size than 3
        expand = (size - 3) // 2
        for _ in range(expand):
            kernel_diff = F.conv1d(kernel_diff, kernel_expansion, padding=2)
            kernel_smooth = F.conv1d(kernel_smooth, kernel_expansion, padding=2)

        return kernel_diff.squeeze(), kernel_smooth.squeeze()

    def __call__(self, image: NdarrayOrTensor) -> torch.Tensor:
        image_tensor = convert_to_tensor(image, track_meta=get_track_meta())

        # Check/set spatial axes
        n_spatial_dims = image_tensor.ndim - 1  # excluding the channel dimension
        valid_spatial_axes = list(range(n_spatial_dims)) + list(range(-n_spatial_dims, 0))

        # Check gradient axes to be valid
        if self.spatial_axes is None:
            spatial_axes = list(range(n_spatial_dims))
        else:
            invalid_axis = set(ensure_tuple(self.spatial_axes)) - set(valid_spatial_axes)
            if invalid_axis:
                raise ValueError(
                    f"The provide axes to calculate gradient is not valid: {invalid_axis}. "
                    f"The image has {n_spatial_dims} spatial dimensions so it should be: {valid_spatial_axes}."
                )
            spatial_axes = [ax % n_spatial_dims if ax < 0 else ax for ax in ensure_tuple(self.spatial_axes)]

        # Add batch dimension for separable_filtering
        image_tensor = image_tensor.unsqueeze(0)

        # Get the Sobel kernels
        kernel_diff = self.kernel_diff.to(image_tensor.device)
        kernel_smooth = self.kernel_smooth.to(image_tensor.device)

        # Calculate gradient
        grad_list = []
        for ax in spatial_axes:
            kernels = [kernel_smooth] * n_spatial_dims
            kernels[ax] = kernel_diff
            grad = separable_filtering(image_tensor, kernels, mode=self.padding)
            if self.normalize_gradients:
                grad_min = grad.min()
                if grad_min != grad.max():
                    grad -= grad_min
                grad_max = grad.max()
                if grad_max > 0:
                    grad /= grad_max
            grad_list.append(grad)

        grads = torch.cat(grad_list, dim=1)

        # Remove batch dimension and convert the gradient type to be the same as input image
        grads = convert_to_dst_type(grads.squeeze(0), image_tensor)[0]

        return grads


class DistanceTransformEDT(Transform):
    """
    Applies the Euclidean distance transform on the input.
    Either GPU based with CuPy / cuCIM or CPU based with scipy.
    To use the GPU implementation, make sure cuCIM is available and that the data is a `torch.tensor` on a GPU device.

    Note that the results of the libraries can differ, so stick to one if possible.
    For details, check out the `SciPy`_ and `cuCIM`_ documentation and / or :func:`monai.transforms.utils.distance_transform_edt`.

    .. _SciPy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html
    .. _cuCIM: https://docs.rapids.ai/api/cucim/nightly/api/#cucim.core.operations.morphology.distance_transform_edt
    """

    backend = [TransformBackends.NUMPY, TransformBackends.CUPY]

    def __init__(self, sampling: None | float | list[float] = None) -> None:
        super().__init__()
        self.sampling = sampling

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Args:
            img: Input image on which the distance transform shall be run.
                Has to be a channel first array, must have shape: (num_channels, H, W [,D]).
                Can be of any type but will be converted into binary: 1 wherever image equates to True, 0 elsewhere.
                Input gets passed channel-wise to the distance-transform, thus results from this function will differ
                from directly calling ``distance_transform_edt()`` in CuPy or SciPy.
            sampling: Spacing of elements along each dimension. If a sequence, must be of length equal to the input rank -1;
                if a single number, this is used for all axes. If not specified, a grid spacing of unity is implied.

        Returns:
            An array with the same shape and data type as img
        """
        return distance_transform_edt(img=img, sampling=self.sampling)  # type: ignore
