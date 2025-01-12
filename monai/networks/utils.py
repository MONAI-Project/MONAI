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
Utilities and types for defining networks, these depend on PyTorch.
"""

from __future__ import annotations

import io
import re
import tempfile
import warnings
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from monai.apps.utils import get_logger
from monai.config import PathLike
from monai.utils.misc import ensure_tuple, save_obj, set_determinism
from monai.utils.module import look_up_option, optional_import
from monai.utils.type_conversion import convert_to_dst_type, convert_to_tensor

onnx, _ = optional_import("onnx")
onnxreference, _ = optional_import("onnx.reference")
onnxruntime, _ = optional_import("onnxruntime")
polygraphy, polygraphy_imported = optional_import("polygraphy")
torch_tensorrt, _ = optional_import("torch_tensorrt", "1.4.0")

__all__ = [
    "one_hot",
    "predict_segmentation",
    "normalize_transform",
    "to_norm_affine",
    "CastTempType",
    "normal_init",
    "icnr_init",
    "pixelshuffle",
    "eval_mode",
    "train_mode",
    "get_state_dict",
    "copy_model_state",
    "save_state",
    "convert_to_onnx",
    "convert_to_torchscript",
    "convert_to_trt",
    "meshgrid_ij",
    "meshgrid_xy",
    "replace_modules",
    "replace_modules_temp",
    "look_up_named_module",
    "set_named_module",
    "has_nvfuser_instance_norm",
    "get_profile_shapes",
]

logger = get_logger(module_name=__name__)

_has_nvfuser = None


def get_profile_shapes(input_shape: Sequence[int], dynamic_batchsize: Sequence[int] | None):
    """
    Given a sample input shape, calculate min/opt/max shapes according to dynamic_batchsize.
    """

    def scale_batch_size(input_shape: Sequence[int], scale_num: int):
        scale_shape = [*input_shape]
        scale_shape[0] = scale_num
        return scale_shape

    # Use the dynamic batchsize range to generate the min, opt and max model input shape
    if dynamic_batchsize:
        min_input_shape = scale_batch_size(input_shape, dynamic_batchsize[0])
        opt_input_shape = scale_batch_size(input_shape, dynamic_batchsize[1])
        max_input_shape = scale_batch_size(input_shape, dynamic_batchsize[2])
    else:
        min_input_shape = opt_input_shape = max_input_shape = input_shape
    return min_input_shape, opt_input_shape, max_input_shape


def has_nvfuser_instance_norm():
    """whether the current environment has InstanceNorm3dNVFuser
    https://github.com/NVIDIA/apex/blob/23.05-devel/apex/normalization/instance_norm.py#L15-L16
    """
    global _has_nvfuser
    if _has_nvfuser is not None:
        return _has_nvfuser

    _, _has_nvfuser = optional_import("apex.normalization", name="InstanceNorm3dNVFuser")
    if not _has_nvfuser:
        return False
    try:
        import importlib

        importlib.import_module("instance_norm_nvfuser_cuda")
    except ImportError:
        _has_nvfuser = False
    return _has_nvfuser


def look_up_named_module(name: str, mod, print_all_options=False):
    """
    get the named module in `mod` by the attribute name,
    for example ``look_up_named_module(net, "features.3.1.attn")``

    Args:
        name: a string representing the module attribute.
        mod: a pytorch module to be searched (in ``mod.named_modules()``).
        print_all_options: whether to print all named modules when `name` is not found in `mod`. Defaults to False.

    Returns:
        the corresponding pytorch module's subcomponent such as ``net.features[3][1].attn``
    """
    name_str = look_up_option(
        name, {n[0] for n in mod.named_modules()}, default=None, print_all_options=print_all_options
    )
    if name_str is None:
        return None
    if name_str == "":
        return mod
    for n in name_str.split("."):
        if n.isdigit():
            mod = mod[int(n)]
        else:
            n = look_up_option(n, {item[0] for item in mod.named_modules()}, default=None, print_all_options=False)
            if n is None:
                return None
            mod = getattr(mod, n)
    return mod


def set_named_module(mod, name: str, new_layer):
    """
    look up `name` in `mod` and replace the layer with `new_layer`, return the updated `mod`.

    Args:
        mod: a pytorch module to be updated.
        name: a string representing the target module attribute.
        new_layer: a new module replacing the corresponding layer at ``mod.name``.

    Returns:
        an updated ``mod``

    See also: :py:func:`monai.networks.utils.look_up_named_module`.
    """
    mods_attr = name.rsplit(".", 1)
    submods, attr = mods_attr if len(mods_attr) == 2 else ("", name)
    if not attr:
        return new_layer
    _mod = look_up_named_module(submods, mod)
    setattr(_mod, attr, new_layer)
    return mod


def one_hot(labels: torch.Tensor, num_classes: int, dtype: torch.dtype = torch.float, dim: int = 1) -> torch.Tensor:
    """
    For every value v in `labels`, the value in the output will be either 1 or 0. Each vector along the `dim`-th
    dimension has the "one-hot" format, i.e., it has a total length of `num_classes`,
    with a one and `num_class-1` zeros.
    Note that this will include the background label, thus a binary mask should be treated as having two classes.

    Args:
        labels: input tensor of integers to be converted into the 'one-hot' format. Internally `labels` will be
            converted into integers `labels.long()`.
        num_classes: number of output channels, the corresponding length of `labels[dim]` will be converted to
            `num_classes` from `1`.
        dtype: the data type of the output one_hot label.
        dim: the dimension to be converted to `num_classes` channels from `1` channel, should be non-negative number.

    Example:

    For a tensor `labels` of dimensions [B]1[spatial_dims], return a tensor of dimensions `[B]N[spatial_dims]`
    when `num_classes=N` number of classes and `dim=1`.

    .. code-block:: python

        from monai.networks.utils import one_hot
        import torch

        a = torch.randint(0, 2, size=(1, 2, 2, 2))
        out = one_hot(a, num_classes=2, dim=0)
        print(out.shape)  # torch.Size([2, 2, 2, 2])

        a = torch.randint(0, 2, size=(2, 1, 2, 2, 2))
        out = one_hot(a, num_classes=2, dim=1)
        print(out.shape)  # torch.Size([2, 2, 2, 2, 2])

    """

    # if `dim` is bigger, add singleton dim at the end
    if labels.ndim < dim + 1:
        shape = list(labels.shape) + [1] * (dim + 1 - len(labels.shape))
        labels = torch.reshape(labels, shape)

    sh = list(labels.shape)

    if sh[dim] != 1:
        raise AssertionError("labels should have a channel with length equal to one.")

    sh[dim] = num_classes

    o = torch.zeros(size=sh, dtype=dtype, device=labels.device)
    labels = o.scatter_(dim=dim, index=labels.long(), value=1)

    return labels


def predict_segmentation(logits: torch.Tensor, mutually_exclusive: bool = False, threshold: float = 0.0) -> Any:
    """
    Given the logits from a network, computing the segmentation by thresholding all values above 0
    if multi-labels task, computing the `argmax` along the channel axis if multi-classes task,
    logits has shape `BCHW[D]`.

    Args:
        logits: raw data of model output.
        mutually_exclusive: if True, `logits` will be converted into a binary matrix using
            a combination of argmax, which is suitable for multi-classes task. Defaults to False.
        threshold: thresholding the prediction values if multi-labels task.
    """
    if not mutually_exclusive:
        return (logits >= threshold).int()
    if logits.shape[1] == 1:
        warnings.warn("single channel prediction, `mutually_exclusive=True` ignored, use threshold instead.")
        return (logits >= threshold).int()
    return logits.argmax(1, keepdim=True)


def normalize_transform(
    shape,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    align_corners: bool = False,
    zero_centered: bool = False,
) -> torch.Tensor:
    """
    Compute an affine matrix according to the input shape.
    The transform normalizes the homogeneous image coordinates to the
    range of `[-1, 1]`.  Currently the following source coordinates are supported:

        - `align_corners=False`, `zero_centered=False`, normalizing from ``[-0.5, d-0.5]``.
        - `align_corners=True`, `zero_centered=False`, normalizing from ``[0, d-1]``.
        - `align_corners=False`, `zero_centered=True`, normalizing from ``[-(d-1)/2, (d-1)/2]``.
        - `align_corners=True`, `zero_centered=True`, normalizing from ``[-d/2, d/2]``.

    Args:
        shape: input spatial shape, a sequence of integers.
        device: device on which the returned affine will be allocated.
        dtype: data type of the returned affine
        align_corners: if True, consider -1 and 1 to refer to the centers of the
            corner pixels rather than the image corners.
            See also: https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample
        zero_centered: whether the coordinates are normalized from a zero-centered range, default to `False`.
            Setting this flag and `align_corners` will jointly specify the normalization source range.
    """
    shape = convert_to_tensor(shape, torch.float64, device=device, wrap_sequence=True, track_meta=False)
    norm = shape.clone().detach().to(dtype=torch.float64, device=device)  # no in-place change
    if align_corners:
        norm[norm <= 1.0] = 2.0
        norm = 2.0 / (norm if zero_centered else norm - 1.0)
        norm = torch.diag(torch.cat((norm, torch.ones((1,), dtype=torch.float64, device=device))))
        if not zero_centered:  # else shift is 0
            norm[:-1, -1] = -1.0
    else:
        norm[norm <= 0.0] = 2.0
        norm = 2.0 / (norm - 1.0 if zero_centered else norm)
        norm = torch.diag(torch.cat((norm, torch.ones((1,), dtype=torch.float64, device=device))))
        if not zero_centered:
            norm[:-1, -1] = 1.0 / shape - 1.0
    norm = norm.unsqueeze(0).to(dtype=dtype)
    norm.requires_grad = False
    return norm  # type: ignore


def to_norm_affine(
    affine: torch.Tensor,
    src_size: Sequence[int],
    dst_size: Sequence[int],
    align_corners: bool = False,
    zero_centered: bool = False,
) -> torch.Tensor:
    """
    Given ``affine`` defined for coordinates in the pixel space, compute the corresponding affine
    for the normalized coordinates.

    Args:
        affine: Nxdxd batched square matrix
        src_size: source image spatial shape
        dst_size: target image spatial shape
        align_corners: if True, consider -1 and 1 to refer to the centers of the
            corner pixels rather than the image corners.
            See also: https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample
        zero_centered: whether the coordinates are normalized from a zero-centered range, default to `False`.
            See also: :py:func:`monai.networks.utils.normalize_transform`.

    Raises:
        TypeError: When ``affine`` is not a ``torch.Tensor``.
        ValueError: When ``affine`` is not Nxdxd.
        ValueError: When ``src_size`` or ``dst_size`` dimensions differ from ``affine``.

    """
    if not isinstance(affine, torch.Tensor):
        raise TypeError(f"affine must be a torch.Tensor but is {type(affine).__name__}.")
    if affine.ndimension() != 3 or affine.shape[1] != affine.shape[2]:
        raise ValueError(f"affine must be Nxdxd, got {tuple(affine.shape)}.")
    sr = affine.shape[1] - 1
    if sr != len(src_size) or sr != len(dst_size):
        raise ValueError(f"affine suggests {sr}D, got src={len(src_size)}D, dst={len(dst_size)}D.")

    src_xform = normalize_transform(src_size, affine.device, affine.dtype, align_corners, zero_centered)
    dst_xform = normalize_transform(dst_size, "cpu", affine.dtype, align_corners, zero_centered)
    return src_xform @ affine @ convert_to_dst_type(np.linalg.inv(dst_xform.numpy()), dst=affine)[0]  # monai#5983


def normal_init(
    m, std: float = 0.02, normal_func: Callable[[torch.Tensor, float, float], Any] = torch.nn.init.normal_
) -> None:
    """
    Initialize the weight and bias tensors of `m' and its submodules to values from a normal distribution with a
    stddev of `std'. Weight tensors of convolution and linear modules are initialized with a mean of 0, batch
    norm modules with a mean of 1. The callable `normal_func', used to assign values, should have the same arguments
    as its default normal_(). This can be used with `nn.Module.apply` to visit submodules of a network.
    """
    cname = m.__class__.__name__

    if getattr(m, "weight", None) is not None and (cname.find("Conv") != -1 or cname.find("Linear") != -1):
        normal_func(m.weight.data, 0.0, std)
        if getattr(m, "bias", None) is not None:
            nn.init.constant_(m.bias.data, 0.0)

    elif cname.find("BatchNorm") != -1:
        normal_func(m.weight.data, 1.0, std)
        nn.init.constant_(m.bias.data, 0)


def icnr_init(conv, upsample_factor, init=nn.init.kaiming_normal_):
    """
    ICNR initialization for 2D/3D kernels adapted from Aitken et al.,2017 , "Checkerboard artifact free
    sub-pixel convolution".
    """
    out_channels, in_channels, *dims = conv.weight.shape
    scale_factor = upsample_factor ** len(dims)

    oc2 = int(out_channels / scale_factor)

    kernel = torch.zeros([oc2, in_channels] + dims)
    kernel = init(kernel)
    kernel = kernel.transpose(0, 1)
    kernel = kernel.reshape(oc2, in_channels, -1)
    kernel = kernel.repeat(1, 1, scale_factor)
    kernel = kernel.reshape([in_channels, out_channels] + dims)
    kernel = kernel.transpose(0, 1)
    conv.weight.data.copy_(kernel)


def pixelshuffle(x: torch.Tensor, spatial_dims: int, scale_factor: int) -> torch.Tensor:
    """
    Apply pixel shuffle to the tensor `x` with spatial dimensions `spatial_dims` and scaling factor `scale_factor`.

    See: Shi et al., 2016, "Real-Time Single Image and Video Super-Resolution
    Using a nEfficient Sub-Pixel Convolutional Neural Network."

    See: Aitken et al., 2017, "Checkerboard artifact free sub-pixel convolution".

    Args:
        x: Input tensor
        spatial_dims: number of spatial dimensions, typically 2 or 3 for 2D or 3D
        scale_factor: factor to rescale the spatial dimensions by, must be >=1

    Returns:
        Reshuffled version of `x`.

    Raises:
        ValueError: When input channels of `x` are not divisible by (scale_factor ** spatial_dims)
    """
    dim, factor = spatial_dims, scale_factor
    input_size = list(x.size())
    batch_size, channels = input_size[:2]
    scale_divisor = factor**dim

    if channels % scale_divisor != 0:
        raise ValueError(
            f"Number of input channels ({channels}) must be evenly "
            f"divisible by scale_factor ** dimensions ({factor}**{dim}={scale_divisor})."
        )

    org_channels = int(channels // scale_divisor)
    output_size = [batch_size, org_channels] + [d * factor for d in input_size[2:]]

    indices = list(range(2, 2 + 2 * dim))
    indices = indices[dim:] + indices[:dim]
    permute_indices = [0, 1]
    for idx in range(dim):
        permute_indices.extend(indices[idx::dim])

    x = x.reshape([batch_size, org_channels] + [factor] * dim + input_size[2:])
    x = x.permute(permute_indices).reshape(output_size)
    return x


@contextmanager
def eval_mode(*nets: nn.Module):
    """
    Set network(s) to eval mode and then return to original state at the end.

    Args:
        nets: Input network(s)

    Examples

    .. code-block:: python

        t=torch.rand(1,1,16,16)
        p=torch.nn.Conv2d(1,1,3)
        print(p.training)  # True
        with eval_mode(p):
            print(p.training)  # False
            print(p(t).sum().backward())  # will correctly raise an exception as gradients are calculated
    """

    # Get original state of network(s).
    # Check the training attribute in case it's TensorRT based models which don't have this attribute.
    training = [n for n in nets if hasattr(n, "training") and n.training]

    try:
        # set to eval mode
        with torch.no_grad():
            yield [n.eval() if hasattr(n, "eval") else n for n in nets]
    finally:
        # Return required networks to training
        for n in training:
            if hasattr(n, "train"):
                n.train()


@contextmanager
def train_mode(*nets: nn.Module):
    """
    Set network(s) to train mode and then return to original state at the end.

    Args:
        nets: Input network(s)

    Examples

    .. code-block:: python

        t=torch.rand(1,1,16,16)
        p=torch.nn.Conv2d(1,1,3)
        p.eval()
        print(p.training)  # False
        with train_mode(p):
            print(p.training)  # True
            print(p(t).sum().backward())  # No exception
    """

    # Get original state of network(s)
    # Check the training attribute in case it's TensorRT based models which don't have this attribute.
    eval_list = [n for n in nets if hasattr(n, "training") and (not n.training)]

    try:
        # set to train mode
        with torch.set_grad_enabled(True):
            yield [n.train() if hasattr(n, "train") else n for n in nets]
    finally:
        # Return required networks to eval_list
        for n in eval_list:
            if hasattr(n, "eval"):
                n.eval()


def get_state_dict(obj: torch.nn.Module | Mapping):
    """
    Get the state dict of input object if has `state_dict`, otherwise, return object directly.
    For data parallel model, automatically convert it to regular model first.

    Args:
        obj: input object to check and get the state_dict.

    """
    if isinstance(obj, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        obj = obj.module
    return obj.state_dict() if hasattr(obj, "state_dict") else obj


def copy_model_state(
    dst: torch.nn.Module | Mapping,
    src: torch.nn.Module | Mapping,
    dst_prefix="",
    mapping=None,
    exclude_vars=None,
    inplace=True,
    filter_func=None,
):
    """
    Compute a module state_dict, of which the keys are the same as `dst`. The values of `dst` are overwritten
    by the ones from `src` whenever their keys match. The method provides additional `dst_prefix` for
    the `dst` key when matching them. `mapping` can be a `{"src_key": "dst_key"}` dict, indicating
    `dst[dst_prefix + dst_key] = src[src_key]`.
    This function is mainly to return a model state dict
    for loading the `src` model state into the `dst` model, `src` and `dst` can have different dict keys, but
    their corresponding values normally have the same shape.

    Args:
        dst: a pytorch module or state dict to be updated.
        src: a pytorch module or state dict used to get the values used for the update.
        dst_prefix: `dst` key prefix, so that `dst[dst_prefix + src_key]`
            will be assigned to the value of `src[src_key]`.
        mapping: a `{"src_key": "dst_key"}` dict, indicating that `dst[dst_prefix + dst_key]`
            to be assigned to the value of `src[src_key]`.
        exclude_vars: a regular expression to match the `dst` variable names,
            so that their values are not overwritten by `src`.
        inplace: whether to set the `dst` module with the updated `state_dict` via `load_state_dict`.
            This option is only available when `dst` is a `torch.nn.Module`.
        filter_func: a filter function used to filter the weights to be loaded.
            See 'filter_swinunetr' in "monai.networks.nets.swin_unetr.py".

    Examples:
        .. code-block:: python

            from monai.networks.nets import BasicUNet
            from monai.networks.utils import copy_model_state

            model_a = BasicUNet(in_channels=1, out_channels=4)
            model_b = BasicUNet(in_channels=1, out_channels=2)
            model_a_b, changed, unchanged = copy_model_state(
                model_a, model_b, exclude_vars="conv_0.conv_0", inplace=False)
            # dst model updated: 76 of 82 variables.
            model_a.load_state_dict(model_a_b)
            # <All keys matched successfully>

    Returns: an OrderedDict of the updated `dst` state, the changed, and unchanged keys.

    """
    src_dict = get_state_dict(src)
    dst_dict = OrderedDict(get_state_dict(dst))

    to_skip = {s_key for s_key in src_dict if exclude_vars and re.compile(exclude_vars).search(s_key)}

    # update dst with items from src
    all_keys, updated_keys = list(dst_dict), list()
    for s, val in src_dict.items():
        dst_key = f"{dst_prefix}{s}"
        if dst_key in dst_dict and dst_key not in to_skip and dst_dict[dst_key].shape == val.shape:
            dst_dict[dst_key] = val
            updated_keys.append(dst_key)
    for s in mapping if mapping else {}:
        dst_key = f"{dst_prefix}{mapping[s]}"
        if dst_key in dst_dict and dst_key not in to_skip:
            if dst_dict[dst_key].shape != src_dict[s].shape:
                warnings.warn(f"Param. shape changed from {dst_dict[dst_key].shape} to {src_dict[s].shape}.")
            dst_dict[dst_key] = src_dict[s]
            updated_keys.append(dst_key)
    if filter_func is not None:
        for key, value in src_dict.items():
            new_pair = filter_func(key, value)
            if new_pair is not None and new_pair[0] not in to_skip:
                dst_dict[new_pair[0]] = new_pair[1]
                updated_keys.append(new_pair[0])

    updated_keys = sorted(set(updated_keys))
    unchanged_keys = sorted(set(all_keys).difference(updated_keys))
    logger.info(f"'dst' model updated: {len(updated_keys)} of {len(dst_dict)} variables.")
    if inplace and isinstance(dst, torch.nn.Module):
        if isinstance(dst, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            dst = dst.module
        dst.load_state_dict(dst_dict)  # type: ignore
    return dst_dict, updated_keys, unchanged_keys


def save_state(src: torch.nn.Module | dict, path: PathLike, **kwargs):
    """
    Save the state dict of input source data with PyTorch `save`.
    It can save `nn.Module`, `state_dict`, a dictionary of `nn.Module` or `state_dict`.
    And automatically convert the data parallel module to regular module.
    For example::

        save_state(net, path)
        save_state(net.state_dict(), path)
        save_state({"net": net, "opt": opt}, path)
        net_dp = torch.nn.DataParallel(net)
        save_state(net_dp, path)

    Refer to: https://pytorch.org/ignite/v0.4.8/generated/ignite.handlers.DiskSaver.html.

    Args:
        src: input data to save, can be `nn.Module`, `state_dict`, a dictionary of `nn.Module` or `state_dict`.
        path: target file path to save the input object.
        kwargs: other args for the `save_obj` except for the `obj` and `path`.
            default `func` is `torch.save()`, details of the args:
            https://pytorch.org/docs/stable/generated/torch.save.html.

    """

    ckpt: dict = {}
    if isinstance(src, dict):
        for k, v in src.items():
            ckpt[k] = get_state_dict(v)
    else:
        ckpt = get_state_dict(src)

    save_obj(obj=ckpt, path=path, **kwargs)


def convert_to_onnx(
    model: nn.Module,
    inputs: Sequence[Any],
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
    opset_version: int | None = None,
    dynamic_axes: Mapping[str, Mapping[int, str]] | Mapping[str, Sequence[int]] | None = None,
    filename: Any | None = None,
    verify: bool = False,
    device: torch.device | None = None,
    use_ort: bool = False,
    ort_provider: Sequence[str] | None = None,
    rtol: float = 1e-4,
    atol: float = 0.0,
    use_trace: bool = True,
    do_constant_folding: bool = True,
    constant_size_threshold: int = 16 * 1024 * 1024 * 1024,
    **kwargs,
):
    """
    Utility to convert a model into ONNX model and optionally verify with ONNX or onnxruntime.
    See also: https://pytorch.org/docs/stable/onnx.html for how to convert a PyTorch model to ONNX.

    Args:
        model: source PyTorch model to save.
        inputs: input sample data used by pytorch.onnx.export. It is also used in ONNX model verification.
        input_names: optional input names of the ONNX model.
        output_names: optional output names of the ONNX model.
        opset_version: version of the (ai.onnx) opset to target. Must be >= 7 and not exceed
        the latest opset version supported by PyTorch, for more details:
            https://github.com/onnx/onnx/blob/main/docs/Operators.md and
            https://github.com/pytorch/pytorch/blob/master/torch/onnx/_constants.py
        dynamic_axes: specifies axes of tensors as dynamic (i.e. known only at run-time). If set to None,
            the exported model will have the shapes of all input and output tensors set to match given
            ones, for more details: https://pytorch.org/docs/stable/onnx.html#torch.onnx.export.
        filename: optional filename to save the ONNX model, if None, don't save the ONNX model.
        verify: whether to verify the ONNX model with ONNX or onnxruntime.
        device: target PyTorch device to verify the model, if None, use CUDA if available.
        use_ort: whether to use onnxruntime to verify the model.
        ort_provider": onnxruntime provider to use, default is ["CPUExecutionProvider"].
        rtol: the relative tolerance when comparing the outputs of PyTorch model and TorchScript model.
        atol: the absolute tolerance when comparing the outputs of PyTorch model and TorchScript model.
        use_trace: whether to use `torch.jit.trace` to export the torchscript model.
        do_constant_folding: passed to onnx.export(). If True, extra polygraphy folding pass is done.
        constant_size_threshold: passed to polygrapy conatant forling, default = 16M
        kwargs: if use_trace=True: additional arguments to pass to torch.onnx.export()
            else: other arguments except `obj` for `torch.jit.script()` to convert model, for more details:
            https://pytorch.org/docs/master/generated/torch.jit.script.html.

    """
    model.eval()
    with torch.no_grad():
        torch_versioned_kwargs = {}
        if use_trace:
            # let torch.onnx.export to trace the model.
            mode_to_export = model
            torch_versioned_kwargs = kwargs
            if "dynamo" in kwargs and kwargs["dynamo"] and verify:
                torch_versioned_kwargs["verify"] = verify
                verify = False
        else:
            mode_to_export = torch.jit.script(model, **kwargs)

        if torch.is_tensor(inputs) or isinstance(inputs, dict):
            onnx_inputs = (inputs,)
        else:
            onnx_inputs = tuple(inputs)
        temp_file = None
        if filename is None:
            temp_file = tempfile.NamedTemporaryFile()
            f = temp_file.name
        else:
            f = filename
        print(f"torch_versioned_kwargs={torch_versioned_kwargs}")
        torch.onnx.export(
            mode_to_export,
            onnx_inputs,
            f=f,
            input_names=input_names,
            output_names=output_names or None,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=do_constant_folding,
            **torch_versioned_kwargs,
        )
        onnx_model = onnx.load(f)

    if do_constant_folding and polygraphy_imported:
        from polygraphy.backend.onnx.loader import fold_constants, save_onnx

        onnx_model = fold_constants(onnx_model, size_threshold=constant_size_threshold)
        save_onnx(onnx_model, f)

    if verify:
        if isinstance(inputs, dict):
            inputs = list(inputs.values())

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        inputs = [i.to(device) if isinstance(i, torch.Tensor) else i for i in inputs]
        model = model.to(device)

        with torch.no_grad():
            set_determinism(seed=0)
            torch_out = ensure_tuple(model(*inputs), True)

        set_determinism(seed=0)
        model_input_names = [i.name for i in onnx_model.graph.input]
        input_dict = dict(zip(model_input_names, [i.cpu().numpy() for i in inputs]))
        if use_ort:
            ort_sess = onnxruntime.InferenceSession(
                onnx_model.SerializeToString(), providers=ort_provider if ort_provider else ["CPUExecutionProvider"]
            )
            onnx_out = ort_sess.run(None, input_dict)
        else:
            sess = onnxreference.ReferenceEvaluator(onnx_model)
            onnx_out = sess.run(None, input_dict)
        set_determinism(seed=None)
        # compare onnx/ort and PyTorch results
        for r1, r2 in zip(torch_out, onnx_out):
            if isinstance(r1, torch.Tensor):
                torch.testing.assert_close(r1.cpu(), convert_to_tensor(r2, dtype=r1.dtype), rtol=rtol, atol=atol)  # type: ignore

    return onnx_model


def convert_to_torchscript(
    model: nn.Module,
    filename_or_obj: Any | None = None,
    extra_files: dict | None = None,
    verify: bool = False,
    inputs: Sequence[Any] | None = None,
    device: torch.device | None = None,
    rtol: float = 1e-4,
    atol: float = 0.0,
    use_trace: bool = False,
    **kwargs,
):
    """
    Utility to convert a model into TorchScript model and save to file,
    with optional input / output data verification.

    Args:
        model: source PyTorch model to save.
        filename_or_obj: if not None, specify a file-like object (has to implement write and flush)
            or a string containing a file path name to save the TorchScript model.
        extra_files: map from filename to contents which will be stored as part of the save model file.
            for more details: https://pytorch.org/docs/stable/generated/torch.jit.save.html.
        verify: whether to verify the input and output of TorchScript model.
            if `filename_or_obj` is not None, load the saved TorchScript model and verify.
        inputs: input test data to verify model, should be a sequence of data, every item maps to a argument
            of `model()` function.
        device: target device to verify the model, if None, use CUDA if available.
        rtol: the relative tolerance when comparing the outputs of PyTorch model and TorchScript model.
        atol: the absolute tolerance when comparing the outputs of PyTorch model and TorchScript model.
        use_trace: whether to use `torch.jit.trace` to export the TorchScript model.
        kwargs: other arguments except `obj` for `torch.jit.script()` or `torch.jit.trace()` (if use_trace is True)
            to convert model, for more details: https://pytorch.org/docs/master/generated/torch.jit.script.html.

    """
    model.eval()
    with torch.no_grad():
        if use_trace:
            if inputs is None:
                raise ValueError("Missing input data for tracing convert.")
            script_module = torch.jit.trace(model, example_inputs=inputs, **kwargs)
        else:
            script_module = torch.jit.script(model, **kwargs)
        if filename_or_obj is not None:
            torch.jit.save(m=script_module, f=filename_or_obj, _extra_files=extra_files)

    if verify:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if inputs is None:
            raise ValueError("Missing input data for verification.")

        inputs = [i.to(device) if isinstance(i, torch.Tensor) else i for i in inputs]
        ts_model = torch.jit.load(filename_or_obj) if filename_or_obj is not None else script_module
        ts_model.eval().to(device)
        model = model.to(device)

        with torch.no_grad():
            set_determinism(seed=0)
            torch_out = ensure_tuple(model(*inputs))
            set_determinism(seed=0)
            torchscript_out = ensure_tuple(ts_model(*inputs))
            set_determinism(seed=None)
        # compare TorchScript and PyTorch results
        for r1, r2 in zip(torch_out, torchscript_out):
            if isinstance(r1, torch.Tensor) or isinstance(r2, torch.Tensor):
                torch.testing.assert_close(r1, r2, rtol=rtol, atol=atol)  # type: ignore

    return script_module


def _onnx_trt_compile(
    onnx_model,
    min_shape: Sequence[int],
    opt_shape: Sequence[int],
    max_shape: Sequence[int],
    device: int,
    precision: str,
    input_names: Sequence[str] | None,
    output_names: Sequence[str] | None,
):
    """
    This function takes an ONNX model as input, exports it to a TensorRT engine, wraps the TensorRT engine
    to a TensorRT engine-based TorchScript model and return the TorchScript model.

    Args:
        onnx_model: the source ONNX model to compile.
        min_shape: the minimum input shape of the converted TensorRT model.
        opt_shape: the optimization input shape of the model, on which the TensorRT optimizes.
        max_shape: the maximum input shape of the converted TensorRT model.
        device: the target GPU index to convert and verify the model.
        precision: the weight precision of the converted TensorRT engine-based TorchScript model.
            Should be 'fp32' or 'fp16'.
        input_names: optional input names of the ONNX model. Should be a sequence like
            `['input_0', 'input_1', ..., 'input_N']` where N equals to the number of the
            model inputs.
        output_names: optional output names of the ONNX model. Should be a sequence like
            `['output_0', 'output_1', ..., 'output_N']` where N equals to the number of
            the model outputs.

    """
    trt, _ = optional_import("tensorrt", "8.5.3")

    input_shapes = (min_shape, opt_shape, max_shape)
    # default to an empty list to fit the `torch_tensorrt.ts.embed_engine_in_new_module` function.
    input_names = [] if not input_names else input_names
    output_names = [] if not output_names else output_names

    # set up the TensorRT builder
    torch.cuda.set_device(device)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    if input_names:
        profile.set_shape(input_names[0], *input_shapes)

    # parse the ONNX model
    parser = trt.OnnxParser(network, logger)
    success = parser.parse(onnx_model.SerializeToString())
    if not success:
        parser_error_message = ""
        for idx in range(parser.num_errors):
            parser_error_message += parser.get_error(idx).desc() + "\n"
        raise Exception(f"TensorRT cannot parse the ONNX model, due to:\n{parser_error_message}")

    # set up the conversion configuration
    config = builder.create_builder_config()
    config.add_optimization_profile(profile)
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    serialized_engine = builder.build_serialized_network(network, config)
    f = io.BytesIO()
    f.write(serialized_engine)

    # wrap the serialized TensorRT engine back to a TorchScript module.
    trt_model = torch_tensorrt.ts.embed_engine_in_new_module(
        f.getvalue(),
        device=torch_tensorrt.Device(f"cuda:{device}"),
        input_binding_names=input_names,
        output_binding_names=output_names,
    )
    return trt_model


def convert_to_trt(
    model: nn.Module,
    precision: str,
    input_shape: Sequence[int],
    dynamic_batchsize: Sequence[int] | None = None,
    use_trace: bool = False,
    filename_or_obj: Any | None = None,
    verify: bool = False,
    device: int | None = None,
    use_onnx: bool | None = False,
    onnx_input_names: Sequence[str] | None = ("input_0",),
    onnx_output_names: Sequence[str] | None = ("output_0",),
    rtol: float = 1e-2,
    atol: float = 0.0,
    **kwargs,
):
    """
    Utility to export a model into a TensorRT engine-based TorchScript model with optional input / output data verification.

    There are two ways to export a model:
    1, Torch-TensorRT way: PyTorch module ---> TorchScript module ---> TensorRT engine-based TorchScript.
    2, ONNX-TensorRT way: PyTorch module ---> TorchScript module ---> ONNX model ---> TensorRT engine --->
    TensorRT engine-based TorchScript.

    When exporting through the first way, some models suffer from the slowdown problem, since Torch-TensorRT
    may only convert a little part of the PyTorch model to the TensorRT engine. However when exporting through
    the second way, some Python data structures like `dict` are not supported. And some TorchScript models are
    not supported by the ONNX if exported through `torch.jit.script`.

    Args:
        model: a source PyTorch model to convert.
        precision: the weight precision of the converted TensorRT engine based TorchScript models. Should be 'fp32' or 'fp16'.
        input_shape: the input shape that is used to convert the model. Should be a list like [N, C, H, W] or
            [N, C, H, W, D].
        dynamic_batchsize: a sequence with three elements to define the batch size range of the input for the model to be
            converted. Should be a sequence like [MIN_BATCH, OPT_BATCH, MAX_BATCH]. After converted, the batchsize of model
            input should between `MIN_BATCH` and `MAX_BATCH` and the `OPT_BATCH` is the best performance batchsize that the
            TensorRT tries to fit. The `OPT_BATCH` should be the most frequently used input batchsize in the application,
            default to None.
        use_trace: whether using `torch.jit.trace` to convert the PyTorch model to a TorchScript model and then convert to
            a TensorRT engine based TorchScript model or an ONNX model (if `use_onnx` is True), default to False.
        filename_or_obj: if not None, specify a file-like object (has to implement write and flush) or a string containing a
            file path name to load the TensorRT engine based TorchScript model for verifying.
        verify: whether to verify the input and output of the TensorRT engine based TorchScript model.
        device: the target GPU index to convert and verify the model. If None, use #0 GPU.
        use_onnx: whether to use the ONNX-TensorRT way to export the TensorRT engine-based TorchScript model.
        onnx_input_names: optional input names of the ONNX model. This arg is only useful when `use_onnx` is True. Should be
            a sequence like `('input_0', 'input_1', ..., 'input_N')` where N equals to the number of the model inputs. If not
            given, will use `('input_0',)`, which supposes the model only has one input.
        onnx_output_names: optional output names of the ONNX model. This arg is only useful when `use_onnx` is True. Should be
            a sequence like `('output_0', 'output_1', ..., 'output_N')` where N equals to the number of the model outputs. If
            not given, will use `('output_0',)`, which supposes the model only has one output.
        rtol: the relative tolerance when comparing the outputs between the PyTorch model and TensorRT model.
        atol: the absolute tolerance when comparing the outputs between the PyTorch model and TensorRT model.
        kwargs: other arguments except `module`, `inputs`, `enabled_precisions` and `device` for `torch_tensorrt.compile()`
            to compile model, for more details: https://pytorch.org/TensorRT/py_api/torch_tensorrt.html#torch-tensorrt-py.
    """

    if not torch.cuda.is_available():
        raise Exception("Cannot find any GPU devices.")

    if not input_shape:
        raise ValueError("Missing the input shape for model convert.")

    if not dynamic_batchsize:
        warnings.warn(f"There is no dynamic batch range. The converted model only takes {input_shape} shape input.")

    if (dynamic_batchsize is not None) and (len(dynamic_batchsize) != 3):
        warnings.warn(f"The dynamic batch range sequence should have 3 elements, but got {dynamic_batchsize} elements.")

    device = device if device else 0
    target_device = torch.device(f"cuda:{device}")
    convert_precision = torch.float32 if precision == "fp32" else torch.half
    inputs = [torch.rand(ensure_tuple(input_shape)).to(target_device)]

    # convert the torch model to a TorchScript model on target device
    model = model.eval().to(target_device)
    min_input_shape, opt_input_shape, max_input_shape = get_profile_shapes(input_shape, dynamic_batchsize)

    if use_onnx:
        # set the batch dim as dynamic
        dynamic_axes = {k: {0: "batchsize"} for k in onnx_input_names} if onnx_input_names else {}
        dynamic_axes.update({k: {0: "batchsize"} for k in onnx_output_names} if onnx_output_names else {})
        ir_model = convert_to_onnx(
            model, inputs, onnx_input_names, onnx_output_names, use_trace=use_trace, dynamic_axes=dynamic_axes
        )
        # convert the model through the ONNX-TensorRT way
        trt_model = _onnx_trt_compile(
            ir_model,
            min_shape=min_input_shape,
            opt_shape=opt_input_shape,
            max_shape=max_input_shape,
            device=device,
            precision=precision,
            input_names=onnx_input_names,
            output_names=onnx_output_names,
        )
    else:
        ir_model = convert_to_torchscript(model, device=target_device, inputs=inputs, use_trace=use_trace)
        ir_model.eval()
        # convert the model through the Torch-TensorRT way
        ir_model.to(target_device)
        with torch.no_grad():
            with torch.cuda.device(device=device):
                input_placeholder = [
                    torch_tensorrt.Input(
                        min_shape=min_input_shape, opt_shape=opt_input_shape, max_shape=max_input_shape
                    )
                ]
                trt_model = torch_tensorrt.compile(
                    ir_model,
                    inputs=input_placeholder,
                    enabled_precisions=convert_precision,
                    device=torch_tensorrt.Device(f"cuda:{device}"),
                    ir="torchscript",
                    **kwargs,
                )

    # verify the outputs between the TensorRT model and PyTorch model
    if verify:
        if inputs is None:
            raise ValueError("Missing input data for verification.")

        trt_model = torch.jit.load(filename_or_obj) if filename_or_obj is not None else trt_model

        with torch.no_grad():
            set_determinism(seed=0)
            torch_out = ensure_tuple(model(*inputs))
            set_determinism(seed=0)
            trt_out = ensure_tuple(trt_model(*inputs))
            set_determinism(seed=None)
        # compare TorchScript and PyTorch results
        for r1, r2 in zip(torch_out, trt_out):
            if isinstance(r1, torch.Tensor) or isinstance(r2, torch.Tensor):
                torch.testing.assert_close(r1, r2, rtol=rtol, atol=atol)  # type: ignore

    return trt_model


def meshgrid_ij(*tensors):
    if torch.meshgrid.__kwdefaults__ is not None and "indexing" in torch.meshgrid.__kwdefaults__:
        return torch.meshgrid(*tensors, indexing="ij")  # new api pytorch after 1.10

    return torch.meshgrid(*tensors)


def meshgrid_xy(*tensors):
    if torch.meshgrid.__kwdefaults__ is not None and "indexing" in torch.meshgrid.__kwdefaults__:
        return torch.meshgrid(*tensors, indexing="xy")  # new api pytorch after 1.10

    return torch.meshgrid(tensors[1], tensors[0], *tensors[2:])


def _replace_modules(
    parent: torch.nn.Module,
    name: str,
    new_module: torch.nn.Module,
    out: list[tuple[str, torch.nn.Module]],
    strict_match: bool = True,
    match_device: bool = True,
) -> None:
    """
    Helper function for :py:class:`monai.networks.utils.replace_modules`.
    """
    if match_device:
        devices = list({i.device for i in parent.parameters()})
        # if only one device for whole of model
        if len(devices) == 1:
            new_module.to(devices[0])
    idx = name.find(".")
    # if there is "." in name, call recursively
    if idx != -1:
        parent_name = name[:idx]
        parent = getattr(parent, parent_name)
        name = name[idx + 1 :]
        _out: list[tuple[str, torch.nn.Module]] = []
        _replace_modules(parent, name, new_module, _out)
        # prepend the parent name
        out += [(f"{parent_name}.{r[0]}", r[1]) for r in _out]
    # no "." in module name, do the actual replacing
    else:
        if strict_match:
            old_module = getattr(parent, name)
            setattr(parent, name, new_module)
            out += [(name, old_module)]
        else:
            for mod_name, _ in parent.named_modules():
                if name in mod_name:
                    _replace_modules(parent, mod_name, deepcopy(new_module), out, strict_match=True)


def replace_modules(
    parent: torch.nn.Module,
    name: str,
    new_module: torch.nn.Module,
    strict_match: bool = True,
    match_device: bool = True,
) -> list[tuple[str, torch.nn.Module]]:
    """
    Replace sub-module(s) in a parent module.

    The name of the module to be replace can be nested e.g.,
    `features.denseblock1.denselayer1.layers.relu1`. If this is the case (there are "."
    in the module name), then this function will recursively call itself.

    Args:
        parent: module that contains the module to be replaced
        name: name of module to be replaced. Can include ".".
        new_module: `torch.nn.Module` to be placed at position `name` inside `parent`. This will
            be deep copied if `strict_match == False` multiple instances are independent.
        strict_match: if `True`, module name must `== name`. If false then
            `name in named_modules()` will be used. `True` can be used to change just
            one module, whereas `False` can be used to replace all modules with similar
            name (e.g., `relu`).
        match_device: if `True`, the device of the new module will match the model. Requires all
            of `parent` to be on the same device.

    Returns:
        List of tuples of replaced modules. Element 0 is module name, element 1 is the replaced module.

    Raises:
        AttributeError: if `strict_match` is `True` and `name` is not a named module in `parent`.
    """
    out: list[tuple[str, torch.nn.Module]] = []
    _replace_modules(parent, name, new_module, out, strict_match, match_device)
    return out


@contextmanager
def replace_modules_temp(
    parent: torch.nn.Module,
    name: str,
    new_module: torch.nn.Module,
    strict_match: bool = True,
    match_device: bool = True,
):
    """
    Temporarily replace sub-module(s) in a parent module (context manager).

    See :py:class:`monai.networks.utils.replace_modules`.
    """
    replaced: list[tuple[str, torch.nn.Module]] = []
    try:
        # replace
        _replace_modules(parent, name, new_module, replaced, strict_match, match_device)
        yield
    finally:
        # revert
        for name, module in replaced:
            _replace_modules(parent, name, module, [], strict_match=True, match_device=match_device)


def freeze_layers(model: nn.Module, freeze_vars=None, exclude_vars=None):
    """
    A utilty function to help freeze specific layers.

    Args:
        model: a source PyTorch model to freeze layer.
        freeze_vars: a regular expression to match the `model` variable names,
            so that their `requires_grad` will set to `False`.
        exclude_vars: a regular expression to match the `model` variable names,
            except for matched variable names, other `requires_grad` will set to `False`.

    Raises:
        ValueError: when freeze_vars and exclude_vars are both specified.

    """
    if freeze_vars is not None and exclude_vars is not None:
        raise ValueError("Incompatible values: freeze_vars and exclude_vars are both specified.")
    src_dict = get_state_dict(model)

    frozen_keys = list()
    if freeze_vars is not None:
        to_freeze = {s_key for s_key in src_dict if freeze_vars and re.compile(freeze_vars).search(s_key)}
        for name, param in model.named_parameters():
            if name in to_freeze:
                param.requires_grad = False
                frozen_keys.append(name)
            elif not param.requires_grad:
                param.requires_grad = True
                warnings.warn(
                    f"The freeze_vars does not include {param}, but requires_grad is False, change it to True."
                )
    if exclude_vars is not None:
        to_exclude = {s_key for s_key in src_dict if exclude_vars and re.compile(exclude_vars).search(s_key)}
        for name, param in model.named_parameters():
            if name not in to_exclude:
                param.requires_grad = False
                frozen_keys.append(name)
            elif not param.requires_grad:
                param.requires_grad = True
                warnings.warn(f"The exclude_vars includes {param}, but requires_grad is False, change it to True.")

    logger.info(f"{len(frozen_keys)} of {len(src_dict)} variables frozen.")


class CastTempType(nn.Module):
    """
    Cast the input tensor to a temporary type before applying the submodule, and then cast it back to the initial type.
    """

    def __init__(self, initial_type, temporary_type, submodule):
        super().__init__()
        self.initial_type = initial_type
        self.temporary_type = temporary_type
        self.submodule = submodule

    def forward(self, x):
        dtype = x.dtype
        if dtype == self.initial_type:
            x = x.to(self.temporary_type)
        x = self.submodule(x)
        if dtype == self.initial_type:
            x = x.to(self.initial_type)
        return x


def cast_tensor(x, from_dtype=torch.float16, to_dtype=torch.float32):
    """
    Utility function to cast a single tensor from from_dtype to to_dtype
    """
    return x.to(dtype=to_dtype) if x.dtype == from_dtype else x


def cast_all(x, from_dtype=torch.float16, to_dtype=torch.float32):
    """
    Utility function to cast all tensors in a tuple from from_dtype to to_dtype
    """
    if isinstance(x, torch.Tensor):
        return cast_tensor(x, from_dtype=from_dtype, to_dtype=to_dtype)
    else:
        if isinstance(x, dict):
            new_dict = {}
            for k in x.keys():
                new_dict[k] = cast_all(x[k], from_dtype=from_dtype, to_dtype=to_dtype)
            return new_dict
        elif isinstance(x, tuple):
            return tuple(cast_all(y, from_dtype=from_dtype, to_dtype=to_dtype) for y in x)


class CastToFloat(torch.nn.Module):
    """
    Class used to add autocast protection for ONNX export
    for forward methods with single return vaue
    """

    def __init__(self, mod):
        super().__init__()
        self.mod = mod

    def forward(self, x):
        dtype = x.dtype
        with torch.amp.autocast("cuda", enabled=False):
            ret = self.mod.forward(x.to(torch.float32)).to(dtype)
        return ret


class CastToFloatAll(torch.nn.Module):
    """
    Class used to add autocast protection for ONNX export
    for forward methods with multiple return values
    """

    def __init__(self, mod):
        super().__init__()
        self.mod = mod

    def forward(self, *args):
        from_dtype = args[0].dtype
        with torch.amp.autocast("cuda", enabled=False):
            ret = self.mod.forward(*cast_all(args, from_dtype=from_dtype, to_dtype=torch.float32))
        return cast_all(ret, from_dtype=torch.float32, to_dtype=from_dtype)


def wrap_module(base_t: type[nn.Module], dest_t: type[nn.Module]) -> Callable[[nn.Module], nn.Module | None]:
    """
    Generic function generator to replace base_t module with dest_t wrapper.
    Args:
        base_t : module type to replace
        dest_t : destination module type
    Returns:
        swap function to replace base_t module with dest_t
    """

    def expansion_fn(mod: nn.Module) -> nn.Module | None:
        out = dest_t(mod)
        return out

    return expansion_fn


def simple_replace(base_t: type[nn.Module], dest_t: type[nn.Module]) -> Callable[[nn.Module], nn.Module | None]:
    """
    Generic function generator to replace base_t module with dest_t.
    base_t and dest_t should have same atrributes. No weights are copied.
    Args:
        base_t : module type to replace
        dest_t : destination module type
    Returns:
        swap function to replace base_t module with dest_t
    """

    def expansion_fn(mod: nn.Module) -> nn.Module | None:
        if not isinstance(mod, base_t):
            return None
        args = [getattr(mod, name, None) for name in mod.__constants__]
        out = dest_t(*args)
        return out

    return expansion_fn


def _swap_modules(model: nn.Module, mapping: dict[str, nn.Module]) -> nn.Module:
    """
    This function swaps nested modules as specified by "dot paths" in mod with a desired replacement. This allows
    for swapping nested modules through arbitrary levels if children

    NOTE: This occurs in place, if you want to preserve model then make sure to copy it first.

    """
    for path, new_mod in mapping.items():
        expanded_path = path.split(".")
        parent_mod = model
        for sub_path in expanded_path[:-1]:
            submod = parent_mod._modules[sub_path]
            if submod is None:
                break
            else:
                parent_mod = submod
        parent_mod._modules[expanded_path[-1]] = new_mod

    return model


def replace_modules_by_type(
    model: nn.Module, expansions: dict[str, Callable[[nn.Module], nn.Module | None]]
) -> nn.Module:
    """
    Top-level function to replace modules in model, specified by class name with a desired replacement.
    NOTE: This occurs in place, if you want to preserve model then make sure to copy it first.
    Args:
        model : top level module
        expansions : replacement dictionary: module class name -> replacement function generator
    Returns:
        model, possibly modified in-place
    """
    mapping: dict[str, nn.Module] = {}
    for name, m in model.named_modules():
        m_type = type(m).__name__
        if m_type in expansions:
            # print (f"Found {m_type} in expansions ...")
            swapped = expansions[m_type](m)
            if swapped:
                mapping[name] = swapped

    print(f"Swapped {len(mapping)} modules")
    _swap_modules(model, mapping)
    return model


def add_casts_around_norms(model: nn.Module) -> nn.Module:
    """
    Top-level function to add cast wrappers around modules known to cause issues for FP16/autocast ONNX export
    NOTE: This occurs in place, if you want to preserve model then make sure to copy it first.
    Args:
        model : top level module
    Returns:
        model, possibly modified in-place
    """
    print("Adding casts around norms...")
    cast_replacements = {
        "BatchNorm1d": wrap_module(nn.BatchNorm1d, CastToFloat),
        "BatchNorm2d": wrap_module(nn.BatchNorm2d, CastToFloat),
        "BatchNorm3d": wrap_module(nn.BatchNorm2d, CastToFloat),
        "LayerNorm": wrap_module(nn.LayerNorm, CastToFloat),
        "InstanceNorm1d": wrap_module(nn.InstanceNorm1d, CastToFloat),
        "InstanceNorm3d": wrap_module(nn.InstanceNorm3d, CastToFloat),
    }
    replace_modules_by_type(model, cast_replacements)
    return model
