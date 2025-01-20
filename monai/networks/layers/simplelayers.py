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

import math
from copy import deepcopy
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function

from monai.config.type_definitions import NdarrayOrTensor
from monai.networks.layers.convutils import gaussian_1d
from monai.networks.layers.factories import Conv
from monai.utils import (
    ChannelMatching,
    SkipMode,
    convert_to_tensor,
    ensure_tuple_rep,
    issequenceiterable,
    look_up_option,
    optional_import,
)

_C, _ = optional_import("monai._C")
fft, _ = optional_import("torch.fft")

__all__ = [
    "ChannelPad",
    "Flatten",
    "GaussianFilter",
    "HilbertTransform",
    "LLTM",
    "MedianFilter",
    "Reshape",
    "SavitzkyGolayFilter",
    "SkipConnection",
    "apply_filter",
    "median_filter",
    "separable_filtering",
]


class ChannelPad(nn.Module):
    """
    Expand the input tensor's channel dimension from length `in_channels` to `out_channels`,
    by padding or a projection.
    """

    def __init__(
        self, spatial_dims: int, in_channels: int, out_channels: int, mode: ChannelMatching | str = ChannelMatching.PAD
    ):
        """

        Args:
            spatial_dims: number of spatial dimensions of the input image.
            in_channels: number of input channels.
            out_channels: number of output channels.
            mode: {``"pad"``, ``"project"``}
                Specifies handling residual branch and conv branch channel mismatches. Defaults to ``"pad"``.

                - ``"pad"``: with zero padding.
                - ``"project"``: with a trainable conv with kernel size one.
        """
        super().__init__()
        self.project = None
        self.pad = None
        if in_channels == out_channels:
            return
        mode = look_up_option(mode, ChannelMatching)
        if mode == ChannelMatching.PROJECT:
            conv_type = Conv[Conv.CONV, spatial_dims]
            self.project = conv_type(in_channels, out_channels, kernel_size=1)
            return
        if mode == ChannelMatching.PAD:
            if in_channels > out_channels:
                raise ValueError('Incompatible values: channel_matching="pad" and in_channels > out_channels.')
            pad_1 = (out_channels - in_channels) // 2
            pad_2 = out_channels - in_channels - pad_1
            pad = [0, 0] * spatial_dims + [pad_1, pad_2] + [0, 0]
            self.pad = tuple(pad)
            return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.project is not None:
            return torch.as_tensor(self.project(x))  # as_tensor used to get around mypy typing bug
        if self.pad is not None:
            return F.pad(x, self.pad)
        return x


class SkipConnection(nn.Module):
    """
    Combine the forward pass input with the result from the given submodule::

        --+--submodule--o--
          |_____________|

    The available modes are ``"cat"``, ``"add"``, ``"mul"``.
    """

    def __init__(self, submodule, dim: int = 1, mode: str | SkipMode = "cat") -> None:
        """

        Args:
            submodule: the module defines the trainable branch.
            dim: the dimension over which the tensors are concatenated.
                Used when mode is ``"cat"``.
            mode: ``"cat"``, ``"add"``, ``"mul"``. defaults to ``"cat"``.
        """
        super().__init__()
        self.submodule = submodule
        self.dim = dim
        self.mode = look_up_option(mode, SkipMode).value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.submodule(x)

        if self.mode == "cat":
            return torch.cat([x, y], dim=self.dim)
        if self.mode == "add":
            return torch.add(x, y)
        if self.mode == "mul":
            return torch.mul(x, y)
        raise NotImplementedError(f"Unsupported mode {self.mode}.")


class Flatten(nn.Module):
    """
    Flattens the given input in the forward pass to be [B,-1] in shape.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


class Reshape(nn.Module):
    """
    Reshapes input tensors to the given shape (minus batch dimension), retaining original batch size.
    """

    def __init__(self, *shape: int) -> None:
        """
        Given a shape list/tuple `shape` of integers (s0, s1, ... , sn), this layer will reshape input tensors of
        shape (batch, s0 * s1 * ... * sn) to shape (batch, s0, s1, ... , sn).

        Args:
            shape: list/tuple of integer shape dimensions
        """
        super().__init__()
        self.shape = (1,) + tuple(shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = list(self.shape)
        shape[0] = x.shape[0]  # done this way for Torchscript
        return x.reshape(shape)


def _separable_filtering_conv(
    input_: torch.Tensor,
    kernels: list[torch.Tensor],
    pad_mode: str,
    d: int,
    spatial_dims: int,
    paddings: list[int],
    num_channels: int,
) -> torch.Tensor:
    if d < 0:
        return input_

    s = [1] * len(input_.shape)
    s[d + 2] = -1
    _kernel = kernels[d].reshape(s)

    # if filter kernel is unity, don't convolve
    if _kernel.numel() == 1 and _kernel[0] == 1:
        return _separable_filtering_conv(input_, kernels, pad_mode, d - 1, spatial_dims, paddings, num_channels)

    _kernel = _kernel.repeat([num_channels, 1] + [1] * spatial_dims)
    _padding = [0] * spatial_dims
    _padding[d] = paddings[d]
    conv_type = [F.conv1d, F.conv2d, F.conv3d][spatial_dims - 1]

    # translate padding for input to torch.nn.functional.pad
    _reversed_padding_repeated_twice: list[list[int]] = [[p, p] for p in reversed(_padding)]
    _sum_reversed_padding_repeated_twice: list[int] = sum(_reversed_padding_repeated_twice, [])
    padded_input = F.pad(input_, _sum_reversed_padding_repeated_twice, mode=pad_mode)

    return conv_type(
        input=_separable_filtering_conv(padded_input, kernels, pad_mode, d - 1, spatial_dims, paddings, num_channels),
        weight=_kernel,
        groups=num_channels,
    )


def separable_filtering(x: torch.Tensor, kernels: list[torch.Tensor], mode: str = "zeros") -> torch.Tensor:
    """
    Apply 1-D convolutions along each spatial dimension of `x`.

    Args:
        x: the input image. must have shape (batch, channels, H[, W, ...]).
        kernels: kernel along each spatial dimension.
            could be a single kernel (duplicated for all spatial dimensions), or
            a list of `spatial_dims` number of kernels.
        mode (string, optional): padding mode passed to convolution class. ``'zeros'``, ``'reflect'``, ``'replicate'``
            or ``'circular'``. Default: ``'zeros'``. See ``torch.nn.Conv1d()`` for more information.

    Raises:
        TypeError: When ``x`` is not a ``torch.Tensor``.

    Examples:

    .. code-block:: python

        >>> import torch
        >>> from monai.networks.layers import separable_filtering
        >>> img = torch.randn(2, 4, 32, 32)  # batch_size 2, channels 4, 32x32 2D images
        # applying a [-1, 0, 1] filter along each of the spatial dimensions.
        # the output shape is the same as the input shape.
        >>> out = separable_filtering(img, torch.tensor((-1., 0., 1.)))
        # applying `[-1, 0, 1]`, `[1, 0, -1]` filters along two spatial dimensions respectively.
        # the output shape is the same as the input shape.
        >>> out = separable_filtering(img, [torch.tensor((-1., 0., 1.)), torch.tensor((1., 0., -1.))])

    """

    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be a torch.Tensor but is {type(x).__name__}.")

    spatial_dims = len(x.shape) - 2
    if isinstance(kernels, torch.Tensor):
        kernels = [kernels] * spatial_dims
    _kernels = [s.to(x) for s in kernels]
    _paddings = [(k.shape[0] - 1) // 2 for k in _kernels]
    n_chs = x.shape[1]
    pad_mode = "constant" if mode == "zeros" else mode

    return _separable_filtering_conv(x, _kernels, pad_mode, spatial_dims - 1, spatial_dims, _paddings, n_chs)


def apply_filter(x: torch.Tensor, kernel: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Filtering `x` with `kernel` independently for each batch and channel respectively.

    Args:
        x: the input image, must have shape (batch, channels, H[, W, D]).
        kernel: `kernel` must at least have the spatial shape (H_k[, W_k, D_k]).
            `kernel` shape must be broadcastable to the `batch` and `channels` dimensions of `x`.
        kwargs: keyword arguments passed to `conv*d()` functions.

    Returns:
        The filtered `x`.

    Examples:

    .. code-block:: python

        >>> import torch
        >>> from monai.networks.layers import apply_filter
        >>> img = torch.rand(2, 5, 10, 10)  # batch_size 2, channels 5, 10x10 2D images
        >>> out = apply_filter(img, torch.rand(3, 3))   # spatial kernel
        >>> out = apply_filter(img, torch.rand(5, 3, 3))  # channel-wise kernels
        >>> out = apply_filter(img, torch.rand(2, 5, 3, 3))  # batch-, channel-wise kernels

    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be a torch.Tensor but is {type(x).__name__}.")
    batch, chns, *spatials = x.shape
    n_spatial = len(spatials)
    if n_spatial > 3:
        raise NotImplementedError(f"Only spatial dimensions up to 3 are supported but got {n_spatial}.")
    k_size = len(kernel.shape)
    if k_size < n_spatial or k_size > n_spatial + 2:
        raise ValueError(
            f"kernel must have {n_spatial} ~ {n_spatial + 2} dimensions to match the input shape {x.shape}."
        )
    kernel = kernel.to(x)
    # broadcast kernel size to (batch chns, spatial_kernel_size)
    kernel = kernel.expand(batch, chns, *kernel.shape[(k_size - n_spatial) :])
    kernel = kernel.reshape(-1, 1, *kernel.shape[2:])  # group=1
    x = x.view(1, kernel.shape[0], *spatials)
    conv = [F.conv1d, F.conv2d, F.conv3d][n_spatial - 1]
    if "padding" not in kwargs:
        kwargs["padding"] = "same"

    if "stride" not in kwargs:
        kwargs["stride"] = 1
    output = conv(x, kernel, groups=kernel.shape[0], bias=None, **kwargs)
    return output.view(batch, chns, *output.shape[2:])


class SavitzkyGolayFilter(nn.Module):
    """
    Convolve a Tensor along a particular axis with a Savitzky-Golay kernel.

    Args:
        window_length: Length of the filter window, must be a positive odd integer.
        order: Order of the polynomial to fit to each window, must be less than ``window_length``.
        axis (optional): Axis along which to apply the filter kernel. Default 2 (first spatial dimension).
        mode (string, optional): padding mode passed to convolution class. ``'zeros'``, ``'reflect'``, ``'replicate'`` or
        ``'circular'``. Default: ``'zeros'``. See torch.nn.Conv1d() for more information.
    """

    def __init__(self, window_length: int, order: int, axis: int = 2, mode: str = "zeros"):
        super().__init__()
        if order >= window_length:
            raise ValueError("order must be less than window_length.")

        self.axis = axis
        self.mode = mode
        self.coeffs = self._make_coeffs(window_length, order)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor or array-like to filter. Must be real, in shape ``[Batch, chns, spatial1, spatial2, ...]`` and
                have a device type of ``'cpu'``.
        Returns:
            torch.Tensor: ``x`` filtered by Savitzky-Golay kernel with window length ``self.window_length`` using
            polynomials of order ``self.order``, along axis specified in ``self.axis``.
        """

        # Make input a real tensor on the CPU
        x = torch.as_tensor(x, device=x.device if isinstance(x, torch.Tensor) else None)
        if torch.is_complex(x):
            raise ValueError("x must be real.")
        x = x.to(dtype=torch.float)

        if (self.axis < 0) or (self.axis > len(x.shape) - 1):
            raise ValueError(f"Invalid axis for shape of x, got axis {self.axis} and shape {x.shape}.")

        # Create list of filter kernels (1 per spatial dimension). The kernel for self.axis will be the savgol coeffs,
        # while the other kernels will be set to [1].
        n_spatial_dims = len(x.shape) - 2
        spatial_processing_axis = self.axis - 2
        new_dims_before = spatial_processing_axis
        new_dims_after = n_spatial_dims - spatial_processing_axis - 1
        kernel_list = [self.coeffs.to(device=x.device, dtype=x.dtype)]
        for _ in range(new_dims_before):
            kernel_list.insert(0, torch.ones(1, device=x.device, dtype=x.dtype))
        for _ in range(new_dims_after):
            kernel_list.append(torch.ones(1, device=x.device, dtype=x.dtype))

        return separable_filtering(x, kernel_list, mode=self.mode)

    @staticmethod
    def _make_coeffs(window_length, order):
        half_length, rem = divmod(window_length, 2)
        if rem == 0:
            raise ValueError("window_length must be odd.")

        idx = torch.arange(window_length - half_length - 1, -half_length - 1, -1, dtype=torch.float, device="cpu")
        a = idx ** torch.arange(order + 1, dtype=torch.float, device="cpu").reshape(-1, 1)
        y = torch.zeros(order + 1, dtype=torch.float, device="cpu")
        y[0] = 1.0
        return torch.linalg.lstsq(a, y).solution.squeeze()


class HilbertTransform(nn.Module):
    """
    Determine the analytical signal of a Tensor along a particular axis.

    Args:
        axis: Axis along which to apply Hilbert transform. Default 2 (first spatial dimension).
        n: Number of Fourier components (i.e. FFT size). Default: ``x.shape[axis]``.
    """

    def __init__(self, axis: int = 2, n: int | None = None) -> None:
        super().__init__()
        self.axis = axis
        self.n = n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor or array-like to transform. Must be real and in shape ``[Batch, chns, spatial1, spatial2, ...]``.
        Returns:
            torch.Tensor: Analytical signal of ``x``, transformed along axis specified in ``self.axis`` using
            FFT of size ``self.N``. The absolute value of ``x_ht`` relates to the envelope of ``x`` along axis ``self.axis``.
        """

        # Make input a real tensor
        x = torch.as_tensor(x, device=x.device if isinstance(x, torch.Tensor) else None)
        if torch.is_complex(x):
            raise ValueError("x must be real.")
        x = x.to(dtype=torch.float)

        if (self.axis < 0) or (self.axis > len(x.shape) - 1):
            raise ValueError(f"Invalid axis for shape of x, got axis {self.axis} and shape {x.shape}.")

        n = x.shape[self.axis] if self.n is None else self.n
        if n <= 0:
            raise ValueError("N must be positive.")
        x = torch.as_tensor(x, dtype=torch.complex64)
        # Create frequency axis
        f = torch.cat(
            [
                torch.true_divide(torch.arange(0, (n - 1) // 2 + 1, device=x.device), float(n)),
                torch.true_divide(torch.arange(-(n // 2), 0, device=x.device), float(n)),
            ]
        )
        xf = fft.fft(x, n=n, dim=self.axis)
        # Create step function
        u = torch.heaviside(f, torch.tensor([0.5], device=f.device))
        u = torch.as_tensor(u, dtype=x.dtype, device=u.device)
        new_dims_before = self.axis
        new_dims_after = len(xf.shape) - self.axis - 1
        for _ in range(new_dims_before):
            u.unsqueeze_(0)
        for _ in range(new_dims_after):
            u.unsqueeze_(-1)

        ht = fft.ifft(xf * 2 * u, dim=self.axis)

        # Apply transform
        return torch.as_tensor(ht, device=ht.device, dtype=ht.dtype)


def get_binary_kernel(window_size: Sequence[int], dtype=torch.float, device=None) -> torch.Tensor:
    """
    Create a binary kernel to extract the patches.
    The window size HxWxD will create a (H*W*D)xHxWxD kernel.
    """
    win_size = convert_to_tensor(window_size, int, wrap_sequence=True)
    prod = torch.prod(win_size)
    s = [prod, 1, *win_size]
    return torch.diag(torch.ones(prod, dtype=dtype, device=device)).view(s)  # type: ignore


def median_filter(
    in_tensor: torch.Tensor,
    kernel_size: Sequence[int] | int = (3, 3, 3),
    spatial_dims: int = 3,
    kernel: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    """
    Apply median filter to an image.

    Args:
        in_tensor: input tensor; median filtering will be applied to the last `spatial_dims` dimensions.
        kernel_size: the convolution kernel size.
        spatial_dims: number of spatial dimensions to apply median filtering.
        kernel: an optional customized kernel.
        kwargs: additional parameters to the `conv`.

    Returns:
        the filtered input tensor, shape remains the same as ``in_tensor``

    Example::

        >>> from monai.networks.layers import median_filter
        >>> import torch
        >>> x = torch.rand(4, 5, 7, 6)
        >>> output = median_filter(x, (3, 3, 3))
        >>> output.shape
        torch.Size([4, 5, 7, 6])

    """
    if not isinstance(in_tensor, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(in_tensor)}")

    original_shape = in_tensor.shape
    oshape, sshape = original_shape[: len(original_shape) - spatial_dims], original_shape[-spatial_dims:]
    oprod = torch.prod(convert_to_tensor(oshape, int, wrap_sequence=True))
    # prepare kernel
    if kernel is None:
        kernel_size = ensure_tuple_rep(kernel_size, spatial_dims)
        kernel = get_binary_kernel(kernel_size, in_tensor.dtype, in_tensor.device)
    else:
        kernel = kernel.to(in_tensor)
    # map the local window to single vector
    conv = [F.conv1d, F.conv2d, F.conv3d][spatial_dims - 1]
    reshaped_input: torch.Tensor = in_tensor.reshape(oprod, 1, *sshape)  # type: ignore

    # even-sized kernels are not supported
    padding = [(k - 1) // 2 for k in reversed(kernel.shape[2:]) for _ in range(2)]
    padded_input: torch.Tensor = F.pad(reshaped_input, pad=padding, mode="replicate")
    features: torch.Tensor = conv(padded_input, kernel, padding=0, stride=1, **kwargs)

    features = features.view(oprod, -1, *sshape)  # type: ignore

    # compute the median along the feature axis
    median: torch.Tensor = torch.median(features, dim=1)[0]
    median = median.reshape(original_shape)

    return median


class MedianFilter(nn.Module):
    """
    Apply median filter to an image.

    Args:
        radius: the blurring kernel radius (radius of 1 corresponds to 3x3x3 kernel when spatial_dims=3).

    Returns:
        filtered input tensor.

    Example::

        >>> from monai.networks.layers import MedianFilter
        >>> import torch
        >>> in_tensor = torch.rand(4, 5, 7, 6)
        >>> blur = MedianFilter([1, 1, 1])  # 3x3x3 kernel
        >>> output = blur(in_tensor)
        >>> output.shape
        torch.Size([4, 5, 7, 6])

    """

    def __init__(self, radius: Sequence[int] | int, spatial_dims: int = 3, device="cpu") -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.radius: Sequence[int] = ensure_tuple_rep(radius, spatial_dims)
        self.window: Sequence[int] = [1 + 2 * deepcopy(r) for r in self.radius]
        self.kernel = get_binary_kernel(self.window, device=device)

    def forward(self, in_tensor: torch.Tensor, number_of_passes=1) -> torch.Tensor:
        """
        Args:
            in_tensor: input tensor, median filtering will be applied to the last `spatial_dims` dimensions.
            number_of_passes: median filtering will be repeated this many times
        """
        x = in_tensor
        for _ in range(number_of_passes):
            x = median_filter(x, kernel=self.kernel, spatial_dims=self.spatial_dims)
        return x


class GaussianFilter(nn.Module):

    def __init__(
        self,
        spatial_dims: int,
        sigma: Sequence[float] | float | Sequence[torch.Tensor] | torch.Tensor,
        truncated: float = 4.0,
        approx: str = "erf",
        requires_grad: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions of the input image.
                must have shape (Batch, channels, H[, W, ...]).
            sigma: std. could be a single value, or `spatial_dims` number of values.
            truncated: spreads how many stds.
            approx: discrete Gaussian kernel type, available options are "erf", "sampled", and "scalespace".

                - ``erf`` approximation interpolates the error function;
                - ``sampled`` uses a sampled Gaussian kernel;
                - ``scalespace`` corresponds to
                  https://en.wikipedia.org/wiki/Scale_space_implementation#The_discrete_Gaussian_kernel
                  based on the modified Bessel functions.

            requires_grad: whether to store the gradients for sigma.
                if True, `sigma` will be the initial value of the parameters of this module
                (for example `parameters()` iterator could be used to get the parameters);
                otherwise this module will fix the kernels using `sigma` as the std.
        """
        if issequenceiterable(sigma):
            if len(sigma) != spatial_dims:  # type: ignore
                raise ValueError
        else:
            sigma = [deepcopy(sigma) for _ in range(spatial_dims)]  # type: ignore
        super().__init__()
        self.sigma = [
            torch.nn.Parameter(
                torch.as_tensor(s, dtype=torch.float, device=s.device if isinstance(s, torch.Tensor) else None),
                requires_grad=requires_grad,
            )
            for s in sigma  # type: ignore
        ]
        self.truncated = truncated
        self.approx = approx
        for idx, param in enumerate(self.sigma):
            self.register_parameter(f"kernel_sigma_{idx}", param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: in shape [Batch, chns, H, W, D].
        """
        _kernel = [gaussian_1d(s, truncated=self.truncated, approx=self.approx) for s in self.sigma]
        return separable_filtering(x=x, kernels=_kernel)


class LLTMFunction(Function):

    @staticmethod
    def forward(ctx, input, weights, bias, old_h, old_cell):
        outputs = _C.lltm_forward(input, weights, bias, old_h, old_cell)
        new_h, new_cell = outputs[:2]
        variables = outputs[1:] + [weights]
        ctx.save_for_backward(*variables)

        return new_h, new_cell

    @staticmethod
    def backward(ctx, grad_h, grad_cell):
        outputs = _C.lltm_backward(grad_h.contiguous(), grad_cell.contiguous(), *ctx.saved_tensors)
        d_old_h, d_input, d_weights, d_bias, d_old_cell = outputs[:5]

        return d_input, d_weights, d_bias, d_old_h, d_old_cell


class LLTM(nn.Module):
    """
    This recurrent unit is similar to an LSTM, but differs in that it lacks a forget
    gate and uses an Exponential Linear Unit (ELU) as its internal activation function.
    Because this unit never forgets, call it LLTM, or Long-Long-Term-Memory unit.
    It has both C++ and CUDA implementation, automatically switch according to the
    target device where put this module to.

    Args:
        input_features: size of input feature data
        state_size: size of the state of recurrent unit

    Referring to: https://pytorch.org/tutorials/advanced/cpp_extension.html
    """

    def __init__(self, input_features: int, state_size: int):
        super().__init__()
        self.input_features = input_features
        self.state_size = state_size
        self.weights = nn.Parameter(torch.empty(3 * state_size, input_features + state_size))
        self.bias = nn.Parameter(torch.empty(1, 3 * state_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.state_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input, state):
        return LLTMFunction.apply(input, self.weights, self.bias, *state)


class ApplyFilter(nn.Module):
    "Wrapper class to apply a filter to an image."

    def __init__(self, filter: NdarrayOrTensor) -> None:
        super().__init__()

        self.filter = convert_to_tensor(filter, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return apply_filter(x, self.filter)


class MeanFilter(ApplyFilter):
    """
    Mean filtering can smooth edges and remove aliasing artifacts in an segmentation image.
    The mean filter used, is a `torch.Tensor` of all ones.
    """

    def __init__(self, spatial_dims: int, size: int) -> None:
        """
        Args:
            spatial_dims: `int` of either 2 for 2D images and 3 for 3D images
            size: edge length of the filter
        """
        filter = torch.ones([size] * spatial_dims)
        filter = filter
        super().__init__(filter=filter)


class LaplaceFilter(ApplyFilter):
    """
    Laplacian filtering for outline detection in images. Can be used to transform labels to contours.
    The laplace filter used, is a `torch.Tensor` where all values are -1, except the center value
    which is `size` ** `spatial_dims`
    """

    def __init__(self, spatial_dims: int, size: int) -> None:
        """
        Args:
            spatial_dims: `int` of either 2 for 2D images and 3 for 3D images
            size: edge length of the filter
        """
        filter = torch.zeros([size] * spatial_dims).float() - 1  # make all -1
        center_point = tuple([size // 2] * spatial_dims)
        filter[center_point] = (size**spatial_dims) - 1
        super().__init__(filter=filter)


class EllipticalFilter(ApplyFilter):
    """
    Elliptical filter, can be used to dilate labels or label-contours.
    The elliptical filter used here, is a `torch.Tensor` with shape (size, ) * ndim containing a circle/sphere of `1`
    """

    def __init__(self, spatial_dims: int, size: int) -> None:
        """
        Args:
            spatial_dims: `int` of either 2 for 2D images and 3 for 3D images
            size: edge length of the filter
        """
        radius = size // 2
        grid = torch.meshgrid(*[torch.arange(0, size) for _ in range(spatial_dims)])
        squared_distances = torch.stack([(axis - radius) ** 2 for axis in grid], 0).sum(0)
        filter = squared_distances <= radius**2
        super().__init__(filter=filter)


class SharpenFilter(EllipticalFilter):
    """
    Convolutional filter to sharpen a 2D or 3D image.
    The filter used contains a circle/sphere of `-1`, with the center value being
    the absolute sum of all non-zero elements in the kernel
    """

    def __init__(self, spatial_dims: int, size: int) -> None:
        """
        Args:
            spatial_dims: `int` of either 2 for 2D images and 3 for 3D images
            size: edge length of the filter
        """
        super().__init__(spatial_dims=spatial_dims, size=size)
        center_point = tuple([size // 2] * spatial_dims)
        center_value = self.filter.sum()
        self.filter *= -1
        self.filter[center_point] = center_value
