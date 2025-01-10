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

from collections.abc import Sequence

import torch
import torch.nn as nn

import monai

NUM_SPLITS = 16
SPLIT_PADDING = 3



class InplaceGroupNorm3D(torch.nn.GroupNorm):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super().__init__(num_groups, num_channels, eps, affine)

    def forward(self, input):
        # Ensure the tensor is 5D: (N, C, D, H, W)
        if len(input.shape) != 5:
            raise ValueError("Expected a 5D tensor")

        N, C, D, H, W = input.shape

        # Reshape to (N, num_groups, C // num_groups, D, H, W)
        input = input.view(N, self.num_groups, C // self.num_groups, D, H, W)

        if False:
            input = input.to(dtype=torch.float64)
            mean = input.mean([2, 3, 4, 5], keepdim=True)
            input.var([2, 3, 4, 5], unbiased=False, keepdim=True).add_(self.eps).sqrt_()

            input = input.to(dtype=torch.float32)
            mean = mean.to(dtype=torch.float32)
            mean.to(dtype=torch.float32)
        else:
            means, stds = [], []
            inputs = []
            for _i in range(input.size(1)):
                array = input[:, _i:_i + 1, ...]
                array = array.to(dtype=torch.float32)
                _mean = array.mean([2, 3, 4, 5], keepdim=True)
                _std = array.var([2, 3, 4, 5], unbiased=False, keepdim=True).add_(self.eps).sqrt_()

                _mean = _mean.to(dtype=torch.float32)
                _std = _std.to(dtype=torch.float32)

                inputs.append(array.sub_(_mean).div_(_std).to(dtype=torch.float16))

        del input
        torch.cuda.empty_cache()

        if False:
            input = torch.cat([inputs[_k] for _k in range(len(inputs))], dim=1)
        else:
            if max(inputs[0].size()) < 500:
                input = torch.cat([inputs[_k] for _k in range(len(inputs))], dim=1)
            else:
                import gc
                _type = inputs[0].device.type
                if _type == 'cuda':
                    input = inputs[0].clone().to('cpu', non_blocking=True)
                else:
                    input = inputs[0].clone()
                inputs[0] = 0
                torch.cuda.empty_cache()

                for _k in range(len(inputs) - 1):
                    input = torch.cat((input, inputs[_k + 1].cpu()), dim=1)
                    inputs[_k + 1] = 0
                    torch.cuda.empty_cache()
                    gc.collect()

                if _type == 'cuda':
                    input = input.to('cuda', non_blocking=True)

        # Reshape back to original size
        input = input.view(N, C, D, H, W)

        # Apply affine transformation if enabled
        if self.affine:
            input.mul_(self.weight.view(1, C, 1, 1, 1)).add_(self.bias.view(1, C, 1, 1, 1))

        return input


class SplitConvolution(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        strides: Sequence[int] | int = 1,
        kernel_size: Sequence[int] | int = 3,
        adn_ordering: str = "NDA",
        act: tuple | str | None = "PRELU",
        norm: tuple | str | None = "INSTANCE",
        dropout: tuple | str | float | None = None,
        dropout_dim: int | None = 1,
        dilation: Sequence[int] | int = 1,
        groups: int = 1,
        bias: bool = True,
        conv_only: bool = False,
        is_transposed: bool = False,
        padding: Sequence[int] | int | None = None,
        output_padding: Sequence[int] | int | None = None,
    ) -> None:
        super().__init__()
        self.conv = monai.networks.blocks.convolutions.Convolution(
            spatial_dims,
            in_channels,
            out_channels,
            strides,
            kernel_size,
            adn_ordering,
            act,
            norm,
            dropout,
            dropout_dim,
            dilation,
            groups,
            bias,
            conv_only,
            is_transposed,
            padding,
            output_padding,
        )

        self.tp_dim = 1

    def forward(self, x):
        num_splits = NUM_SPLITS
        print("num_splits:", num_splits)
        l = x.size(self.tp_dim + 2)
        split_size = l // num_splits

        if False:
            splits = [x[:, :, i * split_size : (i + 1) * split_size, :, :] for i in range(num_splits)]
        else:
            padding = SPLIT_PADDING
            print("padding:", padding)

            overlaps = [0] + [padding] * (num_splits - 1)
            if self.tp_dim == 0:
                splits = [
                    x[
                        :,
                        :,
                        i * split_size - overlaps[i] : (i + 1) * split_size + (padding if i != num_splits - 1 else 0),
                        :,
                        :,
                    ]
                    for i in range(num_splits)
                ]
            elif self.tp_dim == 1:
                splits = [
                    x[
                        :,
                        :,
                        :,
                        i * split_size - overlaps[i] : (i + 1) * split_size + (padding if i != num_splits - 1 else 0),
                        :,
                    ]
                    for i in range(num_splits)
                ]
            elif self.tp_dim == 2:
                splits = [
                    x[
                        :,
                        :,
                        :,
                        :,
                        i * split_size - overlaps[i] : (i + 1) * split_size + (padding if i != num_splits - 1 else 0),
                    ]
                    for i in range(num_splits)
                ]

            for _j in range(len(splits)):
                print(f"splits {_j + 1}/{len(splits)}:", splits[_j].size())

        del x
        torch.cuda.empty_cache()

        splits_0_size = list(splits[0].size())

        if False:
            outputs = [self.conv(splits[i]) for i in range(num_splits)]
        else:
            outputs = []
            _type = splits[0].device.type
            for _i in range(num_splits):
                if _type == "cuda":
                    outputs.append(self.conv(splits[_i]))
                else:
                    _t = splits[_i]
                    _t1 = self.conv(_t.to("cuda", non_blocking=True))
                    del _t
                    torch.cuda.empty_cache()
                    _t1 = _t1.to("cpu", non_blocking=True)
                    outputs.append(_t1)
                    del _t1
                    torch.cuda.empty_cache()

                splits[_i] = 0
                torch.cuda.empty_cache()

        del splits
        torch.cuda.empty_cache()

        split_size_out = split_size
        padding_s = padding
        non_tp_dim = self.tp_dim + 1 if self.tp_dim < 2 else 0
        if outputs[0].size(non_tp_dim + 2) // splits_0_size[non_tp_dim + 2] == 2:
            split_size_out *= 2
            padding_s *= 2
        elif splits_0_size[non_tp_dim + 2] // outputs[0].size(non_tp_dim + 2) == 2:
            split_size_out = split_size_out // 2
            padding_s = padding_s // 2

        if self.tp_dim == 0:
            outputs[0] = outputs[0][:, :, :split_size_out, :, :]
            for i in range(1, num_splits):
                outputs[i] = outputs[i][:, :, padding_s : padding_s + split_size_out, :, :]
        elif self.tp_dim == 1:
            outputs[0] = outputs[0][:, :, :, :split_size_out, :]
            for i in range(1, num_splits):
                outputs[i] = outputs[i][:, :, :, padding_s : padding_s + split_size_out, :]
        elif self.tp_dim == 2:
            outputs[0] = outputs[0][:, :, :, :, :split_size_out]
            for i in range(1, num_splits):
                outputs[i] = outputs[i][:, :, :, :, padding_s : padding_s + split_size_out]

        if max(outputs[0].size()) < 500:
            x = torch.cat(outputs, dim=self.tp_dim + 2)
        else:
            import gc

            _type = outputs[0].device.type
            if _type == "cuda":
                x = outputs[0].clone().to("cpu", non_blocking=True)
            outputs[0] = 0
            torch.cuda.empty_cache()
            for _k in range(len(outputs) - 1):
                x = torch.cat((x, outputs[_k + 1].cpu()), dim=self.tp_dim + 2)
                outputs[_k + 1] = 0
                torch.cuda.empty_cache()
                gc.collect()
            if _type == "cuda":
                x = x.to("cuda", non_blocking=True)

        del outputs
        torch.cuda.empty_cache()

        return x
