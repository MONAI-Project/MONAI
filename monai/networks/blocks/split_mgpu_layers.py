# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import math

import torch
import torch.distributed as dist


from .halo import _TiledHaloExchange2d, _TiledHaloExchange3d, _TiledHaloExchange_p3d

class _BaseTiledLayer2d(torch.nn.Module):
  def __init__(self, tile_op, kernel_size, stride, padding):
    super().__init__()
    self.tile_op = tile_op
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding

    self.halo_exch_op = _TiledHaloExchange2d.apply

    # Multiply gradients by DDP world size to obtain global sum instead of
    # global average.
    # Note: should disable this is model is not wrapped in DDP to allow tile
    # parameters to update independently.
    if dist.is_initialized():
      ddp_world_size = dist.get_world_size()
      for p in self.tile_op.parameters():
        if p.requires_grad is True:
          p.register_hook(lambda x: ddp_world_size * x)

  def _output_size(self, tdesc):
    # Compute output size
    N, C, H, W = tdesc.global_data_dims
    Hout = math.floor((H + 2*self.padding - self.kernel_size) / self.stride + 1)
    Wout = math.floor((W + 2*self.padding - self.kernel_size) / self.stride + 1)

    return N, C, Hout, Wout

  def _output_chunk_sizes(self, row_tile_sizes, col_tile_sizes):
    # Padding is baked into tiles, hence padding = 0 here
    output_row_chunks = ((row_tile_sizes - self.kernel_size) // self.stride) + 1
    output_col_chunks = ((col_tile_sizes - self.kernel_size) // self.stride) + 1
    return output_row_chunks, output_col_chunks

  def reset_parameters(self):
    if hasattr(self.tile_op, 'reset_parameters'):
      self.tile_op.reset_parameters()

  def forward(self, x, tdesc):
    # Compute global output size
    Nout, Cout, Hout, Wout = self._output_size(tdesc)

    # Halo exchange
    x, tile_sizes = self.halo_exch_op(x, tdesc, self.kernel_size, self.stride, self.padding)

    # Compute expected output sizes by tile sizes
    output_row_chunks, output_col_chunks = self._output_chunk_sizes(*tile_sizes)

    Hout_tile = output_row_chunks[tdesc.tile_grid.tile_idx[0]]
    Wout_tile = output_col_chunks[tdesc.tile_grid.tile_idx[1]]
    if (Hout_tile == 0 or Wout_tile == 0):
      raise RuntimeError("Tiled output contains empty entries. Not allowed currently.")
      x = x.new_empty(Nout, Cout, Hout_tile, Wout_tile)
    else:
      x = self.tile_op(x)

    return x, tdesc.tile_grid.create_tile_descriptor([Nout, Cout, Hout, Wout], output_row_chunks, output_col_chunks)


class _BaseTiledLayer_p3d(torch.nn.Module):
  def __init__(self, tile_op, kernel_size, stride, padding):
    super().__init__()
    self.tile_op = tile_op
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding

    self.halo_exch_op = _TiledHaloExchange_p3d.apply

    # Multiply gradients by DDP world size to obtain global sum instead of
    # global average.
    # Note: should disable this is model is not wrapped in DDP to allow tile
    # parameters to update independently.
    if dist.is_initialized():
      ddp_world_size = dist.get_world_size()
      for p in self.tile_op.parameters():
        if p.requires_grad is True:
          p.register_hook(lambda x: ddp_world_size * x)

  def _output_size(self, tdesc):
    # Compute output size
    N, C, H, W, D = tdesc.global_data_dims
    Hout = math.floor((H + 2*self.padding - self.kernel_size) / self.stride + 1)
    Wout = math.floor((W + 2*self.padding - self.kernel_size) / self.stride + 1)
    # Dout = math.floor((D + 2*self.padding - self.kernel_size) / self.stride + 1)

    return N, C, Hout, Wout, D

  def _output_chunk_sizes(self, row_tile_sizes, col_tile_sizes):
    # Padding is baked into tiles, hence padding = 0 here
    output_row_chunks = ((row_tile_sizes - self.kernel_size) // self.stride) + 1
    output_col_chunks = ((col_tile_sizes - self.kernel_size) // self.stride) + 1
    return output_row_chunks, output_col_chunks

  def reset_parameters(self):
    if hasattr(self.tile_op, 'reset_parameters'):
      self.tile_op.reset_parameters()

  def forward(self, x, tdesc):
    # print("tdesc", tdesc, "\n")

    # Compute global output size
    Nout, Cout, Hout, Wout, Dout = self._output_size(tdesc)
    # print("Nout, Cout, Hout, Wout, Dout", Nout, Cout, Hout, Wout, Dout, "\n")

    # Halo exchange
    x, tile_sizes = self.halo_exch_op(x, tdesc, self.kernel_size, self.stride, self.padding)

    # Compute expected output sizes by tile sizes
    output_row_chunks, output_col_chunks = self._output_chunk_sizes(*tile_sizes)

    Hout_tile = output_row_chunks[tdesc.tile_grid.tile_idx[0]]
    Wout_tile = output_col_chunks[tdesc.tile_grid.tile_idx[1]]
    if (Hout_tile == 0 or Wout_tile == 0):
      raise RuntimeError("Tiled output contains empty entries. Not allowed currently.")
      x = x.new_empty(Nout, Cout, Hout_tile, Wout_tile)
    else:
      x = self.tile_op(x)

    return x, tdesc.tile_grid.create_tile_descriptor([Nout, Cout, Hout, Wout, Dout], output_row_chunks, output_col_chunks)


class TiledConv2d(_BaseTiledLayer2d):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
    # Set padding to zero here as halo exchange code adds explicit pad
    tile_op = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, bias=bias)

    super().__init__(tile_op, kernel_size, stride, padding)
    self.out_channels = out_channels

  def _output_size(self, tdesc):
    # Compute output size
    N, C, H, W = tdesc.global_data_dims
    Hout = math.floor((H + 2*self.padding - self.kernel_size) / self.stride + 1)
    Wout = math.floor((W + 2*self.padding - self.kernel_size) / self.stride + 1)

    return N, self.out_channels, Hout, Wout


class TiledConv_p3d(_BaseTiledLayer_p3d):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
    # Set padding to zero here as halo exchange code adds explicit pad
    tile_op = torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=0, bias=bias)

    super().__init__(tile_op, kernel_size, stride, padding)
    self.out_channels = out_channels

  def _output_size(self, tdesc):
    # Compute output size
    N, _, H, W, D = tdesc.global_data_dims
    Hout = math.floor((H + 2*self.padding - self.kernel_size) / self.stride + 1)
    Wout = math.floor((W + 2*self.padding - self.kernel_size) / self.stride + 1)
    # Dout = math.floor((D + 2*self.padding - self.kernel_size) / self.stride + 1)

    return N, self.out_channels, Hout, Wout, D


class TiledMaxPool2d(_BaseTiledLayer2d):
  def __init__(self, kernel_size, stride=1, padding=0):
    # Set padding to zero here as halo exchange code adds explicit pad
    tile_op = torch.nn.MaxPool2d(kernel_size, stride, padding=0)
    super().__init__(tile_op, kernel_size, stride, padding)


class TiledAvgPool2d(_BaseTiledLayer2d):
  def __init__(self, kernel_size, stride=1, padding=0):
    # Set padding to zero here as halo exchange code adds explicit pad
    tile_op = torch.nn.AvgPool2d(kernel_size, stride, padding=0)
    super().__init__(tile_op, kernel_size, stride, padding)


# Note: TiledBatchNorm is mostly a wrapper around SyncBatchNorm, but with pre-scaled
# gradients to cancel out averaging done by DDP.
class TiledBatchNorm(torch.nn.SyncBatchNorm):
  def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
    super().__init__(num_features, eps, momentum, affine, track_running_stats)

    # Multiply gradients by DDP world size to obtain global sum instead of
    # global average.
    # Note: should disable this is model is not wrapped in DDP to allow tile
    # parameters to update independently.
    if dist.is_initialized():
      ddp_world_size = dist.get_world_size()
      for p in self.parameters():
        if p.requires_grad is True:
          p.register_hook(lambda x: ddp_world_size * x)

  def forward(self, x, tdesc):
    # tdesc is unused, but kept as input and output arg for convenience (or future usage)
    x = super().forward(x)
    return x, tdesc


class _BaseTiledLayer3d(torch.nn.Module):
  def __init__(self, tile_op, kernel_size, stride, padding):
    super().__init__()
    self.tile_op = tile_op
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding

    self.halo_exch_op = _TiledHaloExchange3d.apply

    # Multiply gradients by DDP world size to obtain global sum instead of
    # global average.
    # Note: should disable this is model is not wrapped in DDP to allow tile
    # parameters to update independently.
    if dist.is_initialized():
      ddp_world_size = dist.get_world_size()
      for p in self.tile_op.parameters():
        if p.requires_grad is True:
          p.register_hook(lambda x: ddp_world_size * x)


class TiledMaxPool_p3d(_BaseTiledLayer_p3d):
  def __init__(self, kernel_size, stride=1, padding=0):
    # Set padding to zero here as halo exchange code adds explicit pad
    tile_op = torch.nn.MaxPool3d(kernel_size, stride, padding=0)
    super().__init__(tile_op, kernel_size, stride, padding)


class TiledAvgPool_p3d(_BaseTiledLayer_p3d):
  def __init__(self, kernel_size, stride=1, padding=0):
    # Set padding to zero here as halo exchange code adds explicit pad
    tile_op = torch.nn.AvgPool3d(kernel_size, stride, padding=0)
    super().__init__(tile_op, kernel_size, stride, padding)
