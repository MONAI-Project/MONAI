from typing import Sequence, Union, Optional
import numpy as np

import torch
import torch.nn as nn

from monai.networks.blocks.convolutions import Convolution


def infer_spatial_rank(input_tensor: torch.Tensor):
    # from:
    # https://niftynet.readthedocs.io/en/dev/_modules/niftynet/layer/layer_util.html#expand_spatial_params
    input_shape = input_tensor.shape
    dims = len(input_shape) - 2
    assert dims > 0, "input tensor should have at least one spatial dim, " \
                     "in addition to batch and channel dims"
    return int(dims)

def expand_spatial_params(input_param, spatial_rank, param_type=int):
    # from:
    # https://niftynet.readthedocs.io/en/dev/_modules/niftynet/layer/layer_util.html#expand_spatial_params
    spatial_rank = int(spatial_rank)
    try:
        if param_type == int:
            input_param = int(input_param)
        else:
            input_param = float(input_param)
        return (input_param,) * spatial_rank
    except (ValueError, TypeError):
        pass

    try:
        if param_type == int:
            input_param = \
                np.asarray(input_param).flatten().astype(np.int).tolist()
        else:
            input_param = \
                np.asarray(input_param).flatten().astype(np.float).tolist()
    except (ValueError, TypeError):
        # skip type casting if it's a TF tensor
        pass
    assert len(input_param) >= spatial_rank, \
        'param length should be at least the length of spatial rank'
    return tuple(input_param[:spatial_rank])


class ChannelSparseConvolutionLayer(Convolution):
    def __init__(
        self,

        *args,
        **kwargs,
    ):
        """
        Args:
            TODO
        """
        super().__init__(*args, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        input_mask: Optional[torch.Tensor] = None,
        output_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        sparse_input_shape = x.shape.as_list()
        if not input_mask:
            _input_mask = torch.ones([sparse_input_shape[-1]]) > 0
        else:
            _input_mask = input_mask

        if not output_mask:
            _output_mask = torch.ones([self.out_channels]) > 0
        else:
            _output_mask = output_mask

        _input_mask.shape.as_list()[0]

        # TODO: in NiftyNet this is a seperate function
        # https://niftynet.readthedocs.io/en/dev/_modules/niftynet/layer/layer_util.html#infer_spatial_rank
        spatial_rank = infer_spatial_rank(x)

        expand_spatial_params(
            self.kernel_size, spatial_rank
        )

        expand_spatial_params(
            self.stride, spatial_rank
        )

        expand_spatial_params(
            self.dilation, spatial_rank
        )


class DenseFeatureStackBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        dense_channels: int,
        kernel_size: Union[Sequence[int], int],
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple],
        dilation: Union[Sequence[int], int],
        use_bdo: bool = True,
        dim: Optional[int] = None,
    ):
        """
        Dense Feature Stack Block

        - Stack is initialized with the input from above layers.
        - Iteratively the output of convolution layers is added to the feature stack.
        - Each sequential convolution is performed over all the previous stacked
          channels.

        Diagram example:

            feature_stack = [Input]
            feature_stack = [feature_stack, conv(feature_stack)]
            feature_stack = [feature_stack, conv(feature_stack)]
            feature_stack = [feature_stack, conv(feature_stack)]
            ...
            Output = [feature_stack, conv(feature_stack)]
        """
        super().__init__()

        self.dfs_block = []
        self.use_bdo = use_bdo

        for _ in self.dilation_rates:
            if self.use_bdo:
                conv = ChannelSparseConvolutionLayer(
                    out_channels=dense_channels,
                    kernel_size=kernel_size
                )
            else:
                conv = Convolution(
                    out_channels=dense_channels,
                    kernel_size=kernel_size,
                )
            self.dfs_block.append(conv)

    def forward(self, x: torch.Tensor):
        feature_stack = [x]

        # Literal from niftynet
        # TODO: Check if this works the same in pytorch
        channels = x.shape.as_list()[-1]
        input_mask = torch.ones([channels]) > 0

        for i, conv in enumerate(self.dfs_block):
            if i == len(self.dfs_block) - 1:
                keep_prob = None

            channel_dim = len(x.shape) - 1
            input_features = torch.concat(feature_stack, channel_dim)

            if self.use_bdo:
                output_features, new_input_mask = conv(
                    input_features,
                    input_mask=input_mask,
                    dropout=keep_prob,
                )
                input_mask = torch.concat([input_mask, new_input_mask], 0)
            else:
                output_features = conv(
                    input_features,
                    dropout=keep_prob,
                )

            feature_stack.append(output_features)


class DenseFeatureStackBlockWithSkipAndDownsample(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        dense_channels: int,
        segmentation_channels: int,
        kernel_size: int,
        dilation: int,
        use_bdo: bool,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        down_channels: Optional[int] = None,
        dim: Optional[int] = None,
    ):
        """
        The layer DenseFeatureStackBlockWithSkipAndDownsample layer implements
        [DFS + Conv + Downsampling] in a single module, and outputs 2 elements:
            - Skip layer:          [ DFS + Conv]
            - Downsampled output:  [ DFS + Down]

        Args:
            TODO

        """
        super().__init__()

        self.dfs_block = DenseFeatureStackBlock(
            dense_channels=dense_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            use_bdo=use_bdo,
        )

        self.skip_convolution = Convolution(
            out_channels=segmentation_channels,
            kernel_size=kernel_size,
        )

        self.down_convolution = None
        if down_channels:
            self.down_convolution = Convolution(
                out_channels=down_channels,
                kernel_size=kernel_size,
                stride=2,
            )

    def forward(self, x):
        feature_stack = self.dfs_block(x)
        merged_features = torch.concat(feature_stack, axis=len(x.shape) - 1)
        skip_convolution = self.skip_convolution(
            merged_features,
        )
        down_convolution = None
        if self.down_convolution:
            down_convolution = self.down_convolution(
                merged_features
            )
        return skip_convolution, down_convolution


class DenseVNet(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        initial_features: int = 24,
        segmentation_kernel_size: int = 3,
        dense_channels: Sequence[int] = (4, 8, 16),
        act: Union[str, tuple] = ("relu", {"inplace": True}),
        norm: Union[str, tuple] = ("batch"),
        bias: bool = True,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        dimensions: Optional[int] = None,
        use_bdo: bool = False,
        use_prior: bool = False,
        use_dense_connections: bool = True,
        use_coords: bool = False,
    ):
        """
        A Dense V-Net implementation with 1D/2D/3D supports

        Based on:
            Gibson et al. "Automatic Multi-Organ Segmentation on Abdominal CT
            With Dense V-Networks." IEEE Trans Med Imaging 37,8, 1822-1834 (2018),
            DOI: https://dx.doi.org/10.1109%2FTMI.2018.2806309

        Args:
            TODO

        Examples:
            TODO

        Diagram:

            DFS = Dense Feature Stack Block

            - Initial image is first downsampled to a given size.
            - Each DFS+SD outputs a skip link + a downsampled output.
            - All outputs are upscaled to the initial downsampled size.
            - If initial prior is given add it to the output prediction.

            Input
              |
              --[ DFS ]-----------------------[ Conv ]------------[ Conv ]------[+]-->
                   |                                       |  |              |
                   -----[ DFS ]---------------[ Conv ]------  |              |
                           |                                  |              |
                           -----[ DFS ]-------[ Conv ]---------              |
                                                                  [ Prior ]---


        Constraints:

            - Input size has to be divisible by 2*dilation_rates

        """
        super().__init__()

        self.initial_conv = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=initial_features,
            strides=2,
            kernel_size=5,
            act=act,
            norm=norm,
            bias=bias,
            dropout=dropout)

        self.dense_vblocks = []
        num_blocks = len(dense_channels)
        for i in range(num_blocks):
            vblock = DenseFeatureStackBlockWithSkipAndDownsample()  # TODO
            self.dense_vblocks.append(vblock)

        self.final_conv = Convolution(
            out_channels=out_channels,
            kernel_size=segmentation_kernel_size,
            act=act,
            norm=norm,
            bias=bias
        )

    def forward(self, x):
        return x
