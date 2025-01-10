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

from typing import Any, Dict

import torch

from monai.networks.layers import Conv, get_pool_layer
from monai.networks.utils import look_up_named_module, set_named_module
from monai.utils import look_up_option, optional_import

get_graph_node_names, _has_utils = optional_import("torchvision.models.feature_extraction", name="get_graph_node_names")
create_feature_extractor, _ = optional_import("torchvision.models.feature_extraction", name="create_feature_extractor")


class NetAdapter(torch.nn.Module):
    """
    Wrapper to replace the last layer of model by convolutional layer or FC layer.

    See also: :py:class:`monai.networks.nets.TorchVisionFCModel`

    Args:
        model: a PyTorch model, which can be both 2D and 3D models. typically, it can be a pretrained model
            in Torchvision, like: ``resnet18``, ``resnet34``, ``resnet50``, ``resnet101``, ``resnet152``, etc.
            more details: https://pytorch.org/vision/stable/models.html.
        num_classes: number of classes for the last classification layer. Default to 1.
        dim: number of supported spatial dimensions in the specified model, depends on the model implementation.
            default to 2 as most Torchvision models are for 2D image processing.
        in_channels: number of the input channels of last layer. if None, get it from `in_features` of last layer.
        use_conv: whether to use convolutional layer to replace the last layer, default to False.
        pool: parameters for the pooling layer, it should be a tuple, the first item is name of the pooling layer,
            the second item is dictionary of the initialization args. if None, will not replace the `layers[-2]`.
            default to `("avg", {"kernel_size": 7, "stride": 1})`.
        bias: the bias value when replacing the last layer. if False, the layer will not learn an additive bias,
            default to True.
        fc_name: the corresponding layer attribute of the last fully connected layer. Defaults to ``"fc"``.
        node_name: the corresponding feature extractor node name of `model`.
            Defaults to "", the extractor is not in use.

    """

    def __init__(
        self,
        model: torch.nn.Module,
        num_classes: int = 1,
        dim: int = 2,
        in_channels: int | None = None,
        use_conv: bool = False,
        pool: tuple[str, dict[str, Any]] | None = ("avg", {"kernel_size": 7, "stride": 1}),
        bias: bool = True,
        fc_name: str = "fc",
        node_name: str = "",
    ):
        super().__init__()
        layers = list(model.children())
        orig_fc = look_up_named_module(fc_name, model)
        if orig_fc is None:
            orig_fc = layers[-1]
        # guess the number of input channels of the last fully connected layer
        in_channels_: int
        if in_channels is None:
            if not hasattr(orig_fc, "in_features"):
                raise ValueError("please specify input channels of the last fully connected layer with `in_channels`.")
            in_channels_ = orig_fc.in_features

        else:
            in_channels_ = in_channels

        # modify the input model, depending on whether to replace the last pooling layer ``pool``
        if pool is None:  # no modification of pooling
            if node_name != "":
                raise ValueError("`node_name` is not compatible with `pool=None`, please set `pool=''`.")
            # we just drop the model's fully connected layer or set it to identity
            if look_up_named_module(fc_name, model):
                self.features = set_named_module(model, fc_name, torch.nn.Identity())
            else:
                self.features = torch.nn.Sequential(*layers[:-1])  # assuming FC is the last and model is sequential
            self.pool = None
        else:
            # user-specified new pooling layer, we drop both the pooling and FC layers from the model
            if node_name and _has_utils:
                node_name = look_up_option(node_name, get_graph_node_names(model)[0 if model.training else 1])
                self.features = create_feature_extractor(model, [node_name])
            else:
                self.features = torch.nn.Sequential(*layers[:-2])  # assuming the last 2 layers are pooling&FC
            self.pool = get_pool_layer(name=pool, spatial_dims=dim)

        # create new fully connected layer or kernel size 1 convolutional layer
        self.fc: torch.nn.Linear | torch.nn.Conv2d | torch.nn.Conv3d
        if use_conv:
            self.fc = Conv[Conv.CONV, dim](in_channels=in_channels_, out_channels=num_classes, kernel_size=1, bias=bias)
        else:
            self.fc = torch.nn.Linear(in_features=in_channels_, out_features=num_classes, bias=bias)
        self.use_conv = use_conv
        self.dim = dim
        self.node_name = node_name

    def forward(self, x):
        x = self.features(x)
        if isinstance(x, tuple):
            x = x[0]  # it might be a namedtuple such as torchvision.model.InceptionOutputs
        elif (
            isinstance(x, Dict)
            and all(isinstance(el, str) for el in x.keys())
            and all(isinstance(el, torch.Tensor) for el in x.values())
        ):
            x = x[self.node_name]  # torchvision create_feature_extractor
        if self.pool is not None:
            x = self.pool(x)
        if not self.use_conv:
            x = torch.flatten(x, 1)
        else:  # user specified `use_conv` but the pooling layer removed the spatial dims
            while len(x.shape) < self.dim + 2:
                x = x[..., None]
        x = self.fc(x)

        return x
