from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn

from monai.networks.blocks.text_embedding import TextEncoder
from monai.networks.blocks.head_controller import HeadController

from monai.networks.nets import SwinUNETR

class Universal_model(nn.Module):
    """
    Universal Model for organ and tumor segmentation, based on: "Liu et al.,
    CLIP-Driven Universal Model for Organ Segmentation and Tumor Detection <https://arxiv.org/pdf/2301.00785.pdf>"
    """
    def __init__(
        self,
        img_size,
        in_channels,
        out_channels,
        bottleneck_size: int = 768,
        text_dim: int = 512,
        hidden_size: int = 256,
        backbone: str = 'swinunetr',
        encoding: str = 'clip_embedding',
        logits_options: list = None,
    ):
        super().__init__()
        self.backbone_name = backbone
        if backbone == 'swinunetr':
            self.backbone = SwinUNETR_backbone(
                        img_size=img_size,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        feature_size=48,
                        drop_rate=0.0,
                        attn_drop_rate=0.0,
                        dropout_path_rate=0.0,
                        use_checkpoint=False,
                        )
        else:
            raise Exception(f'{backbone} backbone is not implemented, please add your own')
        self.class_num = out_channels
        self.logits_options = logits_options
        # text encoder
        self.text_encoder = TextEncoder(
            out_channels=self.class_num,
            text_dim=text_dim,
            hidden_size=hidden_size,
            encoding=encoding
        )

        self.head_controller = HeadController(
            out_channels=out_channels,
            text_encoding=True
        )

        self.GAP = nn.Sequential(
            nn.GroupNorm(16, bottleneck_size),
            nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool3d((1,1,1)),
            nn.Conv3d(bottleneck_size, hidden_size, kernel_size=1, stride=1, padding=0)
        )

    def load_params(self, model_dict):
        if self.backbone_name == 'swinunetr':
            store_dict = self.backbone.state_dict()
            for key in model_dict.keys():
                if 'out' not in key:
                    store_dict[key] = model_dict[key]

            self.backbone.load_state_dict(store_dict)
            print('Use swin unetr pretrained weights')
        else:
            raise Exception(f'{self.backbone_name} backbone is not implemented, please add your own')

    def forward(self, x_in):
        # get backbone feature
        dec4, out = self.backbone(x_in)
        # get task text encoding
        text_encoding = self.text_encoder()
        # text controlled outputs
        x_feat = self.GAP(dec4)
        out = self.head_controller(x_feat, out, text_encoding, self.logits_options)

        return out

class SwinUNETR_backbone(SwinUNETR):
    """
    Universal Model uses SwinUNETR as backbone without the segmentation head based on:

    "Hatamizadeh et al.,
    Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images
    <https://arxiv.org/abs/2201.01266>" and

    "Tang et al.,
    Self-Supervised Pre-Training of Swin Transformers for 3D Medical Image Analysis
    <https://arxiv.org/abs/2111.14791>"
    """
    def __init__(
        self,
        img_size: Union[Sequence[int], int],
        in_channels: int,
        out_channels: int,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        feature_size: int = 48,
        norm_name: Union[Tuple, str] = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
    ):
        super().__init__(img_size,in_channels,out_channels,feature_size=48)

    def forward(self, x_in):
        hidden_states_out = self.swinViT(x_in, self.normalize)
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])

        dec3 = self.decoder5(dec4, hidden_states_out[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)

        return dec4, out
