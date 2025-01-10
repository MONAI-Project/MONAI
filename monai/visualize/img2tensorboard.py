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

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from monai.config import NdarrayTensor
from monai.transforms import rescale_array
from monai.utils import convert_data_type, optional_import

PIL, _ = optional_import("PIL")
GifImage, _ = optional_import("PIL.GifImagePlugin", name="Image")

if TYPE_CHECKING:
    from tensorboard.compat.proto.summary_pb2 import Summary
    from tensorboardX import SummaryWriter as SummaryWriterX
    from tensorboardX.proto.summary_pb2 import Summary as SummaryX
    from torch.utils.tensorboard import SummaryWriter

    has_tensorboardx = True
else:
    Summary, _ = optional_import("tensorboard.compat.proto.summary_pb2", name="Summary")
    SummaryX, _ = optional_import("tensorboardX.proto.summary_pb2", name="Summary")
    SummaryWriter, _ = optional_import("torch.utils.tensorboard", name="SummaryWriter")
    SummaryWriterX, has_tensorboardx = optional_import("tensorboardX", name="SummaryWriter")

__all__ = ["make_animated_gif_summary", "add_animated_gif", "plot_2d_or_3d_image"]


def _image3_animated_gif(
    tag: str,
    image: np.ndarray | torch.Tensor,
    writer: SummaryWriter | SummaryWriterX | None,
    frame_dim: int = 0,
    scale_factor: float = 1.0,
) -> Any:
    """Function to actually create the animated gif.

    Args:
        tag: Data identifier
        image: 3D image tensors expected to be in `HWD` format
        writer: the tensorboard writer to plot image
        frame_dim: the dimension used as frames for GIF image, expect data shape as `HWD`, default to `0`.
        scale_factor: amount to multiply values by. if the image data is between 0 and 1, using 255 for this value will
            scale it to displayable range
    """
    if len(image.shape) != 3:
        raise AssertionError("3D image tensors expected to be in `HWD` format, len(image.shape) != 3")

    image_np, *_ = convert_data_type(image, output_type=np.ndarray)
    ims = [(i * scale_factor).astype(np.uint8, copy=False) for i in np.moveaxis(image_np, frame_dim, 0)]
    ims = [GifImage.fromarray(im) for im in ims]
    img_str = b""
    for b_data in PIL.GifImagePlugin.getheader(ims[0])[0]:
        img_str += b_data
    img_str += b"\x21\xFF\x0B\x4E\x45\x54\x53\x43\x41\x50" b"\x45\x32\x2E\x30\x03\x01\x00\x00\x00"
    for i in ims:
        for b_data in PIL.GifImagePlugin.getdata(i):
            img_str += b_data
    img_str += b"\x3B"

    summary = SummaryX if has_tensorboardx and isinstance(writer, SummaryWriterX) else Summary
    summary_image_str = summary.Image(height=10, width=10, colorspace=1, encoded_image_string=img_str)
    image_summary = summary.Value(tag=tag, image=summary_image_str)
    return summary(value=[image_summary])


def make_animated_gif_summary(
    tag: str,
    image: np.ndarray | torch.Tensor,
    writer: SummaryWriter | SummaryWriterX | None = None,
    max_out: int = 3,
    frame_dim: int = -3,
    scale_factor: float = 1.0,
) -> Summary:
    """Creates an animated gif out of an image tensor in 'CHWD' format and returns Summary.

    Args:
        tag: Data identifier
        image: The image, expected to be in `CHWD` format
        writer: the tensorboard writer to plot image
        max_out: maximum number of image channels to animate through
        frame_dim: the dimension used as frames for GIF image, expect input data shape as `CHWD`,
            default to `-3` (the first spatial dim)
        scale_factor: amount to multiply values by.
            if the image data is between 0 and 1, using 255 for this value will scale it to displayable range
    """

    suffix = "/image" if max_out == 1 else "/image/{}"
    # GIF image has no channel dim, reduce the spatial dim index if positive
    frame_dim = frame_dim - 1 if frame_dim > 0 else frame_dim

    summary_op = []
    for it_i in range(min(max_out, list(image.shape)[0])):
        one_channel_img: torch.Tensor | np.ndarray = (
            image[it_i, :, :, :].squeeze(dim=0) if isinstance(image, torch.Tensor) else image[it_i, :, :, :]
        )
        summary_op.append(
            _image3_animated_gif(tag + suffix.format(it_i), one_channel_img, writer, frame_dim, scale_factor)
        )
    return summary_op


def add_animated_gif(
    writer: SummaryWriter | SummaryWriterX,
    tag: str,
    image_tensor: np.ndarray | torch.Tensor,
    max_out: int = 3,
    frame_dim: int = -3,
    scale_factor: float = 1.0,
    global_step: int | None = None,
) -> None:
    """Creates an animated gif out of an image tensor in 'CHWD' format and writes it with SummaryWriter.

    Args:
        writer: Tensorboard SummaryWriter to write to
        tag: Data identifier
        image_tensor: tensor for the image to add, expected to be in `CHWD` format
        max_out: maximum number of image channels to animate through
        frame_dim: the dimension used as frames for GIF image, expect input data shape as `CHWD`,
            default to `-3` (the first spatial dim)
        scale_factor: amount to multiply values by. If the image data is between 0 and 1, using 255 for this value will
            scale it to displayable range
        global_step: Global step value to record
    """
    summary = make_animated_gif_summary(
        tag=tag, image=image_tensor, writer=writer, max_out=max_out, frame_dim=frame_dim, scale_factor=scale_factor
    )
    for s in summary:
        # add GIF for every channel separately
        writer._get_file_writer().add_summary(s, global_step)


def plot_2d_or_3d_image(
    data: NdarrayTensor | list[NdarrayTensor],
    step: int,
    writer: SummaryWriter | SummaryWriterX,
    index: int = 0,
    max_channels: int = 1,
    frame_dim: int = -3,
    max_frames: int = 24,
    tag: str = "output",
) -> None:
    """Plot 2D or 3D image on the TensorBoard, 3D image will be converted to GIF image.

    Note:
        Plot 3D or 2D image(with more than 3 channels) as separate images.
        And if writer is from TensorBoardX, data has 3 channels and `max_channels=3`, will plot as RGB video.

    Args:
        data: target data to be plotted as image on the TensorBoard.
            The data is expected to have 'NCHW[D]' dimensions or a list of data with `CHW[D]` dimensions,
            and only plot the first in the batch.
        step: current step to plot in a chart.
        writer: specify TensorBoard or TensorBoardX SummaryWriter to plot the image.
        index: plot which element in the input data batch, default is the first element.
        max_channels: number of channels to plot.
        frame_dim: if plotting 3D image as GIF, specify the dimension used as frames,
            expect input data shape as `NCHWD`, default to `-3` (the first spatial dim)
        max_frames: if plot 3D RGB image as video in TensorBoardX, set the FPS to `max_frames`.
        tag: tag of the plotted image on TensorBoard.
    """
    data_index = data[index]
    # as the `d` data has no batch dim, reduce the spatial dim index if positive
    frame_dim = frame_dim - 1 if frame_dim > 0 else frame_dim

    d: np.ndarray = (
        data_index.detach().cpu().numpy() if isinstance(data_index, torch.Tensor) else np.asarray(data_index)
    )

    if d.ndim == 2:
        d = rescale_array(d, 0, 1)  # type: ignore
        dataformats = "HW"
        writer.add_image(f"{tag}_{dataformats}", d, step, dataformats=dataformats)
        return

    if d.ndim == 3:
        if d.shape[0] == 3 and max_channels == 3:  # RGB
            dataformats = "CHW"
            writer.add_image(f"{tag}_{dataformats}", d, step, dataformats=dataformats)
            return
        dataformats = "HW"
        for j, d2 in enumerate(d[:max_channels]):
            d2 = rescale_array(d2, 0, 1)
            writer.add_image(f"{tag}_{dataformats}/{j}", d2, step, dataformats=dataformats)
        return

    if d.ndim >= 4:
        spatial = d.shape[-3:]
        d = d.reshape([-1] + list(spatial))
        if d.shape[0] == 3 and max_channels == 3 and has_tensorboardx and isinstance(writer, SummaryWriterX):  # RGB
            # move the expected frame dim to the end as `T` dim for video
            d = np.moveaxis(d, frame_dim, -1)
            writer.add_video(tag, d[None], step, fps=max_frames, dataformats="NCHWT")
            return
        # scale data to 0 - 255 for visualization
        max_channels = min(max_channels, d.shape[0])
        d = np.stack([rescale_array(i, 0, 255) for i in d[:max_channels]], axis=0)
        # will plot every channel as a separate GIF image
        add_animated_gif(writer, f"{tag}_HWD", d, max_out=max_channels, frame_dim=frame_dim, global_step=step)
        return
