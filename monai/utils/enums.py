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

import random
from enum import Enum
from typing import TYPE_CHECKING

from monai.utils.module import min_version, optional_import

__all__ = [
    "StrEnum",
    "NumpyPadMode",
    "GridSampleMode",
    "SplineMode",
    "InterpolateMode",
    "UpsampleMode",
    "BlendMode",
    "PytorchPadMode",
    "NdimageMode",
    "GridSamplePadMode",
    "Average",
    "MultiOutput",
    "MetricReduction",
    "LossReduction",
    "DiceCEReduction",
    "Weight",
    "ChannelMatching",
    "SkipMode",
    "Method",
    "TraceKeys",
    "TraceStatusKeys",
    "CommonKeys",
    "GanKeys",
    "PostFix",
    "ForwardMode",
    "TransformBackends",
    "CompInitMode",
    "BoxModeName",
    "GridPatchSort",
    "FastMRIKeys",
    "SpaceKeys",
    "MetaKeys",
    "ColorOrder",
    "EngineStatsKeys",
    "DataStatsKeys",
    "ImageStatsKeys",
    "LabelStatsKeys",
    "HoVerNetMode",
    "HoVerNetBranch",
    "LazyAttr",
    "BundleProperty",
    "BundlePropertyConfig",
    "AlgoKeys",
    "IgniteInfo",
]


class StrEnum(str, Enum):
    """
    Enum subclass that converts its value to a string.

    .. code-block:: python

        from monai.utils import StrEnum

        class Example(StrEnum):
            MODE_A = "A"
            MODE_B = "B"

        assert (list(Example) == ["A", "B"])
        assert Example.MODE_A == "A"
        assert str(Example.MODE_A) == "A"
        assert monai.utils.look_up_option("A", Example) == "A"
    """

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value


class NumpyPadMode(StrEnum):
    """
    See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
    """

    CONSTANT = "constant"
    EDGE = "edge"
    LINEAR_RAMP = "linear_ramp"
    MAXIMUM = "maximum"
    MEAN = "mean"
    MEDIAN = "median"
    MINIMUM = "minimum"
    REFLECT = "reflect"
    SYMMETRIC = "symmetric"
    WRAP = "wrap"
    EMPTY = "empty"


class NdimageMode(StrEnum):
    """
    The available options determine how the input array is extended beyond its boundaries when interpolating.
    See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
    """

    REFLECT = "reflect"
    GRID_MIRROR = "grid-mirror"
    CONSTANT = "constant"
    GRID_CONSTANT = "grid-constant"
    NEAREST = "nearest"
    MIRROR = "mirror"
    GRID_WRAP = "grid-wrap"
    WRAP = "wrap"


class GridSampleMode(StrEnum):
    """
    See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html

    interpolation mode of `torch.nn.functional.grid_sample`

    Note:
        (documentation from `torch.nn.functional.grid_sample`)
        `mode='bicubic'` supports only 4-D input.
        When `mode='bilinear'` and the input is 5-D, the interpolation mode used internally will actually be trilinear.
        However, when the input is 4-D, the interpolation mode will legitimately be bilinear.
    """

    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"


class SplineMode(StrEnum):
    """
    Order of spline interpolation.

    See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
    """

    ZERO = 0
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5


class InterpolateMode(StrEnum):
    """
    See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
    """

    NEAREST = "nearest"
    NEAREST_EXACT = "nearest-exact"
    LINEAR = "linear"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    TRILINEAR = "trilinear"
    AREA = "area"


class UpsampleMode(StrEnum):
    """
    See also: :py:class:`monai.networks.blocks.UpSample`
    """

    DECONV = "deconv"
    DECONVGROUP = "deconvgroup"
    NONTRAINABLE = "nontrainable"  # e.g. using torch.nn.Upsample
    PIXELSHUFFLE = "pixelshuffle"


class BlendMode(StrEnum):
    """
    See also: :py:class:`monai.data.utils.compute_importance_map`
    """

    CONSTANT = "constant"
    GAUSSIAN = "gaussian"


class PytorchPadMode(StrEnum):
    """
    See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
    """

    CONSTANT = "constant"
    REFLECT = "reflect"
    REPLICATE = "replicate"
    CIRCULAR = "circular"


class GridSamplePadMode(StrEnum):
    """
    See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
    """

    ZEROS = "zeros"
    BORDER = "border"
    REFLECTION = "reflection"


class Average(StrEnum):
    """
    See also: :py:class:`monai.metrics.rocauc.compute_roc_auc`
    """

    MACRO = "macro"
    WEIGHTED = "weighted"
    MICRO = "micro"
    NONE = "none"


class MultiOutput(StrEnum):
    """
    See also: :py:func:`monai.metrics.r2_score.compute_r2_score`
    """

    RAW = "raw_values"
    UNIFORM = "uniform_average"
    VARIANCE = "variance_weighted"


class MetricReduction(StrEnum):
    """
    See also: :py:func:`monai.metrics.utils.do_metric_reduction`
    """

    NONE = "none"
    MEAN = "mean"
    SUM = "sum"
    MEAN_BATCH = "mean_batch"
    SUM_BATCH = "sum_batch"
    MEAN_CHANNEL = "mean_channel"
    SUM_CHANNEL = "sum_channel"


class LossReduction(StrEnum):
    """
    See also:
        - :py:class:`monai.losses.dice.DiceLoss`
        - :py:class:`monai.losses.dice.GeneralizedDiceLoss`
        - :py:class:`monai.losses.focal_loss.FocalLoss`
        - :py:class:`monai.losses.tversky.TverskyLoss`
    """

    NONE = "none"
    MEAN = "mean"
    SUM = "sum"


class DiceCEReduction(StrEnum):
    """
    See also:
        - :py:class:`monai.losses.dice.DiceCELoss`
    """

    MEAN = "mean"
    SUM = "sum"


class Weight(StrEnum):
    """
    See also: :py:class:`monai.losses.dice.GeneralizedDiceLoss`
    """

    SQUARE = "square"
    SIMPLE = "simple"
    UNIFORM = "uniform"


class ChannelMatching(StrEnum):
    """
    See also: :py:class:`monai.networks.nets.HighResBlock`
    """

    PAD = "pad"
    PROJECT = "project"


class SkipMode(StrEnum):
    """
    See also: :py:class:`monai.networks.layers.SkipConnection`
    """

    CAT = "cat"
    ADD = "add"
    MUL = "mul"


class Method(StrEnum):
    """
    See also: :py:class:`monai.transforms.croppad.array.SpatialPad`
    """

    SYMMETRIC = "symmetric"
    END = "end"


class ForwardMode(StrEnum):
    """
    See also: :py:class:`monai.transforms.engines.evaluator.Evaluator`
    """

    TRAIN = "train"
    EVAL = "eval"


class TraceKeys(StrEnum):
    """Extra metadata keys used for traceable transforms."""

    CLASS_NAME: str = "class"
    ID: str = "id"
    ORIG_SIZE: str = "orig_size"
    EXTRA_INFO: str = "extra_info"
    DO_TRANSFORM: str = "do_transforms"
    KEY_SUFFIX: str = "_transforms"
    NONE: str = "none"
    TRACING: str = "tracing"
    STATUSES: str = "statuses"
    LAZY: str = "lazy"


class TraceStatusKeys(StrEnum):
    """Enumerable status keys for the TraceKeys.STATUS flag"""

    PENDING_DURING_APPLY = "pending_during_apply"


class CommonKeys(StrEnum):
    """
    A set of common keys for dictionary based supervised training process.
    `IMAGE` is the input image data.
    `LABEL` is the training or evaluation label of segmentation or classification task.
    `PRED` is the prediction data of model output.
    `LOSS` is the loss value of current iteration.
    `INFO` is some useful information during training or evaluation, like loss value, etc.

    """

    IMAGE = "image"
    LABEL = "label"
    PRED = "pred"
    LOSS = "loss"
    METADATA = "metadata"


class GanKeys(StrEnum):
    """
    A set of common keys for generative adversarial networks.

    """

    REALS = "reals"
    FAKES = "fakes"
    LATENTS = "latents"
    GLOSS = "g_loss"
    DLOSS = "d_loss"


class PostFix(StrEnum):
    """Post-fixes."""

    @staticmethod
    def _get_str(prefix: str | None, suffix: str) -> str:
        return suffix if prefix is None else f"{prefix}_{suffix}"

    @staticmethod
    def meta(key: str | None = None) -> str:
        return PostFix._get_str(key, "meta_dict")

    @staticmethod
    def orig_meta(key: str | None = None) -> str:
        return PostFix._get_str(key, "orig_meta_dict")

    @staticmethod
    def transforms(key: str | None = None) -> str:
        return PostFix._get_str(key, TraceKeys.KEY_SUFFIX[1:])


class TransformBackends(StrEnum):
    """
    Transform backends. Most of `monai.transforms` components first converts the input data into ``torch.Tensor`` or
    ``monai.data.MetaTensor``. Internally, some transforms are made by converting the data into ``numpy.array`` or
    ``cupy.array`` and use the underlying transform backend API to achieve the actual output array and
    converting back to ``Tensor``/``MetaTensor``. Transforms with more than one backend indicate the that they may
    convert the input data types to accommodate the underlying API.
    """

    TORCH = "torch"
    NUMPY = "numpy"
    CUPY = "cupy"


class CompInitMode(StrEnum):
    """
    Mode names for instantiating a class or calling a callable.

    See also: :py:func:`monai.utils.module.instantiate`
    """

    DEFAULT = "default"
    CALLABLE = "callable"
    DEBUG = "debug"


class JITMetadataKeys(StrEnum):
    """
    Keys stored in the metadata file for saved Torchscript models. Some of these are generated by the routines
    and others are optionally provided by users.
    """

    NAME = "name"
    TIMESTAMP = "timestamp"
    VERSION = "version"
    DESCRIPTION = "description"


class BoxModeName(StrEnum):
    """
    Box mode names.
    """

    XYXY = "xyxy"  # [xmin, ymin, xmax, ymax]
    XYZXYZ = "xyzxyz"  # [xmin, ymin, zmin, xmax, ymax, zmax]
    XXYY = "xxyy"  # [xmin, xmax, ymin, ymax]
    XXYYZZ = "xxyyzz"  # [xmin, xmax, ymin, ymax, zmin, zmax]
    XYXYZZ = "xyxyzz"  # [xmin, ymin, xmax, ymax, zmin, zmax]
    XYWH = "xywh"  # [xmin, ymin, xsize, ysize]
    XYZWHD = "xyzwhd"  # [xmin, ymin, zmin, xsize, ysize, zsize]
    CCWH = "ccwh"  # [xcenter, ycenter, xsize, ysize]
    CCCWHD = "cccwhd"  # [xcenter, ycenter, zcenter, xsize, ysize, zsize]


class ProbMapKeys(StrEnum):
    """
    The keys to be used for generating the probability maps from patches
    """

    LOCATION = "mask_location"
    SIZE = "mask_size"
    COUNT = "num_patches"
    NAME = "name"


class GridPatchSort(StrEnum):
    """
    The sorting method for the generated patches in `GridPatch`
    """

    RANDOM = "random"
    MIN = "min"
    MAX = "max"

    @staticmethod
    def min_fn(x):
        return x[0].sum()

    @staticmethod
    def max_fn(x):
        return -x[0].sum()

    @staticmethod
    def get_sort_fn(sort_fn):
        if sort_fn == GridPatchSort.RANDOM:
            return random.random
        elif sort_fn == GridPatchSort.MIN:
            return GridPatchSort.min_fn
        elif sort_fn == GridPatchSort.MAX:
            return GridPatchSort.max_fn
        else:
            raise ValueError(
                f'sort_fn should be one of the following values, "{sort_fn}" was given:',
                [e.value for e in GridPatchSort],
            )


class PatchKeys(StrEnum):
    """
    The keys to be used for metadata of patches extracted from any kind of image
    """

    LOCATION = "location"
    SIZE = "size"
    COUNT = "count"


class WSIPatchKeys(StrEnum):
    """
    The keys to be used for metadata of patches extracted from whole slide images
    """

    LOCATION = PatchKeys.LOCATION
    SIZE = PatchKeys.SIZE
    COUNT = PatchKeys.COUNT
    LEVEL = "level"
    PATH = "path"


class FastMRIKeys(StrEnum):
    """
    The keys to be used for extracting data from the fastMRI dataset
    """

    KSPACE = "kspace"
    MASK = "mask"
    FILENAME = "filename"
    RECON = "reconstruction_rss"
    ACQUISITION = "acquisition"
    MAX = "max"
    NORM = "norm"
    PID = "patient_id"


class SpaceKeys(StrEnum):
    """
    The coordinate system keys, for example, Nifti1 uses Right-Anterior-Superior or "RAS",
    DICOM (0020,0032) uses Left-Posterior-Superior or "LPS". This type does not distinguish spatial 1/2/3D.
    """

    RAS = "RAS"
    LPS = "LPS"


class MetaKeys(StrEnum):
    """
    Typical keys for MetaObj.meta
    """

    AFFINE = "affine"  # MetaTensor.affine
    ORIGINAL_AFFINE = "original_affine"  # the affine after image loading before any data processing
    SPATIAL_SHAPE = "spatial_shape"  # optional key for the length in each spatial dimension
    SPACE = "space"  # possible values of space type are defined in `SpaceKeys`
    ORIGINAL_CHANNEL_DIM = "original_channel_dim"  # an integer or float("nan")
    SAVED_TO = "saved_to"


class ColorOrder(StrEnum):
    """
    Enums for color order. Expand as necessary.
    """

    RGB = "RGB"
    BGR = "BGR"


class EngineStatsKeys(StrEnum):
    """
    Default keys for the statistics of trainer and evaluator engines.

    """

    RANK = "rank"
    CURRENT_ITERATION = "current_iteration"
    CURRENT_EPOCH = "current_epoch"
    TOTAL_EPOCHS = "total_epochs"
    TOTAL_ITERATIONS = "total_iterations"
    BEST_VALIDATION_EPOCH = "best_validation_epoch"
    BEST_VALIDATION_METRIC = "best_validation_metric"


class DataStatsKeys(StrEnum):
    """
    Defaults keys for dataset statistical analysis modules

    """

    SUMMARY = "stats_summary"
    BY_CASE = "stats_by_cases"
    BY_CASE_IMAGE_PATH = "image_filepath"
    BY_CASE_LABEL_PATH = "label_filepath"
    IMAGE_STATS = "image_stats"
    FG_IMAGE_STATS = "image_foreground_stats"
    LABEL_STATS = "label_stats"
    IMAGE_HISTOGRAM = "image_histogram"


class ImageStatsKeys(StrEnum):
    """
    Defaults keys for dataset statistical analysis image modules

    """

    SHAPE = "shape"
    CHANNELS = "channels"
    CROPPED_SHAPE = "cropped_shape"
    SPACING = "spacing"
    SIZEMM = "sizemm"
    INTENSITY = "intensity"
    HISTOGRAM = "histogram"


class LabelStatsKeys(StrEnum):
    """
    Defaults keys for dataset statistical analysis label modules

    """

    LABEL_UID = "labels"
    PIXEL_PCT = "foreground_percentage"
    IMAGE_INTST = "image_intensity"
    LABEL = "label"
    LABEL_SHAPE = "shape"
    LABEL_NCOMP = "ncomponents"


class HoVerNetMode(StrEnum):
    """
    Modes for HoVerNet model:
    `FAST`: a faster implementation (than original)
    `ORIGINAL`: the original implementation
    """

    FAST = "FAST"
    ORIGINAL = "ORIGINAL"


class HoVerNetBranch(StrEnum):
    """
    Three branches of HoVerNet model, which results in three outputs:
    `HV` is horizontal and vertical gradient map of each nucleus (regression),
    `NP` is the pixel prediction of all nuclei (segmentation), and
    `NC` is the type of each nucleus (classification).
    """

    HV = "horizontal_vertical"
    NP = "nucleus_prediction"
    NC = "type_prediction"


class LazyAttr(StrEnum):
    """
    MetaTensor with pending operations requires some key attributes tracked especially when the primary array
    is not up-to-date due to lazy evaluation.
    This class specifies the set of key attributes to be tracked for each MetaTensor.
    See also: :py:func:`monai.transforms.lazy.utils.resample` for more details.
    """

    SHAPE = "lazy_shape"  # spatial shape
    AFFINE = "lazy_affine"
    PADDING_MODE = "lazy_padding_mode"
    INTERP_MODE = "lazy_interpolation_mode"
    DTYPE = "lazy_dtype"
    ALIGN_CORNERS = "lazy_align_corners"
    RESAMPLE_MODE = "lazy_resample_mode"


class BundleProperty(StrEnum):
    """
    Bundle property fields:
    `DESC` is the description of the property.
    `REQUIRED` is flag to indicate whether the property is required or optional.
    """

    DESC = "description"
    REQUIRED = "required"


class BundlePropertyConfig(StrEnum):
    """
    additional bundle property fields for config based bundle workflow:
    `ID` is the config item ID of the property.
    `REF_ID` is the ID of config item which is supposed to refer to this property.
    For properties that do not have `REF_ID`, `None` should be set.
    this field is only useful to check the optional property ID.
    """

    ID = "id"
    REF_ID = "refer_id"


class AlgoKeys(StrEnum):
    """
    Default keys for templated Auto3DSeg Algo.
    `ID` is the identifier of the algorithm. The string has the format of <name>_<idx>_<other>.
    `ALGO` is the Auto3DSeg Algo instance.
    `IS_TRAINED` is the status that shows if the Algo has been trained.
    `SCORE` is the score the Algo has achieved after training.
    """

    ID = "identifier"
    ALGO = "algo_instance"
    IS_TRAINED = "is_trained"
    SCORE = "best_metric"


class AdversarialKeys(StrEnum):
    """
    Keys used by the AdversarialTrainer.
    `REALS` are real images from the batch.
    `FAKES` are fake images generated by the generator. Are the same as PRED.
    `REAL_LOGITS` are logits of the discriminator for the real images.
    `FAKE_LOGIT` are logits of the discriminator for the fake images.
    `RECONSTRUCTION_LOSS` is the loss value computed by the reconstruction loss function.
    `GENERATOR_LOSS` is the loss value computed by the generator loss function. It is the
                discriminator loss for the fake images. That is backpropagated through the generator only.
    `DISCRIMINATOR_LOSS` is the loss value computed by the discriminator loss function. It is the
                discriminator loss for the real images and the fake images. That is backpropagated through the
                discriminator only.
    """

    REALS = "reals"
    REAL_LOGITS = "real_logits"
    FAKES = "fakes"
    FAKE_LOGITS = "fake_logits"
    RECONSTRUCTION_LOSS = "reconstruction_loss"
    GENERATOR_LOSS = "generator_loss"
    DISCRIMINATOR_LOSS = "discriminator_loss"


class OrderingType(StrEnum):
    RASTER_SCAN = "raster_scan"
    S_CURVE = "s_curve"
    RANDOM = "random"


class OrderingTransformations(StrEnum):
    ROTATE_90 = "rotate_90"
    TRANSPOSE = "transpose"
    REFLECT = "reflect"


class IgniteInfo(StrEnum):
    """
    Config information of the PyTorch ignite package.

    """

    OPT_IMPORT_VERSION = "0.4.11"


if TYPE_CHECKING:
    from ignite.engine import EventEnum
else:
    EventEnum, _ = optional_import(
        "ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "EventEnum", as_type="base"
    )


class AdversarialIterationEvents(EventEnum):
    """
    Keys used to define events as used in the AdversarialTrainer.
    """

    RECONSTRUCTION_LOSS_COMPLETED = "reconstruction_loss_completed"
    GENERATOR_FORWARD_COMPLETED = "generator_forward_completed"
    GENERATOR_DISCRIMINATOR_FORWARD_COMPLETED = "generator_discriminator_forward_completed"
    GENERATOR_LOSS_COMPLETED = "generator_loss_completed"
    GENERATOR_BACKWARD_COMPLETED = "generator_backward_completed"
    GENERATOR_MODEL_COMPLETED = "generator_model_completed"
    DISCRIMINATOR_REALS_FORWARD_COMPLETED = "discriminator_reals_forward_completed"
    DISCRIMINATOR_FAKES_FORWARD_COMPLETED = "discriminator_fakes_forward_completed"
    DISCRIMINATOR_LOSS_COMPLETED = "discriminator_loss_completed"
    DISCRIMINATOR_BACKWARD_COMPLETED = "discriminator_backward_completed"
    DISCRIMINATOR_MODEL_COMPLETED = "discriminator_model_completed"
