:github_url: https://github.com/Project-MONAI/MONAI

.. _transform_api:

Transforms
==========

Generic Interfaces
------------------
.. automodule:: monai.transforms
.. currentmodule:: monai.transforms

`Transform`
^^^^^^^^^^^
.. autoclass:: Transform
    :members:
    :special-members: __call__

`MapTransform`
^^^^^^^^^^^^^^
.. autoclass:: MapTransform
    :members:
    :special-members: __call__

`RandomizableTrait`
^^^^^^^^^^^^^^^^^^^
.. autoclass:: RandomizableTrait
    :members:

`LazyTrait`
^^^^^^^^^^^
.. autoclass:: LazyTrait
    :members:

`MultiSampleTrait`
^^^^^^^^^^^^^^^^^^
.. autoclass:: MultiSampleTrait
    :members:

`Randomizable`
^^^^^^^^^^^^^^
.. autoclass:: Randomizable
    :members:

`LazyTransform`
^^^^^^^^^^^^^^^
.. autoclass:: LazyTransform
    :members:

`RandomizableTransform`
^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: RandomizableTransform
    :members:

`Compose`
^^^^^^^^^
.. autoclass:: Compose
    :members:
    :special-members: __call__

`InvertibleTransform`
^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: InvertibleTransform
    :members:

`TraceableTransform`
^^^^^^^^^^^^^^^^^^^^
.. autoclass:: TraceableTransform
    :members:

`BatchInverseTransform`
^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: BatchInverseTransform
    :members:

`Decollated`
^^^^^^^^^^^^
.. autoclass:: Decollated
    :members:

`OneOf`
^^^^^^^
.. autoclass:: OneOf
    :members:

`RandomOrder`
^^^^^^^^^^^^^
.. autoclass:: RandomOrder
    :members:

`SomeOf`
^^^^^^^^^^^^^
.. autoclass:: SomeOf
    :members:

Functionals
-----------

Crop and Pad (functional)
^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: monai.transforms.croppad.functional
    :members:

Spatial (functional)
^^^^^^^^^^^^^^^^^^^^
.. automodule:: monai.transforms.spatial.functional
    :members:

.. currentmodule:: monai.transforms

Vanilla Transforms
------------------

Crop and Pad
^^^^^^^^^^^^

`PadListDataCollate`
""""""""""""""""""""
.. autoclass:: PadListDataCollate
    :members:
    :special-members: __call__

`Pad`
"""""
.. autoclass:: Pad
    :members:
    :special-members: __call__

`SpatialPad`
""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/SpatialPad.png
    :alt: example of SpatialPad
.. autoclass:: SpatialPad
    :members:
    :special-members: __call__

`BorderPad`
"""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/BorderPad.png
    :alt: example of BorderPad
.. autoclass:: BorderPad
    :members:
    :special-members: __call__

`DivisiblePad`
""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/DivisiblePad.png
    :alt: example of DivisiblePad
.. autoclass:: DivisiblePad
    :members:
    :special-members: __call__

`Crop`
""""""
.. autoclass:: Crop
    :members:
    :special-members: __call__

`SpatialCrop`
"""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/SpatialCrop.png
    :alt: example of SpatialCrop
.. autoclass:: SpatialCrop
    :members:
    :special-members: __call__

`CenterSpatialCrop`
"""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/CenterSpatialCrop.png
    :alt: example of CenterSpatialCrop
.. autoclass:: CenterSpatialCrop
    :members:
    :special-members: __call__

`RandSpatialCrop`
"""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RandSpatialCrop.png
    :alt: example of RandSpatialCrop
.. autoclass:: RandSpatialCrop
    :members:
    :special-members: __call__

`RandSpatialCropSamples`
""""""""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RandSpatialCropSamples.png
    :alt: example of RandSpatialCropSamples
.. autoclass:: RandSpatialCropSamples
    :members:
    :special-members: __call__

`CropForeground`
""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/CropForeground.png
    :alt: example of CropForeground
.. autoclass:: CropForeground
    :members:
    :special-members: __call__

`RandWeightedCrop`
""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RandWeightedCrop.png
    :alt: example of RandWeightedCrop
.. autoclass:: RandWeightedCrop
    :members:
    :special-members: __call__

`RandCropByPosNegLabel`
"""""""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RandCropByPosNegLabel.png
    :alt: example of RandCropByPosNegLabel
.. autoclass:: RandCropByPosNegLabel
    :members:
    :special-members: __call__

`RandCropByLabelClasses`
""""""""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RandCropByLabelClasses.png
    :alt: example of RandCropByLabelClasses
.. autoclass:: RandCropByLabelClasses
    :members:
    :special-members: __call__

`ResizeWithPadOrCrop`
"""""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/ResizeWithPadOrCrop.png
    :alt: example of ResizeWithPadOrCrop
.. autoclass:: ResizeWithPadOrCrop
    :members:
    :special-members: __call__

`BoundingRect`
""""""""""""""
.. autoclass:: BoundingRect
    :members:
    :special-members: __call__

`RandScaleCrop`
"""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RandScaleCrop.png
    :alt: example of RandScaleCrop
.. autoclass:: RandScaleCrop
    :members:
    :special-members: __call__

`CenterScaleCrop`
"""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/CenterScaleCrop.png
    :alt: example of CenterScaleCrop
.. autoclass:: CenterScaleCrop
    :members:
    :special-members: __call__

Intensity
^^^^^^^^^

`RandGaussianNoise`
"""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RandGaussianNoise.png
    :alt: example of RandGaussianNoise
.. autoclass:: RandGaussianNoise
    :members:
    :special-members: __call__

`ShiftIntensity`
""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/ShiftIntensity.png
    :alt: example of ShiftIntensity
.. autoclass:: ShiftIntensity
    :members:
    :special-members: __call__

`RandShiftIntensity`
""""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RandShiftIntensity.png
    :alt: example of RandShiftIntensity
.. autoclass:: RandShiftIntensity
    :members:
    :special-members: __call__

`StdShiftIntensity`
"""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/StdShiftIntensity.png
    :alt: example of StdShiftIntensity
.. autoclass:: StdShiftIntensity
    :members:
    :special-members: __call__

`RandStdShiftIntensity`
"""""""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RandStdShiftIntensity.png
    :alt: example of RandStdShiftIntensity
.. autoclass:: RandStdShiftIntensity
    :members:
    :special-members: __call__

`RandBiasField`
"""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RandBiasField.png
    :alt: example of RandBiasField
.. autoclass:: RandBiasField
    :members:
    :special-members: __call__

`ScaleIntensity`
""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/ScaleIntensity.png
    :alt: example of ScaleIntensity
.. autoclass:: ScaleIntensity
    :members:
    :special-members: __call__

`ClipIntensityPercentiles`
""""""""""""""""""""""""""
.. autoclass:: ClipIntensityPercentiles
    :members:
    :special-members: __call__

`RandScaleIntensity`
""""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RandScaleIntensity.png
    :alt: example of RandScaleIntensity
.. autoclass:: RandScaleIntensity
    :members:
    :special-members: __call__

`ScaleIntensityFixedMean`
"""""""""""""""""""""""""
.. autoclass:: ScaleIntensityFixedMean
    :members:
    :special-members: __call__

`RandScaleIntensityFixedMean`
"""""""""""""""""""""""""""""
.. autoclass:: RandScaleIntensityFixedMean
    :members:
    :special-members: __call__

`NormalizeIntensity`
""""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/NormalizeIntensity.png
    :alt: example of NormalizeIntensity
.. autoclass:: NormalizeIntensity
    :members:
    :special-members: __call__

`ThresholdIntensity`
""""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/ThresholdIntensity.png
    :alt: example of ThresholdIntensity
.. autoclass:: ThresholdIntensity
    :members:
    :special-members: __call__

`ScaleIntensityRange`
"""""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/ScaleIntensityRange.png
    :alt: example of ScaleIntensityRange
.. autoclass:: ScaleIntensityRange
    :members:
    :special-members: __call__

`ScaleIntensityRangePercentiles`
""""""""""""""""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/ScaleIntensityRangePercentiles.png
    :alt: example of ScaleIntensityRangePercentiles
.. autoclass:: ScaleIntensityRangePercentiles
    :members:
    :special-members: __call__

`AdjustContrast`
""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/AdjustContrast.png
    :alt: example of AdjustContrast
.. autoclass:: AdjustContrast
    :members:
    :special-members: __call__

`RandAdjustContrast`
""""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RandAdjustContrast.png
    :alt: example of RandAdjustContrast
.. autoclass:: RandAdjustContrast
    :members:
    :special-members: __call__

`MaskIntensity`
"""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/MaskIntensity.png
    :alt: example of MaskIntensity
.. autoclass:: MaskIntensity
    :members:
    :special-members: __call__

`SavitzkyGolaySmooth`
"""""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/SavitzkyGolaySmooth.png
    :alt: example of SavitzkyGolaySmooth
.. autoclass:: SavitzkyGolaySmooth
    :members:
    :special-members: __call__

`MedianSmooth`
""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/MedianSmooth.png
    :alt: example of MedianSmooth
.. autoclass:: MedianSmooth
    :members:
    :special-members: __call__

`GaussianSmooth`
""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/GaussianSmooth.png
    :alt: example of GaussianSmooth
.. autoclass:: GaussianSmooth
    :members:
    :special-members: __call__

`RandGaussianSmooth`
""""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RandGaussianSmooth.png
    :alt: example of RandGaussianSmooth
.. autoclass:: RandGaussianSmooth
    :members:
    :special-members: __call__

`GaussianSharpen`
"""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/GaussianSharpen.png
    :alt: example of GaussianSharpen
.. autoclass:: GaussianSharpen
    :members:
    :special-members: __call__

`RandGaussianSharpen`
"""""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RandGaussianSharpen.png
    :alt: example of RandGaussianSharpen
.. autoclass:: RandGaussianSharpen
    :members:
    :special-members: __call__

`RandHistogramShift`
""""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RandHistogramShift.png
    :alt: example of RandHistogramShift
.. autoclass:: RandHistogramShift
    :members:
    :special-members: __call__

`DetectEnvelope`
""""""""""""""""
.. autoclass:: DetectEnvelope
    :members:
    :special-members: __call__

`GibbsNoise`
""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/GibbsNoise.png
    :alt: example of GibbsNoise
.. autoclass:: GibbsNoise
    :members:
    :special-members: __call__

`RandGibbsNoise`
""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RandGibbsNoise.png
    :alt: example of RandGibbsNoise
.. autoclass:: RandGibbsNoise
    :members:
    :special-members: __call__

`KSpaceSpikeNoise`
""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/KSpaceSpikeNoise.png
    :alt: example of KSpaceSpikeNoise
.. autoclass:: KSpaceSpikeNoise
    :members:
    :special-members: __call__

`RandKSpaceSpikeNoise`
""""""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RandKSpaceSpikeNoise.png
    :alt: example of RandKSpaceSpikeNoise
.. autoclass:: RandKSpaceSpikeNoise
    :members:
    :special-members: __call__

`RandRicianNoise`
"""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RandRicianNoise.png
    :alt: example of RandRicianNoise
.. autoclass:: RandRicianNoise
    :members:
    :special-members: __call__

`RandCoarseTransform`
"""""""""""""""""""""
.. autoclass:: RandCoarseTransform
    :members:
    :special-members: __call__

`RandCoarseDropout`
"""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RandCoarseDropout.png
    :alt: example of RandCoarseDropout
.. autoclass:: RandCoarseDropout
    :members:
    :special-members: __call__

`RandCoarseShuffle`
"""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RandCoarseShuffle.png
    :alt: example of RandCoarseShuffle
.. autoclass:: RandCoarseShuffle
    :members:
    :special-members: __call__

`HistogramNormalize`
""""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/HistogramNormalize.png
    :alt: example of HistogramNormalize
.. autoclass:: HistogramNormalize
    :members:
    :special-members: __call__


`ForegroundMask`
""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/ForegroundMask.png
    :alt: example of ForegroundMask
.. autoclass:: ForegroundMask
    :members:
    :special-members: __call__

`ComputeHoVerMaps`
""""""""""""""""""
.. autoclass:: ComputeHoVerMaps
    :members:
    :special-members: __call__


IO
^^

`LoadImage`
"""""""""""
.. autoclass:: LoadImage
    :members:
    :special-members: __call__

`SaveImage`
"""""""""""
.. autoclass:: SaveImage
    :members:
    :special-members: __call__

`WriteFileMapping`
""""""""""""""""""
.. autoclass:: WriteFileMapping
    :members:
    :special-members: __call__


NVIDIA Tool Extension (NVTX)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`RangePush`
"""""""""""
.. autoclass:: RangePush

`RandRangePush`
"""""""""""""""
.. autoclass:: RandRangePush

`RangePop`
""""""""""
.. autoclass:: RangePop

`RandRangePop`
""""""""""""""
.. autoclass:: RandRangePop

`Mark`
""""""
.. autoclass:: Mark

`RandMark`
""""""""""
.. autoclass:: RandMark


Post-processing
^^^^^^^^^^^^^^^

`Activations`
"""""""""""""
.. autoclass:: Activations
    :members:
    :special-members: __call__

`AsDiscrete`
""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/AsDiscrete.png
    :alt: example of AsDiscrete
.. autoclass:: AsDiscrete
    :members:
    :special-members: __call__

`KeepLargestConnectedComponent`
"""""""""""""""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/KeepLargestConnectedComponent.png
    :alt: example of KeepLargestConnectedComponent
.. autoclass:: KeepLargestConnectedComponent
    :members:
    :special-members: __call__

`DistanceTransformEDT`
"""""""""""""""""""""""""""""""
.. autoclass:: DistanceTransformEDT
    :members:
    :special-members: __call__

`RemoveSmallObjects`
""""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RemoveSmallObjects.png
    :alt: example of RemoveSmallObjects
.. autoclass:: RemoveSmallObjects
    :members:
    :special-members: __call__

`LabelFilter`
"""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/LabelFilter.png
    :alt: example of LabelFilter
.. autoclass:: LabelFilter
    :members:
    :special-members: __call__

`FillHoles`
"""""""""""
.. autoclass:: FillHoles
    :members:
    :special-members: __call__

`LabelToContour`
""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/LabelToContour.png
    :alt: example of LabelToContour
.. autoclass:: LabelToContour
    :members:
    :special-members: __call__

`MeanEnsemble`
""""""""""""""
.. autoclass:: MeanEnsemble
    :members:
    :special-members: __call__

`ProbNMS`
"""""""""
.. autoclass:: ProbNMS
  :members:

`SobelGradients`
""""""""""""""""
.. autoclass:: SobelGradients
  :members:
  :special-members: __call__

`VoteEnsemble`
""""""""""""""
.. autoclass:: VoteEnsemble
    :members:
    :special-members: __call__

`Invert`
"""""""""
.. autoclass:: Invert
    :members:
    :special-members: __call__

Regularization
^^^^^^^^^^^^^^

`CutMix`
""""""""
.. autoclass:: CutMix
    :members:
    :special-members: __call__

`CutOut`
""""""""
.. autoclass:: CutOut
    :members:
    :special-members: __call__

`MixUp`
"""""""
.. autoclass:: MixUp
    :members:
    :special-members: __call__

Signal
^^^^^^^

`SignalRandDrop`
""""""""""""""""
.. autoclass:: SignalRandDrop
    :members:
    :special-members: __call__

`SignalRandScale`
"""""""""""""""""
.. autoclass:: SignalRandScale
    :members:
    :special-members: __call__

`SignalRandShift`
"""""""""""""""""
.. autoclass:: SignalRandShift
    :members:
    :special-members: __call__

`SignalRandAddSine`
"""""""""""""""""""
.. autoclass:: SignalRandAddSine
    :members:
    :special-members: __call__

`SignalRandAddSquarePulse`
""""""""""""""""""""""""""
.. autoclass:: SignalRandAddSquarePulse
    :members:
    :special-members: __call__

`SignalRandAddGaussianNoise`
""""""""""""""""""""""""""""
.. autoclass:: SignalRandAddGaussianNoise
    :members:
    :special-members: __call__

`SignalRandAddSinePartial`
""""""""""""""""""""""""""
.. autoclass:: SignalRandAddSinePartial
    :members:
    :special-members: __call__

`SignalRandAddSquarePulsePartial`
"""""""""""""""""""""""""""""""""
.. autoclass:: SignalRandAddSquarePulsePartial
    :members:
    :special-members: __call__

`SignalFillEmpty`
"""""""""""""""""
.. autoclass:: SignalFillEmpty
    :members:
    :special-members: __call__

`SignalRemoveFrequency`
"""""""""""""""""""""""
.. autoclass:: SignalRemoveFrequency
    :members:
    :special-members: __call__

`SignalContinuousWavelet`
"""""""""""""""""""""""""
.. autoclass:: SignalContinuousWavelet
    :members:
    :special-members: __call__

Spatial
^^^^^^^

`SpatialResample`
"""""""""""""""""
.. autoclass:: SpatialResample
    :members:
    :special-members: __call__

`ResampleToMatch`
"""""""""""""""""
.. autoclass:: ResampleToMatch
    :members:
    :special-members: __call__

`Spacing`
"""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/Spacing.png
    :alt: example of Spacing
.. autoclass:: Spacing
    :members:
    :special-members: __call__

`Orientation`
"""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/Orientation.png
    :alt: example of Orientation
.. autoclass:: Orientation
    :members:
    :special-members: __call__

`RandRotate`
""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RandRotate.png
    :alt: example of RandRotate
.. autoclass:: RandRotate
    :members:
    :special-members: __call__

`RandFlip`
""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RandFlip.png
    :alt: example of RandFlip
.. autoclass:: RandFlip
    :members:
    :special-members: __call__

`RandAxisFlip`
""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RandAxisFlip.png
    :alt: example of RandAxisFlip
.. autoclass:: RandAxisFlip
    :members:
    :special-members: __call__

`RandZoom`
""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RandZoom.png
    :alt: example of RandZoom
.. autoclass:: RandZoom
    :members:
    :special-members: __call__

`Affine`
""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/Affine.png
    :alt: example of Affine
.. autoclass:: Affine
    :members:
    :special-members: __call__

`Resample`
""""""""""
.. autoclass:: Resample
    :members:
    :special-members: __call__

`RandAffine`
""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RandAffine.png
    :alt: example of RandAffine
.. autoclass:: RandAffine
    :members:
    :special-members: __call__

`RandDeformGrid`
""""""""""""""""
.. autoclass:: RandDeformGrid
    :members:
    :special-members: __call__

`AffineGrid`
""""""""""""""""
.. autoclass:: AffineGrid
    :members:
    :special-members: __call__

`RandAffineGrid`
""""""""""""""""
.. autoclass:: RandAffineGrid
    :members:
    :special-members: __call__

`GridDistortion`
""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/GridDistortion.png
    :alt: example of GridDistortion
.. autoclass:: GridDistortion
    :members:
    :special-members: __call__

`RandGridDistortion`
""""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RandGridDistortion.png
    :alt: example of RandGridDistortion
.. autoclass:: RandGridDistortion
    :members:
    :special-members: __call__

`Rand2DElastic`
"""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/Rand2DElastic.png
    :alt: example of Rand2DElastic
.. autoclass:: Rand2DElastic
    :members:
    :special-members: __call__

`Rand3DElastic`
"""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/Rand3DElastic.png
    :alt: example of Rand3DElastic
.. autoclass:: Rand3DElastic
    :members:
    :special-members: __call__

`Rotate90`
""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/Rotate90.png
    :alt: example of Rotate90
.. autoclass:: Rotate90
    :members:
    :special-members: __call__

`RandRotate90`
""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RandRotate90.png
    :alt: example of RandRotate90
.. autoclass:: RandRotate90
    :members:
    :special-members: __call__

`Flip`
""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/Flip.png
    :alt: example of Flip
.. autoclass:: Flip
    :members:
    :special-members: __call__

`Resize`
""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/Resize.png
    :alt: example of Resize
.. autoclass:: Resize
    :members:
    :special-members: __call__

`Rotate`
""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/Rotate.png
    :alt: example of Rotate
.. autoclass:: Rotate
    :members:
    :special-members: __call__

`Zoom`
""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/Zoom.png
    :alt: example of Zoom
.. autoclass:: Zoom
    :members:
    :special-members: __call__

`GridPatch`
"""""""""""
.. autoclass:: GridPatch
    :members:
    :special-members: __call__

`RandGridPatch`
"""""""""""""""
.. autoclass:: RandGridPatch
    :members:
    :special-members: __call__

`GridSplit`
"""""""""""
.. autoclass:: GridSplit
    :members:
    :special-members: __call__

`RandSimulateLowResolution`
"""""""""""""""""""""""""""
.. autoclass:: RandSimulateLowResolution
    :members:
    :special-members: __call__

`ConvertBoxToPoints`
""""""""""""""""""""
.. autoclass:: ConvertBoxToPoints
    :members:
    :special-members: __call__

`ConvertPointsToBoxes`
""""""""""""""""""""""
.. autoclass:: ConvertPointsToBoxes
    :members:
    :special-members: __call__


Smooth Field
^^^^^^^^^^^^

`RandSmoothFieldAdjustContrast`
"""""""""""""""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RandSmoothFieldAdjustContrast.png
    :alt: example of RandSmoothFieldAdjustContrast
.. autoclass:: RandSmoothFieldAdjustContrast
    :members:
    :special-members: __call__

`RandSmoothFieldAdjustIntensity`
""""""""""""""""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RandSmoothFieldAdjustIntensity.png
    :alt: example of RandSmoothFieldAdjustIntensity
.. autoclass:: RandSmoothFieldAdjustIntensity
    :members:
    :special-members: __call__

`RandSmoothDeform`
""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RandSmoothDeform.png
    :alt: example of RandSmoothDeform
.. autoclass:: RandSmoothDeform
    :members:
    :special-members: __call__


MRI Transforms
^^^^^^^^^^^^^^

`Kspace under-sampling`
"""""""""""""""""""""""
.. autoclass:: monai.apps.reconstruction.transforms.array.KspaceMask
    :members:
    :special-members: __call__

.. autoclass:: monai.apps.reconstruction.transforms.array.RandomKspaceMask
    :special-members: __call__

.. autoclass:: monai.apps.reconstruction.transforms.array.EquispacedKspaceMask
    :special-members: __call__


Lazy
^^^^

`ApplyPending`
""""""""""""""

.. autoclass:: ApplyPending
    :members:
    :special-members: __call__


Utility
^^^^^^^

`Identity`
""""""""""
.. autoclass:: Identity
    :members:
    :special-members: __call__

`AsChannelLast`
"""""""""""""""
.. autoclass:: AsChannelLast
    :members:
    :special-members: __call__

`EnsureChannelFirst`
""""""""""""""""""""
.. autoclass:: EnsureChannelFirst
    :members:
    :special-members: __call__

`RepeatChannel`
"""""""""""""""
.. autoclass:: RepeatChannel
    :members:
    :special-members: __call__

`SplitDim`
""""""""""
.. autoclass:: SplitDim
    :members:
    :special-members: __call__

`CastToType`
""""""""""""
.. autoclass:: CastToType
    :members:
    :special-members: __call__

`ToTensor`
""""""""""
.. autoclass:: ToTensor
    :members:
    :special-members: __call__

`ToNumpy`
"""""""""
.. autoclass:: ToNumpy
    :members:
    :special-members: __call__

`ToCupy`
""""""""
.. autoclass:: ToCupy
    :members:
    :special-members: __call__

`Transpose`
"""""""""""
.. autoclass:: Transpose
    :members:
    :special-members: __call__

`SqueezeDim`
""""""""""""
.. autoclass:: SqueezeDim
    :members:
    :special-members: __call__

`DataStats`
"""""""""""
.. autoclass:: DataStats
    :members:
    :special-members: __call__

`SimulateDelay`
"""""""""""""""
.. autoclass:: SimulateDelay
    :members:
    :special-members: __call__


`Lambda`
""""""""
.. autoclass:: Lambda
    :members:
    :special-members: __call__

`RandLambda`
""""""""""""
.. autoclass:: RandLambda
    :members:
    :special-members: __call__

`RemoveRepeatedChannel`
"""""""""""""""""""""""
.. autoclass:: RemoveRepeatedChannel
    :members:
    :special-members: __call__

`LabelToMask`
"""""""""""""
.. autoclass:: LabelToMask
    :members:
    :special-members: __call__

`FgBgToIndices`
"""""""""""""""
.. autoclass:: FgBgToIndices
    :members:
    :special-members: __call__

`ClassesToIndices`
""""""""""""""""""
.. autoclass:: ClassesToIndices
    :members:
    :special-members: __call__

`ConvertToMultiChannelBasedOnBratsClasses`
""""""""""""""""""""""""""""""""""""""""""
.. autoclass:: ConvertToMultiChannelBasedOnBratsClasses
    :members:
    :special-members: __call__

`AddExtremePointsChannel`
"""""""""""""""""""""""""
.. autoclass:: AddExtremePointsChannel
    :members:
    :special-members: __call__

`TorchVision`
"""""""""""""
.. autoclass:: TorchVision
    :members:
    :special-members: __call__

`TorchIO`
"""""""""
.. autoclass:: TorchIO
    :members:
    :special-members: __call__

`RandTorchIO`
"""""""""""""
.. autoclass:: RandTorchIO
    :members:
    :special-members: __call__

`MapLabelValue`
"""""""""""""""
.. autoclass:: MapLabelValue
    :members:
    :special-members: __call__

`EnsureType`
""""""""""""
.. autoclass:: EnsureType
    :members:
    :special-members: __call__

`IntensityStats`
""""""""""""""""
.. autoclass:: IntensityStats
    :members:
    :special-members: __call__

`ToDevice`
""""""""""
.. autoclass:: ToDevice
     :members:
     :special-members: __call__

`CuCIM`
"""""""
.. autoclass:: CuCIM
    :members:
    :special-members: __call__

`RandCuCIM`
"""""""""""
.. autoclass:: RandCuCIM
    :members:
    :special-members: __call__

`AddCoordinateChannels`
"""""""""""""""""""""""
.. autoclass:: AddCoordinateChannels
    :members:
    :special-members: __call__

`ImageFilter`
"""""""""""""
.. autoclass:: ImageFilter
    :members:
    :special-members: __call__

`RandImageFilter`
"""""""""""""""""
.. autoclass:: RandImageFilter
    :members:
    :special-members: __call__

`ApplyTransformToPoints`
""""""""""""""""""""""""
.. autoclass:: ApplyTransformToPoints
    :members:
    :special-members: __call__

Dictionary Transforms
---------------------

Crop and Pad (Dict)
^^^^^^^^^^^^^^^^^^^

`Padd`
""""""
.. autoclass:: Padd
    :members:
    :special-members: __call__

`SpatialPadd`
"""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/SpatialPadd.png
    :alt: example of SpatialPadd
.. autoclass:: SpatialPadd
    :members:
    :special-members: __call__

`BorderPadd`
""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/BorderPadd.png
    :alt: example of BorderPadd
.. autoclass:: BorderPadd
    :members:
    :special-members: __call__

`DivisiblePadd`
"""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/DivisiblePadd.png
    :alt: example of DivisiblePadd
.. autoclass:: DivisiblePadd
    :members:
    :special-members: __call__

`Cropd`
"""""""
.. autoclass:: Cropd
    :members:
    :special-members: __call__

`RandCropd`
"""""""""""
.. autoclass:: RandCropd
    :members:
    :special-members: __call__

`SpatialCropd`
""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/SpatialCropd.png
    :alt: example of SpatialCropd
.. autoclass:: SpatialCropd
    :members:
    :special-members: __call__

`CenterSpatialCropd`
""""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/CenterSpatialCropd.png
    :alt: example of CenterSpatialCropd
.. autoclass:: CenterSpatialCropd
    :members:
    :special-members: __call__

`RandSpatialCropd`
""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RandSpatialCropd.png
    :alt: example of RandSpatialCropd
.. autoclass:: RandSpatialCropd
    :members:
    :special-members: __call__

`RandSpatialCropSamplesd`
"""""""""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RandSpatialCropSamplesd.png
    :alt: example of RandSpatialCropSamplesd
.. autoclass:: RandSpatialCropSamplesd
    :members:
    :special-members: __call__

`CropForegroundd`
"""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/CropForegroundd.png
    :alt: example of CropForegroundd
.. autoclass:: CropForegroundd
    :members:
    :special-members: __call__

`RandWeightedCropd`
"""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RandWeightedCropd.png
    :alt: example of RandWeightedCropd
.. autoclass:: RandWeightedCropd
    :members:
    :special-members: __call__

`RandCropByPosNegLabeld`
""""""""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RandCropByPosNegLabeld.png
    :alt: example of RandCropByPosNegLabeld
.. autoclass:: RandCropByPosNegLabeld
    :members:
    :special-members: __call__

`RandCropByLabelClassesd`
"""""""""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RandCropByLabelClassesd.png
    :alt: example of RandCropByLabelClassesd
.. autoclass:: RandCropByLabelClassesd
    :members:
    :special-members: __call__

`ResizeWithPadOrCropd`
""""""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/ResizeWithPadOrCropd.png
    :alt: example of ResizeWithPadOrCropd
.. autoclass:: ResizeWithPadOrCropd
    :members:
    :special-members: __call__

`BoundingRectd`
"""""""""""""""
.. autoclass:: BoundingRectd
    :members:
    :special-members: __call__

`RandScaleCropd`
""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RandScaleCropd.png
    :alt: example of RandScaleCropd
.. autoclass:: RandScaleCropd
    :members:
    :special-members: __call__

`CenterScaleCropd`
""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/CenterScaleCropd.png
    :alt: example of CenterScaleCropd
.. autoclass:: CenterScaleCropd
    :members:
    :special-members: __call__

Intensity (Dict)
^^^^^^^^^^^^^^^^

`RandGaussianNoised`
""""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RandGaussianNoised.png
    :alt: example of RandGaussianNoised
.. autoclass:: RandGaussianNoised
    :members:
    :special-members: __call__

`ShiftIntensityd`
"""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/ShiftIntensityd.png
    :alt: example of ShiftIntensityd
.. autoclass:: ShiftIntensityd
    :members:
    :special-members: __call__

`RandShiftIntensityd`
"""""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RandShiftIntensityd.png
    :alt: example of RandShiftIntensityd
.. autoclass:: RandShiftIntensityd
    :members:
    :special-members: __call__

`StdShiftIntensityd`
""""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/StdShiftIntensityd.png
    :alt: example of StdShiftIntensityd
.. autoclass:: StdShiftIntensityd
    :members:
    :special-members: __call__

`RandStdShiftIntensityd`
""""""""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RandStdShiftIntensityd.png
    :alt: example of RandStdShiftIntensityd
.. autoclass:: RandStdShiftIntensityd
    :members:
    :special-members: __call__

`RandBiasFieldd`
""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RandBiasFieldd.png
    :alt: example of RandBiasFieldd
.. autoclass:: RandBiasFieldd
    :members:
    :special-members: __call__

`ScaleIntensityd`
"""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/ScaleIntensityd.png
    :alt: example of ScaleIntensityd
.. autoclass:: ScaleIntensityd
    :members:
    :special-members: __call__

`ClipIntensityPercentilesd`
"""""""""""""""""""""""""""
.. autoclass:: ClipIntensityPercentilesd
    :members:
    :special-members: __call__

`RandScaleIntensityd`
"""""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RandScaleIntensityd.png
    :alt: example of RandScaleIntensityd
.. autoclass:: RandScaleIntensityd
    :members:
    :special-members: __call__

`RandScaleIntensityFixedMeand`
"""""""""""""""""""""""""""""""
.. autoclass:: RandScaleIntensityFixedMeand
    :members:
    :special-members: __call__

`NormalizeIntensityd`
"""""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/NormalizeIntensityd.png
    :alt: example of NormalizeIntensityd
.. autoclass:: NormalizeIntensityd
    :members:
    :special-members: __call__

`ThresholdIntensityd`
"""""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/ThresholdIntensityd.png
    :alt: example of ThresholdIntensityd
.. autoclass:: ThresholdIntensityd
    :members:
    :special-members: __call__

`ScaleIntensityRanged`
""""""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/ScaleIntensityRanged.png
    :alt: example of ScaleIntensityRanged
.. autoclass:: ScaleIntensityRanged
    :members:
    :special-members: __call__

`GibbsNoised`
""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/GibbsNoised.png
    :alt: example of GibbsNoised
.. autoclass:: GibbsNoised
    :members:
    :special-members: __call__

`RandGibbsNoised`
""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RandGibbsNoised.png
    :alt: example of RandGibbsNoised
.. autoclass:: RandGibbsNoised
    :members:
    :special-members: __call__

`KSpaceSpikeNoised`
""""""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/KSpaceSpikeNoised.png
    :alt: example of KSpaceSpikeNoised
.. autoclass:: KSpaceSpikeNoised
    :members:
    :special-members: __call__

`RandKSpaceSpikeNoised`
"""""""""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RandKSpaceSpikeNoised.png
    :alt: example of RandKSpaceSpikeNoised
.. autoclass:: RandKSpaceSpikeNoised
    :members:
    :special-members: __call__

`RandRicianNoised`
""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RandRicianNoised.png
    :alt: example of RandRicianNoised
.. autoclass:: RandRicianNoised
    :members:
    :special-members: __call__

`ScaleIntensityRangePercentilesd`
"""""""""""""""""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/ScaleIntensityRangePercentilesd.png
    :alt: example of ScaleIntensityRangePercentilesd
.. autoclass:: ScaleIntensityRangePercentilesd
    :members:
    :special-members: __call__

`AdjustContrastd`
"""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/AdjustContrastd.png
    :alt: example of AdjustContrastd
.. autoclass:: AdjustContrastd
    :members:
    :special-members: __call__

`RandAdjustContrastd`
"""""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RandAdjustContrastd.png
    :alt: example of RandAdjustContrastd
.. autoclass:: RandAdjustContrastd
    :members:
    :special-members: __call__

`MaskIntensityd`
""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/MaskIntensityd.png
    :alt: example of MaskIntensityd
.. autoclass:: MaskIntensityd
    :members:
    :special-members: __call__

`SavitzkyGolaySmoothd`
""""""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/SavitzkyGolaySmoothd.png
    :alt: example of SavitzkyGolaySmoothd
.. autoclass:: SavitzkyGolaySmoothd
    :members:
    :special-members: __call__

`MedianSmoothd`
"""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/MedianSmoothd.png
    :alt: example of MedianSmoothd
.. autoclass:: MedianSmoothd
    :members:
    :special-members: __call__

`GaussianSmoothd`
"""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/GaussianSmoothd.png
    :alt: example of GaussianSmoothd
.. autoclass:: GaussianSmoothd
    :members:
    :special-members: __call__

`RandGaussianSmoothd`
"""""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RandGaussianSmoothd.png
    :alt: example of RandGaussianSmoothd
.. autoclass:: RandGaussianSmoothd
    :members:
    :special-members: __call__

`GaussianSharpend`
""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/GaussianSharpend.png
    :alt: example of GaussianSharpend
.. autoclass:: GaussianSharpend
    :members:
    :special-members: __call__

`RandGaussianSharpend`
""""""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RandGaussianSharpend.png
    :alt: example of RandGaussianSharpend
.. autoclass:: RandGaussianSharpend
    :members:
    :special-members: __call__

`RandHistogramShiftd`
"""""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RandHistogramShiftd.png
    :alt: example of RandHistogramShiftd
.. autoclass:: RandHistogramShiftd
    :members:
    :special-members: __call__

`RandCoarseDropoutd`
""""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RandCoarseDropoutd.png
    :alt: example of RandCoarseDropoutd
.. autoclass:: RandCoarseDropoutd
    :members:
    :special-members: __call__

`RandCoarseShuffled`
""""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RandCoarseShuffled.png
    :alt: example of RandCoarseShuffled
.. autoclass:: RandCoarseShuffled
    :members:
    :special-members: __call__

`HistogramNormalized`
"""""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/HistogramNormalized.png
    :alt: example of HistogramNormalized
.. autoclass:: HistogramNormalized
    :members:
    :special-members: __call__

`ForegroundMaskd`
"""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/ForegroundMaskd.png
    :alt: example of ForegroundMaskd
.. autoclass:: ForegroundMaskd
    :members:
    :special-members: __call__

`ComputeHoVerMapsd`
"""""""""""""""""""
.. autoclass:: ComputeHoVerMapsd
    :members:
    :special-members: __call__

IO (Dict)
^^^^^^^^^

`LoadImaged`
""""""""""""
.. autoclass:: LoadImaged
    :members:
    :special-members: __call__

`SaveImaged`
""""""""""""
.. autoclass:: SaveImaged
    :members:
    :special-members: __call__

`WriteFileMappingd`
"""""""""""""""""""
.. autoclass:: WriteFileMappingd
    :members:
    :special-members: __call__

Post-processing (Dict)
^^^^^^^^^^^^^^^^^^^^^^

`Activationsd`
""""""""""""""
.. autoclass:: Activationsd
    :members:
    :special-members: __call__

`AsDiscreted`
"""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/AsDiscreted.png
    :alt: example of AsDiscreted
.. autoclass:: AsDiscreted
    :members:
    :special-members: __call__

`KeepLargestConnectedComponentd`
""""""""""""""""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/KeepLargestConnectedComponentd.png
    :alt: example of KeepLargestConnectedComponentd
.. autoclass:: KeepLargestConnectedComponentd
    :members:
    :special-members: __call__

`DistanceTransformEDTd`
""""""""""""""""""""""""""""""""
.. autoclass:: DistanceTransformEDTd
    :members:
    :special-members: __call__

`RemoveSmallObjectsd`
"""""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RemoveSmallObjectsd.png
    :alt: example of RemoveSmallObjectsd
.. autoclass:: RemoveSmallObjectsd
    :members:
    :special-members: __call__

`LabelFilterd`
""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/LabelFilterd.png
    :alt: example of LabelFilterd
.. autoclass:: LabelFilterd
    :members:
    :special-members: __call__

`FillHolesd`
""""""""""""
.. autoclass:: FillHolesd
    :members:
    :special-members: __call__

`LabelToContourd`
"""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/LabelToContourd.png
    :alt: example of LabelToContourd
.. autoclass:: LabelToContourd
    :members:
    :special-members: __call__

`Ensembled`
"""""""""""
.. autoclass:: Ensembled
    :members:
    :special-members: __call__

`MeanEnsembled`
"""""""""""""""
.. autoclass:: MeanEnsembled
    :members:
    :special-members: __call__

`VoteEnsembled`
"""""""""""""""
.. autoclass:: VoteEnsembled
    :members:
    :special-members: __call__

`Invertd`
"""""""""
.. autoclass:: Invertd
    :members:
    :special-members: __call__

`SaveClassificationd`
"""""""""""""""""""""
.. autoclass:: SaveClassificationd
    :members:
    :special-members: __call__

`ProbNMSd`
""""""""""
.. autoclass:: ProbNMSd
  :members:
  :special-members: __call__


`SobelGradientsd`
"""""""""""""""""
.. autoclass:: SobelGradientsd
  :members:
  :special-members: __call__

Regularization (Dict)
^^^^^^^^^^^^^^^^^^^^^

`CutMixd`
"""""""""
.. autoclass:: CutMixd
    :members:
    :special-members: __call__

`CutOutd`
"""""""""
.. autoclass:: CutOutd
    :members:
    :special-members: __call__

`MixUpd`
""""""""
.. autoclass:: MixUpd
    :members:
    :special-members: __call__

Signal (Dict)
^^^^^^^^^^^^^

`SignalFillEmptyd`
""""""""""""""""""
.. autoclass:: SignalFillEmptyd
    :members:
    :special-members: __call__


Spatial (Dict)
^^^^^^^^^^^^^^

`SpatialResampled`
""""""""""""""""""
.. autoclass:: SpatialResampled
    :members:
    :special-members: __call__

`ResampleToMatchd`
""""""""""""""""""
.. autoclass:: ResampleToMatchd
    :members:
    :special-members: __call__

`Spacingd`
""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/Spacingd.png
    :alt: example of Spacingd
.. autoclass:: Spacingd
    :members:
    :special-members: __call__

`Orientationd`
""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/Orientationd.png
    :alt: example of Orientationd
.. autoclass:: Orientationd
    :members:
    :special-members: __call__

`Flipd`
"""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/Flipd.png
    :alt: example of Flipd
.. autoclass:: Flipd
    :members:
    :special-members: __call__

`RandFlipd`
"""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RandFlipd.png
    :alt: example of RandFlipd
.. autoclass:: RandFlipd
    :members:
    :special-members: __call__

`RandAxisFlipd`
"""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RandAxisFlipd.png
    :alt: example of RandAxisFlipd
.. autoclass:: RandAxisFlipd
    :members:
    :special-members: __call__

`Rotated`
"""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/Rotated.png
    :alt: example of Rotated
.. autoclass:: Rotated
    :members:
    :special-members: __call__

`RandRotated`
"""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RandRotated.png
    :alt: example of RandRotated
.. autoclass:: RandRotated
    :members:
    :special-members: __call__

`Zoomd`
"""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/Zoomd.png
    :alt: example of Zoomd
.. autoclass:: Zoomd
    :members:
    :special-members: __call__

`RandZoomd`
"""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RandZoomd.png
    :alt: example of RandZoomd
.. autoclass:: RandZoomd
    :members:
    :special-members: __call__

`GridPatchd`
""""""""""""
.. autoclass:: GridPatchd
    :members:
    :special-members: __call__

`RandGridPatchd`
""""""""""""""""
.. autoclass:: RandGridPatchd
    :members:
    :special-members: __call__

`GridSplitd`
""""""""""""
.. autoclass:: GridSplitd
    :members:
    :special-members: __call__


`RandRotate90d`
"""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RandRotate90d.png
    :alt: example of RandRotate90d
.. autoclass:: RandRotate90d
    :members:
    :special-members: __call__

`Rotate90d`
"""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/Rotate90d.png
    :alt: example of Rotate90d
.. autoclass:: Rotate90d
    :members:
    :special-members: __call__

`Resized`
"""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/Resized.png
    :alt: example of Resized
.. autoclass:: Resized
    :members:
    :special-members: __call__

`Affined`
"""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/Affined.png
    :alt: example of Affined
.. autoclass:: Affined
    :members:
    :special-members: __call__

`RandAffined`
"""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RandAffined.png
    :alt: example of RandAffined
.. autoclass:: RandAffined
    :members:
    :special-members: __call__

`Rand2DElasticd`
""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/Rand2DElasticd.png
    :alt: example of Rand2DElasticd
.. autoclass:: Rand2DElasticd
    :members:
    :special-members: __call__

`Rand3DElasticd`
""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/Rand3DElasticd.png
    :alt: example of Rand3DElasticd
.. autoclass:: Rand3DElasticd
    :members:
    :special-members: __call__

`GridDistortiond`
"""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/GridDistortiond.png
    :alt: example of GridDistortiond
.. autoclass:: GridDistortiond
    :members:
    :special-members: __call__

`RandGridDistortiond`
"""""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RandGridDistortiond.png
    :alt: example of RandGridDistortiond
.. autoclass:: RandGridDistortiond
    :members:
    :special-members: __call__

`RandSimulateLowResolutiond`
""""""""""""""""""""""""""""
.. autoclass:: RandSimulateLowResolutiond
    :members:
    :special-members: __call__

`ConvertBoxToPointsd`
"""""""""""""""""""""
.. autoclass:: ConvertBoxToPointsd
    :members:
    :special-members: __call__

`ConvertPointsToBoxesd`
"""""""""""""""""""""""
.. autoclass:: ConvertPointsToBoxesd
    :members:
    :special-members: __call__


Smooth Field (Dict)
^^^^^^^^^^^^^^^^^^^

`RandSmoothFieldAdjustContrastd`
""""""""""""""""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RandSmoothFieldAdjustContrastd.png
    :alt: example of RandSmoothFieldAdjustContrastd
.. autoclass:: RandSmoothFieldAdjustContrastd
    :members:
    :special-members: __call__

`RandSmoothFieldAdjustIntensityd`
"""""""""""""""""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RandSmoothFieldAdjustIntensityd.png
    :alt: example of RandSmoothFieldAdjustIntensityd
.. autoclass:: RandSmoothFieldAdjustIntensityd
    :members:
    :special-members: __call__

`RandSmoothDeformd`
"""""""""""""""""""
.. image:: https://raw.githubusercontent.com/Project-MONAI/DocImages/main/transforms/RandSmoothDeformd.png
    :alt: example of RandSmoothDeformd
.. autoclass:: RandSmoothDeformd
    :members:
    :special-members: __call__


`MRI transforms (Dict)`
^^^^^^^^^^^^^^^^^^^^^^^

`Kspace under-sampling (Dict)`
""""""""""""""""""""""""""""""
.. autoclass:: monai.apps.reconstruction.transforms.dictionary.RandomKspaceMaskd
    :special-members: __call__

.. autoclass:: monai.apps.reconstruction.transforms.dictionary.EquispacedKspaceMaskd
    :special-members: __call__

`ExtractDataKeyFromMetaKeyd`
""""""""""""""""""""""""""""
.. autoclass:: monai.apps.reconstruction.transforms.dictionary.ExtractDataKeyFromMetaKeyd
    :special-members: __call__

`ReferenceBasedSpatialCropd`
""""""""""""""""""""""""""""
.. autoclass:: monai.apps.reconstruction.transforms.dictionary.ReferenceBasedSpatialCropd
    :special-members: __call__

`ReferenceBasedNormalizeIntensityd`
"""""""""""""""""""""""""""""""""""
.. autoclass:: monai.apps.reconstruction.transforms.dictionary.ReferenceBasedNormalizeIntensityd
    :special-members: __call__


Lazy (Dict)
^^^^^^^^^^^

`ApplyPendingd`
"""""""""""""""

.. autoclass:: ApplyPendingd
    :members:
    :special-members: __call__


Utility (Dict)
^^^^^^^^^^^^^^

`Identityd`
"""""""""""
.. autoclass:: Identityd
    :members:
    :special-members: __call__

`AsChannelLastd`
""""""""""""""""
.. autoclass:: AsChannelLastd
    :members:
    :special-members: __call__

`EnsureChannelFirstd`
"""""""""""""""""""""
.. autoclass:: EnsureChannelFirstd
    :members:
    :special-members: __call__

`RepeatChanneld`
""""""""""""""""
.. autoclass:: RepeatChanneld
    :members:
    :special-members: __call__

`SplitDimd`
"""""""""""
.. autoclass:: SplitDimd
    :members:
    :special-members: __call__

`CastToTyped`
"""""""""""""
.. autoclass:: CastToTyped
    :members:
    :special-members: __call__

`ToTensord`
"""""""""""
.. autoclass:: ToTensord
    :members:
    :special-members: __call__

`ToNumpyd`
""""""""""
.. autoclass:: ToNumpyd
    :members:
    :special-members: __call__

`ToPIL`
"""""""
.. autoclass:: ToPIL
    :members:
    :special-members: __call__

`ToCupyd`
"""""""""
.. autoclass:: ToCupyd
    :members:
    :special-members: __call__

`ToPILd`
""""""""
.. autoclass:: ToPILd
    :members:
    :special-members: __call__

`DeleteItemsd`
""""""""""""""
.. autoclass:: DeleteItemsd
    :members:
    :special-members: __call__

`SelectItemsd`
""""""""""""""
.. autoclass:: SelectItemsd
    :members:
    :special-members: __call__

`FlattenSubKeysd`
"""""""""""""""""
.. autoclass:: FlattenSubKeysd
    :members:
    :special-members: __call__

`Transposed`
""""""""""""
.. autoclass:: Transposed
    :members:
    :special-members: __call__

`SqueezeDimd`
"""""""""""""
.. autoclass:: SqueezeDimd
    :members:
    :special-members: __call__

`DataStatsd`
""""""""""""
.. autoclass:: DataStatsd
    :members:
    :special-members: __call__

`SimulateDelayd`
""""""""""""""""
.. autoclass:: SimulateDelayd
    :members:
    :special-members: __call__

`CopyItemsd`
""""""""""""
.. autoclass:: CopyItemsd
    :members:
    :special-members: __call__

`SubtractItemsd`
""""""""""""""""
.. autoclass:: SubtractItemsd
    :members:
    :special-members: __call__

`ConcatItemsd`
""""""""""""""
.. autoclass:: ConcatItemsd
    :members:
    :special-members: __call__

`Lambdad`
"""""""""
.. autoclass:: Lambdad
    :members:
    :special-members: __call__

`RandLambdad`
"""""""""""""
.. autoclass:: RandLambdad
    :members:
    :special-members: __call__

`RemoveRepeatedChanneld`
""""""""""""""""""""""""
.. autoclass:: RemoveRepeatedChanneld
    :members:
    :special-members: __call__

`LabelToMaskd`
""""""""""""""
.. autoclass:: LabelToMaskd
    :members:
    :special-members: __call__

`FgBgToIndicesd`
""""""""""""""""
.. autoclass:: FgBgToIndicesd
    :members:
    :special-members: __call__

`ClassesToIndicesd`
"""""""""""""""""""
.. autoclass:: ClassesToIndicesd
    :members:
    :special-members: __call__

`ConvertToMultiChannelBasedOnBratsClassesd`
"""""""""""""""""""""""""""""""""""""""""""
.. autoclass:: ConvertToMultiChannelBasedOnBratsClassesd
    :members:
    :special-members: __call__

`AddExtremePointsChanneld`
""""""""""""""""""""""""""
.. autoclass:: AddExtremePointsChanneld
    :members:
    :special-members: __call__

`TorchVisiond`
""""""""""""""
.. autoclass:: TorchVisiond
    :members:
    :special-members: __call__

`RandTorchVisiond`
""""""""""""""""""
.. autoclass:: RandTorchVisiond
    :members:
    :special-members: __call__

`TorchIOd`
""""""""""
.. autoclass:: TorchIOd
    :members:
    :special-members: __call__

`RandTorchIOd`
""""""""""""""
.. autoclass:: RandTorchIOd
    :members:
    :special-members: __call__

`MapLabelValued`
""""""""""""""""
.. autoclass:: MapLabelValued
    :members:
    :special-members: __call__

`EnsureTyped`
"""""""""""""
.. autoclass:: EnsureTyped
    :members:
    :special-members: __call__

`IntensityStatsd`
"""""""""""""""""
.. autoclass:: IntensityStatsd
    :members:
    :special-members: __call__

`ToDeviced`
"""""""""""
.. autoclass:: ToDeviced
    :members:
    :special-members: __call__

`CuCIMd`
""""""""
.. autoclass:: CuCIMd
    :members:
    :special-members: __call__

`RandCuCIMd`
""""""""""""
.. autoclass:: RandCuCIMd
    :members:
    :special-members: __call__

`AddCoordinateChannelsd`
""""""""""""""""""""""""
.. autoclass:: AddCoordinateChannelsd
    :members:
    :special-members: __call__

`ImageFilterd`
""""""""""""""
.. autoclass:: ImageFilterd
    :members:
    :special-members: __call__

`RandImageFilterd`
""""""""""""""""""
.. autoclass:: RandImageFilterd
    :members:
    :special-members: __call__

`ApplyTransformToPointsd`
"""""""""""""""""""""""""
.. autoclass:: ApplyTransformToPointsd
    :members:
    :special-members: __call__


MetaTensor
^^^^^^^^^^

`ToMetaTensord`
"""""""""""""""
.. autoclass:: ToMetaTensord
    :members:
    :special-members: __call__

`FromMetaTensord`
"""""""""""""""""
.. autoclass:: FromMetaTensord
    :members:
    :special-members: __call__

Transform Adaptors
------------------
.. automodule:: monai.transforms.adaptors

`FunctionSignature`
^^^^^^^^^^^^^^^^^^^
.. autoclass:: FunctionSignature
    :members:

`adaptor`
^^^^^^^^^
.. autofunction:: monai.transforms.adaptors.adaptor

`apply_alias`
^^^^^^^^^^^^^
.. autofunction:: monai.transforms.adaptors.apply_alias

`to_kwargs`
^^^^^^^^^^^
.. autofunction:: monai.transforms.adaptors.to_kwargs

Utilities
---------
.. automodule:: monai.transforms.utils
    :members:

.. automodule:: monai.transforms.utils_pytorch_numpy_unification
    :members:

.. automodule:: monai.transforms.utils_morphological_ops
    :members:

By Categories
-------------
.. toctree::
   :maxdepth: 1

   transforms_idx
