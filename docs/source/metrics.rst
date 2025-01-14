:github_url: https://github.com/Project-MONAI/MONAI

.. _metrics:

Metrics
=======
.. currentmodule:: monai.metrics

`FROC`
------
.. autofunction:: compute_fp_tp_probs
.. autofunction:: compute_froc_curve_data
.. autofunction:: compute_froc_score

`Metric`
--------
.. autoclass:: Metric
    :members:

`Variance`
--------------
.. autofunction:: compute_variance

.. autoclass:: VarianceMetric
    :members:

`LabelQualityScore`
--------------------
.. autofunction:: label_quality_score

.. autoclass:: LabelQualityScore
    :members:

`IterationMetric`
-----------------
.. autoclass:: IterationMetric
    :members:

`Cumulative`
------------
.. autoclass:: Cumulative
    :members:

`CumulativeIterationMetric`
---------------------------
.. autoclass:: CumulativeIterationMetric
    :members:

`LossMetric`
------------
.. autoclass:: LossMetric
    :members:

`Mean Dice`
-----------
.. autoclass:: DiceMetric
    :members:

.. autoclass:: DiceHelper
    :members:

`Mean IoU`
----------
.. autofunction:: compute_iou

.. autoclass:: MeanIoU
    :members:

`Generalized Dice Score`
------------------------
.. autofunction:: compute_generalized_dice

.. autoclass:: GeneralizedDiceScore
    :members:

`Area under the ROC curve`
--------------------------
.. autofunction:: compute_roc_auc

.. autoclass:: ROCAUCMetric
    :members:

`Average Precision`
--------------------------
.. autofunction:: compute_average_precision

.. autoclass:: AveragePrecisionMetric
    :members:

`Confusion matrix`
------------------
.. autofunction:: get_confusion_matrix
.. autofunction:: compute_confusion_matrix_metric

.. autoclass:: ConfusionMatrixMetric
    :members:

`Hausdorff distance`
--------------------
.. autofunction:: compute_hausdorff_distance
.. autofunction:: compute_percent_hausdorff_distance

.. autoclass:: HausdorffDistanceMetric
    :members:

`Average surface distance`
--------------------------
.. autofunction:: compute_average_surface_distance

.. autoclass:: SurfaceDistanceMetric
    :members:

`Surface dice`
--------------
.. autofunction:: compute_surface_dice

.. autoclass:: SurfaceDiceMetric
    :members:

`PanopticQualityMetric`
-----------------------
.. autofunction:: compute_panoptic_quality

.. autoclass:: PanopticQualityMetric
    :members:

`Mean squared error`
--------------------
.. autoclass:: MSEMetric
    :members:

`Mean absolute error`
---------------------
.. autoclass:: MAEMetric
    :members:

`Root mean squared error`
-------------------------
.. autoclass:: RMSEMetric
    :members:

`Peak signal to noise ratio`
----------------------------
.. autoclass:: PSNRMetric
    :members:

`Structural similarity index measure`
-------------------------------------
.. autoclass:: monai.metrics.regression.SSIMMetric

`Multi-scale structural similarity index measure`
-------------------------------------------------
.. autoclass:: MultiScaleSSIMMetric

`Fréchet Inception Distance`
------------------------------
.. autofunction:: compute_frechet_distance

.. autoclass:: FIDMetric
    :members:

`Maximum Mean Discrepancy`
------------------------------
.. autofunction:: compute_mmd

.. autoclass:: MMDMetric
    :members:

`Cumulative average`
--------------------
.. autoclass:: CumulativeAverage
    :members:

`Metrics reloaded binary`
-------------------------
.. autoclass:: MetricsReloadedBinary
    :members:

`Metrics reloaded categorical`
------------------------------
.. autoclass:: MetricsReloadedCategorical
    :members:



Utilities
---------
.. automodule:: monai.metrics.utils
  :members:
