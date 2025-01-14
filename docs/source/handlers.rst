:github_url: https://github.com/Project-MONAI/MONAI

.. _handlers:

Event handlers
==============
.. currentmodule:: monai.handlers

Model checkpoint loader
-----------------------
.. autoclass:: CheckpointLoader
  :members:

Model checkpoint saver
----------------------
.. autoclass:: CheckpointSaver
  :members:


Metrics saver
-------------
.. autoclass:: MetricsSaver
    :members:


CSV saver
---------
.. autoclass:: ClassificationSaver
    :members:


Ignite Metric Handler
---------------------
.. autoclass:: IgniteMetricHandler
    :members:


Mean Dice metrics handler
-------------------------
.. autoclass:: MeanDice
    :members:


Mean IoU metric handler
-----------------------
.. autoclass:: MeanIoUHandler
    :members:


ROC AUC metrics handler
-----------------------
.. autoclass:: ROCAUC
    :members:


Average Precision metric handler
--------------------------------
.. autoclass:: AveragePrecision
    :members:


Confusion matrix metrics handler
--------------------------------
.. autoclass:: ConfusionMatrix
    :members:


Hausdorff distance metrics handler
----------------------------------
.. autoclass:: HausdorffDistance
    :members:


Surface distance metrics handler
--------------------------------
.. autoclass:: SurfaceDistance
    :members:


Panoptic Quality metrics handler
--------------------------------
.. autoclass:: PanopticQuality
    :members:


Mean squared error metrics handler
----------------------------------
.. autoclass:: MeanSquaredError
    :members:


Mean absolute error metrics handler
-----------------------------------
.. autoclass:: MeanAbsoluteError
    :members:


Root mean squared error metrics handler
---------------------------------------
.. autoclass:: RootMeanSquaredError
    :members:


Peak signal to noise ratio metrics handler
------------------------------------------
.. autoclass:: PeakSignalToNoiseRatio
    :members:


Metrics reloaded binary handler
-------------------------------
.. autoclass:: MetricsReloadedBinaryHandler
    :members:


Metrics reloaded categorical handler
------------------------------------
.. autoclass:: MetricsReloadedCategoricalHandler
    :members:


Metric logger
-------------
.. autoclass:: MetricLogger
    :members:


Logfile handler
---------------
.. autoclass:: LogfileHandler
    :members:


Training stats handler
----------------------
.. autoclass:: StatsHandler
    :members:


Tensorboard handlers
--------------------
.. autoclass:: TensorBoardHandler
    :members:

.. autoclass:: TensorBoardStatsHandler
    :members:

.. autoclass:: TensorBoardImageHandler
    :members:


LR Schedule handler
-------------------
.. autoclass:: LrScheduleHandler
    :members:


Validation handler
------------------
.. autoclass:: ValidationHandler
    :members:

SmartCache handler
------------------
.. autoclass:: SmartCacheHandler
    :members:

Parameter Scheduler handler
---------------------------
.. autoclass:: ParamSchedulerHandler
    :members:

EarlyStop handler
-----------------
.. autoclass:: EarlyStopHandler
    :members:

GarbageCollector handler
------------------------
.. autoclass:: GarbageCollector
    :members:

Post processing
---------------
.. autoclass:: PostProcessing
    :members:

Decollate batch
---------------
.. autoclass:: DecollateBatch
    :members:

MLFlow handler
--------------
.. autoclass:: MLFlowHandler
    :members:

ClearML handlers
----------------
.. autoclass:: ClearMLHandler
    :members:

.. autoclass:: ClearMLStatsHandler
    :members:

.. autoclass:: ClearMLImageHandler
    :members:

NVTX Handlers
-------------
.. automodule:: monai.handlers.nvtx_handlers
  :members:

Utilities
---------
.. automodule:: monai.handlers.utils
  :members:

Probability Map Handlers
------------------------
.. automodule:: monai.handlers.probability_maps
  :members:
