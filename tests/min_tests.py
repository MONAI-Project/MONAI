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

import os
import sys
import unittest
from pathlib import Path


def run_testsuit():
    """
    Load test cases by excluding those need external dependencies.
    The loaded cases should work with "requirements-min.txt"::

        # in the monai repo folder:
        pip install -r requirements-min.txt
        QUICKTEST=true python -m tests.min_tests

    :return: a test suite
    """
    exclude_cases = [  # these cases use external dependencies
        "test_ahnet",
        "test_arraydataset",
        "test_auto3dseg_bundlegen",
        "test_auto3dseg_ensemble",
        "test_auto3dseg_hpo",
        "test_auto3dseg",
        "test_bundle_onnx_export",
        "test_bundle_trt_export",
        "test_bundle_push_to_hf_hub",
        "test_cachedataset",
        "test_cachedataset_parallel",
        "test_cachedataset_persistent_workers",
        "test_cachentransdataset",
        "test_check_missing_files",
        "test_compute_f_beta",
        "test_compute_ho_ver_maps",
        "test_compute_ho_ver_maps_d",
        "test_compute_panoptic_quality",
        "test_contrastive_loss",
        "test_convert_to_onnx",
        "test_convert_to_trt",
        "test_csv_dataset",
        "test_csv_iterable_dataset",
        "test_cumulative_average_dist",
        "test_sampler_dist",
        "test_dataset",
        "test_dataset_summary",
        "test_deepedit_transforms",
        "test_deepedit_interaction",
        "test_deepgrow_dataset",
        "test_deepgrow_interaction",
        "test_deepgrow_transforms",
        "test_detect_envelope",
        "test_dints_network",
        "test_distance_transform_edt",
        "test_efficientnet",
        "test_ensemble_evaluator",
        "test_ensure_channel_first",
        "test_ensure_channel_firstd",
        "test_fill_holes",
        "test_fill_holesd",
        "test_foreground_mask",
        "test_foreground_maskd",
        "test_global_mutual_information_loss",
        "test_grid_patch",
        "test_gmm",
        "test_handler_metrics_reloaded",
        "test_handler_checkpoint_loader",
        "test_handler_checkpoint_saver",
        "test_handler_classification_saver",
        "test_handler_classification_saver_dist",
        "test_handler_confusion_matrix",
        "test_handler_confusion_matrix_dist",
        "test_handler_decollate_batch",
        "test_handler_early_stop",
        "test_handler_garbage_collector",
        "test_handler_hausdorff_distance",
        "test_handler_ignite_metric",
        "test_handler_lr_scheduler",
        "test_handler_mean_dice",
        "test_handler_panoptic_quality",
        "test_handler_mean_iou",
        "test_handler_metrics_saver",
        "test_handler_metrics_saver_dist",
        "test_handler_mlflow",
        "test_handler_nvtx",
        "test_handler_parameter_scheduler",
        "test_handler_post_processing",
        "test_handler_prob_map_producer",
        "test_handler_regression_metrics",
        "test_handler_regression_metrics_dist",
        "test_handler_rocauc",
        "test_handler_rocauc_dist",
        "test_handler_smartcache",
        "test_handler_stats",
        "test_handler_surface_distance",
        "test_handler_tb_image",
        "test_handler_tb_stats",
        "test_handler_validation",
        "test_hausdorff_distance",
        "test_header_correct",
        "test_hilbert_transform",
        "test_hovernet_loss",
        "test_image_dataset",
        "test_image_rw",
        "test_img2tensorboard",
        "test_integration_fast_train",
        "test_integration_gpu_customization",
        "test_integration_segmentation_3d",
        "test_integration_lazy_samples",
        "test_integration_sliding_window",
        "test_integration_unet_2d",
        "test_integration_workflows",
        "test_integration_workflows_gan",
        "test_integration_bundle_run",
        "test_integration_autorunner",
        "test_integration_nnunetv2_runner",
        "test_invert",
        "test_invertd",
        "test_iterable_dataset",
        "test_keep_largest_connected_component",
        "test_keep_largest_connected_componentd",
        "test_label_filter",
        "test_lltm",
        "test_lmdbdataset",
        "test_lmdbdataset_dist",
        "test_load_image",
        "test_load_imaged",
        "test_load_spacing_orientation",
        "test_mednistdataset",
        "test_milmodel",
        "test_mlp",
        "test_nifti_header_revise",
        "test_nifti_rw",
        "test_nuclick_transforms",
        "test_nrrd_reader",
        "test_occlusion_sensitivity",
        "test_orientation",
        "test_orientationd",
        "test_patchembedding",
        "test_persistentdataset",
        "test_pil_reader",
        "test_plot_2d_or_3d_image",
        "test_png_rw",
        "test_prepare_batch_default",
        "test_prepare_batch_diffusion",
        "test_prepare_batch_extra_input",
        "test_prepare_batch_hovernet",
        "test_rand_grid_patch",
        "test_rand_rotate",
        "test_rand_rotated",
        "test_rand_zoom",
        "test_rand_zoomd",
        "test_randtorchvisiond",
        "test_rankfilter_dist",
        "test_resample_backends",
        "test_resize",
        "test_resized",
        "test_resample_to_match",
        "test_resample_to_matchd",
        "test_rotate",
        "test_rotated",
        "test_save_image",
        "test_save_imaged",
        "test_selfattention",
        "test_senet",
        "test_smartcachedataset",
        "test_spacing",
        "test_spacingd",
        "test_splitdimd",
        "test_surface_distance",
        "test_surface_dice",
        "test_testtimeaugmentation",
        "test_torchvision",
        "test_torchvisiond",
        "test_transchex",
        "test_transformerblock",
        "test_trt_compile",
        "test_unetr",
        "test_unetr_block",
        "test_vit",
        "test_vitautoenc",
        "test_vnet",
        "test_write_metrics_reports",
        "test_wsireader",
        "test_zoom",
        "test_zoom_affine",
        "test_zoomd",
        "test_prepare_batch_default_dist",
        "test_bundle_verify_metadata",
        "test_bundle_verify_net",
        "test_bundle_ckpt_export",
        "test_bundle_utils",
        "test_bundle_init_bundle",
        "test_fastmri_reader",
        "test_metrics_reloaded",
        "test_spatial_combine_transforms",
        "test_bundle_workflow",
        "test_zarr_avg_merger",
        "test_perceptual_loss",
        "test_ultrasound_confidence_map_transform",
        "test_vista3d_utils",
        "test_vista3d_transforms",
        "test_matshow3d",
    ]
    assert sorted(exclude_cases) == sorted(set(exclude_cases)), f"Duplicated items in {exclude_cases}"

    files = [f.relative_to(Path(__file__).parent.parent) for f in Path(__file__).parent.rglob("test_*.py")]
    files = [str(f).replace(os.sep, ".").replace(".py", "") for f in files]

    cases = []
    for test_module in files:
        test_case = test_module.split(".")[-1]
        if test_case in exclude_cases:
            exclude_cases.remove(test_case)
            print(f"skipping {test_module}")
        else:
            print(f"adding {test_module}")
            cases.append(test_module)
    exclude_cases = [str(list(Path(__file__).parent.rglob(f"*{et}*"))[0]) for et in exclude_cases]
    assert not exclude_cases, f"items in exclude_cases not used: {' '.join(exclude_cases)}"
    test_suite = unittest.TestLoader().loadTestsFromNames(cases)
    return test_suite


if __name__ == "__main__":
    # testing import submodules
    from monai.utils.module import load_submodules

    _, err_mod = load_submodules(sys.modules["monai"], True)
    assert not err_mod, f"err_mod={err_mod} not empty"

    # testing all modules
    test_runner = unittest.TextTestRunner(stream=sys.stdout, verbosity=2)
    result = test_runner.run(run_testsuit())
    sys.exit(int(not result.wasSuccessful()))
