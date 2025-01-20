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

from copy import deepcopy

from monai.data import MetaTensor, set_track_meta
from monai.transforms import InvertibleTransform, MapTransform, Randomizable
from monai.transforms.lazy.functional import apply_pending
from tests.util import assert_allclose

apply_transforms_kwargs = ("pending", "mode", "padding_mode", "dtype", "align_corners")


def get_apply_param(init_param=None, call_param=None, params=apply_transforms_kwargs):
    apply_param = {}
    for key in apply_transforms_kwargs:
        if init_param and key in init_param.keys():
            apply_param[key] = init_param[key]
        if call_param and key in call_param.keys():
            apply_param[key] = call_param[key]
    return apply_param


def test_resampler_lazy(
    resampler,
    expected_output,
    init_param=None,
    call_param=None,
    output_key=None,
    output_idx=None,
    rtol=1e-5,
    atol=1e-7,
    skip_shape_check=False,
    seed=None,
):
    """
    This test function is used to test the consistency between non-lazy and lazy transforms.
    Args:
        resampler: instance of a resampling transform.
        expected_output: output of non-lazy transform.
        init_param: parameters that are used to initialize the transform.
        call_param: parameters that are used when calling the transform.
        output_key: key to get the output of the transform. This argument is used for dictionary based transforms.
        output_idx: index to get the expected output from multiple outputs of the transform.
        rtol: relative tolerance. This argument is only used to compare the output.
        atol: absolute tolerance. This argument is only used to compare the output.
        skip_shape_check: skip the check of shapes.
        seed: set the random state with an integer seed. This argument is used for randomizable transforms.

    """
    if isinstance(resampler, Randomizable):
        resampler.set_random_state(seed=seed)
    set_track_meta(True)
    resampler.lazy = True
    pending_output = resampler(**deepcopy(call_param))
    if output_idx is not None:
        expected_output, pending_output = (expected_output[output_idx], pending_output[output_idx])
    if output_key is not None:
        non_lazy_out, lazy_out = expected_output[output_key], pending_output[output_key]
    else:
        non_lazy_out, lazy_out = expected_output, pending_output
    assert_allclose(lazy_out.peek_pending_affine(), non_lazy_out.affine)
    if not skip_shape_check:
        assert_allclose(lazy_out.peek_pending_shape(), non_lazy_out.shape[1:4])
    apply_param = get_apply_param(init_param, call_param)
    lazy_out = apply_pending(lazy_out, overrides=apply_param)[0]
    assert_allclose(lazy_out, non_lazy_out, rtol=rtol, atol=atol)
    if (
        isinstance(resampler, InvertibleTransform)
        and (not isinstance(resampler, MapTransform))
        and isinstance(lazy_out, MetaTensor)
        and isinstance(non_lazy_out, MetaTensor)
        and non_lazy_out.applied_operations
    ):
        resampler.lazy = False
        out = resampler.inverse(lazy_out.clone())
        ref = resampler.inverse(non_lazy_out.clone())
        assert_allclose(out.applied_operations, [])
        assert_allclose(out.pending_operations, [])
        assert_allclose(ref, out, type_test=False, rtol=1e-3, atol=1e-3)
        resampler.lazy = True
