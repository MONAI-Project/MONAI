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

from .component_store import ComponentStore
from .decorators import MethodReplacer, RestartGenerator
from .deprecate_utils import DeprecatedError, deprecated, deprecated_arg, deprecated_arg_default
from .dist import RankFilter, evenly_divisible_all_gather, get_dist_device, string_list_all_gather
from .enums import (
    AdversarialIterationEvents,
    AdversarialKeys,
    AlgoKeys,
    Average,
    BlendMode,
    BoxModeName,
    BundleProperty,
    BundlePropertyConfig,
    ChannelMatching,
    ColorOrder,
    CommonKeys,
    CompInitMode,
    DiceCEReduction,
    EngineStatsKeys,
    FastMRIKeys,
    ForwardMode,
    GanKeys,
    GridPatchSort,
    GridSampleMode,
    GridSamplePadMode,
    HoVerNetBranch,
    HoVerNetMode,
    IgniteInfo,
    InterpolateMode,
    JITMetadataKeys,
    LazyAttr,
    LossReduction,
    MetaKeys,
    Method,
    MetricReduction,
    NdimageMode,
    NumpyPadMode,
    OrderingTransformations,
    OrderingType,
    PatchKeys,
    PostFix,
    ProbMapKeys,
    PytorchPadMode,
    SkipMode,
    SpaceKeys,
    SplineMode,
    StrEnum,
    TraceKeys,
    TraceStatusKeys,
    TransformBackends,
    UpsampleMode,
    DataCollateMode,
    Weight,
    WSIPatchKeys,
)
from .jupyter_utils import StatusMembers, ThreadContainer
from .misc import (
    MAX_SEED,
    ImageMetaKey,
    MONAIEnvVars,
    check_kwargs_exist_in_class_init,
    check_parent_dir,
    copy_to_device,
    ensure_tuple,
    ensure_tuple_rep,
    ensure_tuple_size,
    fall_back_tuple,
    first,
    flatten_dict,
    get_seed,
    has_option,
    is_immutable,
    is_module_ver_at_least,
    is_scalar,
    is_scalar_tensor,
    is_sqrt,
    issequenceiterable,
    list_to_dict,
    path_to_uri,
    pprint_edges,
    progress_bar,
    run_cmd,
    sample_slices,
    save_obj,
    set_determinism,
    star_zip_with,
    str2bool,
    str2list,
    to_tuple_of_dictionaries,
    unsqueeze_left,
    unsqueeze_right,
    zip_with,
)
from .module import (
    InvalidPyTorchVersionError,
    OptionalImportError,
    allow_missing_reference,
    compute_capabilities_after,
    damerau_levenshtein_distance,
    exact_version,
    get_full_type_name,
    get_package_version,
    get_torch_version_tuple,
    instantiate,
    load_submodules,
    look_up_option,
    min_version,
    optional_import,
    pytorch_after,
    require_pkg,
    run_debug,
    run_eval,
    version_geq,
    version_leq,
)
from .nvtx import Range
from .ordering import Ordering
from .profiling import (
    PerfContext,
    ProfileHandler,
    WorkflowProfiler,
    select_transform_call,
    torch_profiler_full,
    torch_profiler_time_cpu_gpu,
    torch_profiler_time_end_to_end,
)
from .state_cacher import StateCacher
from .tf32 import detect_default_tf32, has_ampere_or_later
from .type_conversion import (
    convert_data_type,
    convert_to_cupy,
    convert_to_dst_type,
    convert_to_list,
    convert_to_numpy,
    convert_to_tensor,
    dtype_numpy_to_torch,
    dtype_torch_to_numpy,
    get_dtype,
    get_dtype_string,
    get_equivalent_dtype,
    get_numpy_dtype_from_string,
    get_torch_dtype_from_string,
)

# have to explicitly bring these in here to resolve circular import issues
