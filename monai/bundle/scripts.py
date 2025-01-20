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

import ast
import json
import os
import re
import warnings
import zipfile
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from pydoc import locate
from shutil import copyfile
from textwrap import dedent
from typing import Any, Callable

import torch
from torch.cuda import is_available

from monai._version import get_versions
from monai.apps.utils import _basename, download_url, extractall, get_logger
from monai.bundle.config_item import ConfigComponent
from monai.bundle.config_parser import ConfigParser
from monai.bundle.utils import DEFAULT_INFERENCE, DEFAULT_METADATA, merge_kv
from monai.bundle.workflows import BundleWorkflow, ConfigWorkflow
from monai.config import PathLike
from monai.data import load_net_with_metadata, save_net_with_metadata
from monai.networks import (
    convert_to_onnx,
    convert_to_torchscript,
    convert_to_trt,
    copy_model_state,
    get_state_dict,
    save_state,
)
from monai.utils import (
    IgniteInfo,
    check_parent_dir,
    deprecated_arg,
    ensure_tuple,
    get_equivalent_dtype,
    min_version,
    optional_import,
    pprint_edges,
)

validate, _ = optional_import("jsonschema", name="validate")
ValidationError, _ = optional_import("jsonschema.exceptions", name="ValidationError")
Checkpoint, has_ignite = optional_import("ignite.handlers", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Checkpoint")
requests_get, has_requests = optional_import("requests", name="get")
onnx, _ = optional_import("onnx")
huggingface_hub, _ = optional_import("huggingface_hub")

logger = get_logger(module_name=__name__)

# set BUNDLE_DOWNLOAD_SRC="ngc" to use NGC source in default for bundle download
# set BUNDLE_DOWNLOAD_SRC="github" to use github source in default for bundle download
DEFAULT_DOWNLOAD_SOURCE = os.environ.get("BUNDLE_DOWNLOAD_SRC", "monaihosting")
PPRINT_CONFIG_N = 5

MONAI_HOSTING_BASE_URL = "https://api.ngc.nvidia.com/v2/models/nvidia/monaihosting"
NGC_BASE_URL = "https://api.ngc.nvidia.com/v2/models/nvidia/monaitoolkit"


def update_kwargs(args: str | dict | None = None, ignore_none: bool = True, **kwargs: Any) -> dict:
    """
    Update the `args` dictionary with the input `kwargs`.
    For dict data, recursively update the content based on the keys.

    Example::

        from monai.bundle import update_kwargs
        update_kwargs({'exist': 1}, exist=2, new_arg=3)
        # return {'exist': 2, 'new_arg': 3}

    Args:
        args: source `args` dictionary (or a json/yaml filename to read as dictionary) to update.
        ignore_none: whether to ignore input args with None value, default to `True`.
        kwargs: key=value pairs to be merged into `args`.

    """
    args_: dict = args if isinstance(args, dict) else {}
    if isinstance(args, str):
        # args are defined in a structured file
        args_ = ConfigParser.load_config_file(args)
    if isinstance(args, (tuple, list)) and all(isinstance(x, str) for x in args):
        primary, overrides = args
        args_ = update_kwargs(primary, ignore_none, **update_kwargs(overrides, ignore_none, **kwargs))
    if not isinstance(args_, dict):
        return args_
    # recursively update the default args with new args
    for k, v in kwargs.items():
        if ignore_none and v is None:
            continue
        if isinstance(v, dict) and isinstance(args_.get(k), dict):
            args_[k] = update_kwargs(args_[k], ignore_none, **v)
        else:
            merge_kv(args_, k, v)
    return args_


_update_args = update_kwargs  # backward compatibility


def _pop_args(src: dict, *args: Any, **kwargs: Any) -> tuple:
    """
    Pop args from the `src` dictionary based on specified keys in `args` and (key, default value) pairs in `kwargs`.

    """
    return tuple([src.pop(i) for i in args] + [src.pop(k, v) for k, v in kwargs.items()])


def _log_input_summary(tag: str, args: dict) -> None:
    logger.info(f"--- input summary of monai.bundle.scripts.{tag} ---")
    for name, val in args.items():
        logger.info(f"> {name}: {pprint_edges(val, PPRINT_CONFIG_N)}")
    logger.info("---\n\n")


def _get_var_names(expr: str) -> list[str]:
    """
    Parse the expression and discover what variables are present in it based on ast module.

    Args:
        expr: source expression to parse.

    """
    tree = ast.parse(expr)
    return [m.id for m in ast.walk(tree) if isinstance(m, ast.Name)]


def _get_fake_spatial_shape(shape: Sequence[str | int], p: int = 1, n: int = 1, any: int = 1) -> tuple:
    """
    Get spatial shape for fake data according to the specified shape pattern.
    It supports `int` number and `string` with formats like: "32", "32 * n", "32 ** p", "32 ** p *n".

    Args:
        shape: specified pattern for the spatial shape.
        p: power factor to generate fake data shape if dim of expected shape is "x**p", default to 1.
        p: multiply factor to generate fake data shape if dim of expected shape is "x*n", default to 1.
        any: specified size to generate fake data shape if dim of expected shape is "*", default to 1.

    """
    ret = []
    for i in shape:
        if isinstance(i, int):
            ret.append(i)
        elif isinstance(i, str):
            if i == "*":
                ret.append(any)
            else:
                for c in _get_var_names(i):
                    if c not in ["p", "n"]:
                        raise ValueError(f"only support variables 'p' and 'n' so far, but got: {c}.")
                ret.append(eval(i, {"p": p, "n": n}))
        else:
            raise ValueError(f"spatial shape items must be int or string, but got: {type(i)} {i}.")
    return tuple(ret)


def _get_git_release_url(repo_owner: str, repo_name: str, tag_name: str, filename: str) -> str:
    return f"https://github.com/{repo_owner}/{repo_name}/releases/download/{tag_name}/{filename}"


def _get_ngc_bundle_url(model_name: str, version: str) -> str:
    return f"{NGC_BASE_URL}/{model_name.lower()}/versions/{version}/files"


def _get_ngc_private_base_url(repo: str) -> str:
    return f"https://api.ngc.nvidia.com/v2/{repo}/models"


def _get_ngc_private_bundle_url(model_name: str, version: str, repo: str) -> str:
    return f"{_get_ngc_private_base_url(repo)}/{model_name.lower()}/versions/{version}/zip"


def _get_monaihosting_bundle_url(model_name: str, version: str) -> str:
    return f"{MONAI_HOSTING_BASE_URL}/{model_name.lower()}/versions/{version}/files/{model_name}_v{version}.zip"


def _download_from_github(repo: str, download_path: Path, filename: str, progress: bool = True) -> None:
    repo_owner, repo_name, tag_name = repo.split("/")
    if ".zip" not in filename:
        filename += ".zip"
    url = _get_git_release_url(repo_owner, repo_name, tag_name=tag_name, filename=filename)
    filepath = download_path / f"{filename}"
    download_url(url=url, filepath=filepath, hash_val=None, progress=progress)
    extractall(filepath=filepath, output_dir=download_path, has_base=True)


def _download_from_monaihosting(download_path: Path, filename: str, version: str, progress: bool) -> None:
    url = _get_monaihosting_bundle_url(model_name=filename, version=version)
    filepath = download_path / f"{filename}_v{version}.zip"
    download_url(url=url, filepath=filepath, hash_val=None, progress=progress)
    extractall(filepath=filepath, output_dir=download_path, has_base=True)


def _add_ngc_prefix(name: str, prefix: str = "monai_") -> str:
    if name.startswith(prefix):
        return name
    return f"{prefix}{name}"


def _remove_ngc_prefix(name: str, prefix: str = "monai_") -> str:
    if name.startswith(prefix):
        return name[len(prefix) :]
    return name


def _get_all_download_files(request_url: str, headers: dict | None = None) -> list[str]:
    if not has_requests:
        raise ValueError("requests package is required, please install it.")
    headers = {} if headers is None else headers
    response = requests_get(request_url, headers=headers)
    response.raise_for_status()
    model_info = json.loads(response.text)

    if not isinstance(model_info, dict) or "modelFiles" not in model_info:
        raise ValueError("The data is not a dictionary or it does not have the key 'modelFiles'.")

    model_files = model_info["modelFiles"]
    return [f["path"] for f in model_files]


def _download_from_ngc(
    download_path: Path,
    filename: str,
    version: str,
    prefix: str = "monai_",
    remove_prefix: str | None = "monai_",
    progress: bool = True,
) -> None:
    # ensure prefix is contained
    filename = _add_ngc_prefix(filename, prefix=prefix)
    url = _get_ngc_bundle_url(model_name=filename, version=version)
    if remove_prefix:
        filename = _remove_ngc_prefix(filename, prefix=remove_prefix)
    filepath = download_path / filename
    filepath.mkdir(parents=True, exist_ok=True)
    for file in _get_all_download_files(url):
        download_url(url=f"{url}/{file}", filepath=f"{filepath}/{file}", hash_val=None, progress=progress)


def _download_from_ngc_private(
    download_path: Path,
    filename: str,
    version: str,
    repo: str,
    prefix: str = "monai_",
    remove_prefix: str | None = "monai_",
    headers: dict | None = None,
) -> None:
    # ensure prefix is contained
    filename = _add_ngc_prefix(filename, prefix=prefix)
    request_url = _get_ngc_private_bundle_url(model_name=filename, version=version, repo=repo)
    if has_requests:
        headers = {} if headers is None else headers
        response = requests_get(request_url, headers=headers)
        response.raise_for_status()
    else:
        raise ValueError("NGC API requires requests package. Please install it.")

    os.makedirs(download_path, exist_ok=True)
    zip_path = download_path / f"{filename}_v{version}.zip"
    with open(zip_path, "wb") as f:
        f.write(response.content)
    logger.info(f"Downloading: {zip_path}.")
    if remove_prefix:
        filename = _remove_ngc_prefix(filename, prefix=remove_prefix)
    extract_path = download_path / f"{filename}"
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_path)
        logger.info(f"Writing into directory: {extract_path}.")


def _get_ngc_token(api_key, retry=0):
    """Try to connect to NGC."""
    url = "https://authn.nvidia.com/token?service=ngc"
    headers = {"Accept": "application/json", "Authorization": "ApiKey " + api_key}
    if has_requests:
        response = requests_get(url, headers=headers)
        if not response.ok:
            # retry 3 times, if failed, raise an error.
            if retry < 3:
                logger.info(f"Retrying {retry} time(s) to GET {url}.")
                return _get_ngc_token(url, retry + 1)
            raise RuntimeError("NGC API response is not ok. Failed to get token.")
        else:
            token = response.json()["token"]
        return token


def _get_latest_bundle_version_monaihosting(name):
    full_url = f"{MONAI_HOSTING_BASE_URL}/{name.lower()}"
    requests_get, has_requests = optional_import("requests", name="get")
    if has_requests:
        resp = requests_get(full_url)
        resp.raise_for_status()
    else:
        raise ValueError("NGC API requires requests package. Please install it.")
    model_info = json.loads(resp.text)
    return model_info["model"]["latestVersionIdStr"]


def _examine_monai_version(monai_version: str) -> tuple[bool, str]:
    """Examine if the package version is compatible with the MONAI version in the metadata."""
    version_dict = get_versions()
    package_version = version_dict.get("version", "0+unknown")
    if package_version == "0+unknown":
        return False, "Package version is not available. Skipping version check."
    if monai_version == "0+unknown":
        return False, "MONAI version is not specified in the bundle. Skipping version check."
    # treat rc versions as the same as the release version
    package_version = re.sub(r"rc\d.*", "", package_version)
    monai_version = re.sub(r"rc\d.*", "", monai_version)
    if package_version < monai_version:
        return (
            False,
            f"Your MONAI version is {package_version}, but the bundle is built on MONAI version {monai_version}.",
        )
    return True, ""


def _check_monai_version(bundle_dir: PathLike, name: str) -> None:
    """Get the `monai_version` from the metadata.json and compare if it is smaller than the installed `monai` package version"""
    metadata_file = Path(bundle_dir) / name / "configs" / "metadata.json"
    if not metadata_file.exists():
        logger.warning(f"metadata file not found in {metadata_file}.")
        return
    with open(metadata_file) as f:
        metadata = json.load(f)
    is_compatible, msg = _examine_monai_version(metadata.get("monai_version", "0+unknown"))
    if not is_compatible:
        logger.warning(msg)


def _list_latest_versions(data: dict, max_versions: int = 3) -> list[str]:
    """
    Extract the latest versions from the data dictionary.

    Args:
        data: the data dictionary.
        max_versions: the maximum number of versions to return.

    Returns:
        versions of the latest models in the reverse order of creation date, e.g. ['1.0.0', '0.9.0', '0.8.0'].
    """
    # Check if the data is a dictionary and it has the key 'modelVersions'
    if not isinstance(data, dict) or "modelVersions" not in data:
        raise ValueError("The data is not a dictionary or it does not have the key 'modelVersions'.")

    # Extract the list of model versions
    model_versions = data["modelVersions"]

    if (
        not isinstance(model_versions, list)
        or len(model_versions) == 0
        or "createdDate" not in model_versions[0]
        or "versionId" not in model_versions[0]
    ):
        raise ValueError(
            "The model versions are not a list or it is empty or it does not have the keys 'createdDate' and 'versionId'."
        )

    # Sort the versions by the 'createdDate' in descending order
    sorted_versions = sorted(model_versions, key=lambda x: x["createdDate"], reverse=True)
    return [v["versionId"] for v in sorted_versions[:max_versions]]


def _get_latest_bundle_version_ngc(name: str, repo: str | None = None, headers: dict | None = None) -> str:
    base_url = _get_ngc_private_base_url(repo) if repo else NGC_BASE_URL
    version_endpoint = base_url + f"/{name.lower()}/versions/"

    if not has_requests:
        raise ValueError("requests package is required, please install it.")

    version_header = {"Accept-Encoding": "gzip, deflate"}  # Excluding 'zstd' to fit NGC requirements
    if headers:
        version_header.update(headers)
    resp = requests_get(version_endpoint, headers=version_header)
    resp.raise_for_status()
    model_info = json.loads(resp.text)
    latest_versions = _list_latest_versions(model_info)

    for version in latest_versions:
        file_endpoint = base_url + f"/{name.lower()}/versions/{version}/files/configs/metadata.json"
        resp = requests_get(file_endpoint, headers=headers)
        metadata = json.loads(resp.text)
        resp.raise_for_status()
        # if the package version is not available or the model is compatible with the package version
        is_compatible, _ = _examine_monai_version(metadata["monai_version"])
        if is_compatible:
            if version != latest_versions[0]:
                logger.info(f"Latest version is {latest_versions[0]}, but the compatible version is {version}.")
            return version

    # if no compatible version is found, return the latest version
    return latest_versions[0]


def _get_latest_bundle_version(
    source: str, name: str, repo: str, **kwargs: Any
) -> dict[str, list[str] | str] | Any | None:
    if source == "ngc":
        name = _add_ngc_prefix(name)
        return _get_latest_bundle_version_ngc(name)
    elif source == "monaihosting":
        return _get_latest_bundle_version_monaihosting(name)
    elif source == "ngc_private":
        headers = kwargs.pop("headers", {})
        name = _add_ngc_prefix(name)
        return _get_latest_bundle_version_ngc(name, repo=repo, headers=headers)
    elif source == "github":
        repo_owner, repo_name, tag_name = repo.split("/")
        return get_bundle_versions(name, repo=f"{repo_owner}/{repo_name}", tag=tag_name)["latest_version"]
    elif source == "huggingface_hub":
        refs = huggingface_hub.list_repo_refs(repo_id=repo)
        if len(refs.tags) > 0:
            all_versions = [t.name for t in refs.tags]  # git tags, not to be confused with `tag`
            latest_version = ["latest_version" if "latest_version" in all_versions else all_versions[-1]][0]
        else:
            latest_version = [b.name for b in refs.branches][0]  # use the branch that was last updated
        return latest_version
    else:
        raise ValueError(
            f"To get the latest bundle version, source should be 'github', 'monaihosting' or 'ngc', got {source}."
        )


def _process_bundle_dir(bundle_dir: PathLike | None = None) -> Path:
    if bundle_dir is None:
        get_dir, has_home = optional_import("torch.hub", name="get_dir")
        if has_home:
            bundle_dir = Path(get_dir()) / "bundle"
        else:
            raise ValueError("bundle_dir=None, but no suitable default directory computed. Upgrade Pytorch to 1.6+ ?")
    return Path(bundle_dir)


def download(
    name: str | None = None,
    version: str | None = None,
    bundle_dir: PathLike | None = None,
    source: str = DEFAULT_DOWNLOAD_SOURCE,
    repo: str | None = None,
    url: str | None = None,
    remove_prefix: str | None = "monai_",
    progress: bool = True,
    args_file: str | None = None,
) -> None:
    """
    download bundle from the specified source or url. The bundle should be a zip file and it
    will be extracted after downloading.
    This function refers to:
    https://pytorch.org/docs/stable/_modules/torch/hub.html

    Typical usage examples:

    .. code-block:: bash

        # Execute this module as a CLI entry, and download bundle from the model-zoo repo:
        python -m monai.bundle download --name <bundle_name> --version "0.1.0" --bundle_dir "./"

        # Execute this module as a CLI entry, and download bundle from specified github repo:
        python -m monai.bundle download --name <bundle_name> --source "github" --repo "repo_owner/repo_name/release_tag"

        # Execute this module as a CLI entry, and download bundle from ngc with latest version:
        python -m monai.bundle download --name <bundle_name> --source "ngc" --bundle_dir "./"

        # Execute this module as a CLI entry, and download bundle from monaihosting with latest version:
        python -m monai.bundle download --name <bundle_name> --source "monaihosting" --bundle_dir "./"

        # Execute this module as a CLI entry, and download bundle from Hugging Face Hub:
        python -m monai.bundle download --name "bundle_name" --source "huggingface_hub" --repo "repo_owner/repo_name"

        # Execute this module as a CLI entry, and download bundle via URL:
        python -m monai.bundle download --name <bundle_name> --url <url>

        # Execute this module as a CLI entry, and download bundle from ngc_private with latest version:
        python -m monai.bundle download --name <bundle_name> --source "ngc_private" --bundle_dir "./" --repo "org/org_name"

        # Set default args of `run` in a JSON / YAML file, help to record and simplify the command line.
        # Other args still can override the default args at runtime.
        # The content of the JSON / YAML file is a dictionary. For example:
        # {"name": "spleen", "bundle_dir": "download", "source": ""}
        # then do the following command for downloading:
        python -m monai.bundle download --args_file "args.json" --source "github"

    Args:
        name: bundle name. If `None` and `url` is `None`, it must be provided in `args_file`.
            for example:
            "spleen_ct_segmentation", "prostate_mri_anatomy" in model-zoo:
            https://github.com/Project-MONAI/model-zoo/releases/tag/hosting_storage_v1.
            "monai_brats_mri_segmentation" in ngc:
            https://catalog.ngc.nvidia.com/models?filters=&orderBy=scoreDESC&query=monai.
        version: version name of the target bundle to download, like: "0.1.0". If `None`, will download
            the latest version (or the last commit to the `main` branch in the case of Hugging Face Hub).
        bundle_dir: target directory to store the downloaded data.
            Default is `bundle` subfolder under `torch.hub.get_dir()`.
        source: storage location name. This argument is used when `url` is `None`.
            In default, the value is achieved from the environment variable BUNDLE_DOWNLOAD_SRC, and
            it should be "ngc", "monaihosting", "github", "ngc_private", or "huggingface_hub".
            If source is "ngc_private", you need specify the NGC_API_KEY in the environment variable.
        repo: repo name. This argument is used when `url` is `None` and `source` is "github" or "huggingface_hub".
            If `source` is "github", it should be in the form of "repo_owner/repo_name/release_tag".
            If `source` is "huggingface_hub", it should be in the form of "repo_owner/repo_name".
            If `source` is "ngc_private", it should be in the form of "org/org_name" or "org/org_name/team/team_name",
            or you can specify the environment variable NGC_ORG and NGC_TEAM.
        url: url to download the data. If not `None`, data will be downloaded directly
            and `source` will not be checked.
            If `name` is `None`, filename is determined by `monai.apps.utils._basename(url)`.
        remove_prefix: This argument is used when `source` is "ngc" or "ngc_private". Currently, all ngc bundles
            have the ``monai_`` prefix, which is not existing in their model zoo contrasts. In order to
            maintain the consistency between these two sources, remove prefix is necessary.
            Therefore, if specified, downloaded folder name will remove the prefix.
        progress: whether to display a progress bar.
        args_file: a JSON or YAML file to provide default values for all the args in this function.
            so that the command line inputs can be simplified.

    """
    _args = update_kwargs(
        args=args_file,
        name=name,
        version=version,
        bundle_dir=bundle_dir,
        source=source,
        repo=repo,
        url=url,
        remove_prefix=remove_prefix,
        progress=progress,
    )

    _log_input_summary(tag="download", args=_args)
    source_, progress_, remove_prefix_, repo_, name_, version_, bundle_dir_, url_ = _pop_args(
        _args, "source", "progress", remove_prefix=None, repo=None, name=None, version=None, bundle_dir=None, url=None
    )

    bundle_dir_ = _process_bundle_dir(bundle_dir_)
    if repo_ is None:
        org_ = os.getenv("NGC_ORG", None)
        team_ = os.getenv("NGC_TEAM", None)
        if org_ is not None and source_ == "ngc_private":
            repo_ = f"org/{org_}/team/{team_}" if team_ is not None else f"org/{org_}"
        else:
            repo_ = "Project-MONAI/model-zoo/hosting_storage_v1"
    if len(repo_.split("/")) not in (2, 4) and source_ == "ngc_private":
        raise ValueError(f"repo should be in the form of `org/org_name/team/team_name` or `org/org_name`, got {repo_}.")
    if len(repo_.split("/")) != 3 and source_ == "github":
        raise ValueError(f"repo should be in the form of `repo_owner/repo_name/release_tag`, got {repo_}.")
    elif len(repo_.split("/")) != 2 and source_ == "huggingface_hub":
        raise ValueError(f"Hugging Face Hub repo should be in the form of `repo_owner/repo_name`, got {repo_}.")
    if url_ is not None:
        if name_ is not None:
            filepath = bundle_dir_ / f"{name_}.zip"
        else:
            filepath = bundle_dir_ / f"{_basename(url_)}"
        download_url(url=url_, filepath=filepath, hash_val=None, progress=progress_)
        extractall(filepath=filepath, output_dir=bundle_dir_, has_base=True)
    else:
        headers = {}
        if name_ is None:
            raise ValueError(f"To download from source: {source_}, `name` must be provided.")
        if source == "ngc_private":
            api_key = os.getenv("NGC_API_KEY", None)
            if api_key is None:
                raise ValueError("API key is required for ngc_private source.")
            else:
                token = _get_ngc_token(api_key)
                headers = {"Authorization": f"Bearer {token}"}

        if version_ is None:
            version_ = _get_latest_bundle_version(source=source_, name=name_, repo=repo_, headers=headers)
        if source_ == "github":
            name_ver = "_v".join([name_, version_]) if version_ is not None else name_
            _download_from_github(repo=repo_, download_path=bundle_dir_, filename=name_ver, progress=progress_)
        elif source_ == "monaihosting":
            _download_from_monaihosting(download_path=bundle_dir_, filename=name_, version=version_, progress=progress_)
        elif source_ == "ngc":
            _download_from_ngc(
                download_path=bundle_dir_,
                filename=name_,
                version=version_,
                remove_prefix=remove_prefix_,
                progress=progress_,
            )
        elif source_ == "ngc_private":
            _download_from_ngc_private(
                download_path=bundle_dir_,
                filename=name_,
                version=version_,
                remove_prefix=remove_prefix_,
                repo=repo_,
                headers=headers,
            )
        elif source_ == "huggingface_hub":
            extract_path = os.path.join(bundle_dir_, name_)
            huggingface_hub.snapshot_download(repo_id=repo_, revision=version_, local_dir=extract_path)
        else:
            raise NotImplementedError(
                "Currently only download from `url`, source 'github', 'monaihosting', 'huggingface_hub' or 'ngc' are implemented,"
                f"got source: {source_}."
            )

    _check_monai_version(bundle_dir_, name_)


@deprecated_arg("net_name", since="1.2", removed="1.5", msg_suffix="please use ``model`` instead.")
@deprecated_arg("net_kwargs", since="1.2", removed="1.5", msg_suffix="please use ``model`` instead.")
@deprecated_arg("return_state_dict", since="1.2", removed="1.5")
def load(
    name: str,
    model: torch.nn.Module | None = None,
    version: str | None = None,
    workflow_type: str = "train",
    model_file: str | None = None,
    load_ts_module: bool = False,
    bundle_dir: PathLike | None = None,
    source: str = DEFAULT_DOWNLOAD_SOURCE,
    repo: str | None = None,
    remove_prefix: str | None = "monai_",
    progress: bool = True,
    device: str | None = None,
    key_in_ckpt: str | None = None,
    config_files: Sequence[str] = (),
    workflow_name: str | BundleWorkflow | None = None,
    args_file: str | None = None,
    copy_model_args: dict | None = None,
    return_state_dict: bool = True,
    net_override: dict | None = None,
    net_name: str | None = None,
    **net_kwargs: Any,
) -> object | tuple[torch.nn.Module, dict, dict] | Any:
    """
    Load model weights or TorchScript module of a bundle.

    Args:
        name: bundle name. If `None` and `url` is `None`, it must be provided in `args_file`.
            for example:
            "spleen_ct_segmentation", "prostate_mri_anatomy" in model-zoo:
            https://github.com/Project-MONAI/model-zoo/releases/tag/hosting_storage_v1.
            "monai_brats_mri_segmentation" in ngc:
            https://catalog.ngc.nvidia.com/models?filters=&orderBy=scoreDESC&query=monai.
            "mednist_gan" in monaihosting:
            https://api.ngc.nvidia.com/v2/models/nvidia/monaihosting/mednist_gan/versions/0.2.0/files/mednist_gan_v0.2.0.zip
        model: a pytorch module to be updated. Default to None, using the "network_def" in the bundle.
        version: version name of the target bundle to download, like: "0.1.0". If `None`, will download
            the latest version. If `source` is "huggingface_hub", this argument is a Git revision id.
        workflow_type: specifies the workflow type: "train" or "training" for a training workflow,
            or "infer", "inference", "eval", "evaluation" for a inference workflow,
            other unsupported string will raise a ValueError.
            default to `train` for training workflow.
        model_file: the relative path of the model weights or TorchScript module within bundle.
            If `None`, "models/model.pt" or "models/model.ts" will be used.
        load_ts_module: a flag to specify if loading the TorchScript module.
        bundle_dir: directory the weights/TorchScript module will be loaded from.
            Default is `bundle` subfolder under `torch.hub.get_dir()`.
        source: storage location name. This argument is used when `model_file` is not existing locally and need to be
            downloaded first.
            In default, the value is achieved from the environment variable BUNDLE_DOWNLOAD_SRC, and
            it should be "ngc", "monaihosting", "github", or "huggingface_hub".
        repo: repo name. This argument is used when `url` is `None` and `source` is "github" or "huggingface_hub".
            If `source` is "github", it should be in the form of "repo_owner/repo_name/release_tag".
            If `source` is "huggingface_hub", it should be in the form of "repo_owner/repo_name".
        remove_prefix: This argument is used when `source` is "ngc". Currently, all ngc bundles
            have the ``monai_`` prefix, which is not existing in their model zoo contrasts. In order to
            maintain the consistency between these three sources, remove prefix is necessary.
            Therefore, if specified, downloaded folder name will remove the prefix.
        progress: whether to display a progress bar when downloading.
        device: target device of returned weights or module, if `None`, prefer to "cuda" if existing.
        key_in_ckpt: for nested checkpoint like `{"model": XXX, "optimizer": XXX, ...}`, specify the key of model
            weights. if not nested checkpoint, no need to set.
        config_files: extra filenames would be loaded. The argument only works when loading a TorchScript module,
            see `_extra_files` in `torch.jit.load` for more details.
        workflow_name: specified bundle workflow name, should be a string or class, default to "ConfigWorkflow".
        args_file: a JSON or YAML file to provide default values for all the args in "download" function.
        copy_model_args: other arguments for the `monai.networks.copy_model_state` function.
        return_state_dict: whether to return state dict, if True, return state_dict, else a corresponding network
            from `_workflow.network_def` will be instantiated and load the achieved weights.
        net_override: id-value pairs to override the parameters in the network of the bundle, default to `None`.
        net_name: if not `None`, a corresponding network will be instantiated and load the achieved weights.
            This argument only works when loading weights.
        net_kwargs: other arguments that are used to instantiate the network class defined by `net_name`.

    Returns:
        1. If `load_ts_module` is `False` and `model` is `None`,
            return model weights if can't find "network_def" in the bundle,
            else return an instantiated network that loaded the weights.
        2. If `load_ts_module` is `False` and `model` is not `None`,
            return an instantiated network that loaded the weights.
        3. If `load_ts_module` is `True`, return a triple that include a TorchScript module,
            the corresponding metadata dict, and extra files dict.
            please check `monai.data.load_net_with_metadata` for more details.
        4. If `return_state_dict` is True, return model weights, only used for compatibility
            when `model` and `net_name` are all `None`.

    """
    if return_state_dict and (model is not None or net_name is not None):
        warnings.warn("Incompatible values: model and net_name are all specified, return state dict instead.")

    bundle_dir_ = _process_bundle_dir(bundle_dir)
    net_override = {} if net_override is None else net_override
    copy_model_args = {} if copy_model_args is None else copy_model_args

    if device is None:
        device = "cuda:0" if is_available() else "cpu"
    if model_file is None:
        model_file = os.path.join("models", "model.ts" if load_ts_module is True else "model.pt")
    if source == "ngc":
        name = _add_ngc_prefix(name)
        if remove_prefix:
            name = _remove_ngc_prefix(name, prefix=remove_prefix)
    full_path = os.path.join(bundle_dir_, name, model_file)
    if not os.path.exists(full_path):
        download(
            name=name,
            version=version,
            bundle_dir=bundle_dir_,
            source=source,
            repo=repo,
            remove_prefix=remove_prefix,
            progress=progress,
            args_file=args_file,
        )

    # loading with `torch.jit.load`
    if load_ts_module is True:
        return load_net_with_metadata(full_path, map_location=torch.device(device), more_extra_files=config_files)
    # loading with `torch.load`
    model_dict = torch.load(full_path, map_location=torch.device(device))

    if not isinstance(model_dict, Mapping):
        warnings.warn(f"the state dictionary from {full_path} should be a dictionary but got {type(model_dict)}.")
        model_dict = get_state_dict(model_dict)

    if return_state_dict:
        return model_dict

    _workflow = None
    if model is None and net_name is None:
        bundle_config_file = bundle_dir_ / name / "configs" / f"{workflow_type}.json"
        if bundle_config_file.is_file():
            _net_override = {f"network_def#{key}": value for key, value in net_override.items()}
            _workflow = create_workflow(
                workflow_name=workflow_name,
                args_file=args_file,
                config_file=str(bundle_config_file),
                workflow_type=workflow_type,
                **_net_override,
            )
        else:
            warnings.warn(f"Cannot find the config file: {bundle_config_file}, return state dict instead.")
            return model_dict
        if _workflow is not None:
            if not hasattr(_workflow, "network_def"):
                warnings.warn("No available network definition in the bundle, return state dict instead.")
                return model_dict
            else:
                model = _workflow.network_def
    elif net_name is not None:
        net_kwargs["_target_"] = net_name
        configer = ConfigComponent(config=net_kwargs)
        model = configer.instantiate()  # type: ignore

    model.to(device)  # type: ignore

    copy_model_state(
        dst=model, src=model_dict if key_in_ckpt is None else model_dict[key_in_ckpt], **copy_model_args  # type: ignore
    )

    return model


def _get_all_bundles_info(
    repo: str = "Project-MONAI/model-zoo", tag: str = "dev", auth_token: str | None = None
) -> dict[str, dict[str, dict[str, Any]]]:
    if has_requests:
        if tag == "hosting_storage_v1":
            request_url = f"https://api.github.com/repos/{repo}/releases"
        else:
            request_url = f"https://raw.githubusercontent.com/{repo}/{tag}/models/model_info.json"

        if auth_token is not None:
            headers = {"Authorization": f"Bearer {auth_token}"}
            resp = requests_get(request_url, headers=headers)
        else:
            resp = requests_get(request_url)
        resp.raise_for_status()
    else:
        raise ValueError("requests package is required, please install it.")
    releases_list = json.loads(resp.text)
    bundle_name_pattern = re.compile(r"_v\d*.")
    bundles_info: dict[str, dict[str, dict[str, Any]]] = {}

    if tag == "hosting_storage_v1":
        for release in releases_list:
            if release["tag_name"] == tag:
                for asset in release["assets"]:
                    asset_name = bundle_name_pattern.split(asset["name"])[0]
                    if asset_name not in bundles_info:
                        bundles_info[asset_name] = {}
                    asset_version = asset["name"].split(f"{asset_name}_v")[-1].replace(".zip", "")
                    bundles_info[asset_name][asset_version] = dict(asset)
                return bundles_info
    else:
        for asset in releases_list.keys():
            asset_name = bundle_name_pattern.split(asset)[0]
            if asset_name not in bundles_info:
                bundles_info[asset_name] = {}
            asset_version = asset.split(f"{asset_name}_v")[-1]
            bundles_info[asset_name][asset_version] = {
                "name": asset,
                "browser_download_url": releases_list[asset]["source"],
            }
    return bundles_info


def get_all_bundles_list(
    repo: str = "Project-MONAI/model-zoo", tag: str = "dev", auth_token: str | None = None
) -> list[tuple[str, str]]:
    """
    Get all bundles names (and the latest versions) that are stored in the release of specified repository
    with the provided tag. If tag is "dev", will get model information from
    https://raw.githubusercontent.com/repo_owner/repo_name/dev/models/model_info.json.
    The default values of arguments correspond to the release of MONAI model zoo. In order to increase the
    rate limits of calling Github APIs, you can input your personal access token.
    Please check the following link for more details about rate limiting:
    https://docs.github.com/en/rest/overview/resources-in-the-rest-api#rate-limiting

    The following link shows how to create your personal access token:
    https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token

    Args:
        repo: it should be in the form of "repo_owner/repo_name/".
        tag: the tag name of the release.
        auth_token: github personal access token.

    Returns:
        a list of tuple in the form of (bundle name, latest version).

    """

    bundles_info = _get_all_bundles_info(repo=repo, tag=tag, auth_token=auth_token)
    bundles_list = []
    for bundle_name in bundles_info:
        latest_version = sorted(bundles_info[bundle_name].keys())[-1]
        bundles_list.append((bundle_name, latest_version))

    return bundles_list


def get_bundle_versions(
    bundle_name: str, repo: str = "Project-MONAI/model-zoo", tag: str = "dev", auth_token: str | None = None
) -> dict[str, list[str] | str]:
    """
    Get the latest version, as well as all existing versions of a bundle that is stored in the release of specified
    repository with the provided tag. If tag is "dev", will get model information from
    https://raw.githubusercontent.com/repo_owner/repo_name/dev/models/model_info.json.
    In order to increase the rate limits of calling Github APIs, you can input your personal access token.
    Please check the following link for more details about rate limiting:
    https://docs.github.com/en/rest/overview/resources-in-the-rest-api#rate-limiting

    The following link shows how to create your personal access token:
    https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token

    Args:
        bundle_name: bundle name.
        repo: it should be in the form of "repo_owner/repo_name/".
        tag: the tag name of the release.
        auth_token: github personal access token.

    Returns:
        a dictionary that contains the latest version and all versions of a bundle.

    """

    bundles_info = _get_all_bundles_info(repo=repo, tag=tag, auth_token=auth_token)
    if bundle_name not in bundles_info:
        raise ValueError(f"bundle: {bundle_name} is not existing in repo: {repo}.")
    bundle_info = bundles_info[bundle_name]
    all_versions = sorted(bundle_info.keys())

    return {"latest_version": all_versions[-1], "all_versions": all_versions}


def get_bundle_info(
    bundle_name: str,
    version: str | None = None,
    repo: str = "Project-MONAI/model-zoo",
    tag: str = "dev",
    auth_token: str | None = None,
) -> dict[str, Any]:
    """
    Get all information (include "name" and "browser_download_url") of a bundle
    with the specified bundle name and version which is stored in the release of specified repository with the provided tag.
    In order to increase the rate limits of calling Github APIs, you can input your personal access token.
    Please check the following link for more details about rate limiting:
    https://docs.github.com/en/rest/overview/resources-in-the-rest-api#rate-limiting

    The following link shows how to create your personal access token:
    https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token

    Args:
        bundle_name: bundle name.
        version: version name of the target bundle, if None, the latest version will be used.
        repo: it should be in the form of "repo_owner/repo_name/".
        tag: the tag name of the release.
        auth_token: github personal access token.

    Returns:
        a dictionary that contains the bundle's information.

    """

    bundles_info = _get_all_bundles_info(repo=repo, tag=tag, auth_token=auth_token)
    if bundle_name not in bundles_info:
        raise ValueError(f"bundle: {bundle_name} is not existing.")
    bundle_info = bundles_info[bundle_name]
    if version is None:
        version = sorted(bundle_info.keys())[-1]
    if version not in bundle_info:
        raise ValueError(f"version: {version} of bundle: {bundle_name} is not existing.")

    return bundle_info[version]


def run(
    run_id: str | None = None,
    init_id: str | None = None,
    final_id: str | None = None,
    meta_file: str | Sequence[str] | None = None,
    config_file: str | Sequence[str] | None = None,
    logging_file: str | None = None,
    tracking: str | dict | None = None,
    args_file: str | None = None,
    **override: Any,
) -> None:
    """
    Specify `config_file` to run monai bundle components and workflows.

    Typical usage examples:

    .. code-block:: bash

        # Execute this module as a CLI entry:
        python -m monai.bundle run --meta_file <meta path> --config_file <config path>

        # Execute with specified `run_id=training`:
        python -m monai.bundle run training --meta_file <meta path> --config_file <config path>

        # Execute with all specified `run_id=runtest`, `init_id=inittest`, `final_id=finaltest`:
        python -m monai.bundle run --run_id runtest --init_id inittest --final_id finaltest ...

        # Override config values at runtime by specifying the component id and its new value:
        python -m monai.bundle run --net#input_chns 1 ...

        # Override config values with another config file `/path/to/another.json`:
        python -m monai.bundle run --net %/path/to/another.json ...

        # Override config values with part content of another config file:
        python -m monai.bundle run --net %/data/other.json#net_arg ...

        # Set default args of `run` in a JSON / YAML file, help to record and simplify the command line.
        # Other args still can override the default args at runtime:
        python -m monai.bundle run --args_file "/workspace/data/args.json" --config_file <config path>

    Args:
        run_id: ID name of the expected config expression to run, default to "run".
            to run the config, the target config must contain this ID.
        init_id: ID name of the expected config expression to initialize before running, default to "initialize".
            it's optional for both configs and this `run` function.
        final_id: ID name of the expected config expression to finalize after running, default to "finalize".
            it's optional for both configs and this `run` function.
        meta_file: filepath of the metadata file, if it is a list of file paths, the content of them will be merged.
            Default to None.
        config_file: filepath of the config file, if `None`, must be provided in `args_file`.
            if it is a list of file paths, the content of them will be merged.
        logging_file: config file for `logging` module in the program. for more details:
            https://docs.python.org/3/library/logging.config.html#logging.config.fileConfig.
            Default to None.
        tracking: if not None, enable the experiment tracking at runtime with optionally configurable and extensible.
            If "mlflow", will add `MLFlowHandler` to the parsed bundle with default tracking settings where a set of
            common parameters shown below will be added and can be passed through the `override` parameter of this method.

            - ``"output_dir"``: the path to save mlflow tracking outputs locally, default to "<bundle root>/eval".
            - ``"tracking_uri"``: uri to save mlflow tracking outputs, default to "/output_dir/mlruns".
            - ``"experiment_name"``: experiment name for this run, default to "monai_experiment".
            - ``"run_name"``: the name of current run.
            - ``"save_execute_config"``: whether to save the executed config files. It can be `False`, `/path/to/artifacts`
              or `True`. If set to `True`, will save to the default path "<bundle_root>/eval". Default to `True`.

            If other string, treat it as file path to load the tracking settings.
            If `dict`, treat it as tracking settings.
            Will patch the target config content with `tracking handlers` and the top-level items of `configs`.
            for detailed usage examples, please check the tutorial:
            https://github.com/Project-MONAI/tutorials/blob/main/experiment_management/bundle_integrate_mlflow.ipynb.
        args_file: a JSON or YAML file to provide default values for `run_id`, `meta_file`,
            `config_file`, `logging`, and override pairs. so that the command line inputs can be simplified.
        override: id-value pairs to override or add the corresponding config content.
            e.g. ``--net#input_chns 42``, ``--net %/data/other.json#net_arg``.

    """

    workflow = create_workflow(
        config_file=config_file,
        args_file=args_file,
        meta_file=meta_file,
        logging_file=logging_file,
        init_id=init_id,
        run_id=run_id,
        final_id=final_id,
        tracking=tracking,
        **override,
    )
    workflow.run()
    workflow.finalize()


def run_workflow(
    workflow_name: str | BundleWorkflow | None = None, args_file: str | None = None, **kwargs: Any
) -> None:
    """
    Specify `bundle workflow` to run monai bundle components and workflows.
    The workflow should be subclass of `BundleWorkflow` and be available to import.
    It can be MONAI existing bundle workflows or user customized workflows.

    Typical usage examples:

    .. code-block:: bash

        # Execute this module as a CLI entry with default ConfigWorkflow:
        python -m monai.bundle run_workflow --meta_file <meta path> --config_file <config path>

        # Set the workflow to other customized BundleWorkflow subclass:
        python -m monai.bundle run_workflow --workflow_name CustomizedWorkflow ...

    Args:
        workflow_name: specified bundle workflow name, should be a string or class, default to "ConfigWorkflow".
        args_file: a JSON or YAML file to provide default values for this API.
            so that the command line inputs can be simplified.
        kwargs: arguments to instantiate the workflow class.

    """

    workflow_ = create_workflow(workflow_name=workflow_name, args_file=args_file, **kwargs)
    workflow_.run()
    workflow_.finalize()


def verify_metadata(
    meta_file: str | Sequence[str] | None = None,
    filepath: PathLike | None = None,
    create_dir: bool | None = None,
    hash_val: str | None = None,
    hash_type: str | None = None,
    args_file: str | None = None,
    **kwargs: Any,
) -> None:
    """
    Verify the provided `metadata` file based on the predefined `schema`.
    `metadata` content must contain the `schema` field for the URL of schema file to download.
    The schema standard follows: http://json-schema.org/.

    Args:
        meta_file: filepath of the metadata file to verify, if `None`, must be provided in `args_file`.
            if it is a list of file paths, the content of them will be merged.
        filepath: file path to store the downloaded schema.
        create_dir: whether to create directories if not existing, default to `True`.
        hash_val: if not None, define the hash value to verify the downloaded schema file.
        hash_type: if not None, define the hash type to verify the downloaded schema file. Defaults to "md5".
        args_file: a JSON or YAML file to provide default values for all the args in this function.
            so that the command line inputs can be simplified.
        kwargs: other arguments for `jsonschema.validate()`. for more details:
            https://python-jsonschema.readthedocs.io/en/stable/validate/#jsonschema.validate.

    """

    _args = update_kwargs(
        args=args_file,
        meta_file=meta_file,
        filepath=filepath,
        create_dir=create_dir,
        hash_val=hash_val,
        hash_type=hash_type,
        **kwargs,
    )
    _log_input_summary(tag="verify_metadata", args=_args)
    filepath_, meta_file_, create_dir_, hash_val_, hash_type_ = _pop_args(
        _args, "filepath", "meta_file", create_dir=True, hash_val=None, hash_type="md5"
    )

    check_parent_dir(path=filepath_, create_dir=create_dir_)
    metadata = ConfigParser.load_config_files(files=meta_file_)
    url = metadata.get("schema")
    if url is None:
        raise ValueError("must provide the `schema` field in the metadata for the URL of schema file.")
    download_url(url=url, filepath=filepath_, hash_val=hash_val_, hash_type=hash_type_, progress=True)
    schema = ConfigParser.load_config_file(filepath=filepath_)

    try:
        # the rest key-values in the _args are for `validate` API
        validate(instance=metadata, schema=schema, **_args)
    except ValidationError as e:  # pylint: disable=E0712
        # as the error message is very long, only extract the key information
        raise ValueError(
            re.compile(r".*Failed validating", re.S).findall(str(e))[0] + f" against schema `{url}`."
        ) from e
    logger.info("metadata is verified with no error.")


def _get_net_io_info(parser: ConfigParser | None = None, prefix: str = "_meta_#network_data_format") -> tuple:
    """
    Get the input and output information defined in the metadata.

    Args:
        parser: a ConfigParser of the given bundle.
        prefix: a prefix for the input and output ID, which will be combined as `prefix#inputs` and
            `prefix#outputs` to parse the input and output information in the `metadata.json` file of
            a bundle, default to `meta_#network_data_format`.

    Returns:
        input_channels: the channel number of the `image` input.
        input_spatial_shape: the spatial shape of the `image` input.
        input_dtype: the data type of the `image` input.
        output_channels: the channel number of the output.
        output_dtype: the data type of the output.
    """
    if not isinstance(parser, ConfigParser):
        raise AttributeError(f"Parameter parser should be a ConfigParser, got {type(parser)}.")

    prefix_key = f"{prefix}#inputs"
    key = f"{prefix_key}#image#num_channels"
    input_channels = parser.get(key)
    key = f"{prefix_key}#image#spatial_shape"
    input_spatial_shape = tuple(parser.get(key))
    key = f"{prefix_key}#image#dtype"
    input_dtype = get_equivalent_dtype(parser.get(key), torch.Tensor)

    prefix_key = f"{prefix}#outputs"
    key = f"{prefix_key}#pred#num_channels"
    output_channels = parser.get(key)
    key = f"{prefix_key}#pred#dtype"
    output_dtype = get_equivalent_dtype(parser.get(key), torch.Tensor)

    return input_channels, input_spatial_shape, input_dtype, output_channels, output_dtype


def _get_fake_input_shape(parser: ConfigParser) -> tuple:
    """
    Get a fake input shape e.g. [N, C, H, W] or [N, C, H, W, D], whose batch size is 1, from the given parser.

    Args:
        parser: a ConfigParser which contains the i/o information of a bundle.
    """
    input_channels, input_spatial_shape, _, _, _ = _get_net_io_info(parser=parser)
    spatial_shape = _get_fake_spatial_shape(input_spatial_shape)
    input_shape = (1, input_channels, *spatial_shape)
    return input_shape


def verify_net_in_out(
    net_id: str | None = None,
    meta_file: str | Sequence[str] | None = None,
    config_file: str | Sequence[str] | None = None,
    device: str | None = None,
    p: int | None = None,
    n: int | None = None,
    any: int | None = None,
    extra_forward_args: dict | None = None,
    args_file: str | None = None,
    **override: Any,
) -> None:
    """
    Verify the input and output data shape and data type of network defined in the metadata.
    Will test with fake Tensor data according to the required data shape in `metadata`.

    Typical usage examples:

    .. code-block:: bash

        python -m monai.bundle verify_net_in_out network --meta_file <meta path> --config_file <config path>

    Args:
        net_id: ID name of the network component to verify, it must be `torch.nn.Module`.
        meta_file: filepath of the metadata file to get network args, if `None`, must be provided in `args_file`.
            if it is a list of file paths, the content of them will be merged.
        config_file: filepath of the config file to get network definition, if `None`, must be provided in `args_file`.
            if it is a list of file paths, the content of them will be merged.
        device: target device to run the network forward computation, if None, prefer to "cuda" if existing.
        p: power factor to generate fake data shape if dim of expected shape is "x**p", default to 1.
        n: multiply factor to generate fake data shape if dim of expected shape is "x*n", default to 1.
        any: specified size to generate fake data shape if dim of expected shape is "*", default to 1.
        extra_forward_args: a dictionary that contains other args for the forward function of the network.
            Default to an empty dictionary.
        args_file: a JSON or YAML file to provide default values for `net_id`, `meta_file`, `config_file`,
            `device`, `p`, `n`, `any`, and override pairs. so that the command line inputs can be simplified.
        override: id-value pairs to override or add the corresponding config content.
            e.g. ``--_meta#network_data_format#inputs#image#num_channels 3``.

    """

    _args = update_kwargs(
        args=args_file,
        net_id=net_id,
        meta_file=meta_file,
        config_file=config_file,
        device=device,
        p=p,
        n=n,
        any=any,
        extra_forward_args=extra_forward_args,
        **override,
    )
    _log_input_summary(tag="verify_net_in_out", args=_args)
    config_file_, meta_file_, net_id_, device_, p_, n_, any_, extra_forward_args_ = _pop_args(
        _args,
        "config_file",
        "meta_file",
        net_id="",
        device="cuda:0" if is_available() else "cpu",
        p=1,
        n=1,
        any=1,
        extra_forward_args={},
    )

    parser = ConfigParser()
    parser.read_config(f=config_file_)
    parser.read_meta(f=meta_file_)

    # the rest key-values in the _args are to override config content
    for k, v in _args.items():
        parser[k] = v

    input_channels, input_spatial_shape, input_dtype, output_channels, output_dtype = _get_net_io_info(parser=parser)
    try:
        key: str = net_id_  # mark the full id when KeyError
        net = parser.get_parsed_content(key).to(device_)
    except KeyError as e:
        raise KeyError(f"Failed to verify due to missing expected key in the config: {key}.") from e

    net.eval()
    with torch.no_grad():
        spatial_shape = _get_fake_spatial_shape(input_spatial_shape, p=p_, n=n_, any=any_)
        test_data = torch.rand(*(1, input_channels, *spatial_shape), dtype=input_dtype, device=device_)
        if input_dtype == torch.float16:
            # fp16 can only be executed in gpu mode
            net.to("cuda")
            from torch.cuda.amp import autocast

            with autocast():
                output = net(test_data.cuda(), **extra_forward_args_)
            net.to(device_)
        else:
            output = net(test_data, **extra_forward_args_)
        if output.shape[1] != output_channels:
            raise ValueError(f"output channel number `{output.shape[1]}` doesn't match: `{output_channels}`.")
        if output.dtype != output_dtype:
            raise ValueError(f"dtype of output data `{output.dtype}` doesn't match: {output_dtype}.")
    logger.info("data shape of network is verified with no error.")


def _export(
    converter: Callable,
    saver: Callable,
    parser: ConfigParser,
    net_id: str,
    filepath: str,
    ckpt_file: str,
    config_file: str,
    key_in_ckpt: str,
    **kwargs: Any,
) -> None:
    """
    Export a model defined in the parser to a new one specified by the converter.

    Args:
        converter: a callable object that takes a torch.nn.module and kwargs as input and
            converts the module to another type.
        saver: a callable object that accepts the converted model to save, a filepath to save to, meta values
            (extracted from the parser), and a dictionary of extra JSON files (name -> contents) as input.
        parser: a ConfigParser of the bundle to be converted.
        net_id: ID name of the network component in the parser, it must be `torch.nn.Module`.
        filepath: filepath to export, if filename has no extension, it becomes `.ts`.
        ckpt_file: filepath of the model checkpoint to load.
        config_file: filepath of the config file to save in the converted model,the saved key in the converted
            model is the config filename without extension, and the saved config value is always serialized in
            JSON format no matter the original file format is JSON or YAML. it can be a single file or a list
            of files.
        key_in_ckpt: for nested checkpoint like `{"model": XXX, "optimizer": XXX, ...}`, specify the key of model
            weights. if not nested checkpoint, no need to set.
        kwargs: key arguments for the converter.

    """
    net = parser.get_parsed_content(net_id)
    if has_ignite:
        # here we use ignite Checkpoint to support nested weights and be compatible with MONAI CheckpointSaver
        Checkpoint.load_objects(to_load={key_in_ckpt: net}, checkpoint=ckpt_file)
    else:
        ckpt = torch.load(ckpt_file)
        copy_model_state(dst=net, src=ckpt if key_in_ckpt == "" else ckpt[key_in_ckpt])

    # Use the given converter to convert a model and save with metadata, config content
    net = converter(model=net, **kwargs)

    extra_files: dict = {}
    for i in ensure_tuple(config_file):
        # split the filename and directory
        filename = os.path.basename(i)
        # remove extension
        filename, _ = os.path.splitext(filename)
        # because all files are stored as JSON their name parts without extension must be unique
        if filename in extra_files:
            raise ValueError(f"Filename part '{filename}' is given multiple times in config file list.")
        # the file may be JSON or YAML but will get loaded and dumped out again as JSON
        extra_files[filename] = json.dumps(ConfigParser.load_config_file(i)).encode()

    # add .json extension to all extra files which are always encoded as JSON
    extra_files = {k + ".json": v for k, v in extra_files.items()}

    meta_values = parser.get().pop("_meta_", None)
    saver(net, filepath, meta_values=meta_values, more_extra_files=extra_files)

    logger.info(f"exported to file: {filepath}.")


def onnx_export(
    net_id: str | None = None,
    filepath: PathLike | None = None,
    ckpt_file: str | None = None,
    meta_file: str | Sequence[str] | None = None,
    config_file: str | Sequence[str] | None = None,
    key_in_ckpt: str | None = None,
    use_trace: bool | None = None,
    input_shape: Sequence[int] | None = None,
    args_file: str | None = None,
    converter_kwargs: Mapping | None = None,
    **override: Any,
) -> None:
    """
    Export the model checkpoint to an onnx model.

    Typical usage examples:

    .. code-block:: bash

        python -m monai.bundle onnx_export network --filepath <export path> --ckpt_file <checkpoint path> ...

    Args:
        net_id: ID name of the network component in the config, it must be `torch.nn.Module`.
        filepath: filepath where the onnx model is saved to.
        ckpt_file: filepath of the model checkpoint to load.
        meta_file: filepath of the metadata file, if it is a list of file paths, the content of them will be merged.
        config_file: filepath of the config file that contains extract network information,
        key_in_ckpt: for nested checkpoint like `{"model": XXX, "optimizer": XXX, ...}`, specify the key of model
            weights. if not nested checkpoint, no need to set.
        use_trace: whether using `torch.jit.trace` to convert the pytorch model to torchscript model.
        input_shape: a shape used to generate the random input of the network, when converting the model to an
            onnx model. Should be a list like [N, C, H, W] or [N, C, H, W, D]. If not given, will try to parse from
            the `metadata` config.
        args_file: a JSON or YAML file to provide default values for all the parameters of this function, so that
            the command line inputs can be simplified.
        converter_kwargs: extra arguments that are needed by `convert_to_onnx`, except ones that already exist in the
            input parameters.
        override: id-value pairs to override or add the corresponding config content.
            e.g. ``--_meta#network_data_format#inputs#image#num_channels 3``.

    """
    _args = update_kwargs(
        args=args_file,
        net_id=net_id,
        filepath=filepath,
        meta_file=meta_file,
        config_file=config_file,
        ckpt_file=ckpt_file,
        key_in_ckpt=key_in_ckpt,
        use_trace=use_trace,
        input_shape=input_shape,
        converter_kwargs=converter_kwargs,
        **override,
    )
    _log_input_summary(tag="onnx_export", args=_args)
    (
        filepath_,
        ckpt_file_,
        config_file_,
        net_id_,
        meta_file_,
        key_in_ckpt_,
        use_trace_,
        input_shape_,
        converter_kwargs_,
    ) = _pop_args(
        _args,
        "filepath",
        "ckpt_file",
        "config_file",
        net_id="",
        meta_file=None,
        key_in_ckpt="",
        use_trace=False,
        input_shape=None,
        converter_kwargs={},
    )

    parser = ConfigParser()

    parser.read_config(f=config_file_)
    if meta_file_ is not None:
        parser.read_meta(f=meta_file_)

    # the rest key-values in the _args are to override config content
    for k, v in _args.items():
        parser[k] = v

    # The convert_to_onnx must have an `inputs` as input, no matter what the `use_trace` is.
    # If the `input_shape` is not provided, will try to parse it from the parser to generate a random `inputs`.
    if not input_shape_:
        input_shape_ = _get_fake_input_shape(parser=parser)

    inputs_ = [torch.rand(input_shape_)]

    converter_kwargs_.update({"inputs": inputs_, "use_trace": use_trace_})

    def save_onnx(onnx_obj: Any, filename_prefix_or_stream: str, **kwargs: Any) -> None:
        onnx.save(onnx_obj, filename_prefix_or_stream)

    _export(
        convert_to_onnx,
        save_onnx,
        parser,
        net_id=net_id_,
        filepath=filepath_,
        ckpt_file=ckpt_file_,
        config_file=config_file_,
        key_in_ckpt=key_in_ckpt_,
        **converter_kwargs_,
    )


def ckpt_export(
    net_id: str | None = None,
    filepath: PathLike | None = None,
    ckpt_file: str | None = None,
    meta_file: str | Sequence[str] | None = None,
    config_file: str | Sequence[str] | None = None,
    key_in_ckpt: str | None = None,
    use_trace: bool | None = None,
    input_shape: Sequence[int] | None = None,
    args_file: str | None = None,
    converter_kwargs: Mapping | None = None,
    **override: Any,
) -> None:
    """
    Export the model checkpoint to the given filepath with metadata and config included as JSON files.

    Typical usage examples:

    .. code-block:: bash

        python -m monai.bundle ckpt_export network --filepath <export path> --ckpt_file <checkpoint path> ...

    Args:
        net_id: ID name of the network component in the config, it must be `torch.nn.Module`.
            Default to "network_def".
        filepath: filepath to export, if filename has no extension it becomes `.ts`.
            Default to "models/model.ts" under "os.getcwd()" if `bundle_root` is not specified.
        ckpt_file: filepath of the model checkpoint to load.
            Default to "models/model.pt" under "os.getcwd()" if `bundle_root` is not specified.
        meta_file: filepath of the metadata file, if it is a list of file paths, the content of them will be merged.
            Default to "configs/metadata.json" under "os.getcwd()" if `bundle_root` is not specified.
        config_file: filepath of the config file to save in TorchScript model and extract network information,
            the saved key in the TorchScript model is the config filename without extension, and the saved config
            value is always serialized in JSON format no matter the original file format is JSON or YAML.
            it can be a single file or a list of files. if `None`, must be provided in `args_file`.
        key_in_ckpt: for nested checkpoint like `{"model": XXX, "optimizer": XXX, ...}`, specify the key of model
            weights. if not nested checkpoint, no need to set.
        use_trace: whether using `torch.jit.trace` to convert the PyTorch model to TorchScript model.
        input_shape: a shape used to generate the random input of the network, when converting the model to a
            TorchScript model. Should be a list like [N, C, H, W] or [N, C, H, W, D]. If not given, will try to
            parse from the `metadata` config.
        args_file: a JSON or YAML file to provide default values for all the parameters of this function, so that
            the command line inputs can be simplified.
        converter_kwargs: extra arguments that are needed by `convert_to_torchscript`, except ones that already exist
            in the input parameters.
        override: id-value pairs to override or add the corresponding config content.
            e.g. ``--_meta#network_data_format#inputs#image#num_channels 3``.

    """
    _args = update_kwargs(
        args=args_file,
        net_id=net_id,
        filepath=filepath,
        meta_file=meta_file,
        config_file=config_file,
        ckpt_file=ckpt_file,
        key_in_ckpt=key_in_ckpt,
        use_trace=use_trace,
        input_shape=input_shape,
        converter_kwargs=converter_kwargs,
        **override,
    )
    _log_input_summary(tag="ckpt_export", args=_args)
    (
        config_file_,
        filepath_,
        ckpt_file_,
        net_id_,
        meta_file_,
        key_in_ckpt_,
        use_trace_,
        input_shape_,
        converter_kwargs_,
    ) = _pop_args(
        _args,
        "config_file",
        filepath=None,
        ckpt_file=None,
        net_id=None,
        meta_file=None,
        key_in_ckpt="",
        use_trace=False,
        input_shape=None,
        converter_kwargs={},
    )
    bundle_root = _args.get("bundle_root", os.getcwd())

    parser = ConfigParser()
    parser.read_config(f=config_file_)
    meta_file_ = os.path.join(bundle_root, "configs", "metadata.json") if meta_file_ is None else meta_file_
    if os.path.exists(meta_file_):
        parser.read_meta(f=meta_file_)

    # the rest key-values in the _args are to override config content
    for k, v in _args.items():
        parser[k] = v

    filepath_ = os.path.join(bundle_root, "models", "model.ts") if filepath_ is None else filepath_
    ckpt_file_ = os.path.join(bundle_root, "models", "model.pt") if ckpt_file_ is None else ckpt_file_
    if not os.path.exists(ckpt_file_):
        raise FileNotFoundError(f'Checkpoint file "{ckpt_file_}" not found, please specify it in argument "ckpt_file".')

    net_id_ = "network_def" if net_id_ is None else net_id_
    try:
        parser.get_parsed_content(net_id_)
    except ValueError as e:
        raise ValueError(
            f'Network definition "{net_id_}" cannot be found in "{config_file_}", specify name with argument "net_id".'
        ) from e

    # When export through torch.jit.trace without providing input_shape, will try to parse one from the parser.
    if (not input_shape_) and use_trace:
        input_shape_ = _get_fake_input_shape(parser=parser)

    inputs_: Sequence[Any] | None = [torch.rand(input_shape_)] if input_shape_ else None

    converter_kwargs_.update({"inputs": inputs_, "use_trace": use_trace_})
    # Use the given converter to convert a model and save with metadata, config content

    save_ts = partial(save_net_with_metadata, include_config_vals=False, append_timestamp=False)

    _export(
        convert_to_torchscript,
        save_ts,
        parser,
        net_id=net_id_,
        filepath=filepath_,
        ckpt_file=ckpt_file_,
        config_file=config_file_,
        key_in_ckpt=key_in_ckpt_,
        **converter_kwargs_,
    )


def trt_export(
    net_id: str | None = None,
    filepath: PathLike | None = None,
    ckpt_file: str | None = None,
    meta_file: str | Sequence[str] | None = None,
    config_file: str | Sequence[str] | None = None,
    key_in_ckpt: str | None = None,
    precision: str | None = None,
    input_shape: Sequence[int] | None = None,
    use_trace: bool | None = None,
    dynamic_batchsize: Sequence[int] | None = None,
    device: int | None = None,
    use_onnx: bool | None = None,
    onnx_input_names: Sequence[str] | None = None,
    onnx_output_names: Sequence[str] | None = None,
    args_file: str | None = None,
    converter_kwargs: Mapping | None = None,
    **override: Any,
) -> None:
    """
    Export the model checkpoint to the given filepath as a TensorRT engine-based TorchScript.
    Currently, this API only supports converting models whose inputs are all tensors.
    Note: NVIDIA Volta support (GPUs with compute capability 7.0) has been removed starting with TensorRT 10.5.
    Review the TensorRT Support Matrix for which GPUs are supported.

    There are two ways to export a model:
    1, Torch-TensorRT way: PyTorch module ---> TorchScript module ---> TensorRT engine-based TorchScript.
    2, ONNX-TensorRT way: PyTorch module ---> TorchScript module ---> ONNX model ---> TensorRT engine --->
    TensorRT engine-based TorchScript.

    When exporting through the first way, some models suffer from the slowdown problem, since Torch-TensorRT
    may only convert a little part of the PyTorch model to the TensorRT engine. However when exporting through
    the second way, some Python data structures like `dict` are not supported. And some TorchScript models are
    not supported by the ONNX if exported through `torch.jit.script`.

    Typical usage examples:

    .. code-block:: bash

        python -m monai.bundle trt_export --net_id <network definition> --filepath <export path> \
            --ckpt_file <checkpoint path> --input_shape <input shape> --dynamic_batchsize <batch range> ...

    Args:
        net_id: ID name of the network component in the config, it must be `torch.nn.Module`.
        filepath: filepath to export, if filename has no extension, it becomes `.ts`.
        ckpt_file: filepath of the model checkpoint to load.
        meta_file: filepath of the metadata file, if it is a list of file paths, the content of them will be merged.
        config_file: filepath of the config file to save in the TensorRT based TorchScript model and extract network
            information, the saved key in the model is the config filename without extension, and the saved config
            value is always serialized in JSON format no matter the original file format is JSON or YAML.
            it can be a single file or a list of files. if `None`, must be provided in `args_file`.
        key_in_ckpt: for nested checkpoint like `{"model": XXX, "optimizer": XXX, ...}`, specify the key of model
            weights. if not nested checkpoint, no need to set.
        precision: the weight precision of the converted TensorRT engine based TorchScript models. Should be 'fp32' or 'fp16'.
        input_shape: the input shape that is used to convert the model. Should be a list like [N, C, H, W] or
            [N, C, H, W, D]. If not given, will try to parse from the `metadata` config.
        use_trace: whether using `torch.jit.trace` to convert the PyTorch model to a TorchScript model and then convert to
            a TensorRT engine based TorchScript model or an ONNX model (if `use_onnx` is True).
        dynamic_batchsize: a sequence with three elements to define the batch size range of the input for the model to be
            converted. Should be a sequence like [MIN_BATCH, OPT_BATCH, MAX_BATCH]. After converted, the batchsize of
            model input should between `MIN_BATCH` and `MAX_BATCH` and the `OPT_BATCH` is the best performance batchsize
            that the TensorRT tries to fit. The `OPT_BATCH` should be the most frequently used input batchsize in
            the application.
        device: the target GPU index to convert and verify the model.
        use_onnx: whether using the ONNX-TensorRT way to export the TensorRT engine-based TorchScript model.
        onnx_input_names: optional input names of the ONNX model. This arg is only useful when `use_onnx` is True. Should be
            a sequence like `['input_0', 'input_1', ..., 'input_N']` where N equals to the number of the model inputs. If not
            given, will use `['input_0']`, which supposes the model only has one input.
        onnx_output_names: optional output names of the ONNX model. This arg is only useful when `use_onnx` is True. Should be
            a sequence like `['output_0', 'output_1', ..., 'output_N']` where N equals to the number of the model outputs. If
            not given, will use `['output_0']`, which supposes the model only has one output.
        args_file: a JSON or YAML file to provide default values for all the parameters of this function, so that
            the command line inputs can be simplified.
        converter_kwargs: extra arguments that are needed by `convert_to_trt`, except ones that already exist in the
            input parameters.
        override: id-value pairs to override or add the corresponding config content.
            e.g. ``--_meta#network_data_format#inputs#image#num_channels 3``.

    """
    _args = update_kwargs(
        args=args_file,
        net_id=net_id,
        filepath=filepath,
        meta_file=meta_file,
        config_file=config_file,
        ckpt_file=ckpt_file,
        key_in_ckpt=key_in_ckpt,
        precision=precision,
        input_shape=input_shape,
        use_trace=use_trace,
        dynamic_batchsize=dynamic_batchsize,
        device=device,
        use_onnx=use_onnx,
        onnx_input_names=onnx_input_names,
        onnx_output_names=onnx_output_names,
        converter_kwargs=converter_kwargs,
        **override,
    )
    _log_input_summary(tag="trt_export", args=_args)
    (
        filepath_,
        ckpt_file_,
        config_file_,
        net_id_,
        meta_file_,
        key_in_ckpt_,
        precision_,
        input_shape_,
        use_trace_,
        dynamic_batchsize_,
        device_,
        use_onnx_,
        onnx_input_names_,
        onnx_output_names_,
        converter_kwargs_,
    ) = _pop_args(
        _args,
        "filepath",
        "ckpt_file",
        "config_file",
        net_id="",
        meta_file=None,
        key_in_ckpt="",
        precision="fp32",
        input_shape=[],
        use_trace=False,
        dynamic_batchsize=None,
        device=None,
        use_onnx=False,
        onnx_input_names=["input_0"],
        onnx_output_names=["output_0"],
        converter_kwargs={},
    )

    parser = ConfigParser()

    parser.read_config(f=config_file_)
    if meta_file_ is not None:
        parser.read_meta(f=meta_file_)

    # the rest key-values in the _args are to override config content
    for k, v in _args.items():
        parser[k] = v

    # The convert_to_trt must have an `input_shape_` as input, no matter what the `use_trace` is.
    # If the `input_shape` is not provided, will try to parse it from the parser`.
    if not input_shape_:
        input_shape_ = _get_fake_input_shape(parser=parser)

    trt_api_parameters = {
        "precision": precision_,
        "input_shape": input_shape_,
        "dynamic_batchsize": dynamic_batchsize_,
        "use_trace": use_trace_,
        "device": device_,
        "use_onnx": use_onnx_,
        "onnx_input_names": onnx_input_names_,
        "onnx_output_names": onnx_output_names_,
    }
    converter_kwargs_.update(trt_api_parameters)

    save_ts = partial(save_net_with_metadata, include_config_vals=False, append_timestamp=False)

    _export(
        convert_to_trt,
        save_ts,
        parser,
        net_id=net_id_,
        filepath=filepath_,
        ckpt_file=ckpt_file_,
        config_file=config_file_,
        key_in_ckpt=key_in_ckpt_,
        **converter_kwargs_,
    )


def init_bundle(
    bundle_dir: PathLike,
    ckpt_file: PathLike | None = None,
    network: torch.nn.Module | None = None,
    dataset_license: bool = False,
    metadata_str: dict | str | None = None,
    inference_str: dict | str | None = None,
) -> None:
    """
    Initialise a new bundle directory with some default configuration files and optionally network weights.

    Typical usage example:

    .. code-block:: bash

        python -m monai.bundle init_bundle /path/to/bundle_dir network_ckpt.pt

    Args:
        bundle_dir: directory name to create, must not exist but parent direct must exist
        ckpt_file: optional checkpoint file to copy into bundle
        network: if given instead of ckpt_file this network's weights will be stored in bundle
        dataset_license: if `True`, a default license file called "data_license.txt" will be produced. This
            file is required if there are any license conditions stated for data your bundle uses.
        metadata_str: optional metadata string to write to bundle, if not given a default will be used.
        inference_str: optional inference string to write to bundle, if not given a default will be used.
    """
    if metadata_str is None:
        metadata_str = DEFAULT_METADATA
    if inference_str is None:
        inference_str = DEFAULT_INFERENCE

    bundle_dir = Path(bundle_dir).absolute()

    if bundle_dir.exists():
        raise ValueError(f"Specified bundle directory '{str(bundle_dir)}' already exists")

    if not bundle_dir.parent.is_dir():
        raise ValueError(f"Parent directory of specified bundle directory '{str(bundle_dir)}' does not exist")

    configs_dir = bundle_dir / "configs"
    models_dir = bundle_dir / "models"
    docs_dir = bundle_dir / "docs"

    bundle_dir.mkdir()
    configs_dir.mkdir()
    models_dir.mkdir()
    docs_dir.mkdir()

    if isinstance(metadata_str, dict):
        metadata_str = json.dumps(metadata_str, indent=4)

    if isinstance(inference_str, dict):
        inference_str = json.dumps(inference_str, indent=4)

    with open(str(configs_dir / "metadata.json"), "w") as o:
        o.write(metadata_str)

    with open(str(configs_dir / "inference.json"), "w") as o:
        o.write(inference_str)

    with open(str(docs_dir / "README.md"), "w") as o:
        readme = """
        # Your Model Name

        Describe your model here and how to run it, for example using `inference.json`:

        ```
        python -m monai.bundle run \
            --meta_file /path/to/bundle/configs/metadata.json \
            --config_file /path/to/bundle/configs/inference.json \
            --dataset_dir ./input \
            --bundle_root /path/to/bundle
        ```
        """

        o.write(dedent(readme))

    with open(str(bundle_dir / "LICENSE"), "w") as o:
        o.write("Select a license and place its terms here\n")

    if dataset_license is True:
        with open(str(docs_dir / "data_license.txt"), "w") as o:
            o.write("Select a license for dataset and place its terms here\n")

    if ckpt_file is not None:
        copyfile(str(ckpt_file), str(models_dir / "model.pt"))
    elif network is not None:
        save_state(network, str(models_dir / "model.pt"))


def _add_model_card_metadata(new_modelcard_path):
    # Extract license from LICENSE file
    license_name = "unknown"
    license_path = os.path.join(os.path.dirname(new_modelcard_path), "LICENSE")
    if os.path.exists(license_path):
        with open(license_path) as file:
            content = file.read()
        if "Apache License" in content and "Version 2.0" in content:
            license_name = "apache-2.0"
        elif "MIT License" in content:
            license_name = "mit"
    # Add relevant tags
    tags = "- monai\n- medical\nlibrary_name: monai\n"
    # Create tag section
    tag_content = f"---\ntags:\n{tags}license: {license_name}\n---"

    # Update model card
    with open(new_modelcard_path) as file:
        content = file.read()
    new_content = tag_content + "\n" + content
    with open(new_modelcard_path, "w") as file:
        file.write(new_content)


def push_to_hf_hub(
    repo: str,
    name: str,
    bundle_dir: str,
    token: str | None = None,
    private: bool | None = True,
    version: str | None = None,
    tag_as_latest_version: bool | None = False,
    **upload_folder_kwargs: Any,
) -> Any:
    """
    Push a MONAI bundle to the Hugging Face Hub.

    Typical usage examples:

    .. code-block:: bash

        python -m monai.bundle push_to_hf_hub --repo <HF repository id> --name <bundle name> \
            --bundle_dir <bundle directory> --version <version> ...

    Args:
        repo: namespace (user or organization) and a repo name separated by a /, e.g. `hf_username/bundle_name`
        bundle_name: name of the bundle directory to push.
        bundle_dir: path to the bundle directory.
        token: Hugging Face authentication token. Default is `None` (will default to the stored token).
        private: Private visibility of the repository on Hugging Face. Default is `True`.
        version_name: Name of the version tag to create. Default is `None` (no version tag is created).
        tag_as_latest_version: Whether to tag the commit as `latest_version`.
            This version will downloaded by default when using `bundle.download()`. Default is `False`.
        upload_folder_kwargs: Keyword arguments to pass to `HfApi.upload_folder`.

    Returns:
        repo_url: URL of the Hugging Face repo
    """
    # Connect to API and create repo
    hf_api = huggingface_hub.HfApi(token=token)
    hf_api.create_repo(repo_id=repo, private=private, exist_ok=True)

    # Create model card in bundle directory
    new_modelcard_path = os.path.join(bundle_dir, name, "README.md")
    modelcard_path = os.path.join(bundle_dir, name, "docs", "README.md")
    if os.path.exists(modelcard_path):
        # Copy README from old path if it exists
        copyfile(modelcard_path, new_modelcard_path)
        _add_model_card_metadata(new_modelcard_path)

    # Upload bundle folder to repo
    repo_url = hf_api.upload_folder(repo_id=repo, folder_path=os.path.join(bundle_dir, name), **upload_folder_kwargs)

    # Create version tag if specified
    if version is not None:
        hf_api.create_tag(repo_id=repo, tag=version, exist_ok=True)

    # Optionally tag as `latest_version`
    if tag_as_latest_version:
        hf_api.create_tag(repo_id=repo, tag="latest_version", exist_ok=True)

    return repo_url


def create_workflow(
    workflow_name: str | BundleWorkflow | None = None,
    config_file: str | Sequence[str] | None = None,
    args_file: str | None = None,
    **kwargs: Any,
) -> Any:
    """
    Specify `bundle workflow` to create monai bundle workflows.
    The workflow should be subclass of `BundleWorkflow` and be available to import.
    It can be MONAI existing bundle workflows or user customized workflows.

    Typical usage examples:

    .. code-block:: python

        # Specify config_file path to create workflow:
        workflow = create_workflow(config_file="/workspace/spleen_ct_segmentation/configs/train.json", workflow_type="train")

        # Set the workflow to other customized BundleWorkflow subclass to create workflow:
        workflow = create_workflow(workflow_name=CustomizedWorkflow)

    Args:
        workflow_name: specified bundle workflow name, should be a string or class, default to "ConfigWorkflow".
        config_file: filepath of the config file, if it is a list of file paths, the content of them will be merged.
        args_file: a JSON or YAML file to provide default values for this API.
            so that the command line inputs can be simplified.
        kwargs: arguments to instantiate the workflow class.

    """
    _args = update_kwargs(args=args_file, workflow_name=workflow_name, config_file=config_file, **kwargs)
    (workflow_name, config_file) = _pop_args(
        _args, workflow_name=ConfigWorkflow, config_file=None
    )  # the default workflow name is "ConfigWorkflow"
    if isinstance(workflow_name, str):
        workflow_class, has_built_in = optional_import("monai.bundle", name=str(workflow_name))  # search built-in
        if not has_built_in:
            workflow_class = locate(str(workflow_name))  # search dotted path
        if workflow_class is None:
            raise ValueError(f"cannot locate specified workflow class: {workflow_name}.")
    elif issubclass(workflow_name, BundleWorkflow):  # type: ignore
        workflow_class = workflow_name
    else:
        raise ValueError(
            "Argument `workflow_name` must be a bundle workflow class name"
            f"or subclass of BundleWorkflow, got: {workflow_name}."
        )

    if config_file is not None:
        workflow_ = workflow_class(config_file=config_file, **_args)
    else:
        workflow_ = workflow_class(**_args)

    workflow_.initialize()
    _log_input_summary(tag="run", args=_args)
    return workflow_


def download_large_files(bundle_path: str | None = None, large_file_name: str | None = None) -> None:
    """
    This utility allows you to download large files from a bundle. It supports file suffixes like ".yml", ".yaml", and ".json".
    If you don't specify a `large_file_name`, it will automatically search for large files among the supported suffixes.

    Typical usage examples:
    .. code-block:: bash

        # Execute this module as a CLI entry to download large files from a bundle path:
        python -m monai.bundle download_large_files --bundle_path <bundle_path>

        # Execute this module as a CLI entry to download large files from the bundle path with a specified `large_file_name`:
        python -m monai.bundle download_large_files --bundle_path <bundle_path> --large_file_name large_files.yaml

    Args:
        bundle_path: (Optional) The path to the bundle where the files are located. Default is `os.getcwd()`.
        large_file_name: (Optional) The name of the large file to be downloaded.

    """
    bundle_path = os.getcwd() if bundle_path is None else bundle_path
    if large_file_name is None:
        large_file_path = list(Path(bundle_path).glob("large_files*"))
        large_file_path = list(filter(lambda x: x.suffix in [".yml", ".yaml", ".json"], large_file_path))
        if len(large_file_path) == 0:
            raise FileNotFoundError(f"Cannot find the large_files.yml/yaml/json under {bundle_path}.")

    parser = ConfigParser()
    parser.read_config(large_file_path)
    large_files_list = parser.get()["large_files"]
    for lf_data in large_files_list:
        lf_data["fuzzy"] = True
        if "hash_val" in lf_data and lf_data.get("hash_val", "") == "":
            lf_data.pop("hash_val")
        if "hash_type" in lf_data and lf_data.get("hash_type", "") == "":
            lf_data.pop("hash_type")
        lf_data["filepath"] = os.path.join(bundle_path, lf_data["path"])
        lf_data.pop("path")
        download_url(**lf_data)
