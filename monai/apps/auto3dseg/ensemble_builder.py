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


import json
import logging
import os
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from monai.apps.auto3dseg.bundle_gen import BundleAlgo
from monai.auto3dseg import concat_val_to_np
from monai.bundle import ConfigParser
from monai.utils.enums import BundleEnsembleKeys

logger = logging.getLogger(__name__)


class BundleEnsemble(ABC):
    """
    The base class of Ensemble methods
    """

    def __init__(self):
        self.algos = []
        self.infer_files = []
        self.algo_ensemble = []

    def set_algos(self, infer_algos):
        """
        Register model in the ensemble
        """
        self.algos = deepcopy(infer_algos)

    def get_algo(self, identifier):
        """
        Get a model by identifier.

        Args:
            identifier: the name of the bundleAlgo
        """
        for algo in self.algos:
            if identifier == algo[BundleEnsembleKeys.ID]:
                return algo

    def get_algo_ensemble(self):
        """
        Get the algo ensemble after ranking or a empty list if ranking was not started.

        Returns:
            A list of Algo
        """
        return self.algo_ensemble

    def set_infer_files(self, dataroot: str, data_src_cfg_file: str):
        """
        Set the files to perform model inference.

        Args:
            dataroot: the path of the files
            data_src_cfg_file: the data source file path
        """
        with open(data_src_cfg_file) as f:
            datalist = json.load(f)

        for d in datalist["testing"]:
            self.infer_files.append({"image": os.path.join(dataroot, d["image"])})

    @abstractmethod
    def rank_algos(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self):
        """
        predict results after the models are ranked/weighted
        """
        raise NotImplementedError


class BundleEnsembleBestN(BundleEnsemble):
    """
    Ensemble method that select N model out of all using the models' best_metric scores

    Args:
        n_best: number of models to pick for ensemble (N).
    """

    def __init__(self, n_best: int = 5):

        super().__init__()
        self.n_best = n_best

    def sort_score(self):
        """
        Sort the best_metrics
        """
        scores = concat_val_to_np(self.algos, [BundleEnsembleKeys.SCORE])
        return np.argsort(scores).tolist()

    def rank_algos(self):
        """
        Rank the algos by finding the top N (n_best) validation scores.
        """
        ranks = self.sort_score()
        if len(ranks) < self.n_best:
            raise ValueError("Number of available algos is less than user-defined N")

        # get the indices that the rank is larger than N
        indices = [i for (i, r) in enumerate(ranks) if r >= self.n_best]

        # remove the found indices
        indices = sorted(indices, reverse=True)

        self.algo_ensemble = deepcopy(self.algos)
        for idx in indices:
            if idx < len(self.algo_ensemble):
                self.algo_ensemble.pop(idx)

    def predict(self, pred_param: Optional[Dict[str, Any]] = None):
        """
        Use the ensembled model to predict result

        Args:
            pred_param: prediction parameter dictionary. The key has two groups. The first group only has 'files_slices' key with
                a value type of `slice`. The files_slices will slice the infer_files and only make prediction on the
                infer_files[file_slices]. The second group of params will be passed to the `InferClass` to override the parameters
                of the class functions.

        Returns:
            A tensor.
        """
        if pred_param is None:
            param = {}
        else:
            param = deepcopy(pred_param)

        files = self.infer_files
        if "files_slices" in param:
            slices = param.pop("files_slices")
            files = self.infer_files[slices]

        outputs = []
        for i in range(len(files)):
            print(i)
            preds = []
            infer_filename = self.infer_files[i]
            for algo in self.algo_ensemble:
                infer_instance = algo[BundleEnsembleKeys.ALGO]
                param.update({"files": [infer_filename]})
                pred = infer_instance.predict(param)
                preds.append(pred[0])
            outputs.append(sum(preds) / len(preds))
        return outputs


class BundleEnsembleBuilder:
    """
    Build ensemble workflow from configs and arguments.

    Args:
        history: a collection of trained bundleAlgo algorithms.
        data_src_cfg_filename: filename of the data source.

    Examples:

        ..code-block:: python
            builder = BundleEnsembleBuilder(history, data_src_cfg)
            builder.set_ensemble_method(BundleBundleEnsembleBestN(3))
            ensemble = builder.get_ensemble()

            result = ensemble.predict()
    """

    def __init__(self, history: Sequence[Dict], data_src_cfg_filename: Optional[str] = None):
        self.infer_algos: List[Dict[BundleEnsembleKeys, Any]] = []
        self.ensemble: BundleEnsemble
        self.data_src_cfg = ConfigParser(globals=False)

        if data_src_cfg_filename is not None and os.path.exists(str(data_src_cfg_filename)):
            self.data_src_cfg.read_config(data_src_cfg_filename)

        for h in history:
            # load inference_config_paths
            # raise warning/error if not found
            if len(h.keys()) > 1:
                raise ValueError(f"{h} should only contain one set of genAlgo key-value")

            name = list(h.keys())[0]
            gen_algo = h[name]
            best_metric = gen_algo.get_score()
            algo_path = gen_algo.output_path
            infer_path = os.path.join(algo_path, "scripts", "infer.py")

            if not os.path.isdir(algo_path):
                raise ValueError(f"{gen_algo.output_path} is not a directory. Please check the path.")

            if not os.path.isfile(infer_path):
                raise ValueError(f"{infer_path} is not found. Please check the path.")

            self.add_inferer(name, gen_algo, best_metric)

    def add_inferer(self, identifier: str, gen_algo: BundleAlgo, best_metric: Optional[float] = None):
        """
        Add model inferer to the builder.

        Args:
            identifier: name of the bundleAlgo.
            gen_algo: a trained BundleAlgo model object.
            best_metric: the best metric in validation of the trained model.
        """

        if best_metric is None:
            raise ValueError("Feature to re-valiate is to be implemented")

        algo = {
            BundleEnsembleKeys.ID: identifier,
            BundleEnsembleKeys.ALGO: gen_algo,
            BundleEnsembleKeys.SCORE: best_metric,
        }
        self.infer_algos.append(algo)

    def set_ensemble_method(self, ensemble: BundleEnsemble):
        """
        Set the ensemble method.

        Args:
            ensemble: the BundleEnsemble to build.
        """

        ensemble.set_algos(self.infer_algos)
        ensemble.rank_algos()
        ensemble.set_infer_files(self.data_src_cfg["dataroot"], self.data_src_cfg["datalist"])

        self.ensemble = ensemble

    def get_ensemble(self):
        """Get the ensemble"""

        return self.ensemble
