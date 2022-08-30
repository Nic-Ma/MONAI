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

import os
import shutil
from copy import deepcopy
from glob import glob
from typing import Sequence, Union

from monai.apps.auto3dseg import BundleAlgo


class Segresnet2DAlgo(BundleAlgo):
    def fill_template_config(self, data_stats_filename):
        if data_stats_filename is None or not os.path.exists(str(data_stats_filename)):
            return
        data_cfg = ConfigParser(globals=False)
        data_cfg.read_config(data_stats_filename)
        patch_size = [320, 320]
        max_shape = data_cfg["stats_summary#image_stats#shape#max"]
        patch_size = [
            max(32, shape_k // 32 * 32) if shape_k < p_k else p_k for p_k, shape_k in zip(patch_size, max_shape)
        ]
        self.cfg["patch_size#0"], self.cfg["patch_size#1"] = patch_size
        self.cfg["patch_size_valid#0"], self.cfg["patch_size_valid#1"] = patch_size
        data_src_cfg = ConfigParser(globals=False)
        if self.data_list_file is not None and os.path.exists(str(self.data_list_file)):
            data_src_cfg.read_config(self.data_list_file)
            self.cfg.update(
                {
                    "data_file_base_dir": data_src_cfg["dataroot"],
                    "data_list_file_path": data_src_cfg["datalist"],
                    "input_channels": data_cfg["stats_summary#image_stats#channels#max"],
                    "output_classes": data_cfg["stats_summary#label_stats#labels"],
                }
            )
        modality = data_src_cfg.get("modality", "ct").lower()
        spacing = data_cfg["stats_summary#image_stats#spacing#median"]
        spacing[-1] = -1.0

        intensity_upper_bound = float(data_cfg["stats_summary#image_foreground_stats#intensity#percentile_99_5"])
        intensity_lower_bound = float(data_cfg["stats_summary#image_foreground_stats#intensity#percentile_00_5"])
        ct_intensity_xform = {
            "_target_": "Compose",
            "transforms": [
                {
                    "_target_": "ScaleIntensityRanged",
                    "keys": "@image_key",
                    "a_min": intensity_lower_bound,
                    "a_max": intensity_upper_bound,
                    "b_min": 0.0,
                    "b_max": 1.0,
                    "clip": True,
                },
                {"_target_": "CropForegroundd", "keys": ["@image_key", "@label_key"], "source_key": "@image_key"},
            ],
        }
        mr_intensity_transform = {
            "_target_": "NormalizeIntensityd",
            "keys": "@image_key",
            "nonzero": True,
            "channel_wise": True,
        }
        for key in ["transforms_infer", "transforms_train", "transforms_validate"]:
            for idx, xform in enumerate(self.cfg[f"{key}#transforms"]):
                if isinstance(xform, dict) and xform.get("_target_", "").startswith("Spacing"):
                    xform["pixdim"] = deepcopy(spacing)
                elif isinstance(xform, str) and xform.startswith("PLACEHOLDER_INTENSITY_NORMALIZATION"):
                    if modality.startswith("ct"):
                        self.cfg[f"{key}#transforms#{idx}"] = deepcopy(ct_intensity_xform)
                    else:
                        self.cfg[f"{key}#transforms#{idx}"] = deepcopy(mr_intensity_transform)
        # self.algorithm_dir = os.path.join(self.output_path, "segresnet2d")
        # self.inference_script = "scripts/infer.py"


class DintsAlgo(BundleAlgo):
    def fill_template_config(self, data_stats_filename):
        print("implement dints template filling method")


default_algos = {
    "segresnet2d": dict(
        _target_="SegresnetAlgo",
        template_configs="../algorithms/templates/segresnet2d/configs",
        scripts_path="../algorithms/templates/segresnet2d/scripts",
    ),
    "dints": dict(
        _target_="DintsAlgo",
        template_configs="../algorithms/templates/dints/configs",
        scripts_path="../algorithms/templates/dints/scripts",
    ),
}


class BundleGen(AlgoGen):
    """
    This class generates a set of bundles according to the cross-validation folds, each of them can run independently.

    .. code-block:: bash

        python -m monai.apps.auto3dseg BundleGen generate --data_stats_filename="../algorithms/data_stats.yaml"

    """

    def __init__(self, algos=None, data_stats_filename=None, data_lists_filename=None):
        self.algos = []
        if algos is None:
            for algo_name, algo_params in default_algos.items():
                self.algos.append(ConfigParser(algo_params).get_parsed_content())
                self.algos[-1].name = algo_name
        else:
            self.algos = ensure_tuple(algos)

        self.data_stats_filename = data_stats_filename
        self.data_lists_filename = data_lists_filename
        self.history = []

    def set_data_stats(self, data_stats_filename):
        self.data_stats_filename = data_stats_filename

    def get_data_stats(self, fold_idx=0):
        return self.data_stats_filename

    def set_data_lists(self, data_lists_filename):
        self.data_lists_filename = data_lists_filename

    def get_data_lists(self, fold_idx=0):
        return self.data_lists_filename

    def get_history(self, *args, **kwargs):
        return self.history

    def generate(self, fold_idx: Union[int, Sequence[int]] = 0):
        for algo in self.algos:
            for f_id in ensure_tuple(fold_idx):
                data_stats = self.get_data_stats(fold_idx)
                data_lists = self.get_data_lists(fold_idx)
                algo.set_data_stats(data_stats)
                algo.set_data_source(data_lists)
                name = f"{algo.name}_{f_id}"
                algo.export_to_disk(".", algo_name=name)
                self.history.append({name: deepcopy(algo)})  # track the previous, may create a persistent history
