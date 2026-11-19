# Copyright (C) 2023 CERN for the benefit of the ATLAS collaboration

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This class represents the entire logic of the graph construction stage. In particular, it
1. Loads events from the Athena-dumped csv files
2. Processes them into PyG Data objects with the specificied structure (see docs)
3. Runs the training of the metric learning or module map
4. Can run inference to build graphs
5. Can run evaluation to plot/print the performance of the graph construction

TODO: Update structure with the latest Gravnet base class
"""

import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import LightningModule
from torch_geometric.data import Dataset
from tqdm import tqdm

from chestnut.utils.loading_utils import (
    add_variable_name_prefix_in_pyg,
    get_condition_lambda,
    get_pyg_data_keys,
    handle_hard_node_cuts,
    infer_num_nodes,
    load_pyg,
    pyg_exists,
    run_data_tests,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# from chestnut.utils import (
#     get_pyg_data_keys,
#     # plot_eff_pur_region,
#     # eval_utils,
#     # get_condition_lambda,
#     handle_hard_node_cuts,
#     run_data_tests,
# )


class GraphConstructionStage:

    @classmethod
    def infer(cls, config):
        """
        The gateway for the inference stage. This class method is called from the infer_stage.py script.
        """
        if isinstance(cls, LightningModule):
            graph_constructor = cls.load_from_checkpoint(
                os.path.join(config["input_dir"], "checkpoints", "last.ckpt")
            )
            graph_constructor.hparams.update(
                config
            )  # Update the configs used in training with those to be used in inference
        else:
            graph_constructor = cls(config)

        graph_constructor.setup(stage="predict")

        for data_name in ["trainset", "valset", "testset"]:
            if hasattr(graph_constructor, data_name):
                graph_constructor.build_graphs(
                    dataset=getattr(graph_constructor, data_name), data_name=data_name
                )

    def build_graphs(self, dataset, data_name):
        """
        Build the graphs using the trained model. This is the only function that needs to be overwritten by the child class.
        """
        pass

    @classmethod
    def evaluate(cls, config, *args):
        """
        The gateway for the evaluation stage. This class method is called from the eval_stage.py script.
        """

        # Load data from testset directory
        graph_constructor = cls(config).to(device)
        graph_constructor.use_csv = False
        graph_constructor.setup(stage="test")

        all_plots = config["plots"]

        # TODO: Handle the list of plots properly
        for plot_function, plot_config in all_plots.items():
            if hasattr(eval_utils, plot_function):
                getattr(eval_utils, plot_function)(
                    graph_constructor, plot_config, config
                )
            else:
                print(f"Plot {plot_function} not implemented")

    def graph_region_efficiency_purity(self, plot_config, config):
        edge_truth, edge_regions = [], []
        node_r, node_z, node_regions = [], [], []

        for event in tqdm(self.testset):
            edge_truth.append(event.edge_y)
            edge_regions.append(
                event.x_region[event.edge_index[0]]
            )  # Assign region depending on first node in edge

            node_r.append(event.x_r)
            node_z.append(event.x_z)
            node_regions.append(event.x_region)

        edge_truth = torch.cat(edge_truth).cpu().numpy()
        edge_positive = np.ones(len(edge_truth))
        edge_regions = torch.cat(edge_regions).cpu().numpy()

        node_r = torch.cat(node_r).cpu().numpy()
        node_z = torch.cat(node_z).cpu().numpy()
        node_regions = torch.cat(node_regions).cpu().numpy()

        fig, ax = plot_eff_pur_region(
            edge_truth,
            edge_positive,
            edge_regions,
            node_r,
            node_z,
            node_regions,
            plot_config,
        )
        fig.savefig(os.path.join(config["stage_dir"], "region_eff_pur.png"))

    def apply_target_conditions(self, event, target_tracks):
        """
        Apply the target conditions to the event. This is used for the evaluation stage.
        Target_tracks is a list of dictionaries, each of which contains the conditions to be applied to the event.
        """
        passing_tracks = torch.ones(
            event.track_to_edge_map.shape[0], dtype=torch.bool
        ).to(self.device)

        for condition_key, condition_val in target_tracks.items():
            condition_lambda = get_condition_lambda(condition_key, condition_val)
            passing_tracks = passing_tracks * condition_lambda(event).to(self.device)

        event.track_target_mask = passing_tracks

