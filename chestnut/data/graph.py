

from typing import Any

import torch
from lightning import LightningDataModule
from lightning_utilities.core.rank_zero import rank_zero_warn
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Dataset

from chestnut.utils.loading_utils import (
    VariableType,
    add_variable_name_prefix_in_pyg,
    gather_inputs,
    get_pyg_data_keys,
    get_variable_type,
    handle_hard_cuts,
    handle_weighting,
    infer_num_nodes,
    load_datafiles_in_dir,
    load_pyg,
)
from chestnut.utils.mapping_utils import remap_from_mask

from .utils import GraphUtilMixin, collate_graph


class GraphDataset(Dataset, GraphUtilMixin):
    """
    The custom default GNN dataset to load graphs off the disk
    """

    def __init__(
        self,
        input_dir,
        node_features,
        data_name=None,
        num_events=None,
        edge_features: list[str] | None = None,
        node_scales: list[float] | None = None,
        preprocess=True,
        return_raw_event=False,
        return_processed_event=False,
        variable_with_prefix: bool = False,
        add_variable_name_prefix_in_pyg: bool = False,
        hard_cuts: dict | None = None,
        weighting: list[dict] | None = None,
        input_cut: float | None = None,
        max_training_graph_size: int | None = None
    ):

        super().__init__()

        self.input_dir = input_dir
        self.data_name = data_name
        self.num_events = num_events
        self.preprocess = preprocess
        self.node_features = node_features
        self.edge_features = edge_features
        if node_scales is None:
            rank_zero_warn(
                "No node scales provided. Defaulting to 1.0 for all node features."
            )
        self.node_scales = node_scales if node_scales is not None else [1.0] * len(node_features)
        self.return_raw_event = return_raw_event
        self.return_processed_event = return_processed_event
        self.variable_with_prefix = variable_with_prefix
        self.hard_cuts = hard_cuts
        self.add_variable_name_prefix_in_pyg = add_variable_name_prefix_in_pyg
        self.weighting = weighting
        self.input_cut = input_cut

        self.input_paths = load_datafiles_in_dir(
            self.input_dir, self.data_name, self.num_events
        )
        self.input_paths.sort()  # We sort here for reproducibility
        self.max_training_graph_size = max_training_graph_size

    def len(self):
        return len(self.input_paths)

    def __getitem__(self, idx: int):
        event_path = self.input_paths[idx]
        event = load_pyg(
            event_path, map_location=torch.device("cpu"), weights_only=False
        )
        event = Data(**event.to_dict())  # convert DataBatch to Data instance because some transformations don't work on DataBatch
        if (not self.variable_with_prefix) or self.add_variable_name_prefix_in_pyg:
            event = add_variable_name_prefix_in_pyg(event)
        if not self.preprocess:
            return {
                "event": event if self.return_processed_event else None,
                "raw_event": event if self.return_raw_event else None,
            }
        processed_event = self.preprocess_event(event)

        # return (event, event_path) if self.stage == "predict" else event
        return {
            "event": processed_event if self.return_processed_event else None,
            "raw_event": event if self.return_raw_event else None,
        }

    def preprocess_event(self, event):
        """
        Process event before it is used in training and validation loops
        """
        infer_num_nodes(event)
        event = self.apply_hard_cuts(event)
        event = self.construct_weighting(event)
        event = self.handle_edge_list(event)

        event = self.add_node_features(
            event
        )  # derived angles need the unscaled values
        event = self.scale_features(event)
        if self.edge_features is not None:
            event = self.add_edge_features(
                event
            )  # scaling must be done before adding features
            event.edge_features = gather_inputs(event, self.edge_features)
            for edge_feat in self.edge_features:
                event.pop(edge_feat, None)  # remove the individual edge features after gathering

        event.node_features = gather_inputs(event, self.node_features)
        for node_feat in self.node_features:
            event.pop(node_feat, None)  # remove the individual node features after scaling
        return event

    def apply_hard_cuts(self, event):
        """
        Apply hard cuts to the event. This is implemented by
        1. Finding which true edges are from tracks that pass the hard cut.
        2. Pruning the input graph to only include nodes that are connected to these edges.
        """

        if self.hard_cuts is not None:
            assert isinstance(
                self.hard_cuts, dict
            ), "Hard cuts must be a dictionary"
            handle_hard_cuts(event, self.hard_cuts)

        return event

    def construct_weighting(self, event):
        """
        Construct the weighting for the event
        """

        assert event.edge_y.shape[0] == event.edge_index.shape[1], (
            f"Input graph has {event.edge_index.shape[1]} edges, but"
            f" {event.edge_y.shape[0]} truth labels"
        )

        if self.weighting is not None:
            assert isinstance(self.weighting, list) & isinstance(
                self.weighting[0], dict
            ), "Weighting must be a list of dictionaries"
            event.edge_weights = handle_weighting(event, self.weighting)
        else:
            event.edge_weights = torch.ones_like(event.edge_y, dtype=torch.float32)

        return event

    def handle_edge_list(self, event):
        if (
            self.input_cut is not None
        ):
            # Apply a score cut to the event
            if not "edge_scores" in get_pyg_data_keys(event):
                rank_zero_warn(
                    "No edge_scores found in event. Skipping score cut."
                )
                return event
            self.apply_score_cut(event, self.input_cut)

        # we don't do this anymore. leaving it here as dead code to remind ourselves. The method is deleted
        # if "undirected" in self.hparams.keys() and self.hparams["undirected"]:
        #     # Flip event.edge_index and concat together
        #     self.to_undirected(event)
        if self.max_training_graph_size is not None and event.edge_index.shape[1] > self.max_training_graph_size:
            edge_keep = torch.randperm(event.edge_index.shape[1])[: self.max_training_graph_size]
            event.edge_scores = torch.zeros_like(event.edge_index[0], dtype=torch.float32)
            event.edge_scores[edge_keep] = 1.0
            self.apply_score_cut(event, 0.5)
        return event
    

    def apply_score_cut(self, event, score_cut):
        """
        Apply a score cut to the event. This is used for the evaluation stage.
        """
        passing_edges_mask = event.edge_scores >= score_cut
        for key in get_pyg_data_keys(event):
            if (
                isinstance(event[key], torch.Tensor)
                and get_variable_type(key) == VariableType.EDGE_LIKE
            ):
                event[key] = event[key][..., passing_edges_mask]

        remap_from_mask(event, passing_edges_mask)
        return event

    def get_y_node(self, event):
        y_node = torch.zeros(event.z.size(0))
        y_node[event.track_edges.view(-1)] = 1
        event.y_node = y_node
        return event



class GraphDataModule(LightningDataModule):
    def __init__(self,
        input_dir,
        node_features,
        data_split,
        edge_features: list[str] | None = None,
        node_scales: list[float] | None = None,
        preprocess=True,
        return_raw_event=False,
        return_processed_event=False,
        variable_with_prefix: bool = False,
        add_variable_name_prefix_in_pyg: bool = False,
        hard_cuts: dict | None = None,
        weighting: list[dict] | None = None,
        input_cut: float | None = None,
        max_training_graph_size: int | None = None,
        num_workers: int = 1,
    ) -> None:
        super().__init__()

        self.input_dir = input_dir
        self.node_features = node_features
        self.data_split = data_split
        self.edge_features = edge_features
        self.node_scales = node_scales
        self.preprocess = preprocess
        self.return_raw_event = return_raw_event
        self.return_processed_event = return_processed_event
        self.variable_with_prefix = variable_with_prefix
        self.add_variable_name_prefix_in_pyg = add_variable_name_prefix_in_pyg
        self.hard_cuts = hard_cuts
        self.weighting = weighting
        self.input_cut = input_cut
        self.max_training_graph_size = max_training_graph_size


    def setup(self, stage: str | None = None):

        if stage == "predict":
            if not self.return_raw_event:
                rank_zero_warn(
                    "return_raw_event is set to False. Setting it to True for prediction."
                )
                self.return_raw_event = True

        if stage != "fit":
            if self.max_training_graph_size is not None:
                rank_zero_warn(
                    "max_training_graph_size is set. This will be ignored for validation and test datasets."
                )
                self.max_training_graph_size = None

        for data_name, data_num in zip(
            ["trainset", "valset", "testset"], self.data_split
        ):
            dataset = GraphDataset(
                input_dir=self.input_dir,
                node_features=self.node_features,
                data_name=data_name,
                num_events=data_num,
                edge_features=self.edge_features,
                node_scales=self.node_scales,
                preprocess=self.preprocess,
                return_raw_event=self.return_raw_event,
                return_processed_event=self.return_processed_event,
                variable_with_prefix=self.variable_with_prefix,
                add_variable_name_prefix_in_pyg=self.add_variable_name_prefix_in_pyg,
                hard_cuts=self.hard_cuts,
                weighting=self.weighting,
                input_cut=self.input_cut,
                max_training_graph_size=self.max_training_graph_size,
            )
            setattr(self, data_name, dataset)

    