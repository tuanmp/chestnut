
import os

import pandas as pd
from lightning import LightningDataModule
from lightning.pytorch.utils.rank_zero import rank_zero_warn
from torch_geometric.data import Dataset

from chestnut.utils.loading_utils import (
    add_variable_name_prefix_in_pyg,
    handle_hard_node_cuts,
    infer_num_nodes,
    load_pyg,
    pyg_exists,
)

from .utils import GraphUtilMixin


class EventDataset(Dataset, GraphUtilMixin):
    """
    The custom default GNN dataset to load graphs off the disk
    """

    def __init__(
        self,
        input_dir: str,
        node_features: list[str],
        node_scales: list[float],
        data_name: str,
        num_events: int=0,
        use_csv: bool=False,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        event_prefix: str | None = None,
        add_variable_name_prefix_in_pyg: bool = False,
        variable_with_prefix: bool = False,
        hard_cuts: dict | None = None,
        return_raw_events: bool = False,
        **kwargs,
    ):
        super().__init__(input_dir, transform, pre_transform, pre_filter)
        self.node_features = node_features
        self.node_scales = node_scales

        self.input_dir = input_dir
        self.data_name = data_name
        self.num_events = num_events
        self.use_csv = use_csv
        self.event_prefix = event_prefix
        self.add_variable_name_prefix_in_pyg = add_variable_name_prefix_in_pyg
        self.variable_with_prefix = variable_with_prefix
        self.hard_cuts = hard_cuts
        self.evt_ids = self.find_evt_ids()

        self.return_raw_events = return_raw_events

    def len(self):
        return len(self.evt_ids)

    def get(self, idx):
        """
        Handles the iteration through the dataset. Depending on how dataset is configured for PyG and CSV file loading,
        will return either PyG events (automatically batched by PyG), CSV files (not batched/concatenated), or both (not batched/concatenated).
        Assumes files are saved correctly from EventReader as: event000...-truth.csv, -particles.csv and -graph.pyg
        """
        event_file_root_name = f"event{self.evt_ids[idx]}"
        if self.event_prefix is not None:
            event_file_root_name = (
                f"{self.event_prefix}_{event_file_root_name}"
            )

        event_path = os.path.join(self.input_dir, self.data_name, event_file_root_name)

        graph_path = (
            f"{event_path}-graph.pyg"
            if pyg_exists(f"{event_path}-graph.pyg")
            else f"{event_path}.pyg"
        )
        graph = load_pyg(graph_path, weights_only=False)
        processed_graph = self.preprocess_graph(graph)

        if not self.use_csv:
            return {
                'event': processed_graph,
                "raw_event": graph if self.return_raw_events else None,
            }

        particles = pd.read_csv(f"{event_path}-particles.csv")
        hits = pd.read_csv(f"{event_path}-truth.csv")
        hits = self.preprocess_hits(hits, graph)

        return graph, particles, hits

    def preprocess_graph(self, graph):
        """Preprocess the PyG graph before returning it."""
        if (not self.variable_with_prefix) or self.add_variable_name_prefix_in_pyg:
            graph = add_variable_name_prefix_in_pyg(graph)
        infer_num_nodes(graph)
        graph = self.apply_hard_cuts(graph)
        self.scale_features(graph)
        return graph

    def preprocess_hits(self, hits, graph):
        """Preprocess the hits dataframe before returning it."""
        hits = self.apply_hard_cuts(hits, passing_hit_ids=graph.hit_id.numpy())
        return hits

    def find_evt_ids(self):
        """
        Returns a list of all event ids, which are the numbers in filenames that end in .csv and .pyg
        """

        input_data_dir = os.path.join(self.input_dir, self.data_name)
        all_files = os.listdir(input_data_dir) if os.path.isdir(input_data_dir) else []
        all_files = [
            f
            for f in all_files
            if f.endswith(".csv") or f.endswith(".pyg") or f.endswith(".pyg.gz")
        ]
        all_event_ids = sorted(
            list({re.findall("[0-9]+", file)[-1] for file in all_files})
        )

        if len(all_event_ids) == 0:
            rank_zero_warn(f"No events found in {self.input_dir}/{self.data_name}")

        if self.num_events is not None:
            assert self.num_events <= len(
                all_event_ids
            ), f"Requested {self.num_events} events, but only found {len(all_event_ids)} in {self.input_dir}/{self.data_name}"
            all_event_ids = all_event_ids[: self.num_events]

        # Check that events are present for the requested filetypes
        if self.event_prefix is not None:
            prefix = self.event_prefix + "_"
        else:
            prefix = ""

        csv_event_ids = []
        if self.use_csv:
            csv_event_ids = [
                evt_id
                for evt_id in all_event_ids
                if (f"{prefix}event{evt_id}-truth.csv" in all_files)
                and (f"{prefix}event{evt_id}-particles.csv" in all_files)
            ]

        pyg_event_ids = [
            evt_id
            for evt_id in all_event_ids
            if f"{prefix}event{evt_id}-graph.pyg" in all_files
            or f"{prefix}event{evt_id}-graph.pyg.gz" in all_files
            or f"{prefix}event{evt_id}.pyg" in all_files
            or f"{prefix}event{evt_id}.pyg.gz" in all_files
        ]

        if self.use_csv:
            all_event_ids = list(set(csv_event_ids) & set(pyg_event_ids))
        else:
            all_event_ids = pyg_event_ids

        return all_event_ids

    def apply_hard_cuts(self, event, passing_hit_ids=None):
        """
        Apply hard cuts to the event. This is implemented by
        1. Finding which true edges are from tracks that pass the hard cut.
        2. Pruning the input graph to only include nodes that are connected to these edges.
        """

        if self.hard_cuts is not None and len(self.hard_cuts) > 0:
            assert isinstance(
                self.hard_cuts, dict
            ), "Hard cuts must be a dictionary"
            event = handle_hard_node_cuts(
                event, self.hard_cuts, passing_hit_ids
            )

        return event


class EventDataModule(LightningDataModule):
    def __init__(self,
        input_dir,
        node_features,
        data_split,
        node_scales: list[float] | None = None,
        preprocess=True,
        return_raw_event=False,
        return_processed_event=False,
        variable_with_prefix: bool = False,
        add_variable_name_prefix_in_pyg: bool = False,
        hard_cuts: dict | None = None,
        num_workers: int | None = None,
    ) -> None:
        super().__init__()

        self.input_dir = input_dir
        self.node_features = node_features
        self.data_split = data_split
        self.node_scales = node_scales or [1.0] * len(node_features)
        self.preprocess = preprocess
        self.return_raw_event = return_raw_event
        self.return_processed_event = return_processed_event
        self.variable_with_prefix = variable_with_prefix
        self.add_variable_name_prefix_in_pyg = add_variable_name_prefix_in_pyg
        self.hard_cuts = hard_cuts
        self.num_workers = num_workers


    def setup(self, stage: str | None = None):

        if stage == "predict":
            if not self.return_raw_event:
                rank_zero_warn(
                    "return_raw_event is set to False. Setting it to True for prediction."
                )
                self.return_raw_event = True

        for data_name, data_num in zip(
            ["trainset", "valset", "testset"], self.data_split
        ):
            dataset = EventDataset(
                input_dir=self.input_dir,
                node_features=self.node_features,
                data_name=data_name,
                num_events=data_num,
                node_scales=self.node_scales,
                preprocess=self.preprocess,
                return_raw_event=self.return_raw_event,
                return_processed_event=self.return_processed_event,
                variable_with_prefix=self.variable_with_prefix,
                add_variable_name_prefix_in_pyg=self.add_variable_name_prefix_in_pyg,
                hard_cuts=self.hard_cuts,
            )
            setattr(self, data_name, dataset)

    


