from typing import Any

from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data

from chestnut.utils.loading_utils import (
    gather_inputs,
    get_pyg_data_keys,
    handle_edge_features,
    handle_node_features,
)


class GraphUtilMixin:

    def add_node_features(self, event):
    
        handle_node_features(event, self.node_features)  # noqa: F821
        return event

    def add_edge_features(self, event):
        if self.edge_features is not None:
            handle_edge_features(event, self.edge_features)
        return event

    def scale_features(self, event):
        """
        Handle feature scaling for the event
        """

        for i, feature in enumerate(self.node_features):
            assert feature in get_pyg_data_keys(
                event
            ), f"Feature {feature} not found in event"
            event[feature] = event[feature] / self.node_scales[i]

        return event

    @staticmethod
    def gather_node_input(event, features):
        """
        Gather the input features for the event
        """

        event.node_features = gather_inputs(event, features)
        for feat in features:
            event.pop(feat, None)

    @staticmethod
    def gather_edge_input(event, features):
        """
        Gather the input features for the event
        """

        event.edge_features = gather_inputs(event, features)
        for feat in features:
            event.pop(feat, None)

def collate_graph(batch):
    """
    Collate a batch of graphs into a single graph. This is used for the DataLoader.
    """
    keys = batch[0].keys()
    return {
        key: Batch.from_data_list([data[key] for data in batch])
        if isinstance(batch[0][key], Data)
        else [data[key] for data in batch]
        for key in keys
    }

class DataModuleMixin:
    
    def _dataloader_kwargs(self, shuffle: bool = False, batch_size: int = 1, num_workers: int = 1):
        return {
            "shuffle": shuffle,
            "batch_size": batch_size,
            "num_workers": self.num_workers or num_workers,
            "collate_fn": collate_graph,
        }

    def train_dataloader(self, shuffle=True, batch_size: int = 1, num_workers: int = 1):
        if getattr(self, "trainset", None) is None:
            raise ValueError("Train dataset is not set up. Call setup() first.")
        return DataLoader(
            self.trainset, **self._dataloader_kwargs(shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)
        )

    def val_dataloader(self, batch_size: int = 1, num_workers: int = 1):
        if getattr(self, "valset", None) is None:
            raise ValueError("Validation dataset is not set up. Call setup() first.")
        return DataLoader(
            self.valset, **self._dataloader_kwargs(shuffle=False, batch_size=batch_size, num_workers=num_workers)
        )

    def test_dataloader(self, batch_size: int = 1, num_workers: int = 1):
        if getattr(self, "testset", None) is None:
            raise ValueError("Test dataset is not set up. Call setup() first.")
        return DataLoader(
            self.testset, **self._dataloader_kwargs(shuffle=False, batch_size=batch_size, num_workers=num_workers)
        )

    def predict_dataloader(self) -> Any:
        return [
            self.train_dataloader(shuffle=False),
            self.val_dataloader(),
            self.test_dataloader(),
        ]
