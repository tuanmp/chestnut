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

"""Shared graph-construction stage and metric-learning base classes."""

import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule

from chestnut.utils.graph_utils import (
    build_edges,
    graph_intersection,
    rearrange_by_distance,
)
from chestnut.utils.loading_utils import build_signal_edges, handle_weighting


class GraphConstructionStage:
    """Marker/protocol shared by graph-construction stages."""


class MetricLearningBase(GraphConstructionStage, LightningModule):
    """Shared training and evaluation logic for metric-learning graph builders.

    Subclasses implement ``build_network`` and ``forward``. ``forward`` returns
    a pair ``(src, tgt)``; a single-embedding model returns the same tensor
    twice.
    """

    #: Flip predicted edges to point outward before matching truth.
    rearrange_edges: bool = False

    def __init__(
        self,
        emb_hidden: int,
        emb_dim: int,
        activation: str,
        nb_layer: int,
        r_train: float,
        r_val: float,
        knn: int,
        knn_val: int,
        weighting: list[dict],
        margin: float,
        points_per_batch: int,
        randomisation: float,
    ):
        super().__init__()

        self.emb_hidden = emb_hidden
        self.emb_dim = emb_dim
        self.activation = activation
        self.nb_layer = nb_layer
        self.r_train = r_train
        self.r_val = r_val
        self.knn = knn
        self.knn_val = knn_val
        self.weighting = weighting
        self.margin = margin
        self.points_per_batch = points_per_batch
        self.randomisation = randomisation

        self.save_hyperparameters()
        self.build_network()

    def build_network(self):
        raise NotImplementedError(
            "This method should be implemented in the child class."
        )

    def get_input_data(self, event):
        return torch.nan_to_num(event.node_features.float(), nan=0.0)

    def get_query_points(self, spatial):
        query_indices = torch.arange(len(spatial), device=self.device)
        query_indices = query_indices[torch.randperm(len(query_indices))][
            : self.points_per_batch
        ]
        return query_indices, spatial[query_indices]

    def append_hnm_pairs(self, e_spatial, query, query_indices, spatial):
        knn_edges = build_edges(
            query=query,
            database=spatial,
            indices=query_indices,
            r_max=self.r_train,
            k_max=self.knn,
            backend="FRNN",
            return_indices=False,
        )
        return torch.cat([e_spatial, knn_edges], dim=-1)

    def append_random_pairs(self, e_spatial, query_indices, spatial):
        n_random = int(self.randomisation * len(query_indices))
        indices_src = torch.randint(
            0, len(query_indices), (n_random,), device=self.device
        )
        indices_dest = torch.randint(0, len(spatial), (n_random,), device=self.device)
        random_pairs = torch.stack([query_indices[indices_src], indices_dest])
        return torch.cat([e_spatial, random_pairs], dim=-1)

    def append_signal_edges(self, event, edges):
        signal_true_edges = build_signal_edges(
            event, self.weighting, event.track_edges
        )
        return torch.cat([edges, signal_true_edges], dim=-1)

    def apply_embedding(
        self,
        input_data,
        src_inplace=None,
        tgt_inplace=None,
        edges=None,
    ):
        if src_inplace is None or tgt_inplace is None or edges is None:
            return self(input_data)

        included_hits = edges.unique().long()
        src, tgt = self(input_data[included_hits])
        src_inplace[included_hits] = src
        if tgt_inplace is not src_inplace:
            tgt_inplace[included_hits] = tgt
        return src_inplace, tgt_inplace

    def get_training_edges(self, event):
        input_data = self.get_input_data(event)

        with torch.no_grad():
            src, tgt = self.apply_embedding(input_data)

        query_indices, query = self.get_query_points(src)

        training_edges = torch.empty([2, 0], dtype=torch.int64, device=self.device)
        training_edges = self.append_hnm_pairs(
            training_edges, query, query_indices, tgt
        )
        training_edges = self.append_random_pairs(
            training_edges, query_indices, tgt
        )
        training_edges = self.append_signal_edges(event, training_edges)

        return torch.unique(training_edges, dim=-1), src, tgt

    def get_truth(self, event, pred_edges):
        if self.rearrange_edges:
            pred_edges = rearrange_by_distance(event, pred_edges)

        pred_edges, edge_y, truth_to_pred = graph_intersection(
            pred_edges,
            event.track_edges,
            return_y_pred=True,
            return_truth_to_pred=True,
            unique_pred=False,
            undirected=False,
        )
        return pred_edges, edge_y, truth_to_pred, event.track_edges

    def get_weights(self, event):
        return handle_weighting(event, self.weighting)

    def get_distances(self, src, tgt, pred_edges):
        reference = src[pred_edges[0]]
        neighbors = tgt[pred_edges[1]]

        try:  # This can be resource intensive, so we chunk it if it fails
            return torch.sum((reference - neighbors) ** 2, dim=-1)
        except RuntimeError:
            return torch.cat(
                [
                    torch.sum((ref - nei) ** 2, dim=-1)
                    for ref, nei in zip(reference.chunk(10), neighbors.chunk(10))
                ]
            )

    def loss_function(self, src, tgt, edge_y, weights, pred_edges):
        d = self.get_distances(src, tgt, pred_edges)
        return self.weighted_hinge_loss(edge_y, d, weights)

    def weighted_hinge_loss(self, truth, d, weights):
        """Weighted hinge loss.

        True edges with positive weight are pulled together; false edges with
        non-zero weight (and true edges with negative weight) are pushed apart.
        """
        negative_mask = ((truth == 0) & (weights != 0)) | (weights < 0)
        negative_loss = F.hinge_embedding_loss(
            d[negative_mask],
            -torch.ones_like(d[negative_mask]),
            margin=self.margin**2,
            reduction="none",
        )
        negative_loss = (
            torch.mean(negative_loss * weights[negative_mask].abs())
            if negative_mask.any()
            else d.new_zeros(())
        )

        positive_mask = (truth == 1) & (weights > 0)
        positive_loss = F.hinge_embedding_loss(
            d[positive_mask],
            torch.ones_like(d[positive_mask]),
            margin=self.margin**2,
            reduction="none",
        )
        positive_loss = (
            torch.mean(positive_loss * weights[positive_mask].abs())
            if positive_mask.any()
            else d.new_zeros(())
        )

        return negative_loss + positive_loss

    def training_step(self, batch, batch_idx):
        event = batch["event"]

        edges, src, tgt = self.get_training_edges(event)
        src, tgt = self.apply_embedding(
            self.get_input_data(event), src, tgt, edges
        )

        (
            event.edge_index,
            event.edge_y,
            event.track_to_edge_map,
            _,
        ) = self.get_truth(event, edges)

        weights = self.get_weights(event)
        loss = self.loss_function(
            src, tgt, event.edge_y, weights, event.edge_index
        )

        self.log("train_loss", loss, batch_size=1)
        return loss

    def shared_evaluation(self, event, knn_radius, knn_num):
        src, tgt = self.apply_embedding(self.get_input_data(event))

        event.edge_index = build_edges(
            query=src,
            database=tgt,
            indices=None,
            r_max=knn_radius,
            k_max=knn_num,
            backend="FRNN",
        )

        (
            event.edge_index,
            event.edge_y,
            event.track_to_edge_map,
            true_edges,
        ) = self.get_truth(event, event.edge_index)

        weights = self.get_weights(event)
        d = self.get_distances(src, tgt, event.edge_index)
        loss = self.weighted_hinge_loss(event.edge_y, d, weights)

        trainer = self._trainer
        if trainer is not None and trainer.state.stage in ("train", "validate"):
            self.log_metrics(
                event,
                loss,
                event.edge_index,
                true_edges,
                event.edge_y,
                weights,
            )

        return {
            "loss": loss,
            "distances": d,
            "src_embedding": src,
            "tgt_embedding": tgt,
            "truth_graph": true_edges,
        }

    def log_metrics(self, event, loss, pred_edges, true_edges, truth, weights):
        signal_true_edges = build_signal_edges(
            event, self.weighting, true_edges
        )
        true_pred_edges = pred_edges[:, truth == 1]
        signal_true_pred_edges = pred_edges[:, (truth == 1) & (weights > 0)]

        n_true = true_edges.shape[1]
        n_signal_true = signal_true_edges.shape[1]
        n_pred = pred_edges.shape[1]
        n_true_pred = true_pred_edges.shape[1]
        n_signal_true_pred = signal_true_pred_edges.shape[1]

        total_eff = n_true_pred / n_true if n_true else 0.0
        signal_eff = (
            n_signal_true_pred / n_signal_true if n_signal_true else 0.0
        )
        total_pur = n_true_pred / n_pred if n_pred else 0.0
        signal_pur = n_signal_true_pred / n_pred if n_pred else 0.0
        f1 = (
            2 * signal_eff * signal_pur / (signal_eff + signal_pur)
            if signal_eff + signal_pur > 0
            else 0.0
        )

        self.log_dict(
            {
                "val_loss": loss,
                "total_eff": total_eff,
                "total_pur": total_pur,
                "signal_eff": signal_eff,
                "signal_pur": signal_pur,
                "f1": f1,
            },
            batch_size=1,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )

    def validation_step(self, batch, batch_idx):
        event = batch["event"]
        outputs = self.shared_evaluation(event, self.r_val, self.knn_val)
        return outputs["loss"]

    def test_step(self, batch, batch_idx):
        event = batch["event"]
        return self.shared_evaluation(event, self.r_val, self.knn_val)
