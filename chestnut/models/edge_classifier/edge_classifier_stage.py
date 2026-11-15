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

import importlib
import os
import warnings
from itertools import product
from typing import Any, Mapping

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from class_resolver import ClassResolver
from lightning.pytorch import LightningModule
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torchmetrics import AUROC, Metric
from torchmetrics.classification import BinaryPrecision, BinaryRecall

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: What is this for??
torch.multiprocessing.set_sharing_strategy("file_system")


class RatioMetric(Metric):
    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = eps
        self.add_state("numerator", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("denominator", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, numerator, denominator):
        self.numerator += numerator.float()
        self.denominator += denominator.float()

    def compute(self):
        return self.numerator / (self.denominator + self.eps)

class GraphDataMixin:

    def get_node_features(self, event: Data):
        if self.node_features is None:
            return None
        return torch.stack([event[feature] for feature in self.node_features], dim=-1).float()

    def get_edge_features(self, event: Data):
        if self.edge_features is None:
            return None
        return torch.stack([event[feature] for feature in self.edge_features], dim=-1).float()


class EdgeClassifierStage(LightningModule):
    def __init__(self):
        super().__init__()
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """

        # Assign hyperparameters
        self.total_auroc = AUROC(task="binary", thresholds=200)
        self.target_auroc = AUROC(task="binary", thresholds=200)
        self.target_eff = BinaryRecall()
        self.target_pur = BinaryPrecision()
        self.total_pur = BinaryPrecision()
        self.pur = RatioMetric()
        self.val_pt_cuts = []
        # ModuleList registers the metrics as submodules so Lightning moves them to GPU
        self.val_pt_cut_metrics = torch.nn.ModuleList()
        if "validation_eff_pt_cuts" in self.hparams and self.hparams["validation_eff_pt_cuts"]:
            for pt_cut in self.hparams["validation_eff_pt_cuts"]:
                self.val_pt_cuts.append(pt_cut)
                self.val_pt_cut_metrics.append(BinaryRecall())

    def setup(self, stage="fit"):
        """
        The setup logic of the stage.
        1. Setup the data for training, validation and testing.
        2. Run tests to ensure data is of the right format and loaded correctly.
        3. Construct the truth and weighting labels for the model training
        """

        try:
            print("Defining figures of merit")
            self.logger.experiment.define_metric("val_loss", summary="min")
            self.logger.experiment.define_metric("auc", summary="max")
        except Exception:
            warnings.warn(
                "Failed to define figures of merit, due to logger unavailable"
            )

    def _forward_from_batch(self, batch):
        """Call forward either with a batch object (legacy) or with unpacked tensors.

        Set ``tensor_forward: true`` in the config to unpack node/edge features
        from the batch and pass them as separate tensors to ``forward(x,
        edge_index, edge_attr)``.  This keeps models that require typed tensor
        signatures (TorchScript / torch.export) compatible with the stage while
        preserving backward compatibility for models whose ``forward`` accepts a
        full batch object.

        When ``time_inference: true`` is set in the config, the forward pass is
        timed during evaluation (not training) and logged as
        ``infer_ms`` and ``infer_ms_per_edge``.
        """
        do_time = self.hparams.get("time_inference", False) and not self.training

        if do_time:
            is_cuda = self.device.type == "cuda"
            if is_cuda:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event   = torch.cuda.Event(enable_timing=True)
                start_event.record()
            else:
                import time as _time
                _t0 = _time.perf_counter()

        x = torch.stack(
            [batch[feature] for feature in self.hparams["node_features"]], dim=-1
        ).float()
        edge_index = batch.edge_index
        edge_features = self.hparams.get("edge_features")
        if edge_features:
            edge_attr = torch.stack(
                [batch[feature] for feature in edge_features], dim=-1
            ).float()
            output = self(x, edge_index, edge_attr)
        else:
            output = self(x, edge_index)

        if do_time:
            if is_cuda:
                end_event.record()
                torch.cuda.synchronize()
                elapsed_ms = start_event.elapsed_time(end_event)
            else:
                elapsed_ms = (_time.perf_counter() - _t0) * 1000
            n_edges = batch.edge_index.shape[1]
            self.log("infer_ms",          elapsed_ms,
                     on_step=True, on_epoch=True, batch_size=1, sync_dist=False)
            self.log("infer_ms_per_edge", elapsed_ms / max(n_edges, 1),
                     on_step=True, on_epoch=True, batch_size=1, sync_dist=False)

        return output

    # def configure_optimizers(self):
    #     optimizer, scheduler = get_optimizers(self.parameters(), self.hparams)
    #     return optimizer, scheduler

    def training_step(self, batch: dict[str, Data], batch_idx):
        event = batch["event"]
        node_features = event.node_features.to(self.node_encoder[0].weight.dtype)
        edge_features = event.edge_features.to(self.edge_encoder[0].weight.dtype)
        edge_index = event.edge_index
        output = self(node_features, edge_index, edge_features)
        loss, pos_loss, neg_loss = self.loss_function(output, event)

        return {"loss": loss, "pos_loss": pos_loss, "neg_loss": neg_loss}

    def on_train_batch_end(self, outputs: Mapping[str, torch.Tensor], batch: Any, batch_idx: int) -> None:
        self.log(
            "train_loss",
            outputs["loss"],
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )
        self.log(
            "train_pos_loss",
            outputs["pos_loss"],
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )
        self.log(
            "train_neg_loss",
            outputs["neg_loss"],
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )

    def loss_function(self, output, batch, balance="proportional"):
        """
        Applies the loss function to the output of the model and the truth labels.
        To balance the positive and negative contribution, simply take the means of each separately.
        Any further fine tuning to the balance of true target, true background and fake can be handled
        with the `weighting` config option.
        """

        assert hasattr(batch, "edge_y"), (
            "The batch does not have a truth label. Please ensure the batch has a `y`"
            " attribute."
        )
        assert hasattr(batch, "edge_weights"), (
            "The batch does not have a weighting label. Please ensure the batch"
            " weighting is handled in preprocessing."
        )

        if balance not in ["equal", "proportional"]:
            warnings.warn(
                f"{balance} is not a proper choice for the loss balance. Use either 'equal' or 'proportional'. Automatically switching to 'proportional' instead."
            )
            balance = "proportional"

        negative_mask = ((batch.edge_y == 0) & (batch.edge_weights != 0)) | (
            batch.edge_weights < 0
        )

        negative_loss = F.binary_cross_entropy_with_logits(
            output[negative_mask],
            torch.zeros_like(output[negative_mask]),
            weight=batch.edge_weights[negative_mask].abs(),
            reduction="sum",
        )

        positive_mask = (batch.edge_y == 1) & (batch.edge_weights > 0)
        positive_loss = F.binary_cross_entropy_with_logits(
            output[positive_mask],
            torch.ones_like(output[positive_mask]),
            weight=batch.edge_weights[positive_mask].abs(),
            reduction="sum",
        )

        if balance == "proportional":
            sow = batch.edge_weights.abs().sum()
            return (
                (positive_loss + negative_loss) / sow,
                positive_loss.detach() / sow,
                negative_loss.detach() / sow,
            )
        else:
            n_pos, n_neg = positive_mask.sum(), negative_mask.sum()
            sow = (
                batch.edge_weights[positive_mask].abs().sum() / n_pos
                + batch.edge_weights[negative_mask].abs().sum() / n_neg
            )
            return (
                (positive_loss / n_pos + negative_loss / n_neg) / sow,
                positive_loss.detach() / n_pos / sow,
                negative_loss.detach() / n_neg / sow,
            )

    def shared_evaluation(self, batch, batch_idx):

        event = batch["event"]
        node_features = event.node_features.to(self.node_encoder[0].weight.dtype)
        edge_features = event.edge_features.to(self.edge_encoder[0].weight.dtype)
        edge_index = event.edge_index
        output = self(node_features, edge_index, edge_features)
        loss, pos_loss, neg_loss = self.loss_function(output, event)

        scores = torch.sigmoid(output)
        event.edge_scores = scores.detach()

        all_truth = event.edge_y.bool()
        target_truth = (event.edge_weights > 0) & all_truth

        return {
            "loss": loss,
            "all_truth": all_truth,
            "target_truth": target_truth,
            "output": output,
            "event": event,
            "pos_loss": pos_loss,
            "neg_loss": neg_loss,
        }

    def validation_step(self, batch, batch_idx):
        output_dict = self.shared_evaluation(batch, batch_idx)
        self.log_metrics(
            output_dict["event"],
            output_dict["output"],
            output_dict["all_truth"],
            output_dict["target_truth"],
            output_dict["loss"],
        )
        self.log(
            "val_loss",
            output_dict["loss"],
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )
        self.log(
            "val_pos_loss",
            output_dict["pos_loss"],
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )
        self.log(
            "val_neg_loss",
            output_dict["neg_loss"],
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )

    def test_step(self, batch, batch_idx):
        return self.shared_evaluation(batch, batch_idx)

    def log_metrics(self, event, output, all_truth, target_truth, loss):
        scores = torch.sigmoid(output).float()  # float() guards 16-bit training
        preds = scores > 0.5

        edge_positive = preds.sum().float()
        target_true_positive = (target_truth.bool() & preds).sum().float()
        true_and_fake_positive = (
            edge_positive - (preds & (~target_truth) & all_truth).sum().float()
        )

        # Torchmetrics accumulates the pooled epoch counts internally.
        self.target_eff.update(preds, target_truth.bool())
        self.target_pur.update(preds, target_truth.bool())
        self.total_pur.update(preds, all_truth.bool())
        self.pur.update(target_true_positive, true_and_fake_positive)

        # AUROC: torchmetrics accumulates internally (stays on GPU)
        self.total_auroc.update(scores, all_truth.long())
        mask = target_truth.bool() | (~all_truth.bool())
        self.target_auroc.update(scores[mask], all_truth[mask].long())

        if self.val_pt_cut_metrics:
            # track_target_mask may be absent if the graph wasn't built with target_tracks
            # conditions; default to "all tracks are targets" like eval_utils.py does.
            track_target_mask = getattr(
                event,
                "track_target_mask",
                torch.ones_like(event.track_to_edge_map, dtype=torch.bool),
            )
            has_edge = track_target_mask & (event.track_to_edge_map >= 0)
            track_pt = event.track_particle_pt

            edge_pt = torch.full_like(target_truth, -1, dtype=track_pt.dtype)
            edge_pt[event.track_to_edge_map[has_edge]] = track_pt[has_edge]

            for pt_cut, metric in zip(self.val_pt_cuts, self.val_pt_cut_metrics):
                mask = edge_pt >= pt_cut
                if mask.any():
                    metric.update(preds[mask], target_truth[mask])

        return preds

    def on_validation_epoch_end(self):
        self.log_dict(
            {
                "target_eff": self.target_eff.compute(),
                "target_pur": self.target_pur.compute(),
                "total_pur": self.total_pur.compute(),
                "pur": self.pur.compute(),
                "auc": self.target_auroc.compute(),
                "total_auc": self.total_auroc.compute(),
            },
            sync_dist=True,
        )

        for m in (
            self.target_eff,
            self.target_pur,
            self.total_pur,
            self.pur,
            self.total_auroc,
            self.target_auroc,
        ):
            m.reset()

        if self.val_pt_cut_metrics:
            pt_log_dict = {}
            for pt_cut, metric in zip(self.val_pt_cuts, self.val_pt_cut_metrics):
                pt_log_dict[f"target_eff_pt{pt_cut:.0f}"] = metric.compute()
                metric.reset()
            self.log_dict(pt_log_dict, sync_dist=True)


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        This function handles the prediction of each graph. It is called in the `infer.py` script.
        It can be overwritted in your custom stage, but it should implement three simple steps:
        1. Run an edge-scoring model on the input graph
        2. Add the scored edges to the graph, as `scores` attribute
        3. Append the stage config to the `config` attribute of the graph
        """
        event = batch["event"]
        node_features = event.node_features
        edge_features = event.edge_features
        edge_index = event.edge_index
        output = self(node_features, edge_index, edge_features)
        scores = torch.sigmoid(output)
        return scores
        # dataset = self.predict_dataloader()[dataloader_idx].dataset
        # event_id = (
        #     batch.event_id[0] if isinstance(batch.event_id, list) else batch.event_id
        # )
        # if pyg_exists(
        #     os.path.join(
        #         self.hparams["stage_dir"],
        #         dataset.data_name,
        #         f"event{event_id}.pyg",
        #     ),
        #     subdir=self.hparams.get("subdir_size", None),
        # ) and self.hparams.get("skip_existing"):
        #     return
        # if batch.edge_index.shape[1] == 0:
        #     batch.edge_scores = torch.tensor([], device=self.device)
        # else:
        #     eval_dict = self.shared_evaluation(batch, batch_idx)
        #     batch = eval_dict["batch"]
        # self.save_edge_scores(batch, dataset)
