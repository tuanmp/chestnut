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

import torch.nn.functional as F

from chestnut.utils.ml_utils import make_mlp

from ..graph_construction_stage import MetricLearningBase


class DoubleMetricLearning(MetricLearningBase):
    """Double metric learning with separate source and target embeddings."""

    rearrange_edges = False

    def build_network(self):
        self.network = make_mlp(
            None,
            [self.emb_hidden] * self.nb_layer,
            hidden_activation=self.activation,
            output_activation=None,
            layer_norm=True,
        )

        self.src_decoder = make_mlp(
            self.emb_hidden,
            [self.emb_dim],
            hidden_activation=self.activation,
            output_activation=None,
            layer_norm=True,
        )

        self.tgt_decoder = make_mlp(
            self.emb_hidden,
            [self.emb_dim],
            hidden_activation=self.activation,
            output_activation=None,
            layer_norm=True,
        )

    def forward(self, x):
        hidden = self.network(x)
        return (
            F.normalize(self.src_decoder(hidden)),
            F.normalize(self.tgt_decoder(hidden)),
        )
