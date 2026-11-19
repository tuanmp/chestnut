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


class MetricLearning(MetricLearningBase):
    """Single-embedding metric learning."""

    rearrange_edges = True

    def build_network(self):
        self.network = make_mlp(
            None,
            [self.emb_hidden] * self.nb_layer + [self.emb_dim],
            hidden_activation=self.activation,
            output_activation=None,
            layer_norm=True,
        )

    def forward(self, x):
        embedding = F.normalize(self.network(x))
        return embedding, embedding
