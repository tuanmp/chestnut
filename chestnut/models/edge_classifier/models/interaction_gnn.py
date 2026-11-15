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

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import aggr
from torch_scatter import scatter_add, scatter_mean

from chestnut.utils.ml_utils import make_mlp

from ..edge_classifier_stage import EdgeClassifierStage


class InteractionGNN2(EdgeClassifierStage):
    """
    Interaction Network (L2IT version).
    Operates on directed graphs.
    Aggregate and reduce (sum) separately incomming and outcoming edges latents.
    """

    def __init__(self, 
        hidden: int,
        layernorm: bool,
        concat: bool,
        in_out_diff_agg: bool,
        n_node_net_layers: int,
        n_edge_net_layers: int,
        n_edge_decoder_layers: int,
        n_graph_iters: int,
        hidden_activation: str,
        output_activation: str,
        edge_output_transform_final_activation: str | None,
        checkpointing: bool,
        aggr: str,
        edge_chunk_size: int,
    ):
        super().__init__()

        # TODO: Add equivalent check and default values for other model parameters ?
        # TODO: Use get() method

        # Define the dataset to be used, if not using the default
        self.save_hyperparameters()

        # self.setup_layer_sizes()

        assert n_graph_iters > 0, "Number of graph iterations must be greater than 0"

        if concat:
            if in_out_diff_agg:
                in_node_net = hidden * 4
            else:
                in_node_net = hidden * 3
            in_edge_net = hidden * 6
        else:
            if in_out_diff_agg:
                in_node_net = hidden * 3
            else:
                in_node_net = hidden * 2
            in_edge_net = hidden * 3
        # node encoder
        self.node_encoder = make_mlp(
            input_size=None,
            sizes=[hidden] * n_node_net_layers,
            output_activation=output_activation,
            hidden_activation=hidden_activation,
            layer_norm=layernorm,
        )
        # edge encoder

        self.edge_encoder = make_mlp(
            input_size=None,
            sizes=[hidden] * n_edge_net_layers,
            output_activation=output_activation,
            hidden_activation=hidden_activation,
            layer_norm=layernorm,
        )

        # edge network
        self.edge_network = nn.ModuleList(
            [
                make_mlp(
                    input_size=in_edge_net,
                    sizes=[hidden] * n_edge_net_layers,
                    output_activation=output_activation,
                    hidden_activation=hidden_activation,
                    layer_norm=layernorm,
                )
                for i in range(n_graph_iters)
            ]
        )
        # node network
        self.node_network = nn.ModuleList(
            [
                make_mlp(
                    input_size=in_node_net,
                    sizes=[hidden] * n_node_net_layers,
                    output_activation=output_activation,
                    hidden_activation=hidden_activation,
                    layer_norm=layernorm,
                )
                for i in range(n_graph_iters)
            ]
        )

        # edge decoder
        self.edge_decoder = make_mlp(
            input_size=hidden,
            sizes=[hidden] * n_edge_decoder_layers,
            output_activation=output_activation,
            hidden_activation=hidden_activation,
            layer_norm=layernorm,
        )
        # edge output transform layer
        self.edge_output_transform = make_mlp(
            input_size=hidden,
            sizes=[hidden, 1],
            output_activation=edge_output_transform_final_activation,
            hidden_activation=hidden_activation,
            layer_norm=layernorm,
        )

        # hyperparams
        # self.hparams = hparams

        self.aggr_function = (
            scatter_add if self.hparams.get("aggr", "sum") == "sum" else scatter_mean
        )

    def forward(self, x, edge_index, edge_attr=None):


        # Get src and dst
        src, dst = edge_index

        # Encode nodes and edges features into latent spaces
        if self.hparams["checkpointing"]:
            x = checkpoint(self.node_encoder, x, use_reentrant=False)
            if edge_attr is not None:
                e = checkpoint(self.edge_encoder, edge_attr, use_reentrant=False)
            else:
                e = checkpoint(self.edge_encoder, torch.cat([x[src], x[dst]], dim=-1), use_reentrant=False)
        else:
            x = self.node_encoder(x)
            if edge_attr is not None:
                e = self.edge_encoder(edge_attr)
            else:
                e = self.edge_encoder(torch.cat([x[src], x[dst]], dim=-1))
        # Apply dropout
        # x = self.dropout(x)
        # e = self.dropout(e)

        # memorize initial encodings for concatenate in the gnn loop if request
        if self.hparams["concat"]:
            input_x = x
            input_e = e
        # Loop over gnn layers
        for i in range(self.hparams["n_graph_iters"]):
            if self.hparams["checkpointing"]:
                if self.hparams["concat"]:
                    x = checkpoint(self.concat, x, input_x)
                    e = checkpoint(self.concat, e, input_e)

                x, e = checkpoint(self.message_step, x, e, src, dst, i, use_reentrant=False)
            else:
                if self.hparams["concat"]:
                    x = torch.cat([x, input_x], dim=-1)
                    e = torch.cat([e, input_e], dim=-1)

                x, e = self.message_step(x, e, src, dst, i)
        return self.edge_output_transform(self.edge_decoder(e)).squeeze(-1)

    def message_step(self, x, e, src, dst, i=None):

        # chunk the edges edgewise reduce transient memory use
        e_chunks = []
        for s in range(0, e.shape[0], self.hparams["edge_chunk_size"]):
            sl = slice(s, min(s + self.hparams["edge_chunk_size"], e.shape[0]))
            chunk_edge_inputs = torch.cat([e[sl], x[src[sl]], x[dst[sl]]], dim=-1)
            e_chunks.append(self.edge_network[i](chunk_edge_inputs))
        
        e_updated = torch.cat(e_chunks, dim=0)

        # edge_inputs = torch.cat([e, x[src], x[dst]], dim=-1)  # order dst src x ?

        # e_updated = self.edge_network[i](edge_inputs)
        # Update nodes
        edge_messages_from_src = self.aggr_function(
            e_updated, dst, dim=0, dim_size=x.shape[0]
        )
        edge_messages_from_dst = self.aggr_function(
            e_updated, src, dim=0, dim_size=x.shape[0]
        )
        if self.hparams["in_out_diff_agg"]:
            node_inputs = torch.cat(
                [edge_messages_from_src, edge_messages_from_dst, x], dim=-1
            )  # to check : the order dst src  x ?
        else:
            # add message from src and dst ?? # edge_messages = edge_messages_from_src + edge_messages_from_dst
            edge_messages = edge_messages_from_src + edge_messages_from_dst
            node_inputs = torch.cat([edge_messages, x], dim=-1)

        x_updated = self.node_network[i](node_inputs)

        return x_updated, e_updated

    def concat(self, x, y):
        return torch.cat([x, y], dim=-1)

