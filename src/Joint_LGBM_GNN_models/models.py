"""Temporal relation-aware GNN and LightGBM logit-fusion models."""

import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def scalar_logit(p):
    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    return math.log(p / (1.0 - p))

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class TemporalRelationAwareMeanLayer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        relation_names,
        attention_hidden_dim=32,
    ):
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.relation_names = tuple(relation_names)

        self.self_linear = nn.Linear(
            self.hidden_dim,
            self.hidden_dim,
        )

        self.relation_linears = nn.ModuleDict({
            name: nn.Linear(
                self.hidden_dim,
                self.hidden_dim,
                bias=False,
            )
            for name in self.relation_names
        })

        self.attention_mlps = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(
                    2 * self.hidden_dim,
                    int(attention_hidden_dim),
                ),
                nn.ReLU(),
                nn.Linear(
                    int(attention_hidden_dim),
                    1,
                ),
            )
            for name in self.relation_names
        })

    @staticmethod
    def relation_weighted_mean(
        h,
        edge_index,
        edge_weight,
    ):
        num_nodes = h.size(0)
        hidden_dim = h.size(1)

        weighted_sum = torch.zeros(
            (num_nodes, hidden_dim),
            dtype=h.dtype,
            device=h.device,
        )
        weight_sum = torch.zeros(
            num_nodes,
            dtype=h.dtype,
            device=h.device,
        )

        if edge_index.numel() == 0:
            active = torch.zeros(
                num_nodes,
                dtype=torch.bool,
                device=h.device,
            )
            return weighted_sum, active

        src = edge_index[0]
        dst = edge_index[1]

        edge_weight = edge_weight.to(
            dtype=h.dtype,
            device=h.device,
        )

        weighted_message = (
            h[src]
            * edge_weight.unsqueeze(1)
        )

        weighted_sum.index_add_(
            0,
            dst,
            weighted_message,
        )

        weight_sum.index_add_(
            0,
            dst,
            edge_weight,
        )

        active = weight_sum > 0

        weighted_mean = (
            weighted_sum
            / weight_sum.clamp_min(1e-12).unsqueeze(1)
        )

        return weighted_mean, active

    def forward(
        self,
        h,
        relation_edge_index,
        relation_edge_weight,
    ):
        rel_messages = []
        rel_scores = []
        rel_active = []

        for name in self.relation_names:
            edge_idx = relation_edge_index[name]
            edge_weight = relation_edge_weight[name]

            rel_mean, active = self.relation_weighted_mean(
                h,
                edge_idx,
                edge_weight,
            )

            rel_message = self.relation_linears[name](
                rel_mean
            )

            score_input = torch.cat(
                [h, rel_mean],
                dim=1,
            )

            score = self.attention_mlps[name](
                score_input
            ).squeeze(1)

            rel_messages.append(rel_message)
            rel_scores.append(score)
            rel_active.append(active)

        messages = torch.stack(
            rel_messages,
            dim=1,
        )
        scores = torch.stack(
            rel_scores,
            dim=1,
        )
        active_mask = torch.stack(
            rel_active,
            dim=1,
        )

        masked_scores = scores.masked_fill(
            ~active_mask,
            -1e9,
        )

        attention = torch.softmax(
            masked_scores,
            dim=1,
        )

        attention = (
            attention
            * active_mask.to(attention.dtype)
        )

        denom = attention.sum(
            dim=1,
            keepdim=True,
        )

        any_active = denom.squeeze(1) > 0

        attention = torch.where(
            any_active.unsqueeze(1),
            attention / denom.clamp_min(1e-12),
            torch.zeros_like(attention),
        )

        combined_message = (
            attention.unsqueeze(2) * messages
        ).sum(dim=1)

        out = (
            self.self_linear(h)
            + combined_message
        )

        return out, attention

class TemporalRelationAwareGraphSAGEBranch(nn.Module):
    def __init__(
        self,
        input_dim,
        relation_names,
        hidden_dim=256,
        dropout_p=0.2,
        attention_hidden_dim=32,
    ):
        super().__init__()

        self.relation_names = tuple(relation_names)

        self.input_proj = nn.Linear(
            input_dim,
            hidden_dim,
        )

        self.conv1 = TemporalRelationAwareMeanLayer(
            hidden_dim=hidden_dim,
            relation_names=self.relation_names,
            attention_hidden_dim=attention_hidden_dim,
        )

        self.conv2 = TemporalRelationAwareMeanLayer(
            hidden_dim=hidden_dim,
            relation_names=self.relation_names,
            attention_hidden_dim=attention_hidden_dim,
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.dropout_p = float(dropout_p)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x,
        relation_edge_index,
        relation_edge_weight,
    ):
        h = F.relu(
            self.input_proj(x)
        )

        h, attention1 = self.conv1(
            h,
            relation_edge_index,
            relation_edge_weight,
        )
        h = self.norm1(h)
        h = F.relu(h)
        h = F.dropout(
            h,
            p=self.dropout_p,
            training=self.training,
        )

        h, attention2 = self.conv2(
            h,
            relation_edge_index,
            relation_edge_weight,
        )
        h = self.norm2(h)
        h = F.relu(h)
        h = F.dropout(
            h,
            p=self.dropout_p,
            training=self.training,
        )

        logit = self.head(h).squeeze(1)

        return {
            'h': h,
            'logit': logit,
            'attention_layer1': attention1,
            'attention_layer2': attention2,
        }

class LearnedGlobalLogitFusion(nn.Module):
    def __init__(self, alpha_init=0.5):
        super().__init__()

        self.alpha_logit = nn.Parameter(
            torch.tensor(
                scalar_logit(alpha_init),
                dtype=torch.float32,
            )
        )

    def alpha(self):
        return torch.sigmoid(self.alpha_logit)

    def forward(
        self,
        lgb_raw,
        gnn_logit,
        alpha_override=None,
    ):
        if alpha_override is None:
            alpha = self.alpha()
        else:
            alpha = torch.as_tensor(
                float(alpha_override),
                dtype=gnn_logit.dtype,
                device=gnn_logit.device,
            )

        fused_logit = (
            alpha * lgb_raw
            + (1.0 - alpha) * gnn_logit
        )

        return fused_logit, alpha

class JointLightGBMTemporalRelationAwareGraphSAGE(nn.Module):
    def __init__(
        self,
        sage_input_dim,
        num_topology_features,
        relation_names,
        hidden_dim=256,
        dropout_p=0.2,
        attention_hidden_dim=32,
        alpha_init=0.5,
    ):
        super().__init__()

        self.gnn = TemporalRelationAwareGraphSAGEBranch(
            input_dim=int(
                sage_input_dim
                + num_topology_features
            ),
            relation_names=relation_names,
            hidden_dim=hidden_dim,
            dropout_p=dropout_p,
            attention_hidden_dim=attention_hidden_dim,
        )

        self.fusion = LearnedGlobalLogitFusion(
            alpha_init=alpha_init
        )

    def forward(
        self,
        x_sage,
        x_topology,
        lgb_raw,
        relation_edge_index,
        relation_edge_weight,
        alpha_override=None,
    ):
        x_gnn = torch.cat(
            [x_sage, x_topology],
            dim=1,
        )

        gnn_out = self.gnn(
            x_gnn,
            relation_edge_index,
            relation_edge_weight,
        )

        logit_gnn = gnn_out['logit']

        fused_logit, alpha = self.fusion(
            lgb_raw=lgb_raw,
            gnn_logit=logit_gnn,
            alpha_override=alpha_override,
        )

        return {
            'logit': fused_logit,
            'logit_gnn': logit_gnn,
            'alpha': alpha,
            'h_gnn': gnn_out['h'],
            'relation_attention': {
                'layer1': gnn_out['attention_layer1'],
                'layer2': gnn_out['attention_layer2'],
            },
        }

def create_model_and_optimizer(
    *,
    fixed_alpha,
    sage_input_dim,
    num_topology_features,
    relation_names,
    hidden_dim,
    dropout_p,
    attention_hidden_dim,
    alpha_init,
    lr,
    alpha_lr,
    weight_decay,
    random_seed,
    device,
):
    """Create a fresh joint model and optimizer for one experiment."""
    set_all_seeds(random_seed)

    model = JointLightGBMTemporalRelationAwareGraphSAGE(
        sage_input_dim=sage_input_dim,
        num_topology_features=num_topology_features,
        relation_names=relation_names,
        hidden_dim=hidden_dim,
        dropout_p=dropout_p,
        attention_hidden_dim=attention_hidden_dim,
        alpha_init=alpha_init,
    ).to(device)

    if fixed_alpha is None:
        model.fusion.alpha_logit.requires_grad_(True)
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": model.gnn.parameters(),
                    "lr": lr,
                    "weight_decay": weight_decay,
                },
                {
                    "params": [model.fusion.alpha_logit],
                    "lr": alpha_lr,
                    "weight_decay": 0.0,
                },
            ]
        )
    else:
        model.fusion.alpha_logit.requires_grad_(False)
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": model.gnn.parameters(),
                    "lr": lr,
                    "weight_decay": weight_decay,
                },
            ]
        )

    return model, optimizer

