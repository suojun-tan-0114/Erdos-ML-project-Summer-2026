"""Relation-aware incoming-neighbor sampling utilities."""

import math
import random

import numpy as np
import torch

class SimpleNeighborBatch:
    def __init__(
        self,
        x_sage,
        x_topology,
        lgb_raw,
        y,
        edge_index,
        relation_edge_index,
        relation_edge_weight,
        n_id,
        batch_size,
        relation_sample_counts=None,
    ):
        self.x_sage = x_sage
        self.x_topology = x_topology
        self.lgb_raw = lgb_raw
        self.y = y
        self.edge_index = edge_index
        self.relation_edge_index = relation_edge_index
        self.relation_edge_weight = relation_edge_weight
        self.n_id = n_id
        self.batch_size = int(batch_size)
        self.relation_sample_counts = relation_sample_counts or {}

    def to(self, device):
        self.x_sage = self.x_sage.to(device)
        self.x_topology = self.x_topology.to(device)
        self.lgb_raw = self.lgb_raw.to(device)
        self.y = self.y.to(device)
        self.edge_index = self.edge_index.to(device)

        self.relation_edge_index = {
            name: edge_idx.to(device)
            for name, edge_idx in self.relation_edge_index.items()
        }
        self.relation_edge_weight = {
            name: edge_weight.to(device)
            for name, edge_weight in self.relation_edge_weight.items()
        }

        self.n_id = self.n_id.to(device)
        return self

class RelationAwareIncomingNeighborLoader:
    def __init__(
        self,
        data,
        relation_edges,
        relation_tau_days,
        relation_fanouts,
        input_nodes,
        batch_size,
        shuffle,
        seed=42,
        labels=None,
        positive_seed_rate=None,
        epoch_size=None,
    ):
        self.data = data
        self.relation_fanouts = [int(x) for x in relation_fanouts]
        self.input_nodes = np.asarray(input_nodes, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self._epoch = 0

        self.labels = None if labels is None else np.asarray(labels)
        self.positive_seed_rate = (
            None if positive_seed_rate is None else float(positive_seed_rate)
        )
        self.epoch_size = (
            len(self.input_nodes) if epoch_size is None else int(epoch_size)
        )

        self.relation_names = list(relation_edges.keys())
        self.relation_tau_days = {
            name: float(relation_tau_days[name])
            for name in self.relation_names
        }

        if set(self.relation_tau_days) != set(self.relation_names):
            raise ValueError(
                'relation_tau_days must contain exactly the graph relation names'
            )

        if any(tau <= 0.0 for tau in self.relation_tau_days.values()):
            raise ValueError('all temporal decay constants must be positive')

        if self.positive_seed_rate is not None:
            if self.labels is None:
                raise ValueError(
                    'labels are required for positive-seed oversampling'
                )
            if not (0.0 < self.positive_seed_rate < 1.0):
                raise ValueError(
                    'positive_seed_rate must be strictly between 0 and 1'
                )

            seed_y = self.labels[self.input_nodes]
            self.pos_input_nodes = self.input_nodes[seed_y == 1]
            self.neg_input_nodes = self.input_nodes[seed_y == 0]

            if (
                len(self.pos_input_nodes) == 0
                or len(self.neg_input_nodes) == 0
            ):
                raise ValueError(
                    'both positive and negative seed nodes are required'
                )
        else:
            self.pos_input_nodes = None
            self.neg_input_nodes = None

        self.relation_adjacency = {}

        for name in self.relation_names:
            src = relation_edges[name]['src'].astype(
                np.int64,
                copy=False,
            )
            dst = relation_edges[name]['dst'].astype(
                np.int64,
                copy=False,
            )
            gap = relation_edges[name]['gap_days'].astype(
                np.float32,
                copy=False,
            )

            # Sort by target, then ascending gap.
            # Evaluation therefore keeps the most recent relation-specific
            # neighbors first under deterministic fanout truncation.
            order = np.lexsort((gap, dst))
            src = src[order]
            dst = dst[order]
            gap = gap[order]

            tau = self.relation_tau_days[name]
            weight = np.exp(
                -gap.astype(np.float64) / tau
            ).astype(np.float32)

            counts = np.bincount(
                dst,
                minlength=int(data.num_nodes),
            )
            colptr = np.empty(
                int(data.num_nodes) + 1,
                dtype=np.int64,
            )
            colptr[0] = 0
            np.cumsum(counts, out=colptr[1:])

            self.relation_adjacency[name] = {
                'src': src,
                'gap_days': gap,
                'weight': weight,
                'colptr': colptr,
            }

    def reset_epoch(self):
        self._epoch = 0

    def __len__(self):
        n = (
            self.epoch_size
            if self.positive_seed_rate is not None
            else len(self.input_nodes)
        )
        return int(math.ceil(n / self.batch_size))

    def _sample_stratified_seed_batch(
        self,
        rng,
        current_batch_size,
    ):
        n_pos = int(
            round(self.positive_seed_rate * current_batch_size)
        )
        n_neg = current_batch_size - n_pos

        if n_pos > len(self.pos_input_nodes):
            n_pos = len(self.pos_input_nodes)
            n_neg = current_batch_size - n_pos

        if n_neg > len(self.neg_input_nodes):
            n_neg = len(self.neg_input_nodes)
            n_pos = current_batch_size - n_neg

        if (
            n_pos > len(self.pos_input_nodes)
            or n_neg > len(self.neg_input_nodes)
        ):
            raise RuntimeError(
                'Not enough unique class-specific nodes to form one seed batch.'
            )

        pos_nodes = rng.choice(
            self.pos_input_nodes,
            size=n_pos,
            replace=False,
        )
        neg_nodes = rng.choice(
            self.neg_input_nodes,
            size=n_neg,
            replace=False,
        )

        seed_nodes = np.concatenate(
            [pos_nodes, neg_nodes]
        ).astype(np.int64, copy=False)

        rng.shuffle(seed_nodes)
        return seed_nodes

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)

        if self.positive_seed_rate is None:
            nodes = self.input_nodes.copy()

            if self.shuffle:
                rng.shuffle(nodes)

            for start in range(
                0,
                len(nodes),
                self.batch_size,
            ):
                seed_nodes = nodes[
                    start:start + self.batch_size
                ]
                yield self._sample_batch(
                    seed_nodes,
                    rng,
                    random_sample=self.shuffle,
                )
        else:
            n_total = int(self.epoch_size)

            for start in range(
                0,
                n_total,
                self.batch_size,
            ):
                current_bs = min(
                    self.batch_size,
                    n_total - start,
                )
                seed_nodes = self._sample_stratified_seed_batch(
                    rng,
                    current_bs,
                )
                yield self._sample_batch(
                    seed_nodes,
                    rng,
                    random_sample=self.shuffle,
                )

        self._epoch += 1

    @staticmethod
    def _deduplicate_pairs(src_list, dst_list):
        seen = set()
        out_src = []
        out_dst = []

        for s, d in zip(src_list, dst_list):
            pair = (int(s), int(d))
            if pair in seen:
                continue

            seen.add(pair)
            out_src.append(int(s))
            out_dst.append(int(d))

        return out_src, out_dst

    @staticmethod
    def _deduplicate_weighted_pairs(
        src_list,
        dst_list,
        weight_list,
    ):
        seen = set()
        out_src = []
        out_dst = []
        out_weight = []

        for s, d, w in zip(
            src_list,
            dst_list,
            weight_list,
        ):
            pair = (int(s), int(d))
            if pair in seen:
                continue

            seen.add(pair)
            out_src.append(int(s))
            out_dst.append(int(d))
            out_weight.append(float(w))

        return out_src, out_dst, out_weight

    def _sample_batch(
        self,
        seed_nodes,
        rng,
        random_sample,
    ):
        local_nodes = []
        global_to_local = {}

        for node in seed_nodes.tolist():
            node = int(node)

            if node not in global_to_local:
                global_to_local[node] = len(local_nodes)
                local_nodes.append(node)

        frontier = [
            int(x)
            for x in seed_nodes.tolist()
        ]

        sampled_by_relation = {
            name: {
                'src': [],
                'dst': [],
                'weight': [],
            }
            for name in self.relation_names
        }

        relation_counts = {
            name: 0
            for name in self.relation_names
        }

        for fanout in self.relation_fanouts:
            next_frontier = []
            next_seen = set()

            for dst_node in frontier:
                for name in self.relation_names:
                    adj = self.relation_adjacency[name]

                    a = adj['colptr'][dst_node]
                    b = adj['colptr'][dst_node + 1]

                    neighbors = adj['src'][a:b]
                    weights = adj['weight'][a:b]

                    if len(neighbors) == 0:
                        continue

                    if (
                        fanout >= 0
                        and len(neighbors) > fanout
                    ):
                        if random_sample:
                            pick = rng.choice(
                                len(neighbors),
                                size=fanout,
                                replace=False,
                            )
                        else:
                            pick = np.arange(
                                fanout,
                                dtype=np.int64,
                            )

                        chosen = neighbors[pick]
                        chosen_weight = weights[pick]
                    else:
                        chosen = neighbors
                        chosen_weight = weights

                    relation_counts[name] += int(
                        len(chosen)
                    )

                    for src_node, edge_weight in zip(
                        chosen.tolist(),
                        chosen_weight.tolist(),
                    ):
                        src_node = int(src_node)

                        sampled_by_relation[name]['src'].append(
                            src_node
                        )
                        sampled_by_relation[name]['dst'].append(
                            int(dst_node)
                        )
                        sampled_by_relation[name]['weight'].append(
                            float(edge_weight)
                        )

                        if src_node not in global_to_local:
                            global_to_local[src_node] = len(
                                local_nodes
                            )
                            local_nodes.append(src_node)

                        if src_node not in next_seen:
                            next_seen.add(src_node)
                            next_frontier.append(src_node)

            frontier = next_frontier

            if not frontier:
                break

        relation_edge_index_local = {}
        relation_edge_weight_local = {}

        union_src_global = []
        union_dst_global = []

        for name in self.relation_names:
            rel_src, rel_dst, rel_weight = (
                self._deduplicate_weighted_pairs(
                    sampled_by_relation[name]['src'],
                    sampled_by_relation[name]['dst'],
                    sampled_by_relation[name]['weight'],
                )
            )

            union_src_global.extend(rel_src)
            union_dst_global.extend(rel_dst)

            if rel_src:
                local_src = np.fromiter(
                    (
                        global_to_local[x]
                        for x in rel_src
                    ),
                    dtype=np.int64,
                )
                local_dst = np.fromiter(
                    (
                        global_to_local[x]
                        for x in rel_dst
                    ),
                    dtype=np.int64,
                )

                rel_edge_index = torch.from_numpy(
                    np.vstack([local_src, local_dst])
                ).long()

                rel_edge_weight = torch.tensor(
                    rel_weight,
                    dtype=torch.float32,
                )
            else:
                rel_edge_index = torch.empty(
                    (2, 0),
                    dtype=torch.long,
                )
                rel_edge_weight = torch.empty(
                    (0,),
                    dtype=torch.float32,
                )

            relation_edge_index_local[name] = rel_edge_index
            relation_edge_weight_local[name] = rel_edge_weight

        union_src_global, union_dst_global = (
            self._deduplicate_pairs(
                union_src_global,
                union_dst_global,
            )
        )

        if union_src_global:
            union_local_src = np.fromiter(
                (
                    global_to_local[x]
                    for x in union_src_global
                ),
                dtype=np.int64,
            )
            union_local_dst = np.fromiter(
                (
                    global_to_local[x]
                    for x in union_dst_global
                ),
                dtype=np.int64,
            )

            edge_index_local = torch.from_numpy(
                np.vstack(
                    [union_local_src, union_local_dst]
                )
            ).long()
        else:
            edge_index_local = torch.empty(
                (2, 0),
                dtype=torch.long,
            )

        n_id = torch.from_numpy(
            np.asarray(
                local_nodes,
                dtype=np.int64,
            )
        ).long()

        return SimpleNeighborBatch(
            x_sage=self.data.x_sage[n_id],
            x_topology=self.data.x_topology[n_id],
            lgb_raw=self.data.lgb_raw[n_id],
            y=self.data.y[n_id],
            edge_index=edge_index_local,
            relation_edge_index=relation_edge_index_local,
            relation_edge_weight=relation_edge_weight_local,
            n_id=n_id,
            batch_size=len(seed_nodes),
            relation_sample_counts=relation_counts,
        )

def reset_loader_epochs(*loaders):
    """Reset epoch counters on one or more relation-aware loaders."""
    for loader in loaders:
        if hasattr(loader, "set_epoch"):
            loader.set_epoch(0)

