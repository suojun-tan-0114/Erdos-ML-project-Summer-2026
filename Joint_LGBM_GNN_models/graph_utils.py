"""Graph construction and past-only structural feature helpers."""

import numpy as np
import pandas as pd
import torch

SECONDS_PER_DAY = 86400.0

def build_split_aware_relation_graph(
    df,
    relation_specs,
    train_end,
    val_end,
    time_col='TransactionDT',
):
    """Build directed past -> future relation edges with strict split-aware history."""
    n = len(df)
    node_ids = np.arange(n, dtype=np.int64)
    t = pd.to_numeric(df[time_col], errors='coerce').fillna(-1).to_numpy(dtype=np.float64)

    relation_edges = {}
    stats = []
    union_src_parts = []
    union_dst_parts = []

    for spec in relation_specs:
        name = spec['name']
        cols = list(spec['cols'])

        missing = [c for c in cols if c not in df.columns]
        if missing:
            print(f'skipping {name}: missing {missing}')
            continue

        min_age_days = float(spec.get('min_age_days', 0.0))
        window_days = float(spec['window_days'])
        max_history = int(spec['max_history'])

        valid = df[cols].notna().all(axis=1).to_numpy()
        work = df.loc[valid, cols].copy()
        work['_node'] = node_ids[valid]
        work['_time'] = t[valid]

        work = work.sort_values(cols + ['_time', '_node'], kind='mergesort')
        grouped_nodes = work.groupby(cols, sort=False, dropna=False)['_node']
        grouped_times = work.groupby(cols, sort=False, dropna=False)['_time']

        src_parts = []
        dst_parts = []
        gap_parts = []

        for lag in range(1, max_history + 1):
            src_node = grouped_nodes.shift(lag)
            src_time = grouped_times.shift(lag)
            gap_seconds = work['_time'] - src_time

            keep = (
                src_node.notna()
                & src_time.notna()
                & (gap_seconds > 0.0)
                & (gap_seconds <= window_days * SECONDS_PER_DAY)
            )

            if min_age_days > 0.0:
                keep &= gap_seconds > min_age_days * SECONDS_PER_DAY

            if not keep.any():
                continue

            src = src_node.loc[keep].astype(np.int64).to_numpy()
            dst = work.loc[keep, '_node'].to_numpy(dtype=np.int64)
            gap = gap_seconds.loc[keep].to_numpy(dtype=np.float32) / SECONDS_PER_DAY

            # Validation targets: source must be in train.
            val_target = (dst >= train_end) & (dst < val_end)
            allowed_val = (~val_target) | (src < train_end)

            # Test targets: source must be before test (train + validation).
            test_target = dst >= val_end
            allowed_test = (~test_target) | (src < val_end)

            allowed = allowed_val & allowed_test

            src = src[allowed]
            dst = dst[allowed]
            gap = gap[allowed]

            if len(src):
                src_parts.append(src)
                dst_parts.append(dst)
                gap_parts.append(gap)

        if src_parts:
            src = np.concatenate(src_parts)
            dst = np.concatenate(dst_parts)
            gap = np.concatenate(gap_parts)

            order = np.lexsort((gap, dst))
            src = src[order]
            dst = dst[order]
            gap = gap[order]
        else:
            src = np.empty(0, dtype=np.int64)
            dst = np.empty(0, dtype=np.int64)
            gap = np.empty(0, dtype=np.float32)

        relation_edges[name] = {
            'src': src,
            'dst': dst,
            'gap_days': gap,
        }

        union_src_parts.append(src)
        union_dst_parts.append(dst)

        stats.append({
            'relation': name,
            'edges': int(len(src)),
            'unique_targets': int(np.unique(dst).size),
            'window_days': window_days,
            'min_age_days': min_age_days,
            'max_history': max_history,
        })

        print(f'{name:18s} edges={len(src):,} targets={np.unique(dst).size:,}')

    if not union_src_parts:
        raise ValueError('No graph edges were created')

    all_src = np.concatenate(union_src_parts)
    all_dst = np.concatenate(union_dst_parts)

    pair_key = all_src.astype(np.int64) * np.int64(n) + all_dst.astype(np.int64)
    unique_key = np.unique(pair_key)
    union_src = (unique_key // np.int64(n)).astype(np.int64)
    union_dst = (unique_key % np.int64(n)).astype(np.int64)

    edge_index = torch.from_numpy(np.vstack([union_src, union_dst])).long()
    stats_df = pd.DataFrame(stats)

    # Leakage assertions.
    src_np = edge_index[0].numpy()
    dst_np = edge_index[1].numpy()

    assert np.all(t[src_np] < t[dst_np])

    val_edges = (dst_np >= train_end) & (dst_np < val_end)
    assert np.all(src_np[val_edges] < train_end)

    test_edges = dst_np >= val_end
    assert np.all(src_np[test_edges] < val_end)

    return edge_index, relation_edges, stats_df

def build_relation_incoming_index(relation_edges, num_nodes):
    incoming = {}

    for name, payload in relation_edges.items():
        src = payload['src'].astype(np.int64, copy=False)
        dst = payload['dst'].astype(np.int64, copy=False)

        order = np.argsort(dst, kind='stable')
        src = src[order]
        dst = dst[order]

        counts = np.bincount(dst, minlength=int(num_nodes))
        colptr = np.empty(int(num_nodes) + 1, dtype=np.int64)
        colptr[0] = 0
        np.cumsum(counts, out=colptr[1:])

        incoming[name] = {'src': src, 'colptr': colptr}

    return incoming

def local_graph_stats(per_relation_candidates):
    all_nodes = [int(x) for cand in per_relation_candidates for x in cand]

    if not all_nodes:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    unique_nodes = sorted(set(all_nodes))
    n_nodes = len(unique_nodes)
    local_pos = {node: i for i, node in enumerate(unique_nodes)}

    parent = list(range(n_nodes))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    edge_pairs = set()
    active_relations = 0

    for cand in per_relation_candidates:
        if len(cand) == 0:
            continue

        active_relations += 1
        local_nodes = sorted({local_pos[int(node)] for node in cand})

        for i in range(len(local_nodes)):
            for j in range(i + 1, len(local_nodes)):
                a, b = local_nodes[i], local_nodes[j]
                pair = (a, b)
                if pair not in edge_pairs:
                    edge_pairs.add(pair)
                    union(a, b)

    m_edges = len(edge_pairs)
    beta0 = len({find(i) for i in range(n_nodes)})
    beta1 = max(m_edges - n_nodes + beta0, 0)
    density = 0.0 if n_nodes <= 1 else 2.0 * m_edges / (n_nodes * (n_nodes - 1))

    return (
        float(n_nodes),
        float(active_relations),
        float(m_edges),
        float(beta0),
        float(beta1),
        float(density),
    )

def build_past_only_toper_features(relation_edges, num_nodes, verbose=True):
    incoming = build_relation_incoming_index(relation_edges, num_nodes)
    relation_names = list(incoming.keys())
    features = np.zeros((int(num_nodes), 6), dtype=np.float32)

    for v in range(int(num_nodes)):
        if verbose and v > 0 and v % 50000 == 0:
            print(f'TopER features: {v:,}/{num_nodes:,}')

        candidates = []
        for name in relation_names:
            adj = incoming[name]
            a, b = adj['colptr'][v], adj['colptr'][v + 1]
            candidates.append(adj['src'][a:b])

        features[v] = local_graph_stats(candidates)

    return features

def robust_scale_train_only(Z, fit_idx, clip_value=8.0):
    Z = np.asarray(Z, dtype=np.float32)
    fit_idx = np.asarray(fit_idx, dtype=np.int64)

    med = np.nanmedian(Z[fit_idx], axis=0)
    q25 = np.nanpercentile(Z[fit_idx], 25, axis=0)
    q75 = np.nanpercentile(Z[fit_idx], 75, axis=0)
    scale = q75 - q25
    scale[~np.isfinite(scale) | (scale < 1e-12)] = 1.0

    out = (Z - med) / scale
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    out = np.clip(out, -clip_value, clip_value).astype(np.float32)

    return out, {'median': med, 'scale': scale}
