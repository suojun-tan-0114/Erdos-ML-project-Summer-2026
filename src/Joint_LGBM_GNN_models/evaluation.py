"""Validation/test evaluation and alpha-selection helpers."""

import numpy as np
import pandas as pd

from .lgbm_utils import safe_auc, safe_pr_auc, sigmoid_np

def fusion_metrics_from_pred_df(
    pred_df,
    alpha,
):
    """Score one global alpha using already-computed branch logits."""
    alpha = float(alpha)

    y_true = pred_df[
        'isFraud'
    ].to_numpy(dtype=np.float64)

    lgb_raw = pred_df[
        'logit_lgb'
    ].to_numpy(dtype=np.float64)

    gnn_raw = pred_df[
        'logit_gnn'
    ].to_numpy(dtype=np.float64)

    fused_raw = (
        alpha * lgb_raw
        + (1.0 - alpha) * gnn_raw
    )

    p_fused = sigmoid_np(fused_raw)

    return {
        'auc': safe_auc(y_true, p_fused),
        'pr': safe_pr_auc(y_true, p_fused),
        'alpha': alpha,
    }

def select_alpha_on_validation(
    pred_df,
    alpha_grid,
    metric='auc',
):
    if metric not in {'auc', 'pr'}:
        raise ValueError(
            "metric must be 'auc' or 'pr'"
        )

    rows = []

    for alpha in np.asarray(
        alpha_grid,
        dtype=float,
    ):
        rows.append(
            fusion_metrics_from_pred_df(
                pred_df,
                alpha,
            )
        )

    grid_df = pd.DataFrame(rows)
    score = grid_df[
        metric
    ].to_numpy(dtype=float)

    if not np.any(np.isfinite(score)):
        raise RuntimeError(
            'No finite validation score in alpha grid search.'
        )

    best_pos = int(
        np.nanargmax(score)
    )
    best_row = grid_df.iloc[
        best_pos
    ].to_dict()

    return (
        float(best_row['alpha']),
        best_row,
        grid_df,
    )

def add_fused_probability(
    pred_df,
    alpha,
    column_name='pred_fused',
):
    out = pred_df.copy()

    fused_raw = (
        float(alpha)
        * out['logit_lgb'].to_numpy(
            dtype=np.float64
        )
        + (1.0 - float(alpha))
        * out['logit_gnn'].to_numpy(
            dtype=np.float64
        )
    )

    out[column_name] = sigmoid_np(
        fused_raw
    )

    return out

def _attention_summary_from_array(
    attention,
    relation_names,
):
    attention = np.asarray(
        attention,
        dtype=np.float64,
    )

    row_sum = attention.sum(axis=1)
    active = row_sum > 0

    if not np.any(active):
        mean_attention = np.zeros(
            len(relation_names),
            dtype=np.float64,
        )
        coverage = 0.0
    else:
        mean_attention = attention[
            active
        ].mean(axis=0)
        coverage = float(
            active.mean()
        )

    return {
        'mean': {
            name: float(value)
            for name, value in zip(
                relation_names,
                mean_attention,
            )
        },
        'coverage': coverage,
    }

def evaluate_loader(
    model,
    loader,
    device,
    alpha_override=None,
):
    model.eval()

    relation_names = list(
        model.gnn.relation_names
    )

    node_ids = []
    y_true = []
    lgb_raw_all = []
    gnn_raw_all = []

    attn1_all = []
    attn2_all = []

    alpha_value = None

    for batch in loader:
        batch = batch.to(device)

        out = model(
            batch.x_sage,
            batch.x_topology,
            batch.lgb_raw,
            batch.relation_edge_index,
            batch.relation_edge_weight,
            alpha_override=alpha_override,
        )

        bs = int(batch.batch_size)
        seed = slice(0, bs)

        node_ids.append(
            batch.n_id[seed].cpu().numpy()
        )
        y_true.append(
            batch.y[seed].cpu().numpy()
        )
        lgb_raw_all.append(
            batch.lgb_raw[seed].cpu().numpy()
        )
        gnn_raw_all.append(
            out['logit_gnn'][seed]
            .cpu()
            .numpy()
        )

        attn1_all.append(
            out['relation_attention']['layer1'][
                seed
            ].cpu().numpy()
        )
        attn2_all.append(
            out['relation_attention']['layer2'][
                seed
            ].cpu().numpy()
        )

        alpha_value = float(
            out['alpha'].detach().cpu()
        )

    node_ids = np.concatenate(node_ids)
    y_true = np.concatenate(y_true)
    lgb_raw_all = np.concatenate(
        lgb_raw_all
    )
    gnn_raw_all = np.concatenate(
        gnn_raw_all
    )

    attn1_all = np.concatenate(
        attn1_all,
        axis=0,
    )
    attn2_all = np.concatenate(
        attn2_all,
        axis=0,
    )

    preds = pd.DataFrame({
        'node_id': node_ids,
        'isFraud': y_true,
        'logit_lgb': lgb_raw_all,
        'logit_gnn': gnn_raw_all,
    })

    for j, name in enumerate(
        relation_names
    ):
        preds[
            f'attn_l1_{name}'
        ] = attn1_all[:, j]

        preds[
            f'attn_l2_{name}'
        ] = attn2_all[:, j]

    learned = fusion_metrics_from_pred_df(
        preds,
        alpha_value,
    )

    p_lgb = sigmoid_np(
        lgb_raw_all
    )
    p_gnn = sigmoid_np(
        gnn_raw_all
    )

    attn1_summary = _attention_summary_from_array(
        attn1_all,
        relation_names,
    )
    attn2_summary = _attention_summary_from_array(
        attn2_all,
        relation_names,
    )

    metrics = {
        'auc_fused': learned['auc'],
        'pr_fused': learned['pr'],
        'auc_lgb': safe_auc(
            y_true,
            p_lgb,
        ),
        'pr_lgb': safe_pr_auc(
            y_true,
            p_lgb,
        ),
        'auc_gnn': safe_auc(
            y_true,
            p_gnn,
        ),
        'pr_gnn': safe_pr_auc(
            y_true,
            p_gnn,
        ),
        'alpha': alpha_value,
        'attention_layer1': attn1_summary['mean'],
        'attention_layer2': attn2_summary['mean'],
        'attention_coverage_layer1': attn1_summary[
            'coverage'
        ],
        'attention_coverage_layer2': attn2_summary[
            'coverage'
        ],
    }

    preds['pred_lgb'] = p_lgb
    preds['pred_gnn'] = p_gnn

    preds['pred_fused'] = sigmoid_np(
        alpha_value * lgb_raw_all
        + (1.0 - alpha_value)
        * gnn_raw_all
    )

    preds['alpha'] = alpha_value

    return metrics, preds
