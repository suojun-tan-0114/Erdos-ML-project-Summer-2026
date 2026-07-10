"""Data loading, temporal split, and feature-preparation helpers."""

import numpy as np
import pandas as pd

SECONDS_PER_DAY = 86400.0

def add_uid(df):
    """Create the same card-holder style UID used in the residual notebook."""
    needed = {'card1', 'addr1', 'D1', 'TransactionDT'}
    out = df.copy()

    if not needed.issubset(out.columns):
        out['UID'] = pd.NA
        return out

    anchor = ((out['TransactionDT'] / SECONDS_PER_DAY) - out['D1']).round()

    uid = (
        out['card1'].astype('string')
        + '_'
        + out['addr1'].astype('string')
        + '_'
        + anchor.astype('string')
    )

    missing = out['card1'].isna() | out['addr1'].isna() | out['D1'].isna()
    uid[missing] = pd.NA
    out['UID'] = uid
    return out

def nearest_time_boundary(sorted_times, nominal_cut):
    """Return a nearby cut that does not split an equal-timestamp group."""
    sorted_times = np.asarray(sorted_times)
    n = len(sorted_times)
    nominal_cut = int(np.clip(nominal_cut, 1, n - 1))

    pivot = sorted_times[nominal_cut]
    left = int(np.searchsorted(sorted_times, pivot, side='left'))
    right = int(np.searchsorted(sorted_times, pivot, side='right'))

    candidates = [c for c in (left, right) if 0 < c < n]
    if not candidates:
        return nominal_cut

    return min(candidates, key=lambda c: abs(c - nominal_cut))

def make_temporal_80_10_10_split(
    df,
    time_col='TransactionDT',
    train_frac=0.80,
    val_frac=0.10,
):
    if time_col not in df.columns:
        raise KeyError(f'{time_col} not found')

    out = (
        df.sort_values([time_col, 'TransactionID'], kind='mergesort')
        .reset_index(drop=True)
    )

    t = out[time_col].to_numpy()
    n = len(out)

    train_end = nearest_time_boundary(t, int(train_frac * n))
    val_end = nearest_time_boundary(t, int((train_frac + val_frac) * n))

    if not (0 < train_end < val_end < n):
        raise ValueError((train_end, val_end, n))

    train_idx = np.arange(0, train_end, dtype=np.int64)
    val_idx = np.arange(train_end, val_end, dtype=np.int64)
    test_idx = np.arange(val_end, n, dtype=np.int64)

    # No equal timestamp can straddle a split boundary.
    assert out.loc[train_idx[-1], time_col] < out.loc[val_idx[0], time_col]
    assert out.loc[val_idx[-1], time_col] < out.loc[test_idx[0], time_col]

    return out, train_idx, val_idx, test_idx

def make_lgb_matrix(df, fit_idx, use_uid_as_lgb_feature=False, drop_transactiondt=False):
    """
    Residual-notebook preprocessing:
      - fit categorical levels on fit_idx only;
      - encode unseen later categories as -1;
      - leave numeric columns in native scale.
    """
    drop_cols = {'isFraud', 'TransactionID'}

    if not use_uid_as_lgb_feature:
        drop_cols.add('UID')

    if drop_transactiondt:
        drop_cols.add('TransactionDT')

    feat_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feat_cols].copy()

    numeric_cols = set(X.select_dtypes(include=['number']).columns)

    for col in X.columns:
        if col in numeric_cols:
            continue

        fit_values = X.iloc[fit_idx][col].astype('string')
        categories = pd.Index(pd.unique(fit_values.dropna()))

        X[col] = pd.Categorical(
            X[col].astype('string'),
            categories=categories,
        ).codes.astype('int32')

    return X

def make_graphsage_node_features(
    Xmat,
    lgb_model,
    time_values,
    fit_idx,
    top_n=64,
    clip_value=8.0,
):
    """
    Build GraphSAGE node features without LightGBM prediction scores.

    LightGBM is used only to rank transaction columns by training-period
    gain importance. The actual GNN node matrix contains:
      - robust-scaled selected transaction columns;
      - robust-scaled transaction time.

    No lgb_prob or lgb_raw prediction feature is included here.
    """
    Xmat = Xmat.copy()
    fit_idx = np.asarray(fit_idx, dtype=np.int64)

    try:
        importance = pd.DataFrame({
            'feature': lgb_model.feature_name(),
            'importance': lgb_model.feature_importance(importance_type='gain'),
        }).sort_values('importance', ascending=False)
        top_cols = importance.head(int(top_n))['feature'].tolist()
    except Exception:
        top_cols = list(Xmat.columns[:int(top_n)])

    top_cols = [c for c in top_cols if c in Xmat.columns]
    X_small = Xmat[top_cols].astype(float).to_numpy(dtype=np.float32)

    med = np.nanmedian(X_small[fit_idx], axis=0)
    q25 = np.nanpercentile(X_small[fit_idx], 25, axis=0)
    q75 = np.nanpercentile(X_small[fit_idx], 75, axis=0)
    scale = q75 - q25
    scale[~np.isfinite(scale) | (scale < 1e-12)] = 1.0

    X_small = (X_small - med) / scale
    X_small = np.nan_to_num(X_small, nan=0.0, posinf=0.0, neginf=0.0)
    X_small = np.clip(X_small, -clip_value, clip_value).astype(np.float32)

    t = np.asarray(time_values, dtype=np.float64)
    t_med = np.median(t[fit_idx])
    t_scale = np.percentile(t[fit_idx], 75) - np.percentile(t[fit_idx], 25)
    if not np.isfinite(t_scale) or t_scale < 1e-12:
        t_scale = np.std(t[fit_idx]) + 1e-12

    t_feat = ((t - t_med) / t_scale).reshape(-1, 1)
    t_feat = np.clip(t_feat, -clip_value, clip_value).astype(np.float32)

    X_sage = np.column_stack([X_small, t_feat]).astype(np.float32)

    feature_names = (
        [f'lgbfeat::{c}' for c in top_cols]
        + ['time_scaled']
    )

    return X_sage, feature_names
