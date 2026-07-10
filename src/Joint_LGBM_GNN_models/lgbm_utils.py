"""LightGBM fitting, temporal OOF, and metric helpers."""

import numpy as np
import lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score

from .data_utils import nearest_time_boundary

def safe_auc(y_true, score):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, score))

def safe_pr_auc(y_true, score):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(average_precision_score(y_true, score))

def sigmoid_np(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out

def running_past_prior_logits(y, idx_sorted, alpha_prior=1.0, beta_prior=29.0):
    """Past-only smoothed label prior; row i uses labels strictly before i."""
    idx_sorted = np.asarray(idx_sorted, dtype=np.int64)
    y = np.asarray(y, dtype=np.float64)

    logits = np.full(len(y), np.nan, dtype=np.float32)
    pos = float(alpha_prior)
    total = float(alpha_prior + beta_prior)

    for node in idx_sorted:
        p = np.clip(pos / total, 1e-6, 1.0 - 1e-6)
        logits[node] = np.log(p / (1.0 - p))
        pos += float(y[node])
        total += 1.0

    return logits

def split_inner_train_es(train_idx, time_values, es_frac=0.10):
    train_idx = np.asarray(train_idx, dtype=np.int64)
    order = np.argsort(np.asarray(time_values)[train_idx], kind='mergesort')
    s = train_idx[order]
    t = np.asarray(time_values)[s]

    nominal = int((1.0 - es_frac) * len(s))
    cut = nearest_time_boundary(t, nominal)

    fit_idx = s[:cut]
    es_idx = s[cut:]

    assert np.max(time_values[fit_idx]) < np.min(time_values[es_idx])
    return fit_idx, es_idx

def fit_lgb_train_only(
    Xmat,
    y,
    train_idx,
    time_values,
    params,
    inner_es_frac=0.10,
    max_boost_rounds=3000,
    early_stopping_rounds=100,
    fallback_rounds=400,
):
    """Choose best iteration inside train, then refit on all train rows."""
    params = params.copy()
    inner_fit_idx, inner_es_idx = split_inner_train_es(
        train_idx,
        time_values,
        es_frac=inner_es_frac,
    )

    dfit = lgb.Dataset(Xmat.iloc[inner_fit_idx], label=y[inner_fit_idx])
    des = lgb.Dataset(Xmat.iloc[inner_es_idx], label=y[inner_es_idx], reference=dfit)

    probe = lgb.train(
        params,
        dfit,
        num_boost_round=max_boost_rounds,
        valid_sets=[des],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds),
            lgb.log_evaluation(100),
        ],
    )

    best_iteration = int(probe.best_iteration or fallback_rounds)
    print('selected LightGBM rounds:', best_iteration)

    dfull = lgb.Dataset(Xmat.iloc[train_idx], label=y[train_idx])
    model = lgb.train(
        params,
        dfull,
        num_boost_round=best_iteration,
        callbacks=[lgb.log_evaluation(100)],
    )

    raw_all = model.predict(Xmat, raw_score=True).astype(np.float32)
    prob_all = sigmoid_np(raw_all).astype(np.float32)

    return model, raw_all, prob_all, best_iteration

def temporal_fold_edges(train_sorted, time_values, n_splits, min_train_frac):
    """Create expanding-window validation blocks aligned to timestamp boundaries."""
    train_sorted = np.asarray(train_sorted, dtype=np.int64)
    t = np.asarray(time_values)[train_sorted]
    n = len(train_sorted)

    start = max(1, int(min_train_frac * n))
    nominal = np.linspace(start, n, n_splits + 1, dtype=int)

    edges = [nearest_time_boundary(t, int(x)) if x < n else n for x in nominal]
    edges[0] = max(1, edges[0])
    edges[-1] = n

    clean = [edges[0]]
    for x in edges[1:]:
        if x > clean[-1]:
            clean.append(x)

    return clean

def make_lgb_temporal_oof_raw(
    Xmat,
    y,
    train_idx,
    time_values,
    params,
    n_splits=8,
    min_train_frac=0.05,
    es_holdout_frac=0.10,
    max_boost_rounds=3000,
    early_stopping_rounds=100,
    fallback_rounds=400,
):
    """
    Expanding-window temporal OOF raw scores.

    The fold being predicted is never used for early stopping. If a clean
    train-tail early-stop block is unavailable, a fixed fallback round count
    is used instead.
    """
    params = params.copy()

    train_idx = np.asarray(train_idx, dtype=np.int64)
    order = np.argsort(np.asarray(time_values)[train_idx], kind='mergesort')
    train_sorted = train_idx[order]

    edges = temporal_fold_edges(
        train_sorted,
        time_values,
        n_splits=n_splits,
        min_train_frac=min_train_frac,
    )

    oof_raw = np.full(len(y), np.nan, dtype=np.float32)

    for fold, (a, b) in enumerate(zip(edges[:-1], edges[1:]), start=1):
        va_fold = train_sorted[a:b]
        if len(va_fold) == 0:
            continue

        va_min_time = np.min(np.asarray(time_values)[va_fold])
        tr_all = train_sorted[np.asarray(time_values)[train_sorted] < va_min_time]

        if len(tr_all) < 50 or len(np.unique(y[tr_all])) < 2:
            print(f'OOF fold {fold}: skipped, insufficient prior training data')
            continue

        es_size = max(1, int(es_holdout_frac * len(tr_all)))
        provisional_cut = len(tr_all) - es_size
        t_tr = np.asarray(time_values)[tr_all]
        cut = nearest_time_boundary(t_tr, provisional_cut)

        tr_fold = tr_all[:cut]
        es_fold = tr_all[cut:]

        use_es = (
            len(tr_fold) >= 50
            and len(es_fold) >= 20
            and len(np.unique(y[tr_fold])) == 2
            and len(np.unique(y[es_fold])) == 2
        )

        if use_es:
            dtrain = lgb.Dataset(Xmat.iloc[tr_fold], label=y[tr_fold])
            dvalid = lgb.Dataset(Xmat.iloc[es_fold], label=y[es_fold], reference=dtrain)

            model = lgb.train(
                params,
                dtrain,
                num_boost_round=max_boost_rounds,
                valid_sets=[dvalid],
                callbacks=[
                    lgb.early_stopping(early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(0),
                ],
            )
            rounds = int(model.best_iteration or fallback_rounds)
            tag = 'train-tail ES'
        else:
            dtrain = lgb.Dataset(Xmat.iloc[tr_all], label=y[tr_all])
            rounds = fallback_rounds
            model = lgb.train(
                params,
                dtrain,
                num_boost_round=rounds,
                callbacks=[lgb.log_evaluation(0)],
            )
            tag = 'fixed rounds'

        pred = model.predict(Xmat.iloc[va_fold], raw_score=True)
        oof_raw[va_fold] = pred.astype(np.float32)

        print(
            f'OOF fold {fold}/{len(edges)-1}: {tag}, '
            f'train={len(tr_all):,}, predict={len(va_fold):,}, rounds={rounds}'
        )

    return oof_raw
