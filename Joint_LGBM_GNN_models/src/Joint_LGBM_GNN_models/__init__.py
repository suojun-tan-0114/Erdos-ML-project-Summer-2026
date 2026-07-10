"""Helpers for the joint LightGBM + temporal relation-aware GNN experiments."""

from .artifacts import save_single_experiment_artifacts
from .data_utils import (
    add_uid,
    make_graphsage_node_features,
    make_lgb_matrix,
    make_temporal_80_10_10_split,
    nearest_time_boundary,
)
from .evaluation import (
    add_fused_probability,
    evaluate_loader,
    fusion_metrics_from_pred_df,
    select_alpha_on_validation,
)
from .graph_utils import (
    build_past_only_toper_features,
    build_relation_incoming_index,
    build_split_aware_relation_graph,
    local_graph_stats,
    robust_scale_train_only,
)
from .lgbm_utils import (
    fit_lgb_train_only,
    make_lgb_temporal_oof_raw,
    running_past_prior_logits,
    safe_auc,
    safe_pr_auc,
    sigmoid_np,
)
from .models import (
    JointLightGBMTemporalRelationAwareGraphSAGE,
    create_model_and_optimizer,
    set_all_seeds,
)
from .plotting import plot_history_metric
from .sampling import (
    RelationAwareIncomingNeighborLoader,
    SimpleNeighborBatch,
    reset_loader_epochs,
)
from .training_utils import (
    cpu_state_dict,
    grad_norm_from_loss,
    pairwise_auc_loss,
)

__all__ = [
    "add_uid",
    "make_temporal_80_10_10_split",
    "make_lgb_matrix",
    "make_graphsage_node_features",
    "fit_lgb_train_only",
    "make_lgb_temporal_oof_raw",
    "running_past_prior_logits",
    "safe_auc",
    "safe_pr_auc",
    "sigmoid_np",
    "build_split_aware_relation_graph",
    "build_relation_incoming_index",
    "build_past_only_toper_features",
    "local_graph_stats",
    "robust_scale_train_only",
    "RelationAwareIncomingNeighborLoader",
    "SimpleNeighborBatch",
    "reset_loader_epochs",
    "JointLightGBMTemporalRelationAwareGraphSAGE",
    "create_model_and_optimizer",
    "set_all_seeds",
    "fusion_metrics_from_pred_df",
    "select_alpha_on_validation",
    "add_fused_probability",
    "evaluate_loader",
    "cpu_state_dict",
    "grad_norm_from_loss",
    "pairwise_auc_loss",
    "save_single_experiment_artifacts",
    "plot_history_metric",
]
