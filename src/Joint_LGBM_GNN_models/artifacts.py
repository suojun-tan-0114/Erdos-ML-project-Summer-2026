"""Persistence helpers for experiment checkpoints and reusable outputs."""

from pathlib import Path

import torch


def save_single_experiment_artifacts(
    result,
    model_dir,
    checkpoint_metadata,
):
    """
    Persist one completed experiment immediately so later runs can fail
    without losing already-completed training work.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    experiment_name = result["name"]

    checkpoint = {
        "experiment_name": experiment_name,
        "fixed_alpha": result["fixed_alpha"],
        "main_loss_type": result["main_loss_type"],
        "best_epoch": result["best_epoch"],
        "best_val_score": result["best_val_score"],
        "best_selected_alpha_auc": result["best_selected_alpha_auc"],
        "best_selected_alpha_pr": result["best_selected_alpha_pr"],
        "restored_alpha_parameter": result["restored_alpha_parameter"],
        "model_state_dict": result["best_state"],
        **checkpoint_metadata,
    }

    checkpoint_path = (
        model_dir
        / f"temporal_relation_aware_{experiment_name}_best.pt"
    )
    torch.save(checkpoint, checkpoint_path)

    history_path = (
        model_dir
        / f"temporal_relation_aware_{experiment_name}_history.csv"
    )
    result["history_df"].to_csv(history_path, index=False)

    alpha_grid_path = (
        model_dir
        / f"temporal_relation_aware_{experiment_name}_best_alpha_grid.csv"
    )
    result["best_alpha_grid_df"].to_csv(alpha_grid_path, index=False)

    val_pred_path = (
        model_dir
        / f"temporal_relation_aware_{experiment_name}_best_val_predictions.parquet"
    )
    val_pred_to_save = result["best_val_predictions"].copy()
    val_pred_to_save.insert(0, "experiment", experiment_name)
    val_pred_to_save.insert(
        1,
        "main_loss_type",
        result["main_loss_type"],
    )
    val_pred_to_save.to_parquet(val_pred_path, index=False)

    print("saved checkpoint:", checkpoint_path)
    print("saved history:", history_path)
    print("saved best alpha grid:", alpha_grid_path)
    print("saved best validation predictions:", val_pred_path)

    return {
        "checkpoint": checkpoint_path,
        "history": history_path,
        "alpha_grid": alpha_grid_path,
        "validation_predictions": val_pred_path,
    }
