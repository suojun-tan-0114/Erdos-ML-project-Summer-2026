"""Small training utilities and loss functions."""

import torch
import torch.nn.functional as F

def grad_norm_from_loss(
    loss,
    params,
    retain_graph=True,
):
    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=retain_graph,
        allow_unused=True,
    )

    total_sq = 0.0

    for grad in grads:
        if grad is not None:
            total_sq += (
                grad.detach().pow(2).sum().item()
            )

    return total_sq ** 0.5

def cpu_state_dict(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }

def pairwise_auc_loss(
    logits,
    targets,
    max_pairs=8192,
):
    """
    Pairwise logistic ranking surrogate for AUC maximization.

    Encourages positive examples to receive larger logits than negative
    examples. When all positive-negative pairs exceed max_pairs, sample
    uniform positive-negative pairs to bound memory and runtime.
    """
    logits = logits.reshape(-1)
    targets = targets.reshape(-1)

    pos_scores = logits[targets > 0.5]
    neg_scores = logits[targets <= 0.5]

    n_pos = int(pos_scores.numel())
    n_neg = int(neg_scores.numel())

    if n_pos == 0 or n_neg == 0:
        # Keep a differentiable zero so the auxiliary GNN BCE term can
        # still train this batch.
        return logits.sum() * 0.0

    total_pairs = n_pos * n_neg

    if total_pairs <= int(max_pairs):
        differences = (
            neg_scores.unsqueeze(0)
            - pos_scores.unsqueeze(1)
        )
        return F.softplus(differences).mean()

    pos_idx = torch.randint(
        low=0,
        high=n_pos,
        size=(int(max_pairs),),
        device=logits.device,
    )
    neg_idx = torch.randint(
        low=0,
        high=n_neg,
        size=(int(max_pairs),),
        device=logits.device,
    )

    sampled_differences = (
        neg_scores[neg_idx]
        - pos_scores[pos_idx]
    )

    return F.softplus(sampled_differences).mean()
