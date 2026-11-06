"""
Model Evaluation module for Restricted Boltzmann Machine recommender.
Computes real test-set metrics on held-out MovieLens ratings:
- Reconstruction RMSE and MAE
- Precision@K, Recall@K, Hit Rate@K, and MRR@K
"""

from typing import Dict, List, Tuple
import torch
import numpy as np
import pandas as pd

from .rbm import RBM


def compute_reconstruction_metrics(
    model: RBM,
    test_data: torch.Tensor,
    test_mask: torch.Tensor,
    train_data: torch.Tensor,
    k_gibbs: int = 1
) -> Dict[str, float]:
    """
    Computes RMSE and MAE of reconstructed visible probabilities against held-out test ratings.
    Evaluated strictly on items observed in the test set.
    """
    model.eval()
    with torch.no_grad():
        # Condition on user's known training ratings
        reconstructed = model.reconstruct(train_data, k=k_gibbs)

        test_data_cpu = test_data.cpu()
        test_mask_cpu = test_mask.cpu()

        rated_count = torch.sum(test_mask_cpu).item()
        if rated_count == 0:
            return {"rmse": 0.0, "mae": 0.0, "rated_count": 0}

        diff = (test_data_cpu - reconstructed) * test_mask_cpu
        mse = torch.sum(diff ** 2).item() / rated_count
        rmse = float(np.sqrt(mse))
        mae = float(torch.sum(torch.abs(diff)).item() / rated_count)

    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "test_ratings_evaluated": int(rated_count)
    }


def compute_ranking_metrics_at_k(
    model: RBM,
    train_data: torch.Tensor,
    train_mask: torch.Tensor,
    test_data: torch.Tensor,
    test_mask: torch.Tensor,
    k_list: List[int] = [5, 10, 15, 20],
    rating_threshold: float = 1.0,
    k_gibbs: int = 1
) -> Dict[str, Dict[str, float]]:
    """
    Evaluates Top-K recommendation ranking quality:
    - Precision@K: Fraction of top-K recommendations that are in user's test set
    - Recall@K: Fraction of user's test set items discovered in top-K
    - Hit Rate@K: Fraction of users who have at least 1 relevant hit in top-K
    - MRR@K: Mean Reciprocal Rank of the first relevant recommended item
    """
    model.eval()
    num_users = train_data.size(0)

    with torch.no_grad():
        reconstructed = model.reconstruct(train_data, k=k_gibbs).cpu().numpy()

    train_mask_np = train_mask.cpu().numpy()
    test_data_np = test_data.cpu().numpy()
    test_mask_np = test_mask.cpu().numpy()

    # Ground truth: movies user liked in test set (test_data == 1 and test_mask == 1)
    relevant_test = (test_data_np >= rating_threshold) & (test_mask_np == 1)

    metrics_by_k = {k: {"precision": [], "recall": [], "hit_rate": [], "mrr": []} for k in k_list}

    for u in range(num_users):
        u_relevant = np.where(relevant_test[u])[0]
        if len(u_relevant) == 0:
            continue  # User has no positive interactions in test split

        scores = reconstructed[u].copy()
        # Exclude items already observed during training by setting their score to -inf
        scores[train_mask_np[u] == 1] = -1e9

        # Rank candidate movies descending by score
        ranked_indices = np.argsort(scores)[::-1]
        relevant_set = set(u_relevant)

        for k in k_list:
            top_k = ranked_indices[:k]
            hits = [idx for idx in top_k if idx in relevant_set]
            num_hits = len(hits)

            # Precision@K
            p_at_k = num_hits / float(k)
            # Recall@K
            r_at_k = num_hits / float(len(relevant_set))
            # Hit Rate@K
            hr_at_k = 1.0 if num_hits > 0 else 0.0

            # MRR@K
            reciprocal_rank = 0.0
            for rank_pos, idx in enumerate(top_k, start=1):
                if idx in relevant_set:
                    reciprocal_rank = 1.0 / rank_pos
                    break

            metrics_by_k[k]["precision"].append(p_at_k)
            metrics_by_k[k]["recall"].append(r_at_k)
            metrics_by_k[k]["hit_rate"].append(hr_at_k)
            metrics_by_k[k]["mrr"].append(reciprocal_rank)

    # Average across all evaluated test users
    summary = {}
    for k in k_list:
        summary[f"K={k}"] = {
            "Precision@K": round(float(np.mean(metrics_by_k[k]["precision"])), 4),
            "Recall@K": round(float(np.mean(metrics_by_k[k]["recall"])), 4),
            "Hit_Rate@K": round(float(np.mean(metrics_by_k[k]["hit_rate"])), 4),
            "MRR@K": round(float(np.mean(metrics_by_k[k]["mrr"])), 4),
        }

    return summary


def evaluate_rbm_model(
    model: RBM,
    train_data: torch.Tensor,
    train_mask: torch.Tensor,
    test_data: torch.Tensor,
    test_mask: torch.Tensor,
    k_list: List[int] = [5, 10, 15, 20]
) -> Dict:
    """
    Comprehensive evaluation pipeline combining error metrics and ranking metrics.
    """
    recon_metrics = compute_reconstruction_metrics(
        model, test_data, test_mask, train_data, k_gibbs=1
    )
    ranking_metrics = compute_ranking_metrics_at_k(
        model, train_data, train_mask, test_data, test_mask, k_list=k_list, k_gibbs=1
    )

    return {
        "reconstruction": recon_metrics,
        "ranking": ranking_metrics
    }

