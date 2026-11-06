"""
Training script for Restricted Boltzmann Machine on MovieLens 100K.
Executes Contrastive Divergence (CD-k), monitors reconstruction error,
evaluates on test split, and persists the trained checkpoint.
"""

import os
import sys
import json
import time
from typing import Dict, List, Optional
import torch
import numpy as np

# Ensure project root is in python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.rbm import RBM
from models.evaluation import evaluate_rbm_model
from utils.data_loader import load_train_test_matrices, DEFAULT_DATA_DIR

DEFAULT_MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "saved_models", "rbm_movielens100k.pt")
DEFAULT_HISTORY_SAVE_PATH = os.path.join(PROJECT_ROOT, "saved_models", "training_history.json")


def train_rbm(
    data_dir: str = DEFAULT_DATA_DIR,
    num_hidden: int = 100,
    epochs: int = 25,
    batch_size: int = 64,
    learning_rate: float = 0.05,
    momentum: float = 0.8,
    weight_decay: float = 0.0001,
    k_cd: int = 1,
    split_num: int = 1,
    model_save_path: str = DEFAULT_MODEL_SAVE_PATH,
    history_save_path: str = DEFAULT_HISTORY_SAVE_PATH,
    device: Optional[str] = None
) -> Dict:
    """
    Trains RBM on MovieLens 100K dataset using Contrastive Divergence (CD-1).
    """
    print("=" * 65)
    print("  MOVIE RECOMMENDATION SYSTEM: RBM TRAINING PIPELINE")
    print("=" * 65)
    print(f"Loading MovieLens 100K (Split {split_num})...")

    (train_data, train_mask, _), (test_data, test_mask, _) = load_train_test_matrices(
        data_dir=data_dir, split_num=split_num
    )

    num_users, num_movies = train_data.shape
    print(f"Dataset stats: {num_users} users, {num_movies} visible units (movies)")
    print(f"Hyperparameters: Hidden units={num_hidden}, Epochs={epochs}, Batch={batch_size}, LR={learning_rate}")
    print(f"Contrastive Divergence: CD-{k_cd}, Momentum={momentum}, Weight Decay={weight_decay}")

    # Initialize model
    model = RBM(
        num_visible=num_movies,
        num_hidden=num_hidden,
        learning_rate=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        k=k_cd,
        device=device
    )

    history = {
        "epochs": [],
        "train_loss": [],
        "test_rmse": [],
        "epoch_time_sec": []
    }

    start_time = time.time()
    num_batches = int(np.ceil(num_users / batch_size))

    print("-" * 65)
    print(f"{'Epoch':<8} | {'Train Error (MSE)':<18} | {'Test RMSE':<14} | {'Time (s)':<10}")
    print("-" * 65)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        # Shuffle users each epoch
        perm = torch.randperm(num_users)
        train_data_shuffled = train_data[perm]
        train_mask_shuffled = train_mask[perm]

        epoch_errors = []
        for b in range(num_batches):
            start_idx = b * batch_size
            end_idx = min(start_idx + batch_size, num_users)

            v_batch = train_data_shuffled[start_idx:end_idx]
            mask_batch = train_mask_shuffled[start_idx:end_idx]

            batch_error = model.contrastive_divergence(v_batch, mask=mask_batch, k=k_cd)
            epoch_errors.append(batch_error)

        avg_train_loss = float(np.mean(epoch_errors))

        # Evaluate on test set
        with torch.no_grad():
            reconstructed_test = model.reconstruct(train_data, k=1).cpu()
            test_mask_cpu = test_mask.cpu()
            test_data_cpu = test_data.cpu()
            rated_count = torch.sum(test_mask_cpu).item()
            if rated_count > 0:
                diff = (test_data_cpu - reconstructed_test) * test_mask_cpu
                test_rmse = float(np.sqrt(torch.sum(diff ** 2).item() / rated_count))
            else:
                test_rmse = 0.0

        elapsed = time.time() - t0

        history["epochs"].append(epoch)
        history["train_loss"].append(round(avg_train_loss, 5))
        history["test_rmse"].append(round(test_rmse, 5))
        history["epoch_time_sec"].append(round(elapsed, 2))

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"{epoch:<8} | {avg_train_loss:<18.5f} | {test_rmse:<14.5f} | {elapsed:<10.2f}")

    total_training_time = time.time() - start_time
    print("-" * 65)
    print(f"Training completed in {total_training_time:.2f} seconds.")

    # Full evaluation
    print("\nRunning comprehensive model evaluation on test split...")
    eval_results = evaluate_rbm_model(
        model, train_data, train_mask, test_data, test_mask, k_list=[5, 10, 15, 20]
    )

    print("\n--- Test Set Reconstruction Metrics ---")
    for k, v in eval_results["reconstruction"].items():
        print(f"  {k}: {v}")

    print("\n--- Test Set Top-K Ranking Metrics ---")
    print(f"{'Metric':<14} | {'K=5':<10} | {'K=10':<10} | {'K=15':<10} | {'K=20':<10}")
    print("-" * 58)
    for metric_name in ["Precision@K", "Recall@K", "Hit_Rate@K", "MRR@K"]:
        row = [f"{eval_results['ranking'][f'K={k}'][metric_name]:.4f}" for k in [5, 10, 15, 20]]
        print(f"{metric_name:<14} | {row[0]:<10} | {row[1]:<10} | {row[2]:<10} | {row[3]:<10}")

    # Save model checkpoint
    metadata = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "k_cd": k_cd,
        "split_num": split_num,
        "total_training_time_sec": round(total_training_time, 2),
        "evaluation": eval_results,
        "final_train_loss": history["train_loss"][-1],
        "final_test_rmse": history["test_rmse"][-1]
    }
    model.save_checkpoint(model_save_path, extra_meta=metadata)
    print(f"\nModel checkpoint saved to: {model_save_path}")

    # Save training history JSON
    with open(history_save_path, "w") as f:
        json.dump({"history": history, "metadata": metadata}, f, indent=2)
    print(f"Training history saved to: {history_save_path}")
    print("=" * 65)

    return {"model": model, "history": history, "evaluation": eval_results}


if __name__ == "__main__":
    train_rbm()

