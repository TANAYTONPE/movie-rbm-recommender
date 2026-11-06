"""
Restricted Boltzmann Machine (RBM) implementation in PyTorch from scratch.

Features:
- Energy-based stochastic neural network
- Visible layer for movie ratings & Hidden layer for latent preferences
- Sigmoid activation with Bernoulli stochastic sampling
- Gibbs Sampling and Contrastive Divergence (CD-k)
- Conditional masking support for sparse collaborative filtering
- Exact Free Energy computation and parameter checkpointing
"""

import os
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn


class RBM(nn.Module):
    """
    Restricted Boltzmann Machine (RBM) for Collaborative Filtering.

    Parameters
    ----------
    num_visible : int
        Number of visible units (movies, e.g. 1682).
    num_hidden : int
        Number of hidden units representing latent preference factors (e.g. 100).
    learning_rate : float
        Learning rate for parameter updates.
    momentum : float
        Momentum coefficient for SGD updates.
    weight_decay : float
        L2 regularization penalty.
    k : int
        Number of Gibbs sampling steps in Contrastive Divergence (CD-k).
    """

    def __init__(
        self,
        num_visible: int = 1682,
        num_hidden: int = 100,
        learning_rate: float = 0.05,
        momentum: float = 0.8,
        weight_decay: float = 0.0002,
        k: int = 1,
        device: Optional[str] = None
    ):
        super(RBM, self).__init__()
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.k = k

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize weights with standard Gaussian scaled by sqrt(2 / (N_v + N_h))
        std = (2.0 / (num_visible + num_hidden)) ** 0.5
        self.W = nn.Parameter(torch.randn(num_visible, num_hidden, device=self.device) * std)
        self.v_bias = nn.Parameter(torch.zeros(num_visible, device=self.device))
        self.h_bias = nn.Parameter(torch.zeros(num_hidden, device=self.device))

        # Velocity buffers for momentum updates
        self.register_buffer("W_vel", torch.zeros_like(self.W))
        self.register_buffer("v_bias_vel", torch.zeros_like(self.v_bias))
        self.register_buffer("h_bias_vel", torch.zeros_like(self.h_bias))

        self.to(self.device)

    def sample_h(self, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Positive phase / Hidden activation:
        Computes conditional probability P(h_j = 1 | v) and samples binary states.

        Formula:
            P(h_j = 1 | v) = sigmoid(b_j + sum_i(v_i * W_ij))
        """
        # Shape: (batch_size, num_hidden)
        p_h = torch.sigmoid(torch.matmul(v, self.W) + self.h_bias)
        # Bernoulli sampling: h_sample ~ Bernoulli(p_h)
        h_sample = torch.bernoulli(p_h)
        return p_h, h_sample

    def sample_v(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Negative phase / Visible reconstruction:
        Computes conditional probability P(v_i = 1 | h) and samples binary states.

        Formula:
            P(v_i = 1 | h) = sigmoid(a_i + sum_j(h_j * W_ij))
        """
        # Shape: (batch_size, num_visible)
        p_v = torch.sigmoid(torch.matmul(h, self.W.t()) + self.v_bias)
        # Bernoulli sampling: v_sample ~ Bernoulli(p_v)
        v_sample = torch.bernoulli(p_v)
        return p_v, v_sample

    def gibbs_step(
        self,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs a single step of Gibbs Sampling: v -> h -> v' -> h'

        Parameters
        ----------
        v : torch.Tensor
            Current visible states, shape (batch_size, num_visible).
        mask : Optional[torch.Tensor]
            Observation mask (1 for rated, 0 for missing).
            If provided, unobserved units are clamped to 0 so they don't introduce noise.

        Returns
        -------
        p_h : probabilities of hidden layer
        h_sample : sampled hidden binary states
        p_v : reconstructed visible probabilities
        v_sample : sampled visible binary states
        """
        p_h, h_sample = self.sample_h(v)
        p_v, v_sample = self.sample_v(h_sample)

        # Apply mask to prevent hallucinated gradients on unrated items
        if mask is not None:
            v_sample = v_sample * mask
            p_v = p_v * mask

        p_h_prime, h_sample_prime = self.sample_h(v_sample)
        return p_h, h_sample, p_v, v_sample

    def contrastive_divergence(
        self,
        v_data: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        k: Optional[int] = None
    ) -> float:
        """
        Performs Contrastive Divergence (CD-k) learning step.

        1. Positive phase: compute P(h_0 | v_0) and sample h_0.
        2. Run Gibbs chain for k steps: v_0 -> h_0 -> v_1 -> ... -> v_k -> h_k.
        3. Parameter update:
             Delta W = (v_0^T h_0 - v_k^T h_k) / batch_size - weight_decay * W
             Delta a = sum(v_0 - v_k) / batch_size
             Delta b = sum(h_0 - h_k) / batch_size
        """
        if k is None:
            k = self.k

        v_data = v_data.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)

        batch_size = v_data.size(0)

        # Positive phase
        p_h0, h0 = self.sample_h(v_data)

        # Negative phase (Gibbs chain for k steps)
        vk = v_data.clone()
        hk = h0.clone()

        for _ in range(k):
            p_vk, vk = self.sample_v(hk)
            if mask is not None:
                vk = vk * mask
            p_hk, hk = self.sample_h(vk)

        # Positive and negative statistics (outer products)
        # Use probabilities for hidden units to reduce sampling variance (standard Hinton rule)
        pos_grad_w = torch.matmul(v_data.t(), p_h0)
        neg_grad_w = torch.matmul(vk.t(), p_hk)

        # Gradient signals
        grad_W = (pos_grad_w - neg_grad_w) / batch_size
        grad_v = torch.sum(v_data - vk, dim=0) / batch_size
        grad_h = torch.sum(p_h0 - p_hk, dim=0) / batch_size

        # Apply weight decay to W
        grad_W = grad_W - self.weight_decay * self.W.data

        # Update velocities with momentum
        with torch.no_grad():
            self.W_vel = self.momentum * self.W_vel + self.learning_rate * grad_W
            self.v_bias_vel = self.momentum * self.v_bias_vel + self.learning_rate * grad_v
            self.h_bias_vel = self.momentum * self.h_bias_vel + self.learning_rate * grad_h

            # Update weights and biases
            self.W.data += self.W_vel
            self.v_bias.data += self.v_bias_vel
            self.h_bias.data += self.h_bias_vel

        # Compute batch reconstruction error on observed items
        with torch.no_grad():
            if mask is not None:
                rated_count = torch.sum(mask).item()
                if rated_count > 0:
                    diff = (v_data - p_vk) * mask
                    recon_error = torch.sum(diff ** 2).item() / rated_count
                else:
                    recon_error = 0.0
            else:
                recon_error = torch.mean((v_data - p_vk) ** 2).item()

        return float(recon_error)

    def energy(self, v: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Computes joint energy E(v, h) = -a^T v - b^T h - v^T W h
        """
        v = v.to(self.device)
        h = h.to(self.device)
        v_term = torch.matmul(v, self.v_bias)
        h_term = torch.matmul(h, self.h_bias)
        vh_term = torch.sum(torch.matmul(v, self.W) * h, dim=1)
        return -v_term - h_term - vh_term

    def free_energy(self, v: torch.Tensor) -> torch.Tensor:
        """
        Computes exact marginal Free Energy F(v):
            F(v) = -a^T v - sum_j log(1 + exp(b_j + sum_i v_i W_ij))
        """
        v = v.to(self.device)
        v_term = torch.matmul(v, self.v_bias)
        # Log-Sum-Exp trick for numerical stability
        wx_b = torch.matmul(v, self.W) + self.h_bias
        h_term = torch.sum(torch.nn.functional.softplus(wx_b), dim=1)
        return -v_term - h_term

    def reconstruct(
        self,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        k: int = 1
    ) -> torch.Tensor:
        """
        Inference / Recommendation generation:
        Runs Gibbs sampling starting from visible input v and returns predicted
        probabilities P(v_i = 1 | h) across all movies.
        """
        self.eval()
        v = v.to(self.device)
        with torch.no_grad():
            # Initial forward pass to hidden layer
            p_h, h = self.sample_h(v)
            p_v = None

            for _ in range(k):
                p_v, v_sample = self.sample_v(h)
                if mask is not None:
                    # Keep user's known inputs clamped during sampling
                    v_sample = v * mask + v_sample * (1.0 - mask)
                p_h, h = self.sample_h(v_sample)

            # Final reconstruction using probability values for smooth preference ranking
            final_p_v, _ = self.sample_v(p_h)

        return final_p_v.cpu()

    def save_checkpoint(self, path: str, extra_meta: Optional[Dict] = None):
        """
        Saves model weights, architecture hyperparameters, and training metadata.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            "num_visible": self.num_visible,
            "num_hidden": self.num_hidden,
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "k": self.k,
            "state_dict": self.state_dict(),
            "metadata": extra_meta or {}
        }
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path: str, device: Optional[str] = None) -> "RBM":
        """
        Loads a saved RBM checkpoint from disk.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        map_location = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)

        model = cls(
            num_visible=checkpoint["num_visible"],
            num_hidden=checkpoint["num_hidden"],
            learning_rate=checkpoint.get("learning_rate", 0.05),
            momentum=checkpoint.get("momentum", 0.8),
            weight_decay=checkpoint.get("weight_decay", 0.0002),
            k=checkpoint.get("k", 1),
            device=map_location
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.metadata = checkpoint.get("metadata", {})
        return model

