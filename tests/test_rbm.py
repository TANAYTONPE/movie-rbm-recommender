"""
Automated unit tests for Restricted Boltzmann Machine (RBM),
Data Loader, Contrastive Divergence, and Recommendation Engine.
"""

import os
import sys
import unittest
import torch
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.rbm import RBM
from models.recommend import RecommenderEngine
from utils.data_loader import load_movie_metadata, build_rating_matrix, DEFAULT_DATA_DIR
from utils.preprocessing import (
    ratings_dict_to_tensor,
    binary_score_to_stars,
    search_suggestions_by_prompt
)


class TestRBM(unittest.TestCase):
    """Test suite for RBM implementation and components."""

    def setUp(self):
        self.num_visible = 50
        self.num_hidden = 10
        self.rbm = RBM(
            num_visible=self.num_visible,
            num_hidden=self.num_hidden,
            learning_rate=0.05,
            device="cpu"
        )

    def test_parameter_shapes(self):
        """Verify weights and bias tensors have expected dimensions."""
        self.assertEqual(self.rbm.W.shape, (self.num_visible, self.num_hidden))
        self.assertEqual(self.rbm.v_bias.shape, (self.num_visible,))
        self.assertEqual(self.rbm.h_bias.shape, (self.num_hidden,))

    def test_sample_h(self):
        """Hidden layer activation should produce probabilities in [0, 1] and binary samples."""
        batch_size = 4
        v = torch.bernoulli(torch.full((batch_size, self.num_visible), 0.5))
        p_h, h_sample = self.rbm.sample_h(v)

        self.assertEqual(p_h.shape, (batch_size, self.num_hidden))
        self.assertEqual(h_sample.shape, (batch_size, self.num_hidden))
        self.assertTrue(torch.all(p_h >= 0.0) and torch.all(p_h <= 1.0))

        # Check binary values
        unique_vals = set(h_sample.detach().cpu().unique().numpy().tolist())
        self.assertTrue(unique_vals.issubset({0.0, 1.0}))

    def test_sample_v(self):
        """Visible reconstruction should produce probabilities in [0, 1] and binary samples."""
        batch_size = 4
        h = torch.bernoulli(torch.full((batch_size, self.num_hidden), 0.5))
        p_v, v_sample = self.rbm.sample_v(h)

        self.assertEqual(p_v.shape, (batch_size, self.num_visible))
        self.assertEqual(v_sample.shape, (batch_size, self.num_visible))
        self.assertTrue(torch.all(p_v >= 0.0) and torch.all(p_v <= 1.0))

        unique_vals = set(v_sample.detach().cpu().unique().numpy().tolist())
        self.assertTrue(unique_vals.issubset({0.0, 1.0}))

    def test_gibbs_step(self):
        """Gibbs step should preserve tensor shapes."""
        batch_size = 2
        v = torch.bernoulli(torch.full((batch_size, self.num_visible), 0.5))
        p_h, h_s, p_v, v_s = self.rbm.gibbs_step(v)

        self.assertEqual(p_h.shape, (batch_size, self.num_hidden))
        self.assertEqual(p_v.shape, (batch_size, self.num_visible))

    def test_free_energy(self):
        """Free energy should compute a finite scalar per user."""
        v = torch.tensor([[1.0] * self.num_visible, [0.0] * self.num_visible])
        f = self.rbm.free_energy(v)
        self.assertEqual(f.shape, (2,))
        self.assertFalse(torch.isnan(f).any())
        self.assertFalse(torch.isinf(f).any())

    def test_contrastive_divergence_updates_weights(self):
        """CD-1 step must update weights and return positive reconstruction error."""
        W_before = self.rbm.W.clone()
        v = torch.bernoulli(torch.full((8, self.num_visible), 0.4))
        mask = torch.ones_like(v)

        error = self.rbm.contrastive_divergence(v, mask=mask, k=1)
        self.assertGreaterEqual(error, 0.0)
        self.assertFalse(torch.equal(W_before, self.rbm.W))

    def test_data_loader(self):
        """Verify MovieLens metadata loading."""
        df_movies = load_movie_metadata(DEFAULT_DATA_DIR)
        self.assertEqual(len(df_movies), 1682)
        self.assertIn("title", df_movies.columns)
        self.assertIn("genres_str", df_movies.columns)
        self.assertIn("genres_list", df_movies.columns)

    def test_ratings_dict_to_tensor(self):
        """Verify dictionary conversion to PyTorch tensors."""
        user_ratings = {1: 5.0, 50: 1.0}
        v, mask = ratings_dict_to_tensor(user_ratings, num_movies=1682, rating_threshold=3.0)

        self.assertEqual(v.shape, (1, 1682))
        self.assertEqual(mask.shape, (1, 1682))
        # Movie 1 rated 5 (>= 3.0) -> v=1, mask=1
        self.assertEqual(v[0, 0].item(), 1.0)
        self.assertEqual(mask[0, 0].item(), 1.0)
        # Movie 50 rated 1 (< 3.0) -> v=0, mask=1
        self.assertEqual(v[0, 49].item(), 0.0)
        self.assertEqual(mask[0, 49].item(), 1.0)
        # Movie 2 unrated -> v=0, mask=0
        self.assertEqual(v[0, 1].item(), 0.0)
        self.assertEqual(mask[0, 1].item(), 0.0)

    def test_binary_score_to_stars(self):
        """Verify score mapping formula."""
        self.assertEqual(binary_score_to_stars(1.0), 5.0)
        self.assertEqual(binary_score_to_stars(0.0), 1.0)
        self.assertEqual(binary_score_to_stars(0.5), 3.0)

    def test_recommender_engine(self):
        """Verify recommender excludes rated movies and returns requested top_n."""
        engine = RecommenderEngine(data_dir=DEFAULT_DATA_DIR)
        user_ratings = {1: 5.0, 2: 4.0, 50: 5.0}  # Toy Story, GoldenEye, Star Wars
        recs = engine.recommend(user_ratings, top_n=5)

        self.assertEqual(len(recs), 5)
        rec_ids = [r["movie_id"] for r in recs]
        # Exclude rated movies
        for rated_id in [1, 2, 50]:
            self.assertNotIn(rated_id, rec_ids)

        for r in recs:
            self.assertIn("title", r)
            self.assertIn("genres", r)
            self.assertIn("affinity_pct", r)
            self.assertIn("predicted_stars", r)
            self.assertIn("explanation", r)
            self.assertTrue(0.0 <= r["score"] <= 1.0)

    def test_search_suggestions_by_prompt(self):
        """Verify conversational movie suggestion search."""
        df_movies = load_movie_metadata(DEFAULT_DATA_DIR)
        suggestions = search_suggestions_by_prompt(
            prompt="scary horror ghost story",
            df_movies=df_movies,
            top_n=5
        )
        self.assertGreater(len(suggestions), 0)
        self.assertLessEqual(len(suggestions), 5)
        for s in suggestions:
            self.assertIn("title", s)
            self.assertIn("genres_list", s)
            self.assertIn("match_reason", s)
            self.assertIn("predicted_stars", s)


if __name__ == "__main__":
    unittest.main()
