"""
Recommendation Engine module for RBM-based movie recommender.
Converts user ratings into model input, runs Gibbs inference,
ranks unseen items, and formats recommendations with metadata and explanations.
"""

from typing import Dict, List, Optional
import os
import torch
import pandas as pd
import numpy as np

from .rbm import RBM
from utils.data_loader import load_movie_metadata, get_popular_movies, DEFAULT_DATA_DIR
from utils.preprocessing import (
    ratings_dict_to_tensor,
    binary_score_to_stars,
    generate_recommendation_reason
)

DEFAULT_MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "saved_models", "rbm_movielens100k.pt")
)


class RecommenderEngine:
    """
    Inference and ranking engine powered by a trained Restricted Boltzmann Machine.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        data_dir: str = DEFAULT_DATA_DIR,
        device: Optional[str] = None
    ):
        self.data_dir = data_dir
        self.df_movies = load_movie_metadata(data_dir)
        self.num_movies = len(self.df_movies)

        if os.path.exists(model_path):
            self.model = RBM.load_checkpoint(model_path, device=device)
            self.is_trained = True
        else:
            self.model = RBM(num_visible=self.num_movies, num_hidden=100, device=device)
            self.is_trained = False

    def recommend(
        self,
        user_ratings: Dict[int, float],
        top_n: int = 10,
        k_gibbs: int = 1,
        genre_filter: str = "All"
    ) -> List[Dict]:
        """
        Generates top_n movie recommendations for a user given their rated movies.

        Parameters
        ----------
        user_ratings : Dict[int, float]
            Mapping of {movie_id (1-indexed): rating (1.0 - 5.0)}
        top_n : int
            Number of recommendations to return
        k_gibbs : int
            Number of Gibbs sampling steps
        genre_filter : str
            Optional genre filter (e.g. "Action", "Comedy", "All")

        Returns
        -------
        List[Dict] of recommended movies with titles, genres, scores, and explanations.
        """
        # If user has not rated any movies, fall back to popular high-rated items (cold-start)
        if not user_ratings:
            return self._cold_start_recommendations(top_n=top_n, genre_filter=genre_filter)

        # Convert user ratings to RBM visible vector (1, num_movies)
        v_input, mask = ratings_dict_to_tensor(user_ratings, num_movies=self.num_movies)

        # Run RBM reconstruction inference
        p_v = self.model.reconstruct(v_input, mask=mask, k=k_gibbs)
        probabilities = p_v.squeeze(0).numpy()

        # Build candidate list excluding already rated movies
        rated_ids = set(user_ratings.keys())
        candidates = []

        for m_idx, prob in enumerate(probabilities):
            movie_id = m_idx + 1  # 1-indexed movie ID
            if movie_id in rated_ids:
                continue

            # Look up movie metadata
            row = self.df_movies[self.df_movies["movie_id"] == movie_id]
            if row.empty:
                continue
            item = row.iloc[0]

            # Genre filter if specified
            if genre_filter != "All" and genre_filter not in item["genres_list"]:
                continue

            candidates.append({
                "movie_id": movie_id,
                "title": item["title"],
                "clean_title": item.get("clean_title", item["title"]),
                "year": item.get("year", "N/A"),
                "genres": item["genres_str"],
                "genres_list": item["genres_list"],
                "score": float(prob),
                "affinity_pct": round(float(prob) * 100, 1),
                "predicted_stars": binary_score_to_stars(prob)
            })

        # Sort candidates descending by predicted probability score
        candidates.sort(key=lambda x: x["score"], reverse=True)
        top_candidates = candidates[:top_n]

        # Attach personalized rationale and ranking
        for rank, rec in enumerate(top_candidates, start=1):
            rec["rank"] = rank
            rec["explanation"] = generate_recommendation_reason(
                rec["movie_id"], user_ratings, self.df_movies
            )

        return top_candidates

    def _cold_start_recommendations(
        self,
        top_n: int = 10,
        genre_filter: str = "All"
    ) -> List[Dict]:
        """
        Returns top popular movies for cold-start users who haven't rated anything.
        """
        df = self.df_movies
        if genre_filter != "All":
            df = df[df[genre_filter] == 1]

        top_df = df.head(top_n)
        results = []
        for rank, (_, row) in enumerate(top_df.iterrows(), start=1):
            results.append({
                "rank": rank,
                "movie_id": int(row["movie_id"]),
                "title": row["title"],
                "clean_title": row.get("clean_title", row["title"]),
                "year": row.get("year", "N/A"),
                "genres": row["genres_str"],
                "genres_list": row["genres_list"],
                "score": 0.5,
                "affinity_pct": 50.0,
                "predicted_stars": 3.0,
                "explanation": "Cold-start baseline: Popular classic from the MovieLens collection."
            })
        return results

