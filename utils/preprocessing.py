"""
Preprocessing utilities for user ratings, tensor conversion,
score transformations, and recommendation explanation generation.
"""

from typing import Dict, List, Tuple
import torch
import numpy as np
import pandas as pd


def ratings_dict_to_tensor(
    user_ratings: Dict[int, float],
    num_movies: int = 1682,
    rating_threshold: float = 3.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Converts a dictionary of {movie_id (1-indexed): rating (1.0 - 5.0)}
    into (v_input, mask) tensors of shape (1, num_movies).
    
    - v_input[0, i] = 1.0 if user rated movie (i+1) >= rating_threshold, else 0.0
    - mask[0, i] = 1.0 if user rated movie (i+1), else 0.0
    """
    v = np.zeros((1, num_movies), dtype=np.float32)
    mask = np.zeros((1, num_movies), dtype=np.float32)

    for movie_id, rating in user_ratings.items():
        idx = int(movie_id) - 1
        if 0 <= idx < num_movies:
            mask[0, idx] = 1.0
            if rating >= rating_threshold:
                v[0, idx] = 1.0
            else:
                v[0, idx] = 0.0

    return torch.from_numpy(v), torch.from_numpy(mask)


def binary_score_to_stars(prob: float) -> float:
    """
    Maps RBM activation probability P(v_i=1|h) in [0, 1] to 1-5 star scale:
      star_rating = 1.0 + 4.0 * prob
    Example:
      prob = 0.90 -> 1.0 + 3.6 = 4.6 / 5.0 stars
      prob = 0.50 -> 1.0 + 2.0 = 3.0 / 5.0 stars
    """
    clamped = max(0.0, min(1.0, float(prob)))
    stars = 1.0 + 4.0 * clamped
    return round(stars, 1)


def generate_recommendation_reason(
    rec_movie_id: int,
    user_ratings: Dict[int, float],
    df_movies: pd.DataFrame
) -> str:
    """
    Generates a natural language explanation for why a movie was recommended,
    based on genre overlap and high ratings previously provided by the user.
    """
    if not user_ratings:
        return "Popular among all MovieLens viewers (Baseline Discovery)."

    # Look up recommended movie's genres
    rec_rows = df_movies[df_movies["movie_id"] == rec_movie_id]
    if rec_rows.empty:
        return "High latent preference match from RBM hidden representation."

    rec_genres = set(rec_rows.iloc[0]["genres_list"])

    # Find user's highest rated movies
    liked_movies = [m_id for m_id, r in user_ratings.items() if r >= 4.0]
    if not liked_movies:
        liked_movies = list(user_ratings.keys())

    best_match_title = None
    best_overlap = 0
    common_genres = set()

    for m_id in liked_movies:
        m_rows = df_movies[df_movies["movie_id"] == m_id]
        if not m_rows.empty:
            m_genres = set(m_rows.iloc[0]["genres_list"])
            overlap = rec_genres.intersection(m_genres)
            if len(overlap) > best_overlap:
                best_overlap = len(overlap)
                best_match_title = m_rows.iloc[0]["title"]
                common_genres = overlap

    if best_match_title and common_genres:
        genres_str = "/".join(list(common_genres)[:2])
        return f"Because you enjoyed '{best_match_title}' (Shared {genres_str} appeal)."
    elif rec_genres:
        top_genres = "/".join(list(rec_genres)[:2])
        return f"Matches your latent affinity for {top_genres} films."
    else:
        return "Strong collaborative signal from users with similar taste patterns."


KEYWORD_GENRE_MAP = {
    "action": ["Action"],
    "fight": ["Action"],
    "explosive": ["Action"],
    "martial": ["Action"],
    "adventure": ["Adventure"],
    "journey": ["Adventure"],
    "quest": ["Adventure"],
    "space": ["Sci-Fi", "Adventure"],
    "alien": ["Sci-Fi", "Horror"],
    "galaxy": ["Sci-Fi", "Adventure"],
    "star": ["Sci-Fi"],
    "future": ["Sci-Fi"],
    "futuristic": ["Sci-Fi"],
    "robot": ["Sci-Fi"],
    "cyberpunk": ["Sci-Fi"],
    "time travel": ["Sci-Fi"],
    "sci-fi": ["Sci-Fi"],
    "scifi": ["Sci-Fi"],
    "animation": ["Animation", "Children's"],
    "animated": ["Animation", "Children's"],
    "cartoon": ["Animation", "Children's"],
    "disney": ["Animation", "Children's"],
    "pixar": ["Animation", "Children's"],
    "kids": ["Children's", "Animation"],
    "children": ["Children's", "Animation"],
    "family": ["Children's", "Animation", "Comedy"],
    "comedy": ["Comedy"],
    "funny": ["Comedy"],
    "laugh": ["Comedy"],
    "humor": ["Comedy"],
    "hilarious": ["Comedy"],
    "parody": ["Comedy"],
    "crime": ["Crime"],
    "gangster": ["Crime"],
    "mafia": ["Crime"],
    "robbery": ["Crime"],
    "heist": ["Crime", "Action"],
    "detective": ["Mystery", "Crime", "Film-Noir"],
    "police": ["Crime"],
    "cop": ["Crime"],
    "noir": ["Film-Noir", "Mystery"],
    "drama": ["Drama"],
    "emotional": ["Drama"],
    "touching": ["Drama"],
    "heartfelt": ["Drama"],
    "sad": ["Drama"],
    "inspiring": ["Drama"],
    "documentary": ["Documentary"],
    "history": ["Documentary", "Drama"],
    "fantasy": ["Fantasy"],
    "magic": ["Fantasy"],
    "wizard": ["Fantasy"],
    "horror": ["Horror"],
    "scary": ["Horror"],
    "ghost": ["Horror"],
    "spooky": ["Horror"],
    "monster": ["Horror"],
    "zombie": ["Horror"],
    "nightmare": ["Horror"],
    "creepy": ["Horror"],
    "musical": ["Musical"],
    "music": ["Musical"],
    "singing": ["Musical"],
    "dance": ["Musical"],
    "mystery": ["Mystery"],
    "puzzle": ["Mystery", "Thriller"],
    "twist": ["Mystery", "Thriller"],
    "whodunit": ["Mystery"],
    "romance": ["Romance"],
    "romantic": ["Romance"],
    "love": ["Romance"],
    "couple": ["Romance"],
    "dating": ["Romance", "Comedy"],
    "thriller": ["Thriller"],
    "suspense": ["Thriller"],
    "psychological": ["Thriller", "Mystery"],
    "tense": ["Thriller"],
    "mind-bending": ["Sci-Fi", "Thriller", "Mystery"],
    "war": ["War"],
    "soldier": ["War"],
    "army": ["War"],
    "military": ["War"],
    "wwii": ["War"],
    "western": ["Western"],
    "cowboy": ["Western"],
    "outlaw": ["Western"]
}


def search_suggestions_by_prompt(
    prompt: str,
    df_movies: pd.DataFrame,
    df_ratings: Optional[pd.DataFrame] = None,
    user_ratings: Optional[Dict[int, float]] = None,
    engine: Optional[object] = None,
    top_n: int = 8
) -> List[Dict]:
    """
    Intelligent conversational suggestion search:
    Parses natural language prompts (mood, keywords, titles) into genre affinities
    and title matching, then blends with RBM latent predictions if available.
    """
    if not prompt or not prompt.strip():
        # Default to popular movies
        sample_df = df_movies.head(top_n)
        return [
            {
                "movie_id": int(row["movie_id"]),
                "title": row["title"],
                "clean_title": row.get("clean_title", row["title"]),
                "year": row.get("year", "N/A"),
                "genres": row["genres_str"],
                "genres_list": row["genres_list"],
                "match_reason": "Popular title in MovieLens 100K catalog",
                "predicted_stars": 4.5,
                "score": 0.85
            }
            for _, row in sample_df.iterrows()
        ]

    prompt_clean = prompt.lower().strip()
    tokens = set(prompt_clean.replace(",", " ").replace(".", " ").replace("!", " ").split())

    # Detect target genres from keyword mapping
    matched_genres = set()
    for kw, genres in KEYWORD_GENRE_MAP.items():
        if kw in prompt_clean or any(t == kw for t in tokens):
            matched_genres.update(genres)

    # Compute pre-calculated popularity lookup if df_ratings is given
    pop_counts = {}
    pop_means = {}
    if df_ratings is not None:
        stats = df_ratings.groupby("item_id")["rating"].agg(["count", "mean"])
        for item_id, row in stats.iterrows():
            pop_counts[int(item_id)] = int(row["count"])
            pop_means[int(item_id)] = float(row["mean"])

    # If RBM engine is available and user has ratings, get personalized probs
    rbm_scores = {}
    if engine is not None and getattr(engine, "is_trained", False) and user_ratings:
        try:
            v_input, mask = ratings_dict_to_tensor(user_ratings, num_movies=len(df_movies))
            p_v = engine.model.reconstruct(v_input, mask=mask, k=1)
            rbm_probs = p_v.squeeze(0).numpy()
            for idx, p in enumerate(rbm_probs):
                rbm_scores[idx + 1] = float(p)
        except Exception:
            rbm_scores = {}

    candidates = []
    rated_set = set(user_ratings.keys()) if user_ratings else set()

    for _, row in df_movies.iterrows():
        m_id = int(row["movie_id"])
        title_lower = row["title"].lower()
        clean_title_lower = str(row.get("clean_title", "")).lower()
        genres = set(row.get("genres_list", []))

        # Base score components
        score = 0.0
        reasons = []

        # 1. Exact title or keyword match in title
        if prompt_clean in title_lower or prompt_clean in clean_title_lower:
            score += 15.0
            reasons.append(f"Title directly matches '{prompt}'")
        else:
            title_token_matches = [t for t in tokens if len(t) > 3 and t in title_lower]
            if title_token_matches:
                score += 5.0 * len(title_token_matches)
                reasons.append(f"Matches title keywords ({', '.join(title_token_matches)})")

        # 2. Genre overlap with prompt
        overlap = matched_genres.intersection(genres)
        if overlap:
            score += 3.5 * len(overlap)
            reasons.append(f"Matches requested mood/genres ({'/'.join(list(overlap)[:2])})")

        # 3. Community rating boost (MovieLens popularity)
        count = pop_counts.get(m_id, 0)
        mean_r = pop_means.get(m_id, 3.0)
        if count >= 30:
            score += min(3.0, (count / 200.0)) + (mean_r / 2.0)

        # 4. RBM Personalized Latent Score boost
        if m_id in rbm_scores:
            prob = rbm_scores[m_id]
            score += prob * 4.0
            if prob > 0.75:
                reasons.append("High latent compatibility with your taste")

        if score > 1.5:
            match_explanation = " & ".join(reasons) if reasons else "Matches your search criteria"
            predicted_stars = binary_score_to_stars(rbm_scores.get(m_id, min(1.0, max(0.2, score / 20.0))))
            candidates.append({
                "movie_id": m_id,
                "title": row["title"],
                "clean_title": row.get("clean_title", row["title"]),
                "year": row.get("year", "N/A"),
                "genres": row["genres_str"],
                "genres_list": row.get("genres_list", []),
                "match_reason": match_explanation,
                "predicted_stars": predicted_stars,
                "score": score,
                "rating_mean": pop_means.get(m_id, 3.5),
                "rating_count": pop_counts.get(m_id, 0),
                "is_rated": m_id in rated_set
            })

    # Sort descending by composite score
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_n]


