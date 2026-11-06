"""
Data loader module for MovieLens 100K dataset.
Handles downloading, parsing, user-item matrix construction,
and train/test splitting for the RBM collaborative filter.
"""

import os
import ssl
import urllib.request
import zipfile
import pandas as pd
import numpy as np
import torch

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
DEFAULT_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "ml-100k")
)

GENRE_NAMES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western"
]

ITEM_COLS = [
    "movie_id", "title", "release_date", "video_release_date", "imdb_url",
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]

RATING_COLS = ["user_id", "item_id", "rating", "timestamp"]


def download_dataset_if_needed(data_dir: str = DEFAULT_DATA_DIR) -> str:
    """
    Checks if MovieLens 100K files exist locally; if not, downloads and extracts them.
    Returns path to the dataset folder.
    """
    required_file = os.path.join(data_dir, "u.data")
    if os.path.exists(required_file):
        return data_dir

    parent_dir = os.path.dirname(data_dir)
    os.makedirs(parent_dir, exist_ok=True)
    zip_path = os.path.join(parent_dir, "ml-100k.zip")

    print(f"Dataset not found at {data_dir}. Downloading from {DATA_URL}...")
    try:
        ctx = ssl._create_unverified_context()
        with urllib.request.urlopen(DATA_URL, context=ctx) as response, open(zip_path, "wb") as out_file:
            out_file.write(response.read())
        print("Extracting archive...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(parent_dir)
        print("Dataset ready.")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to auto-download MovieLens 100K dataset: {exc}.\n"
            f"Please download '{DATA_URL}' manually and extract it to '{data_dir}'."
        ) from exc

    return data_dir


def load_movie_metadata(data_dir: str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """
    Loads movie items from 'u.item' with titles, release year, and genre labels.
    """
    download_dataset_if_needed(data_dir)
    item_path = os.path.join(data_dir, "u.item")

    df_movies = pd.read_csv(
        item_path,
        sep="|",
        names=ITEM_COLS,
        encoding="latin-1",
        usecols=["movie_id", "title", "release_date"] + GENRE_NAMES
    )

    # Extract human-readable comma-separated genre string and list
    def extract_genres(row):
        active = [g for g in GENRE_NAMES if row[g] == 1]
        return ", ".join(active) if active else "General"

    df_movies["genres_str"] = df_movies.apply(extract_genres, axis=1)
    df_movies["genres_list"] = df_movies.apply(
        lambda r: [g for g in GENRE_NAMES if r[g] == 1], axis=1
    )

    # Extract year from title if present (e.g. 'Toy Story (1995)')
    df_movies["year"] = df_movies["title"].str.extract(r"\((\d{4})\)")
    df_movies["clean_title"] = df_movies["title"].str.replace(r"\s*\(\d{4}\)", "", regex=True)

    return df_movies


def load_ratings_data(data_dir: str = DEFAULT_DATA_DIR, filename: str = "u.data") -> pd.DataFrame:
    """
    Loads raw user-item interactions from u.data or split files (u1.base, u1.test).
    """
    download_dataset_if_needed(data_dir)
    ratings_path = os.path.join(data_dir, filename)
    df = pd.read_csv(ratings_path, sep="\t", names=RATING_COLS)
    return df


def build_rating_matrix(
    df_ratings: pd.DataFrame,
    num_users: int = 943,
    num_movies: int = 1682,
    rating_threshold: float = 3.0
):
    """
    Converts rating DataFrame into:
      1. rating_matrix: Raw ratings (0 for unrated, 1-5 for rated)
      2. binary_data: 1 if rating >= threshold (liked), 0 otherwise
      3. mask: 1 if item was rated by user, 0 if missing
    """
    rating_matrix = np.zeros((num_users, num_movies), dtype=np.float32)
    binary_data = np.zeros((num_users, num_movies), dtype=np.float32)
    mask = np.zeros((num_users, num_movies), dtype=np.float32)

    for row in df_ratings.itertuples(index=False):
        u_idx = int(row.user_id) - 1   # 1-indexed to 0-indexed
        m_idx = int(row.item_id) - 1   # 1-indexed to 0-indexed
        if u_idx < num_users and m_idx < num_movies:
            r = float(row.rating)
            rating_matrix[u_idx, m_idx] = r
            mask[u_idx, m_idx] = 1.0
            binary_data[u_idx, m_idx] = 1.0 if r >= rating_threshold else 0.0

    return (
        torch.from_numpy(rating_matrix),
        torch.from_numpy(binary_data),
        torch.from_numpy(mask)
    )


def load_train_test_matrices(
    data_dir: str = DEFAULT_DATA_DIR,
    split_num: int = 1,
    rating_threshold: float = 3.0,
    num_users: int = 943,
    num_movies: int = 1682
):
    """
    Loads official MovieLens u{split_num}.base (train) and u{split_num}.test (test)
    and constructs PyTorch tensors:
      (train_data, train_mask, train_raw), (test_data, test_mask, test_raw)
    """
    train_df = load_ratings_data(data_dir, f"u{split_num}.base")
    test_df = load_ratings_data(data_dir, f"u{split_num}.test")

    train_raw, train_data, train_mask = build_rating_matrix(
        train_df, num_users, num_movies, rating_threshold
    )
    test_raw, test_data, test_mask = build_rating_matrix(
        test_df, num_users, num_movies, rating_threshold
    )

    return (train_data, train_mask, train_raw), (test_data, test_mask, test_raw)


def get_popular_movies(
    df_ratings: pd.DataFrame,
    df_movies: pd.DataFrame,
    top_n: int = 25,
    min_ratings: int = 50
) -> pd.DataFrame:
    """
    Retrieves most frequently rated and top-rated movies for cold-start or explorer page.
    """
    stats = df_ratings.groupby("item_id")["rating"].agg(["count", "mean"]).reset_index()
    stats.columns = ["movie_id", "rating_count", "rating_mean"]

    merged = df_movies.merge(stats, on="movie_id", how="left")
    merged["rating_count"] = merged["rating_count"].fillna(0).astype(int)
    merged["rating_mean"] = merged["rating_mean"].fillna(0.0).round(2)

    # Bayesian-weighted rating or simple filter on minimum reviews
    qualified = merged[merged["rating_count"] >= min_ratings].copy()
    qualified = qualified.sort_values(
        by=["rating_count", "rating_mean"], ascending=[False, False]
    ).head(top_n)

    return qualified


def search_movies(query: str, df_movies: pd.DataFrame, limit: int = 30) -> pd.DataFrame:
    """
    Case-insensitive search by title or release year.
    """
    if not query or not query.strip():
        return df_movies.head(limit)
    q = query.strip().lower()
    mask = df_movies["title"].str.lower().str.contains(q, na=False)
    return df_movies[mask].head(limit)


def filter_by_genre(genre: str, df_movies: pd.DataFrame, limit: int = 50) -> pd.DataFrame:
    """
    Filter movies by a specific genre column.
    """
    if genre == "All" or genre not in df_movies.columns:
        return df_movies.head(limit)
    return df_movies[df_movies[genre] == 1].head(limit)

