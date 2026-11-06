# utils package initialization
from .data_loader import (
    load_movie_metadata,
    load_ratings_data,
    load_train_test_matrices,
    get_popular_movies,
    search_movies,
    filter_by_genre,
    download_dataset_if_needed
)
from .preprocessing import ratings_dict_to_tensor, binary_score_to_stars

