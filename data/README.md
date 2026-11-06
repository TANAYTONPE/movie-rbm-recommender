# MovieLens 100K Dataset

This directory contains the MovieLens 100K benchmark dataset collected by the GroupLens Research Project at the University of Minnesota.

## Dataset Overview

- **Source:** [GroupLens Research](https://grouplens.org/datasets/movielens/100k/)
- **Total Ratings:** 100,000 ratings (scale 1 to 5)
- **Users:** 943 unique users
- **Movies:** 1,682 unique movies
- **Sparsity:** Approximately 93.7% of the user-item matrix is unrated (sparse)
- **Timeframe:** Collected between September 1997 and April 1998

## File Description

| File | Description |
|---|---|
| `u.data` | Full ratings dataset: 100,000 rows (`user_id \t item_id \t rating \t timestamp`). |
| `u.item` | Movie metadata: 1,682 rows containing `movie_id`, `movie_title`, `release_date`, and 19 binary genre indicators. |
| `u.genre` | List of 19 genre names and their indices. |
| `u.user` | Demographic user data: `user_id \t age \t gender \t occupation \t zip_code`. |
| `u1.base` / `u1.test` | 80%/20% train/test split 1 (80,000 train ratings, 20,000 test ratings). |
| `u2.base` / `u2.test` to `u5.base` / `u5.test` | 5-fold cross-validation splits. |

## Automatic Download

The application includes an automated dataset fetcher in `utils/data_loader.py`. If `data/ml-100k` is missing, the loader attempts to download and extract `ml-100k.zip` automatically.

## Manual Download Instructions

If your machine has restricted internet access:
1. Download `ml-100k.zip` from: https://files.grouplens.org/datasets/movielens/ml-100k.zip
2. Extract the contents so that `u.data` and `u.item` are located in:
   `movie-rbm-recommender/data/ml-100k/`
3. Verify the files:
   - `data/ml-100k/u.data`
   - `data/ml-100k/u.item`
   - `data/ml-100k/u.genre`

## Citation

F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.* ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages. DOI: https://doi.org/10.1145/2827872

