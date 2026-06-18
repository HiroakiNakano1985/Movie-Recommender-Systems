"""Data loading and preprocessing utilities.

Students must implement the TODO sections.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from . import config


def load_ratings(path=config.RATINGS_PATH):
    """Load user-item ratings/interactions.

    Expected MovieLens columns:
        userId, movieId, rating, timestamp

    Returns:
        pandas.DataFrame
    """
    ratings = pd.read_csv(path)

    required = [config.USER_COL, config.ITEM_COL, config.RATING_COL]
    missing = [c for c in required if c not in ratings.columns]
    if missing:
        raise ValueError(
            f"Ratings file {path} is missing required columns: {missing}. "
            f"Found columns: {list(ratings.columns)}"
        )

    return ratings


def load_items(path=config.ITEMS_PATH):
    """Load item metadata.

    Expected MovieLens columns:
        movieId, title, genres

    Returns:
        pandas.DataFrame
    """
    items = pd.read_csv(path)

    required = [config.ITEM_COL, config.TITLE_COL]
    missing = [c for c in required if c not in items.columns]
    if missing:
        raise ValueError(
            f"Items file {path} is missing required columns: {missing}. "
            f"Found columns: {list(items.columns)}"
        )

    return items


def describe_dataset(ratings, items=None):
    """Compute and print basic dataset statistics.

    Returns a dict with the computed statistics so callers (e.g. notebooks)
    can reuse the numbers for tables and plots.
    """
    n_users = ratings[config.USER_COL].nunique()
    n_rated_items = ratings[config.ITEM_COL].nunique()
    n_interactions = len(ratings)

    # Catalog size: prefer the metadata file when available, otherwise fall
    # back to the number of items that actually appear in the ratings.
    n_catalog_items = items[config.ITEM_COL].nunique() if items is not None else n_rated_items

    # Sparsity is computed against the items that can actually be rated
    # (the rated items), i.e. the density of the observed user-item matrix.
    possible = n_users * n_rated_items
    sparsity = 1.0 - (n_interactions / possible) if possible else float("nan")

    rating_distribution = ratings[config.RATING_COL].value_counts().sort_index()

    ratings_per_user = ratings.groupby(config.USER_COL).size()
    ratings_per_item = ratings.groupby(config.ITEM_COL).size()

    most_active_users = ratings_per_user.sort_values(ascending=False).head(10)
    most_popular_items = ratings_per_item.sort_values(ascending=False).head(10)

    print("=" * 50)
    print("Dataset summary")
    print("=" * 50)
    print(f"Users:                 {n_users:,}")
    print(f"Items (rated):         {n_rated_items:,}")
    print(f"Items (catalog):       {n_catalog_items:,}")
    print(f"Interactions:          {n_interactions:,}")
    print(f"Sparsity:              {sparsity:.4%}")
    print(f"Density:               {1 - sparsity:.4%}")
    print(f"Ratings per user:      "
          f"min={ratings_per_user.min()}, "
          f"median={ratings_per_user.median():.0f}, "
          f"max={ratings_per_user.max()}")
    print(f"Ratings per item:      "
          f"min={ratings_per_item.min()}, "
          f"median={ratings_per_item.median():.0f}, "
          f"max={ratings_per_item.max()}")
    print(f"Rating range:          {ratings[config.RATING_COL].min()} - "
          f"{ratings[config.RATING_COL].max()} "
          f"(mean={ratings[config.RATING_COL].mean():.2f})")
    print("\nRating distribution:")
    for value, count in rating_distribution.items():
        print(f"  {value:>4}: {count:>7,} ({count / n_interactions:.1%})")

    print("\nMost active users (by number of ratings):")
    for user_id, count in most_active_users.items():
        print(f"  user {user_id}: {count} ratings")

    if items is not None:
        popular_titles = most_popular_items.rename("n_ratings").to_frame()
        popular_titles = popular_titles.merge(
            items.set_index(config.ITEM_COL)[config.TITLE_COL],
            left_index=True, right_index=True, how="left",
        )
        print("\nMost popular items (by number of ratings):")
        for item_id, row in popular_titles.iterrows():
            print(f"  {row[config.TITLE_COL]} ({int(row['n_ratings'])} ratings)")

    return {
        "n_users": n_users,
        "n_rated_items": n_rated_items,
        "n_catalog_items": n_catalog_items,
        "n_interactions": n_interactions,
        "sparsity": sparsity,
        "rating_distribution": rating_distribution,
        "ratings_per_user": ratings_per_user,
        "ratings_per_item": ratings_per_item,
        "most_active_users": most_active_users,
        "most_popular_items": most_popular_items,
    }


def train_test_split_ratings(ratings, test_size=0.2, random_state=config.RANDOM_STATE):
    """Create a train/test split.

    The split is stratified by user so that every user keeps part of their
    history in the training set and part in the test set. This is what the
    top-N evaluation needs: we must be able to build a profile for a user from
    `train` and check held-out items from `test`.

    For advanced work, students may replace this with a temporal split using
    the timestamp column.
    """
    stratify = ratings[config.USER_COL] if config.USER_COL in ratings.columns else None
    train, test = train_test_split(
        ratings,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def get_seen_items(ratings, user_id):
    """Return the set of items already rated/consumed by one user."""
    user_ratings = ratings.loc[ratings[config.USER_COL] == user_id, config.ITEM_COL]
    return set(user_ratings.tolist())
