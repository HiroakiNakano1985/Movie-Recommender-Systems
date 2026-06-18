"""Non-personalized baseline recommenders.

Students should implement at least two baselines.
"""

import numpy as np
import pandas as pd
from . import config
from .data_loading import get_seen_items


def _top_n_from_ranking(ranking, seen, n, exclude_seen):
    """Walk a pre-computed item ranking and return the first n (item_id, score)
    pairs, optionally skipping items the user has already seen.

    `ranking` is an iterable of (item_id, score) pairs ordered best-first.
    """
    out = []
    for item_id, score in ranking:
        if exclude_seen and item_id in seen:
            continue
        out.append((item_id, float(score)))
        if len(out) >= n:
            break
    return out


class MostPopularRecommender:
    """Recommend the most frequently rated/interacted items.

    Non-personalized: the ranking is identical for every user (only the
    already-seen filtering differs per user).
    """

    def __init__(self):
        self.ranking_ = None

    def fit(self, ratings, items=None):
        counts = ratings.groupby(config.ITEM_COL).size().sort_values(ascending=False)
        # Score = popularity (number of ratings). Stored as (item_id, score) pairs.
        self.ranking_ = list(counts.items())
        return self

    def recommend(self, user_id, ratings_train, n=10, exclude_seen=True):
        seen = get_seen_items(ratings_train, user_id) if exclude_seen else set()
        return _top_n_from_ranking(self.ranking_, seen, n, exclude_seen)


class HighestAverageRatingRecommender:
    """Recommend items with the highest average rating, requiring a minimum
    number of ratings so that a single 5-star rating cannot top the chart."""

    def __init__(self, min_ratings=20):
        self.min_ratings = min_ratings
        self.ranking_ = None

    def fit(self, ratings, items=None):
        agg = ratings.groupby(config.ITEM_COL)[config.RATING_COL].agg(["mean", "count"])
        qualified = agg[agg["count"] >= self.min_ratings]
        qualified = qualified.sort_values("mean", ascending=False)
        self.ranking_ = list(qualified["mean"].items())
        return self

    def recommend(self, user_id, ratings_train, n=10, exclude_seen=True):
        seen = get_seen_items(ratings_train, user_id) if exclude_seen else set()
        return _top_n_from_ranking(self.ranking_, seen, n, exclude_seen)


class RandomRecommender:
    """Optional baseline: recommend random unseen items.

    Useful as a lower bound and as a high-coverage / high-novelty reference
    when comparing beyond-accuracy metrics.
    """

    def __init__(self, random_state=config.RANDOM_STATE):
        self.random_state = random_state
        self.items_ = None

    def fit(self, ratings, items=None):
        self.items_ = ratings[config.ITEM_COL].unique()
        return self

    def recommend(self, user_id, ratings_train, n=10, exclude_seen=True):
        seen = get_seen_items(ratings_train, user_id) if exclude_seen else set()
        # Derive a per-user seed so results are reproducible but differ by user.
        rng = np.random.default_rng(self.random_state + int(user_id))
        candidates = [item for item in self.items_ if not (exclude_seen and item in seen)]
        chosen = rng.choice(candidates, size=min(n, len(candidates)), replace=False)
        return [(item, 0.0) for item in chosen.tolist()]
