"""Content-based recommender placeholder.

Students should implement item feature vectors, user profiles, and cosine scoring.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from . import config
from .data_loading import get_seen_items


class ContentBasedRecommender:
    """Content-based recommender using item metadata such as genres or tags.

    Items are described by TF-IDF vectors over their genres. A user profile is
    the rating-weighted (and mean-centered) average of the vectors of the items
    they rated; recommendations are the unseen items whose vector is most cosine-
    similar to that profile.

    Because it relies only on item metadata, this method is unaffected by the
    user-item sparsity that hurts collaborative filtering and can score even
    rarely-rated ("long tail") items.
    """

    def __init__(self, feature_col=config.GENRES_COL, use_tfidf=True):
        self.feature_col = feature_col
        self.use_tfidf = use_tfidf
        self.vectorizer = None
        self.item_features_ = None
        self.item_ids_ = None
        self.item_id_to_index_ = None

    def _to_text(self, raw):
        # MovieLens genres look like "Adventure|Animation|Children".
        # Turn the pipe separator into spaces so each genre is a token, and
        # drop the explicit "(no genres listed)" marker.
        text = str(raw).replace("|", " ").replace("(no genres listed)", "")
        return text.lower().strip()

    def fit(self, ratings, items):
        """Build the item feature matrix from metadata text."""
        items = items.drop_duplicates(subset=config.ITEM_COL)
        text = items[self.feature_col].fillna("").map(self._to_text)

        # TF-IDF down-weights ubiquitous genres (e.g. Drama) and up-weights
        # distinctive ones. token_pattern keeps multi-word tokens like sci-fi.
        self.vectorizer = TfidfVectorizer(
            token_pattern=r"[^\s]+",
            use_idf=self.use_tfidf,
        )
        self.item_features_ = self.vectorizer.fit_transform(text)

        self.item_ids_ = items[config.ITEM_COL].to_numpy()
        self.item_id_to_index_ = {item_id: idx for idx, item_id in enumerate(self.item_ids_)}
        return self

    def build_user_profile(self, user_id, ratings_train):
        """Build a user profile from mean-centered ratings and item vectors.

            profile(u) = sum_i (rating(u,i) - mean_rating(u)) * vector(i)

        Centering means items rated *below* the user's average push the profile
        away from their genres, so the profile captures relative preference, not
        just which genres the user happened to watch.
        """
        user_ratings = ratings_train[ratings_train[config.USER_COL] == user_id]
        if user_ratings.empty:
            return None

        mean_rating = user_ratings[config.RATING_COL].mean()

        # Gather the rated items that have a metadata vector and their centered
        # weights, then form the profile as a single sparse matrix-vector product
        # (weights · item_vectors) instead of looping row by row — much faster for
        # heavy users (some have 2,000+ ratings).
        idxs, weights = [], []
        for item_id, rating in zip(user_ratings[config.ITEM_COL], user_ratings[config.RATING_COL]):
            idx = self.item_id_to_index_.get(item_id)
            if idx is None:
                continue  # item has no metadata vector
            idxs.append(idx)
            weights.append(rating - mean_rating)

        if not idxs:
            return None

        sub = self.item_features_[idxs]                      # (n_rated x n_features), sparse
        profile = np.asarray(sub.T @ np.asarray(weights)).ravel()
        return profile

    def recommend(self, user_id, ratings_train, n=10, exclude_seen=True):
        """Recommend unseen items whose vectors are most similar to the profile."""
        profile = self.build_user_profile(user_id, ratings_train)
        if profile is None or not np.any(profile):
            return []

        scores = cosine_similarity(profile.reshape(1, -1), self.item_features_).ravel()

        seen = get_seen_items(ratings_train, user_id) if exclude_seen else set()
        order = np.argsort(-scores)

        out = []
        for idx in order:
            item_id = self.item_ids_[idx]
            if exclude_seen and item_id in seen:
                continue
            out.append((item_id, float(scores[idx])))
            if len(out) >= n:
                break
        return out

    def similar_items(self, item_id, n=10):
        """Return the n items most similar to a given item (item-to-item)."""
        idx = self.item_id_to_index_.get(item_id)
        if idx is None:
            return []
        scores = cosine_similarity(self.item_features_[idx], self.item_features_).ravel()
        order = np.argsort(-scores)
        out = []
        for j in order:
            if j == idx:
                continue
            out.append((self.item_ids_[j], float(scores[j])))
            if len(out) >= n:
                break
        return out
