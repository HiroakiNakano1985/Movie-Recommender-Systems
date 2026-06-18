"""Collaborative filtering placeholders.

Students should implement item-item CF or user-user CF.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from . import config


def _keep_top_k_per_row(sim, k):
    """Zero out all but the k largest entries in each row of a similarity matrix.

    Restricting each item/user to its k nearest neighbours both improves quality
    (distant, noisy neighbours are dropped) and is the standard k-NN CF formulation.
    """
    if k is None or k >= sim.shape[1]:
        return sim
    out = np.zeros_like(sim)
    # argpartition picks the top-k indices per row without a full sort.
    top_idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(sim.shape[0])[:, None]
    out[rows, top_idx] = sim[rows, top_idx]
    return out


class ItemItemCollaborativeFiltering:
    """Item-item collaborative filtering with adjusted cosine similarity.

    Sparsity safeguards (see EDA: median 3 ratings/item, heavy long tail):
    - ``min_support``: items rated by fewer than this many users are excluded
      from the neighbourhood model — their co-rating signal is too thin to trust
      (content-based recommendation is the intended fallback for those items).
    - ``shrinkage``: similarities backed by few co-ratings are shrunk towards 0
      via  sim_adj(i,j) = n_ij / (n_ij + shrinkage) * sim(i,j),  where n_ij is the
      number of users who rated both items. This stops a 2-user overlap from
      producing a spurious similarity of 1.0.

    Similarities use *adjusted cosine*: ratings are mean-centered per user before
    the cosine, which removes each user's rating-scale bias.
    """

    def __init__(self, k=20, min_support=5, shrinkage=10.0, similarity="cosine"):
        self.k = k
        self.min_support = min_support
        self.shrinkage = shrinkage
        self.similarity = similarity
        self.centered_matrix_ = None      # users x items, mean-centered (dense)
        self.rated_mask_ = None           # users x items, 1 where rated
        self.item_similarity_ = None      # items x items, shrunk + top-k
        self.user_means_ = None
        self.user_ids_ = None
        self.item_ids_ = None
        self.user_id_to_index_ = None
        self.item_id_to_index_ = None

    def fit(self, ratings):
        """Create the user-item matrix and compute item-item similarities."""
        item_counts = ratings.groupby(config.ITEM_COL).size()
        kept_items = item_counts[item_counts >= self.min_support].index
        ratings = ratings[ratings[config.ITEM_COL].isin(kept_items)]

        self.user_ids_ = np.sort(ratings[config.USER_COL].unique())
        self.item_ids_ = np.sort(ratings[config.ITEM_COL].unique())
        self.user_id_to_index_ = {u: i for i, u in enumerate(self.user_ids_)}
        self.item_id_to_index_ = {it: i for i, it in enumerate(self.item_ids_)}

        u_idx = ratings[config.USER_COL].map(self.user_id_to_index_).to_numpy()
        i_idx = ratings[config.ITEM_COL].map(self.item_id_to_index_).to_numpy()
        vals = ratings[config.RATING_COL].to_numpy(dtype=float)

        n_users, n_items = len(self.user_ids_), len(self.item_ids_)
        R = csr_matrix((vals, (u_idx, i_idx)), shape=(n_users, n_items))
        mask = csr_matrix((np.ones_like(vals), (u_idx, i_idx)), shape=(n_users, n_items))

        # Per-user mean over rated items, then mean-center (adjusted cosine).
        sums = np.asarray(R.sum(axis=1)).ravel()
        counts = np.asarray(mask.sum(axis=1)).ravel()
        self.user_means_ = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)

        R_dense = R.toarray()
        mask_dense = mask.toarray()
        centered = (R_dense - self.user_means_[:, None]) * mask_dense
        self.centered_matrix_ = centered
        self.rated_mask_ = mask_dense

        # Item-item cosine on the centered matrix (items are the columns).
        sim = cosine_similarity(centered.T)

        # Shrinkage by number of co-ratings n_ij = (mask^T mask).
        co_counts = mask_dense.T @ mask_dense
        sim *= co_counts / (co_counts + self.shrinkage)

        np.fill_diagonal(sim, 0.0)
        self.item_similarity_ = _keep_top_k_per_row(sim, self.k)
        return self

    def _ranking_scores(self, user_id):
        """Length-n_items array of *unnormalized* neighbour scores for ranking.

            score(u,i) = sum_j sim(i,j) * centered_r(u,j)

        We deliberately do NOT divide by sum|sim| here. Normalizing is correct
        for *rating prediction* (predict_score), but for top-N *ranking* it lets a
        single weak neighbour produce an extreme score, flooding the list with
        low-support tail items. The raw weighted sum rewards items that are similar
        to *many* of the user's liked items, which is what we want to rank by.
        """
        u = self.user_id_to_index_.get(user_id)
        if u is None:
            return None
        return self.item_similarity_ @ self.centered_matrix_[u]

    def predict_score(self, user_id, item_id):
        """Predict a rating for one user-item pair (mean + normalized deviation)."""
        i = self.item_id_to_index_.get(item_id)
        u = self.user_id_to_index_.get(user_id)
        if i is None or u is None:
            return float("nan")
        sim_i = self.item_similarity_[i]
        numerator = sim_i @ self.centered_matrix_[u]
        denominator = np.abs(sim_i) @ self.rated_mask_[u]
        deviation = numerator / denominator if denominator > 0 else 0.0
        return float(self.user_means_[u] + deviation)

    def recommend(self, user_id, ratings_train, n=10, exclude_seen=True):
        """Generate top-n recommendations for a user."""
        scores = self._ranking_scores(user_id)
        if scores is None:
            return []

        u = self.user_id_to_index_[user_id]
        if exclude_seen:
            # Never recommend an item the user already rated in training.
            scores = np.where(self.rated_mask_[u] > 0, -np.inf, scores)
        # Items the model could not score (no similar rated neighbours) are useless.
        scores = np.where(scores == 0.0, -np.inf, scores)

        order = np.argsort(-scores)
        out = []
        for idx in order:
            if not np.isfinite(scores[idx]):
                break
            out.append((self.item_ids_[idx], float(scores[idx])))
            if len(out) >= n:
                break
        return out


class UserUserCollaborativeFiltering:
    """User-user collaborative filtering (the dual of item-item).

    Predicts a user's interest in an item from how similar users rated it. With
    MovieLens the user axis is denser (median 70 ratings/user) than the item axis,
    so this neighbourhood is often more reliable than item-item — a contrast worth
    showing in the comparison.
    """

    def __init__(self, k=30, shrinkage=10.0, similarity="cosine"):
        self.k = k
        self.shrinkage = shrinkage
        self.similarity = similarity
        self.centered_matrix_ = None
        self.rated_mask_ = None
        self.user_similarity_ = None
        self.user_means_ = None
        self.user_ids_ = None
        self.item_ids_ = None
        self.user_id_to_index_ = None
        self.item_id_to_index_ = None

    def fit(self, ratings):
        self.user_ids_ = np.sort(ratings[config.USER_COL].unique())
        self.item_ids_ = np.sort(ratings[config.ITEM_COL].unique())
        self.user_id_to_index_ = {u: i for i, u in enumerate(self.user_ids_)}
        self.item_id_to_index_ = {it: i for i, it in enumerate(self.item_ids_)}

        u_idx = ratings[config.USER_COL].map(self.user_id_to_index_).to_numpy()
        i_idx = ratings[config.ITEM_COL].map(self.item_id_to_index_).to_numpy()
        vals = ratings[config.RATING_COL].to_numpy(dtype=float)

        n_users, n_items = len(self.user_ids_), len(self.item_ids_)
        R = csr_matrix((vals, (u_idx, i_idx)), shape=(n_users, n_items)).toarray()
        mask = csr_matrix((np.ones_like(vals), (u_idx, i_idx)), shape=(n_users, n_items)).toarray()

        counts = mask.sum(axis=1)
        self.user_means_ = np.divide(R.sum(axis=1), counts,
                                     out=np.zeros(n_users), where=counts > 0)
        centered = (R - self.user_means_[:, None]) * mask
        self.centered_matrix_ = centered
        self.rated_mask_ = mask

        sim = cosine_similarity(centered)
        co_counts = mask @ mask.T
        sim *= co_counts / (co_counts + self.shrinkage)
        np.fill_diagonal(sim, 0.0)
        self.user_similarity_ = _keep_top_k_per_row(sim, self.k)
        return self

    def recommend(self, user_id, ratings_train, n=10, exclude_seen=True):
        u = self.user_id_to_index_.get(user_id)
        if u is None:
            return []

        # Unnormalized weighted sum for ranking (same rationale as item-item):
        # rank by similarity-weighted neighbour preference, not normalized rating.
        sim_u = self.user_similarity_[u]                   # similarity to every user
        scores = sim_u @ self.centered_matrix_             # over items

        if exclude_seen:
            scores = np.where(self.rated_mask_[u] > 0, -np.inf, scores)
        scores = np.where(scores == 0.0, -np.inf, scores)

        order = np.argsort(-scores)
        out = []
        for idx in order:
            if not np.isfinite(scores[idx]):
                break
            out.append((self.item_ids_[idx], float(scores[idx])))
            if len(out) >= n:
                break
        return out
