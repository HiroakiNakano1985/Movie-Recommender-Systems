"""Matrix factorization placeholder.

Students may use a library such as scikit-surprise, or implement a simple model.
"""

import numpy as np
import pandas as pd
from . import config
from .data_loading import get_seen_items


class MatrixFactorizationRecommender:
    """Biased matrix factorization trained with stochastic gradient descent.

    This is the model that won the Netflix Prize and the natural answer to the
    user-item sparsity in this dataset: instead of relying on direct co-ratings
    (which the long tail makes unreliable), it learns a dense ``n_factors``-
    dimensional latent vector for every user and item from the *observed* ratings
    only, and generalizes to unobserved pairs.

    Prediction:
        r_hat(u,i) = mu + b_u + b_i + p_u . q_i

    Training minimizes regularized squared error over observed ratings:
        sum (r_ui - r_hat_ui)^2 + reg * (b_u^2 + b_i^2 + ||p_u||^2 + ||q_i||^2)

    We implement SGD ourselves (numpy only) rather than depend on scikit-surprise,
    which does not reliably build on recent Python/NumPy. This also makes the model
    fully transparent, as the assignment requires it to be explained.
    """

    def __init__(self, n_factors=50, n_epochs=20, lr=0.005, reg=0.02,
                 random_state=config.RANDOM_STATE, verbose=False):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.random_state = random_state
        self.verbose = verbose

        self.mu_ = 0.0
        self.b_u_ = None
        self.b_i_ = None
        self.P_ = None
        self.Q_ = None
        self.user_ids_ = None
        self.all_items_ = None
        self.user_id_to_index_ = None
        self.item_id_to_index_ = None
        self.rating_min_ = None
        self.rating_max_ = None

    def fit(self, ratings):
        """Train the model with SGD over the observed ratings."""
        rng = np.random.default_rng(self.random_state)

        self.user_ids_ = np.sort(ratings[config.USER_COL].unique())
        self.all_items_ = np.sort(ratings[config.ITEM_COL].unique())
        self.user_id_to_index_ = {u: i for i, u in enumerate(self.user_ids_)}
        self.item_id_to_index_ = {it: i for i, it in enumerate(self.all_items_)}

        u_idx = ratings[config.USER_COL].map(self.user_id_to_index_).to_numpy()
        i_idx = ratings[config.ITEM_COL].map(self.item_id_to_index_).to_numpy()
        r = ratings[config.RATING_COL].to_numpy(dtype=float)

        n_users, n_items = len(self.user_ids_), len(self.all_items_)
        self.mu_ = float(r.mean())
        self.rating_min_, self.rating_max_ = float(r.min()), float(r.max())
        self.b_u_ = np.zeros(n_users)
        self.b_i_ = np.zeros(n_items)
        # Small random init breaks symmetry; scale 0.1 keeps early predictions near mu.
        self.P_ = rng.normal(0, 0.1, (n_users, self.n_factors))
        self.Q_ = rng.normal(0, 0.1, (n_items, self.n_factors))

        n = len(r)
        for epoch in range(self.n_epochs):
            order = rng.permutation(n)
            sq_err = 0.0
            for idx in order:
                u, i, rating = u_idx[idx], i_idx[idx], r[idx]
                pred = self.mu_ + self.b_u_[u] + self.b_i_[i] + self.P_[u] @ self.Q_[i]
                err = rating - pred
                sq_err += err * err

                bu, bi = self.b_u_[u], self.b_i_[i]
                pu, qi = self.P_[u].copy(), self.Q_[i]

                self.b_u_[u] += self.lr * (err - self.reg * bu)
                self.b_i_[i] += self.lr * (err - self.reg * bi)
                self.P_[u] += self.lr * (err * qi - self.reg * pu)
                self.Q_[i] += self.lr * (err * pu - self.reg * qi)

            if self.verbose:
                print(f"  epoch {epoch + 1:2d}/{self.n_epochs}  train RMSE={np.sqrt(sq_err / n):.4f}")

        return self

    def predict_score(self, user_id, item_id):
        """Predict the rating for one user-item pair, clipped to the rating range."""
        u = self.user_id_to_index_.get(user_id)
        i = self.item_id_to_index_.get(item_id)
        if u is None and i is None:
            return self.mu_
        bu = self.b_u_[u] if u is not None else 0.0
        bi = self.b_i_[i] if i is not None else 0.0
        dot = self.P_[u] @ self.Q_[i] if (u is not None and i is not None) else 0.0
        pred = self.mu_ + bu + bi + dot
        return float(np.clip(pred, self.rating_min_, self.rating_max_))

    def recommend(self, user_id, ratings_train, n=10, exclude_seen=True):
        """Recommend top-n unseen items by predicted score (vectorized over items)."""
        u = self.user_id_to_index_.get(user_id)
        if u is None:
            return []

        # r_hat(u, :) = mu + b_u + b_i + Q @ p_u  for every item at once.
        scores = self.mu_ + self.b_u_[u] + self.b_i_ + self.Q_ @ self.P_[u]

        if exclude_seen:
            seen = get_seen_items(ratings_train, user_id)
            for item_id in seen:
                j = self.item_id_to_index_.get(item_id)
                if j is not None:
                    scores[j] = -np.inf

        order = np.argsort(-scores)[:n]
        return [(self.all_items_[j], float(scores[j])) for j in order if np.isfinite(scores[j])]
