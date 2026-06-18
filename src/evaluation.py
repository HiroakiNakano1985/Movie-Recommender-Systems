"""Evaluation metrics for recommender systems.

Students should complete these functions and use them to compare all models.
"""

import numpy as np
import pandas as pd
from . import config


def _take_top_k(recommended_items, k):
    """Return the first k recommended item IDs as a plain list."""
    return list(recommended_items)[:k]


def precision_at_k(recommended_items, relevant_items, k=10):
    """Compute Precision@K for one user.

    Precision@K = (# recommended items in top-k that are relevant) / k.
    """
    top_k = _take_top_k(recommended_items, k)
    if k == 0:
        return 0.0
    relevant = set(relevant_items)
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def recall_at_k(recommended_items, relevant_items, k=10):
    """Compute Recall@K for one user.

    Recall@K = (# relevant items found in top-k) / (total # relevant items).
    """
    relevant = set(relevant_items)
    if not relevant:
        return 0.0
    top_k = _take_top_k(recommended_items, k)
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def hit_rate_at_k(recommended_items, relevant_items, k=10):
    """Return 1 if there is at least one relevant item in the top-k, else 0."""
    relevant = set(relevant_items)
    top_k = _take_top_k(recommended_items, k)
    return 1.0 if any(item in relevant for item in top_k) else 0.0


def dcg_at_k(relevance_scores, k=10):
    """Compute DCG@K for relevance scores ordered by recommendation rank.

    Formula: sum_i rel_i / log2(i + 1), with rank i starting at 1.
    """
    relevance_scores = np.asarray(relevance_scores, dtype=float)[:k]
    if relevance_scores.size == 0:
        return 0.0
    ranks = np.arange(1, relevance_scores.size + 1)
    return float(np.sum(relevance_scores / np.log2(ranks + 1)))


def ndcg_at_k(recommended_items, relevant_items, k=10):
    """Compute NDCG@K for one user with binary relevance."""
    relevant = set(relevant_items)
    if not relevant:
        return 0.0

    top_k = _take_top_k(recommended_items, k)
    gains = [1.0 if item in relevant else 0.0 for item in top_k]
    dcg = dcg_at_k(gains, k)

    # Ideal DCG: all relevant items ranked first (but no more than k of them).
    ideal_gains = [1.0] * min(len(relevant), k)
    idcg = dcg_at_k(ideal_gains, k)

    return dcg / idcg if idcg > 0 else 0.0


def mean_reciprocal_rank(recommended_items, relevant_items, k=10):
    """Compute the reciprocal rank of the first relevant item in the top-k."""
    relevant = set(relevant_items)
    top_k = _take_top_k(recommended_items, k)
    for rank, item in enumerate(top_k, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def catalog_coverage(all_recommendations, all_items):
    """Compute catalog coverage.

    all_recommendations: iterable of recommended item IDs across users
    all_items: iterable of all item IDs in catalog

    Returns the fraction of the catalog that appears in at least one
    recommendation list. Low coverage signals popularity bias.
    """
    catalog = set(all_items)
    if not catalog:
        return 0.0
    recommended = set(all_recommendations)
    return len(recommended & catalog) / len(catalog)


# --- Beyond-accuracy metrics -------------------------------------------------

def item_popularity(ratings_train):
    """Return a dict item_id -> P(item) = (# users who rated it) / (# users).

    Used as the popularity prior for the novelty metric.
    """
    n_users = ratings_train[config.USER_COL].nunique()
    counts = ratings_train.groupby(config.ITEM_COL)[config.USER_COL].nunique()
    return (counts / n_users).to_dict()


def novelty_at_k(recommended_items, popularity, k=10):
    """Mean self-information (in bits) of the top-k recommended items.

        novelty = mean( -log2 P(item) )

    Recommending niche (low-probability) items yields high novelty; recommending
    blockbusters yields low novelty. This is the standard popularity-based
    novelty / "surprise" measure.
    """
    top_k = _take_top_k(recommended_items, k)
    infos = [-np.log2(popularity[item]) for item in top_k
             if popularity.get(item, 0) > 0]
    return float(np.mean(infos)) if infos else 0.0


def intra_list_diversity(recommended_items, feature_matrix, item_id_to_index, k=10):
    """Average pairwise dissimilarity (1 - cosine) within the top-k list.

    A diverse list spreads across the content space (e.g. many genres); a
    redundant list keeps recommending near-identical items. Requires an item
    feature matrix (we reuse the content-based TF-IDF genre vectors).
    """
    from sklearn.metrics.pairwise import cosine_similarity

    top_k = _take_top_k(recommended_items, k)
    idxs = [item_id_to_index[item] for item in top_k if item in item_id_to_index]
    if len(idxs) < 2:
        return 0.0
    sims = cosine_similarity(feature_matrix[idxs])
    n = len(idxs)
    iu = np.triu_indices(n, k=1)       # unique unordered pairs
    return float(np.mean(1.0 - sims[iu]))


# --- Rating-prediction metrics (for models that expose predict_score) --------

def rating_prediction_metrics(model, ratings_test):
    """Compute RMSE and MAE of a model's rating predictions on the test set.

    Only meaningful for models with a ``predict_score(user, item)`` method
    (matrix factorization, CF). Returns ``None`` if the model has no such method.
    """
    if not hasattr(model, "predict_score"):
        return None

    preds, actuals = [], []
    for u, i, r in zip(ratings_test[config.USER_COL],
                       ratings_test[config.ITEM_COL],
                       ratings_test[config.RATING_COL]):
        p = model.predict_score(u, i)
        if p is None or (isinstance(p, float) and np.isnan(p)):
            continue
        preds.append(p)
        actuals.append(r)

    if not preds:
        return None
    preds = np.asarray(preds)
    actuals = np.asarray(actuals)
    return {
        "rmse": float(np.sqrt(np.mean((preds - actuals) ** 2))),
        "mae": float(np.mean(np.abs(preds - actuals))),
        "n_predictions": len(preds),
    }


def evaluate_model(model, ratings_train, ratings_test, users=None, k=10,
                   relevance_threshold=4.0, diversity_features=None, verbose=False):
    """Evaluate a recommender over a set of users with a single held-out split.

    Protocol
    --------
    - "Relevant" items for a user are the items in ``ratings_test`` that the user
      rated >= ``relevance_threshold`` (positivity-aware top-N evaluation).
    - For each such user we ask the model for top-k recommendations (computed from
      ``ratings_train`` only, excluding items already seen in train) and compare
      them against the relevant test items.
    - Users with no relevant test items are skipped (no ground truth to score).

    Beyond-accuracy metrics are averaged over the same users: novelty (from train
    popularity) always, and intra-list diversity when ``diversity_features`` is
    given as a ``(feature_matrix, item_id_to_index)`` tuple.

    Returns a dict of averaged metrics plus catalog coverage across all users.
    """
    test_relevant = (
        ratings_test[ratings_test[config.RATING_COL] >= relevance_threshold]
        .groupby(config.USER_COL)[config.ITEM_COL]
        .apply(set)
    )

    if users is None:
        users = test_relevant.index.tolist()

    catalog = set(ratings_train[config.ITEM_COL].unique())
    popularity = item_popularity(ratings_train)
    div_matrix, div_index = diversity_features if diversity_features else (None, None)

    precisions, recalls, ndcgs, mrrs, hits = [], [], [], [], []
    novelties, diversities = [], []
    all_recommended = []
    evaluated_users = 0

    for user_id in users:
        relevant = test_relevant.get(user_id)
        if not relevant:
            continue

        recs = model.recommend(user_id, ratings_train, n=k, exclude_seen=True)
        rec_items = [item for item, _ in recs] if recs and isinstance(recs[0], tuple) else list(recs)

        precisions.append(precision_at_k(rec_items, relevant, k))
        recalls.append(recall_at_k(rec_items, relevant, k))
        ndcgs.append(ndcg_at_k(rec_items, relevant, k))
        mrrs.append(mean_reciprocal_rank(rec_items, relevant, k))
        hits.append(hit_rate_at_k(rec_items, relevant, k))
        novelties.append(novelty_at_k(rec_items, popularity, k))
        if div_matrix is not None:
            diversities.append(intra_list_diversity(rec_items, div_matrix, div_index, k))
        all_recommended.extend(rec_items)
        evaluated_users += 1

        if verbose and evaluated_users % 100 == 0:
            print(f"  ...evaluated {evaluated_users} users")

    return {
        "k": k,
        "n_users_evaluated": evaluated_users,
        "precision@k": float(np.mean(precisions)) if precisions else 0.0,
        "recall@k": float(np.mean(recalls)) if recalls else 0.0,
        "ndcg@k": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "mrr@k": float(np.mean(mrrs)) if mrrs else 0.0,
        "hit_rate@k": float(np.mean(hits)) if hits else 0.0,
        "catalog_coverage": catalog_coverage(all_recommended, catalog),
        "novelty@k": float(np.mean(novelties)) if novelties else 0.0,
        "diversity@k": float(np.mean(diversities)) if diversities else float("nan"),
    }
