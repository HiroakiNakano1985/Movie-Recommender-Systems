"""Main entry point for the individual recommender assignment.

Runs the full prototype pipeline on the MovieLens (movie track) data:
  1. load data + EDA summary
  2. per-user train/test split
  3. train every recommender (baselines, content-based, CF, matrix factorization)
  4. evaluate them with a common top-N protocol (relevance = rating >= 4.0)
  5. save the comparison table to results/metrics.csv
  6. print top-N recommendations for a few example users

Run:  python main.py
"""

import sys

import pandas as pd

# MovieLens titles and our headings contain non-ASCII characters (em-dashes,
# accented film titles like "Amelie"). Force UTF-8 stdout so the pipeline does
# not crash on a Windows cp932 console.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src import config
from src.data_loading import load_ratings, load_items, train_test_split_ratings, describe_dataset
from src.baselines import (
    MostPopularRecommender,
    HighestAverageRatingRecommender,
    RandomRecommender,
)
from src.content_based import ContentBasedRecommender
from src.collaborative_filtering import (
    ItemItemCollaborativeFiltering,
    UserUserCollaborativeFiltering,
)
from src.matrix_factorization import MatrixFactorizationRecommender
from src.evaluation import evaluate_model, rating_prediction_metrics

K = config.TOP_K


def build_models():
    """Instantiate every recommender to compare. Order = roughly increasing
    sophistication, which is also the story we tell in the report."""
    return {
        "random": RandomRecommender(),
        "most_popular": MostPopularRecommender(),
        "highest_average": HighestAverageRatingRecommender(min_ratings=20),
        "content_based": ContentBasedRecommender(),
        "item_item_cf": ItemItemCollaborativeFiltering(k=20, min_support=5, shrinkage=10.0),
        "user_user_cf": UserUserCollaborativeFiltering(k=30, shrinkage=10.0),
        "matrix_factorization": MatrixFactorizationRecommender(
            n_factors=50, n_epochs=30, reg=0.01
        ),
    }


def fit_model(name, model, train, items):
    """Fit a model, passing item metadata only to the models that use it."""
    if name == "content_based":
        return model.fit(train, items)
    if name in ("random", "most_popular", "highest_average"):
        return model.fit(train, items)
    return model.fit(train)


def show_recommendations(models, train, items, user_ids, n=10):
    """Print top-n recommendations (with titles) for a few example users."""
    title_by_id = items.set_index(config.ITEM_COL)[config.TITLE_COL].to_dict()
    for user_id in user_ids:
        seen = train[train[config.USER_COL] == user_id]
        print(f"\n{'=' * 70}\nUser {user_id} — {len(seen)} ratings in train "
              f"(avg {seen[config.RATING_COL].mean():.2f})")
        for name, model in models.items():
            recs = model.recommend(user_id, train, n=n, exclude_seen=True)
            titles = [title_by_id.get(item_id, item_id) for item_id, _ in recs[:5]]
            print(f"  {name:22s}: " + " | ".join(str(t) for t in titles))


def main():
    print("Individual recommender assignment — full pipeline\n")

    ratings = load_ratings()
    items = load_items()
    describe_dataset(ratings, items)

    train, test = train_test_split_ratings(ratings, test_size=0.2)
    print(f"\nTrain: {train.shape[0]:,} ratings | Test: {test.shape[0]:,} ratings\n")

    models = build_models()

    # Fit all models first so we can reuse the content model's genre vectors as
    # the feature space for the intra-list diversity metric.
    for name, model in models.items():
        print(f"Training: {name} ...")
        fit_model(name, model, train, items)

    content_model = models["content_based"]
    diversity_features = (content_model.item_features_, content_model.item_id_to_index_)

    rows, rating_rows = [], []
    for name, model in models.items():
        print(f"Evaluating: {name} ...")
        metrics = evaluate_model(model, train, test, k=K, relevance_threshold=4.0,
                                 diversity_features=diversity_features)
        metrics["model"] = name
        rows.append(metrics)

        rp = rating_prediction_metrics(model, test)
        if rp is not None:
            rp["model"] = name
            rating_rows.append(rp)

    results = pd.DataFrame(rows).set_index("model")
    results = results[[
        "precision@k", "recall@k", "ndcg@k", "mrr@k", "hit_rate@k",
        "catalog_coverage", "novelty@k", "diversity@k", "n_users_evaluated", "k",
    ]].sort_values("ndcg@k", ascending=False)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.RESULTS_DIR / "metrics.csv"
    results.to_csv(out_path)

    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(f"\n{'=' * 70}\nMethod comparison (K={K}, relevance = rating >= 4.0)\n{'=' * 70}")
    print(results.to_string())
    print(f"\nSaved metrics to {out_path}")

    if rating_rows:
        rating_results = (pd.DataFrame(rating_rows).set_index("model")
                          .sort_values("rmse"))
        rating_path = config.RESULTS_DIR / "rating_metrics.csv"
        rating_results.to_csv(rating_path)
        print(f"\n{'=' * 70}\nRating-prediction quality (models with predict_score)\n{'=' * 70}")
        print(rating_results.to_string())
        print(f"\nSaved rating metrics to {rating_path}")

    # Recommendation examples for three users with different profiles.
    example_users = [1, 414, 599]
    show_recommendations(models, train, items, example_users, n=10)


if __name__ == "__main__":
    main()
