"""Pre-train all recommenders and pickle them for the Streamlit app.

Training the models (matrix factorization in particular) takes ~1 minute, which
is too slow to run on every Streamlit interaction. This script trains everything
once and saves the fitted models + data splits to ``results/artifacts.joblib`` so
the app can load them instantly.

Run once before launching the app (re-run whenever the models change):

    python build_artifacts.py
"""

import sys

import joblib

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src import config
from src.data_loading import load_ratings, load_items, train_test_split_ratings
from main import build_models, fit_model

ARTIFACTS_PATH = config.RESULTS_DIR / "artifacts.joblib"


def main():
    ratings = load_ratings()
    items = load_items()
    train, test = train_test_split_ratings(ratings, test_size=0.2)

    models = build_models()
    for name, model in models.items():
        print(f"Training: {name} ...")
        fit_model(name, model, train, items)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"train": train, "test": test, "items": items, "models": models},
        ARTIFACTS_PATH,
        compress=3,
    )
    print(f"\nSaved artifacts to {ARTIFACTS_PATH}")


if __name__ == "__main__":
    main()
