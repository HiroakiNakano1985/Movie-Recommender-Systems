"""Streamlit demo for the movie recommender prototype.

Two tabs:
  - Explore: pick a user, see their taste profile, and compare the top-N
    recommendations of several algorithms side by side, each with a short
    "why" explanation.
  - Evaluation: the quantitative comparison (metrics table + figures) produced
    by `python main.py`.

Before running, build the model artifacts once:
    python build_artifacts.py
Then launch:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import joblib

from src import config

ARTIFACTS_PATH = config.RESULTS_DIR / "artifacts.joblib"
FIG_DIR = config.RESULTS_DIR / "figures"

ALGO_LABELS = {
    "most_popular": "Most Popular",
    "highest_average": "Highest Average",
    "content_based": "Content-Based",
    "item_item_cf": "Item-Item CF",
    "user_user_cf": "User-User CF",
    "matrix_factorization": "Matrix Factorization",
    "random": "Random",
}

st.set_page_config(page_title="Movie Recommender Prototype", layout="wide")


# --- Data / model loading ----------------------------------------------------

@st.cache_resource
def load_artifacts():
    if not ARTIFACTS_PATH.exists():
        return None
    return joblib.load(ARTIFACTS_PATH)


@st.cache_data
def derived_lookups(_train, _items):
    """Cheap per-dataset lookups reused across reruns."""
    title_by_id = _items.set_index(config.ITEM_COL)[config.TITLE_COL].to_dict()
    genres_by_id = _items.set_index(config.ITEM_COL)[config.GENRES_COL].to_dict()
    popularity = _train.groupby(config.ITEM_COL).size().to_dict()
    return title_by_id, genres_by_id, popularity


# --- Helpers -----------------------------------------------------------------

def genre_list(genres_str):
    if not isinstance(genres_str, str):
        return []
    return [g for g in genres_str.split("|") if g and g != "(no genres listed)"]


def user_profile(train, items, user_id):
    ur = train[train[config.USER_COL] == user_id]
    liked = ur[ur[config.RATING_COL] >= 4.0]
    merged = liked.merge(items, on=config.ITEM_COL, how="left")
    genres = (
        merged[config.GENRES_COL].dropna().map(genre_list).explode().value_counts()
    )
    top_liked = (
        ur.sort_values(config.RATING_COL, ascending=False)
        .merge(items, on=config.ITEM_COL, how="left")
        [[config.TITLE_COL, config.RATING_COL, config.GENRES_COL]]
        .head(8)
        .reset_index(drop=True)
    )
    return ur, genres, top_liked


def explain(name, model, user_id, item_id, train, genres_by_id, popularity, user_top_genres):
    """Return a short human-readable reason for recommending item_id."""
    if name == "content_based":
        overlap = [g for g in genre_list(genres_by_id.get(item_id, "")) if g in user_top_genres]
        return ("Matches your taste: " + ", ".join(overlap[:3])) if overlap \
            else "Similar genres to what you rate highly"

    if name == "item_item_cf":
        i = model.item_id_to_index_.get(item_id)
        if i is not None:
            rated = train[train[config.USER_COL] == user_id]
            best_title, best_sim = None, 0.0
            for jid, jr in zip(rated[config.ITEM_COL], rated[config.RATING_COL]):
                j = model.item_id_to_index_.get(jid)
                if j is None:
                    continue
                s = model.item_similarity_[i, j]
                if s > best_sim:
                    best_sim, best_title = s, jid
            if best_title is not None:
                return f"Because you liked “{title_short(best_title, genres_by_id, train)}”"
        return "Similar to items you rated highly"

    if name == "user_user_cf":
        return "Users with similar taste rated this highly"

    if name == "matrix_factorization":
        pred = model.predict_score(user_id, item_id)
        return f"Predicted rating: {pred:.1f}★" if pred == pred else "Learned latent preference"

    if name == "most_popular":
        return f"Popular overall ({popularity.get(item_id, 0)} ratings)"
    if name == "highest_average":
        return "High average rating across users"
    if name == "random":
        return "Random pick (baseline)"
    return ""


# title lookup that also works inside explain (small global set after load)
_TITLES = {}


def title_short(item_id, genres_by_id=None, train=None):
    return _TITLES.get(item_id, str(item_id))


# --- App ---------------------------------------------------------------------

art = load_artifacts()
st.title("\U0001F3AC Movie Recommender Prototype")

if art is None:
    st.error(
        "Model artifacts not found. Run **`python build_artifacts.py`** once to "
        "train and cache the models, then reload this page."
    )
    st.stop()

train, test, items, models = art["train"], art["test"], art["items"], art["models"]
title_by_id, genres_by_id, popularity = derived_lookups(train, items)
_TITLES.update(title_by_id)
user_ids = sorted(train[config.USER_COL].unique().tolist())

# Sidebar controls
st.sidebar.header("Controls")
if "user_id" not in st.session_state:
    st.session_state.user_id = user_ids[0]
if st.sidebar.button("\U0001F3B2 Random user"):
    st.session_state.user_id = int(np.random.default_rng().choice(user_ids))
user_id = st.sidebar.selectbox(
    "User", user_ids, index=user_ids.index(st.session_state.user_id)
)
st.session_state.user_id = user_id

algo_options = [a for a in ALGO_LABELS if a in models]
default_algos = [a for a in ["most_popular", "item_item_cf", "user_user_cf",
                             "content_based", "matrix_factorization"] if a in models]
selected = st.sidebar.multiselect(
    "Algorithms to compare", algo_options, default=default_algos,
    format_func=lambda a: ALGO_LABELS[a],
)
top_n = st.sidebar.slider("Top-N", 5, 20, config.TOP_K)
exclude_seen = st.sidebar.checkbox("Exclude already-seen", value=True)

tab_explore, tab_eval = st.tabs(["\U0001F3AC Explore", "\U0001F4CA Evaluation"])

with tab_explore:
    ur, genres, top_liked = user_profile(train, items, user_id)
    user_top_genres = genres.index.tolist()[:6]

    st.subheader(f"User {user_id} — taste profile")
    c1, c2, c3 = st.columns(3)
    c1.metric("Ratings in train", len(ur))
    c2.metric("Average rating", f"{ur[config.RATING_COL].mean():.2f}")
    c3.metric("Favourite genres", ", ".join(user_top_genres[:3]) or "—")

    with st.expander("Highest-rated movies by this user", expanded=False):
        st.dataframe(top_liked, hide_index=True, width="stretch")

    st.divider()
    st.subheader("Recommendations by algorithm")
    if not selected:
        st.info("Select at least one algorithm in the sidebar.")
    else:
        cols = st.columns(len(selected))
        for col, name in zip(cols, selected):
            with col:
                st.markdown(f"**{ALGO_LABELS[name]}**")
                recs = models[name].recommend(user_id, train, n=top_n, exclude_seen=exclude_seen)
                if not recs:
                    st.caption("No recommendations (model could not score this user).")
                for rank, (item_id, _score) in enumerate(recs, start=1):
                    title = title_by_id.get(item_id, item_id)
                    gl = ", ".join(genre_list(genres_by_id.get(item_id, ""))[:3])
                    why = explain(name, models[name], user_id, item_id, train,
                                  genres_by_id, popularity, user_top_genres)
                    st.markdown(f"**{rank}. {title}**")
                    if gl:
                        st.caption(f"_{gl}_")
                    st.caption(f"ℹ️ {why}")

with tab_eval:
    st.subheader("Quantitative comparison")
    st.caption("Generated by `python main.py` (K=10, relevance = rating ≥ 4.0, 599 users).")

    metrics_path = config.RESULTS_DIR / "metrics.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path, index_col="model")
        show_cols = ["precision@k", "recall@k", "ndcg@k", "mrr@k", "hit_rate@k",
                     "catalog_coverage", "novelty@k", "diversity@k"]
        st.dataframe(
            metrics[show_cols].style.format("{:.4f}")
            .background_gradient(cmap="Blues", subset=["ndcg@k", "precision@k"]),
            width="stretch",
        )
    else:
        st.warning("results/metrics.csv not found — run `python main.py`.")

    rating_path = config.RESULTS_DIR / "rating_metrics.csv"
    if rating_path.exists():
        st.markdown("**Rating-prediction quality** (models with `predict_score`)")
        st.dataframe(pd.read_csv(rating_path, index_col="model").style.format("{:.4f}"),
                     width="stretch")

    st.divider()
    fig_path = FIG_DIR / "accuracy_vs_coverage.png"
    if fig_path.exists():
        _, mid, _ = st.columns([1, 3, 1])
        mid.image(str(fig_path), caption="Accuracy vs. coverage trade-off", width="stretch")
