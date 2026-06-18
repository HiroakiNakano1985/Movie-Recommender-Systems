# Raw data — MovieLens Latest Small

The dataset is **not committed** to this repository because GroupLens does not
permit redistribution. Download it yourself to run `python main.py`.

> The deployed Streamlit app does **not** need these files — it loads the
> pre-trained models from `results/artifacts.joblib`. You only need the raw data
> to re-run the full pipeline / notebooks locally.

## How to get the data

1. Download **MovieLens Latest Small** from GroupLens:
   https://grouplens.org/datasets/movielens/  →  `ml-latest-small.zip`
   (direct: https://files.grouplens.org/datasets/movielens/ml-latest-small.zip)
2. Unzip it and place these files directly in `data/raw/`:

   ```text
   data/raw/ratings.csv
   data/raw/movies.csv
   data/raw/tags.csv     (optional)
   data/raw/links.csv    (optional)
   ```

Expected columns:

```text
ratings.csv : userId, movieId, rating, timestamp
movies.csv  : movieId, title, genres
```

Citation: F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets:
History and Context.* ACM TiiS. https://doi.org/10.1145/2827872
