# Deploying the Streamlit app

The app is served live with **Streamlit Community Cloud** (free), which runs the
app directly from a **GitHub** repository. GitHub alone only stores the code — it
does not run it — so both pieces are needed:

```
push code to GitHub  →  connect the repo on Streamlit Community Cloud  →  public https://<app>.streamlit.app URL
```

## What is in the repo (and why it works without the dataset)

The deployed app loads everything it needs from pre-built artifacts, so it does
**not** require the MovieLens raw data at runtime:

- `app.py` — the Streamlit app (repo root = this folder, so `from src import ...` works)
- `src/` — model classes (needed to un-pickle the trained models)
- `results/artifacts.joblib` (~8 MB) — pre-trained models + train/test/items
- `results/metrics.csv`, `results/rating_metrics.csv`, `results/figures/*.png` — the Evaluation tab
- `requirements.txt` — dependencies Streamlit Cloud installs

The MovieLens data is intentionally **git-ignored** (not redistributable); it is
only needed to re-run `python main.py` locally.

## Step 1 — Push to GitHub

This folder is already a git repository with an initial commit. Create an **empty**
repo on GitHub (no README/.gitignore), then from this folder:

```bash
git remote add origin https://github.com/<your-username>/<your-repo>.git
git branch -M main
git push -u origin main
```

Confirm on GitHub that `app.py`, `src/`, and especially **`results/artifacts.joblib`**
were uploaded.

## Step 2 — Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in with GitHub.
2. **Create app → Deploy a public app from GitHub**.
3. Set:
   - **Repository:** `<your-username>/<your-repo>`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **Deploy**. First build takes a few minutes (it installs `requirements.txt`).
5. You get a public URL like `https://<your-app>.streamlit.app` — put this in your
   report and slides.

## Step 3 — Put the link where it will be graded

- Add the URL to the title slide / "The Prototype" slide and to the report.
- Also include 1–2 screenshots, so the demo is still evident if the free app has
  gone to sleep (Community Cloud idles inactive apps; they wake on the next visit).

## Troubleshooting

- **"Model artifacts not found"** → `results/artifacts.joblib` was not pushed
  (check it is not blocked by `.gitignore`; it is whitelisted on purpose).
- **Build fails on a dependency / version** → in the app's *Advanced settings* pin
  the Python version (3.12 or 3.13 are safe), then reboot the app.
- **Artifacts fail to un-pickle** (rare numpy/scipy version mismatch) → easiest fix
  is to regenerate them in an environment matching `requirements.txt`
  (`python build_artifacts.py`) and push the new file.
