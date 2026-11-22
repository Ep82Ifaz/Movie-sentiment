"""
Movie Word Cloud - Flask app

Type a movie name -> the app finds reviews that mention it, scores them with the
model saved by imdb_sentiment_analysis.ipynb, and returns a word cloud (JSON).

Folder layout (next to this file):
    app.py
    templates/index.html
    IMDB Dataset.csv          <- the same CSV used in the notebook
    outputs/best_model.pkl    <- saved by the notebook (optional but recommended)

Run:
    pip install flask wordcloud pandas scikit-learn joblib
    python app.py            -> http://127.0.0.1:5000

Optional: set TMDB_API_KEY to also pull live reviews for the movie from TMDB.
"""
import base64
import html
import io
import json
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request
from wordcloud import STOPWORDS, WordCloud

BASE = Path(__file__).parent
DATA_PATH = Path(os.environ.get("IMDB_CSV", BASE / "IMDB Dataset.csv"))
MODEL_PATH = BASE / "outputs" / "best_model.pkl"
TMDB_API_KEY = os.environ.get("TMDB_API_KEY")

MAX_REVIEWS_FOR_CLOUD = 400
MIN_REVIEWS = 3

app = Flask(__name__)


# ---------- same cleaning as the notebook ----------
def clean_text(text):
    text = html.unescape(str(text))
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-zA-Z']", " ", text)
    return re.sub(r"\s+", " ", text).lower().strip()


# ---------- load data + model once at startup ----------
df = None
if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH, usecols=["review", "sentiment"]).dropna().drop_duplicates()
    df["clean_review"] = df["review"].map(clean_text)
    df = df[df.clean_review.str.len() > 0].reset_index(drop=True)
    print(f"Loaded {len(df):,} reviews from {DATA_PATH.name}")
else:
    print(f"WARNING: {DATA_PATH} not found - dataset search disabled")

model = None
if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)["model"]
    print("Loaded sentiment model")
else:
    print("WARNING: outputs/best_model.pkl not found - using dataset labels only")

EXTRA_STOPWORDS = set(STOPWORDS) | {
    "movie", "movies", "film", "films", "one", "even", "really", "just", "much",
    "also", "will", "would", "could", "get", "got", "see", "seen", "watch",
    "watching", "watched", "make", "makes", "made", "way", "thing", "things",
    "s", "t", "ve", "ll", "d", "m", "re", "br", "don", "didn", "doesn", "isn",
    "wasn", "can", "like", "story", "time", "character", "characters",
}


# ---------- review sources ----------
def reviews_from_dataset(name):
    """Reviews (with their labels) whose text mentions the movie name."""
    if df is None:
        return []
    pattern = r"\b" + re.escape(clean_text(name)) + r"\b"
    hits = df[df.clean_review.str.contains(pattern, regex=True)]
    return hits.head(2000)[["clean_review", "sentiment"]].to_dict("records")


def reviews_from_tmdb(name):
    """Optional live reviews from TMDB (needs TMDB_API_KEY)."""
    if not TMDB_API_KEY:
        return []
    try:
        q = urllib.parse.quote(name)
        url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={q}"
        with urllib.request.urlopen(url, timeout=8) as r:
            results = json.load(r).get("results", [])
        if not results:
            return []
        url = f"https://api.themoviedb.org/3/movie/{results[0]['id']}/reviews?api_key={TMDB_API_KEY}"
        with urllib.request.urlopen(url, timeout=8) as r:
            revs = json.load(r).get("results", [])
        return [{"clean_review": clean_text(x["content"]), "sentiment": None} for x in revs]
    except Exception as exc:  # network / key problems shouldn't kill the request
        print("TMDB lookup failed:", exc)
        return []


# ---------- word cloud ----------
def make_cloud_png(texts, movie_name):
    stop = EXTRA_STOPWORDS | set(clean_text(movie_name).split())
    cloud = WordCloud(
        width=1100,
        height=600,
        background_color="#0f1220",
        colormap="cool",
        max_words=150,
        collocations=False,
        stopwords=stop,
    ).generate(" ".join(texts))
    buf = io.BytesIO()
    cloud.to_image().save(buf, format="PNG")
    top = sorted(cloud.words_.items(), key=lambda kv: -kv[1])[:10]
    return base64.b64encode(buf.getvalue()).decode(), [w for w, _ in top]


# ---------- routes ----------
@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/wordcloud")
def api_wordcloud():
    payload = request.get_json(silent=True) or {}
    name = (payload.get("movie") or "").strip()
    if len(name) < 2:
        return jsonify(error="Please enter a movie name."), 400

    records = reviews_from_dataset(name) + reviews_from_tmdb(name)
    if len(records) < MIN_REVIEWS:
        return jsonify(
            error=f"Only {len(records)} review(s) mention \"{name}\". Try another title "
                  "or a shorter/more common spelling."
        ), 404

    texts = [r["clean_review"] for r in records]

    # sentiment: model prediction if available, else the dataset label
    if model is not None:
        preds = list(model.predict(texts))
    else:
        preds = [r["sentiment"] for r in records if r["sentiment"]]
    pos = sum(p == "positive" for p in preds)
    neg = sum(p == "negative" for p in preds)
    total = max(pos + neg, 1)

    image, top_words = make_cloud_png(texts[:MAX_REVIEWS_FOR_CLOUD], name)
    return jsonify(
        movie=name,
        review_count=len(records),
        positive=pos,
        negative=neg,
        positive_pct=round(100 * pos / total, 1),
        top_words=top_words,
        image=f"data:image/png;base64,{image}",
    )


if __name__ == "__main__":
    app.run(debug=True)
