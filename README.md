# Movie Word Cloud

A small Flask app. Type a movie name and get a word cloud of what reviewers say about it, plus the positive/negative split and the top words.

## How it works

The IMDB dataset used in `imdb_sentiment_analysis.ipynb` has only `review` and `sentiment` columns, with no movie titles. So the app:

1. Finds every review whose text mentions the movie name you typed.
2. Scores those reviews with the model saved by the notebook (`outputs/best_model.pkl`). If the model is missing, it uses the dataset's own labels.
3. Builds a word cloud from the matching reviews and returns everything as JSON.

At least 3 matching reviews are needed, otherwise the app asks you to try another title.

## Project layout

```
movie-wordcloud/
├── app.py                  # Flask backend + JSON API
├── templates/
│   └── index.html          # Web interface
├── IMDB Dataset.csv        # Same CSV used in the notebook (you provide this)
└── outputs/
    └── best_model.pkl      # Saved by the notebook (recommended)
```

`index.html` may also sit directly next to `app.py`.

## Setup

```bash
pip install flask wordcloud pandas scikit-learn joblib
python app.py
```

Open http://127.0.0.1:5000, type a movie name (e.g. `Titanic`) and press **Generate**.

## API

**POST** `/api/wordcloud`

Request:
```json
{ "movie": "Titanic" }
```

Response (200):
```json
{
  "movie": "Titanic",
  "review_count": 312,
  "positive": 190,
  "negative": 122,
  "positive_pct": 60.9,
  "top_words": ["acting", "music", "love", "..."],
  "image": "data:image/png;base64,..."
}
```

Errors: `400` if the name is empty or too short, `404` if too few reviews mention the title. Both return `{"error": "..."}`.

## Configuration

| Setting | How | Purpose |
|---|---|---|
| `IMDB_CSV` | environment variable | Path to the dataset if it is not next to `app.py` |
| `TMDB_API_KEY` | environment variable | Optional. Also pulls live reviews for the movie from TMDB |
| `MAX_REVIEWS_FOR_CLOUD`, `MIN_REVIEWS` | constants in `app.py` | Reviews used for the cloud, and the minimum required |

## Notes and limits

- Matching is by name in the review text, so very common titles (e.g. "Batman") can include reviews about other films.
- Common filler words ("movie", "film", etc.) and the title itself are removed from the cloud.
- The dataset and model are loaded once at startup, so restart the app after replacing them.

## Troubleshooting

- **Template not found / index.html not found:** put `index.html` in a `templates` folder next to `app.py`.
- **Dataset not found warning:** place `IMDB Dataset.csv` next to `app.py` or set `IMDB_CSV`.
- **Model warning:** run the notebook to create `outputs/best_model.pkl`, or ignore it to use dataset labels only.
