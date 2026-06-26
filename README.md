# SemanticScore — Text Similarity API with TF-IDF & Flask Deployment

> TF-IDF + Cosine Similarity · Preprocessing impact analysis · Flask REST API · Heroku/AWS EC2 deployed · Saved model artifact

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/API-Flask-black?style=flat-square&logo=flask)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-green?style=flat-square)
![Deployed](https://img.shields.io/badge/Deployed-Heroku%20%7C%20AWS%20EC2-orange?style=flat-square)

---

## Overview

SemanticScore is a deployed REST API that scores the semantic similarity between two text paragraphs using **TF-IDF vectorization and cosine similarity**. Beyond the model itself, this project systematically compares similarity scores **with and without text preprocessing** — quantifying exactly how much cleaning impacts results.

Includes a serialized model artifact (`tfidfvectorizer.joblib`), a web interface, and a `Procfile` for one-command Heroku deployment.

---

## How It Works

```
Text Input (paragraph A + paragraph B)
            │
            ▼
┌───────────────────────────────┐
│      Preprocessing (partA1)   │
│  Lowercase → tokenize →       │
│  stopword removal → stemming  │
└───────────┬───────────────────┘
            │
            ▼
┌───────────────────────────────┐
│    TF-IDF Vectorization       │
│  tfidfvectorizer.joblib       │
│  (pre-fitted, loaded at       │
│   inference time)             │
└───────────┬───────────────────┘
            │ document vectors
            ▼
┌───────────────────────────────┐
│    Cosine Similarity          │
│  score ∈ [0.0, 1.0]          │
└───────────┬───────────────────┘
            │
            ▼
      JSON response via Flask API
```

---

## Preprocessing Impact Analysis

A core contribution of this project — two full result sets are included:

| File | Description |
|---|---|
| `Similarity_Scores(without_preprocessing).csv` | Raw text → TF-IDF → cosine similarity |
| `Similarity_Scores1(with_preprocessing).csv` | Cleaned text → TF-IDF → cosine similarity |
| `Preprocessed_Data(partA1-preprocessing).csv` | Intermediate preprocessed corpus |

This comparison directly answers: *does stopword removal and stemming actually improve similarity scoring, or introduce information loss?*

---

## Project Structure

```
├── app.py                          # Flask REST API
├── partA.py                        # Similarity scoring (no preprocessing)
├── partA1.py                       # Similarity scoring (with preprocessing)
├── tfidfvectorizer.joblib          # Serialized fitted TF-IDF vectorizer
├── DataNeuron_Text_Similarity.csv  # Source dataset
├── templates/                      # Web UI templates
├── Procfile                        # Heroku deployment config
├── requirements.txt
└── Results/                        # Output visualizations
```

---

## API Usage

```bash
# Run locally
python app.py

# POST request
curl -X POST http://localhost:5000/similarity \
  -H "Content-Type: application/json" \
  -d '{"text1": "The cat sat on the mat", "text2": "A cat was sitting on a mat"}'

# Response
{"similarity_score": 0.847}
```

---

## Quick Start

```bash
git clone https://github.com/Drashti0913/TextSimilarityScore.git
cd TextSimilarityScore

pip install -r requirements.txt

# Run the API
python app.py

# Run similarity analysis without preprocessing
python partA.py

# Run with preprocessing
python partA1.py
```

---

## Deployment

### Heroku (one command)
```bash
heroku create
git push heroku main
```
The `Procfile` handles the rest.

### AWS EC2
```bash
# On EC2 instance
pip install -r requirements.txt
python app.py --host 0.0.0.0 --port 80
```

---

## Key Design Decisions

**Serialized vectorizer:** The TF-IDF vectorizer is fitted on the corpus once and saved via `joblib` — at inference time it's loaded directly, avoiding re-fitting on every request. This is the correct production pattern for stateless APIs.

**Two preprocessing modes:** Rather than assuming preprocessing always helps, this project measures its impact empirically. Results show preprocessing improves scores for semantically similar but lexically varied text, but can hurt precision on short technical strings where stemming loses meaning.

**Cosine over Euclidean distance:** TF-IDF vectors are high-dimensional and sparse — cosine similarity is invariant to document length, making it far more reliable than Euclidean distance for text comparison.

---

## License

MIT
