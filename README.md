# 🎵 MoodTune Recommender

> A mood-based music recommendation system that maps how you feel into songs you'll love — powered by an audio-feature ML classifier and a Streamlit frontend.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Solution](#solution)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Running the App](#running-the-app)
- [Retraining the Model](#retraining-the-model)
- [Model Metrics](#model-metrics)
- [Dataset](#dataset)
- [Notebook](#notebook)

---

## Problem Statement

Existing music platforms recommend songs based on listening history or genre tags — not the listener's **current emotional state**. When someone feels exhausted, anxious, or joyful, no platform translates that free-form emotional expression into an intelligent recommendation backed by audio features. This leaves users manually browsing playlists to find music that fits the moment.

---

## Solution

MoodTune accepts a plain-text mood description (e.g. `"I am exhausted today"`), maps it to a mood class such as **Calm**, **Happy**, **Energetic**, or **Sad**, then uses a trained ML classifier on audio features — tempo, energy, valence, danceability — to surface the best-matching songs from a curated dataset. The full pipeline runs inside a lightweight Streamlit app with no login required.

---

## Features

| Feature | Description |
|---|---|
| 💬 Free-text mood input | Type any mood description in plain natural language |
| 🧠 Mood classification | Keyword rules map text to mood labels; ML model ranks matches |
| 🎼 Audio feature matching | Songs ranked by tempo, energy, valence, and danceability |
| 📊 Model metrics view | Accuracy, precision, and recall saved as `model_metrics.json` |
| 🔄 Re-trainable pipeline | One-command retraining via `src/train_model.py` |
| 📓 Notebook workflow | Full Jupyter notebook for exploration and demo |

---

## Tech Stack

| Layer | Tools |
|---|---|
| Frontend | Streamlit, Python 3 |
| ML / Modelling | scikit-learn, pickle, NumPy |
| Data processing | pandas, CSV pipeline |
| Notebooks & docs | Jupyter, JSON metrics |
| Runtime | Anaconda (`/opt/anaconda3`) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      DATA LAYER                         │
│          cleaned_mood_music_dataset.csv                 │
│          Audio features + mood labels                   │
└────────────┬──────────────────────────┬─────────────────┘
             │                          │
             ▼                          ▼
┌────────────────────────┐   ┌──────────────────────────┐
│     train_model.py     │   │   song_recommender.pkl   │
│   Trains classifier    │──▶│  Song bundle + rules     │
└────────────┬───────────┘   └──────────────┬───────────┘
             │                              │
             ▼                              │
┌────────────────────────┐                 │
│    mood_model.pkl      │                 │
│    Mood classifier     │                 │
└────────────┬───────────┘                 │
             │                             │
             ▼                             ▼
┌─────────────────────────────────────────────────────────┐
│                    APP LAYER  (app.py)                  │
│   User types mood  ──▶  Streamlit  ──▶  Top-N songs    │
└─────────────────────────────────────────────────────────┘
```

**Flow summary:**
1. The cleaned CSV is loaded from `data/`.
2. Numeric audio features train a mood classifier (`train_model.py`).
3. The trained classifier and recommendation bundle are saved as pickle files in `models/`.
4. The Streamlit app reads the pickle files at startup.
5. The user types a mood description; the app maps it to a mood class.
6. Matching songs are ranked and displayed as recommendations.

---

## Project Structure

```
Mood_Music_Recommender_Project/
├── app.py                          # Streamlit frontend
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── data/
│   └── cleaned_mood_music_dataset.csv   # Preprocessed dataset
├── docs/
│   └── PROJECT_STRUCTURE.txt       # File-by-file hierarchy notes
├── models/
│   ├── model_metrics.json          # Saved accuracy / precision / recall
│   ├── mood_model.pkl              # Trained mood classifier
│   └── song_recommender.pkl        # Song records + mood rules bundle
├── notebooks/
│   └── Mood_Based_Music_Model_Training_Recommender.ipynb
└── src/
    ├── create_notebook.py          # Regenerates the notebook file
    └── train_model.py              # Retrains model and recreates pickles
```

---

## Getting Started

### Prerequisites

- Python 3.8+ (via Anaconda recommended)
- All dependencies listed in `requirements.txt`

### Installation

```bash
# 1. Clone or download the project
cd "/Users/divvyas/Documents/ML FINAL/Mood_Music_Recommender_Project"

# 2. Install dependencies
/opt/anaconda3/bin/pip install -r requirements.txt
```

---

## Running the App

```bash
cd "/Users/divvyas/Documents/ML FINAL/Mood_Music_Recommender_Project"
/opt/anaconda3/bin/streamlit run app.py
```

Open the local URL printed by Streamlit in your browser:

```
http://localhost:8501
```

Type a mood description in the input box — for example:

- `"I am exhausted today"` → recommends **Calm** songs
- `"Feeling pumped and ready"` → recommends **Energetic** songs
- `"Kind of sad and reflective"` → recommends **Sad** songs

---

## Retraining the Model

To retrain the classifier from scratch using the cleaned dataset:

```bash
cd "/Users/divvyas/Documents/ML FINAL/Mood_Music_Recommender_Project"
/opt/anaconda3/bin/python3 src/train_model.py
```

This recreates the following files:

| File | Description |
|---|---|
| `models/mood_model.pkl` | Trained mood classifier |
| `models/song_recommender.pkl` | Song recommendation bundle |
| `models/model_metrics.json` | Updated evaluation metrics |

---

## Model Metrics

After training, evaluation results are stored in `models/model_metrics.json`. Typical fields include:

```json
{
  "accuracy": 0.91,
  "precision": 0.89,
  "recall": 0.88,
  "dataset_summary": {
    "total_songs": 1500,
    "mood_classes": ["Happy", "Sad", "Energetic", "Calm"]
  }
}
```

---

## Dataset

**File:** `data/cleaned_mood_music_dataset.csv`

The dataset has been preprocessed and contains the following audio features used by the classifier:

| Feature | Description |
|---|---|
| `tempo` | Beats per minute |
| `energy` | Intensity and activity level (0–1) |
| `valence` | Musical positiveness / happiness (0–1) |
| `danceability` | How suitable the track is for dancing (0–1) |
| `mood` | Target label (Happy, Sad, Energetic, Calm, …) |

---

## Notebook

An interactive training and recommendation walkthrough is available at:

```
notebooks/Mood_Based_Music_Model_Training_Recommender.ipynb
```

To regenerate the notebook from source:

```bash
/opt/anaconda3/bin/python3 src/create_notebook.py
```

---

*Built as an ML final project — MoodTune Recommender.*
