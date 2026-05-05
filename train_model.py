from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "cleaned_mood_music_dataset.csv"
MODEL_PATH = PROJECT_DIR / "models" / "mood_model.pkl"
RECOMMENDER_PATH = PROJECT_DIR / "models" / "song_recommender.pkl"
METRICS_PATH = PROJECT_DIR / "models" / "model_metrics.json"

FEATURE_COLUMNS = [
    "popularity_norm",
    "duration_norm",
    "danceability",
    "energy",
    "loudness_norm",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo_norm",
    "explicit_numeric",
    "key_norm",
    "time_signature_norm",
    "mode_numeric",
]

DISPLAY_COLUMNS = [
    "track_id",
    "artists",
    "album_name",
    "track_name",
    "popularity",
    "duration_ms",
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "track_genre",
    "mood",
    "mood_score",
    "mood_confidence_gap",
]

SONG_PAYLOAD_COLUMNS = list(dict.fromkeys(DISPLAY_COLUMNS + FEATURE_COLUMNS))

MOOD_INTENT_RULES = {
    "Calm": {
        "keywords": [
            "exhausted",
            "tired",
            "drained",
            "peaceful",
            "relax",
            "relaxed",
            "soothing",
            "sleep",
            "rest",
            "anxious",
            "stress",
            "stressed",
            "calm",
            "quiet",
        ],
        "description": "Soft, low-energy, acoustic-leaning songs for slowing down.",
    },
    "Chill": {
        "keywords": ["chill", "lazy", "evening", "lofi", "soft", "cozy", "unwind", "breathe"],
        "description": "Easygoing tracks that feel relaxed without being too sleepy.",
    },
    "Happy": {
        "keywords": ["happy", "good", "smile", "cheerful", "bright", "positive", "light"],
        "description": "Positive songs with warmer valence and an upbeat feel.",
    },
    "Energetic": {
        "keywords": ["gym", "workout", "run", "running", "energy", "energetic", "boost", "pump"],
        "description": "Fast, active tracks for movement and motivation.",
    },
    "Euphoric": {
        "keywords": ["excited", "celebrate", "celebration", "party", "dance", "festival", "euphoric"],
        "description": "High-energy, high-valence songs for celebration.",
    },
    "Sad": {
        "keywords": ["sad", "cry", "heartbroken", "lonely", "hurt", "miss", "broken"],
        "description": "Gentle emotional tracks for sadness and reflection.",
    },
    "Melancholic": {
        "keywords": ["nostalgic", "memories", "rain", "empty", "bittersweet", "melancholic"],
        "description": "Reflective songs with a deeper emotional tone.",
    },
    "Focus": {
        "keywords": ["study", "focus", "coding", "work", "concentrate", "deep", "productive"],
        "description": "Lower-distraction songs for study, coding, and deep work.",
    },
    "Intense": {
        "keywords": ["angry", "rage", "intense", "power", "aggressive", "hard", "heavy"],
        "description": "Loud, forceful tracks for intense moods.",
    },
    "Neutral": {
        "keywords": ["normal", "okay", "fine", "random", "anything", "neutral"],
        "description": "Balanced recommendations when the input is broad.",
    },
}

FALLBACK_MOOD = "Calm"


def clean_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = list(dict.fromkeys(FEATURE_COLUMNS + ["mood"] + DISPLAY_COLUMNS))
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in cleaned dataset: {missing}")

    model_df = df[required].copy()
    model_df = model_df.dropna(subset=FEATURE_COLUMNS + ["mood"])
    model_df = model_df[model_df["mood"].astype(str).str.len() > 0]
    return model_df


def build_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=220,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=0.02,
        class_weight="balanced",
        random_state=42,
    )


def create_recommender_payload(df: pd.DataFrame) -> dict:
    songs = df[SONG_PAYLOAD_COLUMNS].copy()
    songs = songs.drop_duplicates(subset=["track_id", "track_name", "artists"])
    songs = songs.sort_values(
        by=["mood_score", "popularity", "mood_confidence_gap"],
        ascending=[False, False, False],
    )

    mood_profiles = {}
    for mood, group in df.groupby("mood"):
        mood_profiles[mood] = {
            "count": int(len(group)),
            "avg_popularity": float(group["popularity"].mean()),
            "avg_energy": float(group["energy"].mean()),
            "avg_valence": float(group["valence"].mean()),
            "avg_acousticness": float(group["acousticness"].mean()),
            "avg_danceability": float(group["danceability"].mean()),
            "avg_tempo": float(group["tempo"].mean()),
        }

    return {
        "songs": songs.to_dict(orient="records"),
        "feature_columns": FEATURE_COLUMNS,
        "display_columns": DISPLAY_COLUMNS,
        "song_payload_columns": SONG_PAYLOAD_COLUMNS,
        "mood_profiles": mood_profiles,
        "mood_intent_rules": MOOD_INTENT_RULES,
        "fallback_mood": FALLBACK_MOOD,
    }


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    model_df = clean_training_frame(df)

    x = model_df[FEATURE_COLUMNS]
    y = model_df["mood"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = build_model()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, zero_division=0, output_dict=True)

    joblib.dump(
        {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "classes": sorted(model_df["mood"].unique().tolist()),
        },
        MODEL_PATH,
    )
    joblib.dump(create_recommender_payload(model_df), RECOMMENDER_PATH)

    metrics = {
        "rows": int(len(model_df)),
        "features": FEATURE_COLUMNS,
        "accuracy": float(accuracy),
        "classification_report": report,
        "mood_counts": {mood: int(count) for mood, count in model_df["mood"].value_counts().items()},
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Trained rows: {len(model_df):,}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Saved: {MODEL_PATH.name}, {RECOMMENDER_PATH.name}, {METRICS_PATH.name}")


if __name__ == "__main__":
    main()
