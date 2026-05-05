from __future__ import annotations

from pathlib import Path

import nbformat as nbf


PROJECT_DIR = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = PROJECT_DIR / "notebooks" / "Mood_Based_Music_Model_Training_Recommender.ipynb"


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source.strip())


def markdown(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(source.strip())


nb = nbf.v4.new_notebook()
nb["cells"] = [
    markdown(
        """
        # Mood-Based Music Recommendation System

        This notebook continues after preprocessing. It trains a supervised mood classifier from the cleaned audio-feature dataset, saves pickle files, and demonstrates mood-to-song recommendation for user text such as: `I am exhausted today`.
        """
    ),
    markdown("## 1. Import Libraries"),
    code(
        """
        import json
        from pathlib import Path

        import joblib
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        from sklearn.model_selection import train_test_split

        sns.set_theme(style="whitegrid")
        pd.set_option("display.max_columns", None)
        """
    ),
    markdown("## 2. Load Preprocessed Dataset"),
    code(
        """
        PROJECT_DIR = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
        DATA_PATH = PROJECT_DIR / "data" / "cleaned_mood_music_dataset.csv"

        df = pd.read_csv(DATA_PATH)
        print("Shape:", df.shape)
        df.head()
        """
    ),
    markdown("## 3. Select Model Features"),
    code(
        """
        FEATURE_COLUMNS = [
            "popularity_norm", "duration_norm", "danceability", "energy", "loudness_norm",
            "speechiness", "acousticness", "instrumentalness", "liveness", "valence",
            "tempo_norm", "explicit_numeric", "key_norm", "time_signature_norm", "mode_numeric",
        ]

        DISPLAY_COLUMNS = [
            "track_id", "artists", "album_name", "track_name", "popularity", "duration_ms",
            "danceability", "energy", "loudness", "speechiness", "acousticness",
            "instrumentalness", "liveness", "valence", "tempo", "track_genre",
            "mood", "mood_score", "mood_confidence_gap",
        ]
        SONG_PAYLOAD_COLUMNS = list(dict.fromkeys(DISPLAY_COLUMNS + FEATURE_COLUMNS))

        model_df = df[list(dict.fromkeys(FEATURE_COLUMNS + ["mood"] + DISPLAY_COLUMNS))].dropna(subset=FEATURE_COLUMNS + ["mood"])
        print(model_df["mood"].value_counts())
        """
    ),
    markdown("## 4. Train Mood Classification Model"),
    code(
        """
        X = model_df[FEATURE_COLUMNS]
        y = model_df["mood"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        mood_model = HistGradientBoostingClassifier(
            max_iter=220,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=0.02,
            class_weight="balanced",
            random_state=42,
        )

        mood_model.fit(X_train, y_train)
        y_pred = mood_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print("Test accuracy:", round(accuracy, 3))
        print(classification_report(y_test, y_pred, zero_division=0))
        """
    ),
    markdown("## 5. Confusion Matrix"),
    code(
        """
        labels = sorted(model_df["mood"].unique())
        cm = confusion_matrix(y_test, y_pred, labels=labels)

        plt.figure(figsize=(12, 8))
        sns.heatmap(cm, cmap="viridis", xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted mood")
        plt.ylabel("Actual mood")
        plt.title("Mood classifier confusion matrix")
        plt.tight_layout()
        plt.show()
        """
    ),
    markdown("## 6. User Text to Mood Intent Mapping"),
    code(
        """
        MOOD_INTENT_RULES = {
            "Calm": ["exhausted", "tired", "drained", "peaceful", "relax", "soothing", "sleep", "rest", "stress", "calm", "quiet"],
            "Chill": ["chill", "lazy", "evening", "lofi", "soft", "cozy", "unwind"],
            "Happy": ["happy", "good", "smile", "cheerful", "bright", "positive"],
            "Energetic": ["gym", "workout", "run", "energy", "boost", "pump"],
            "Euphoric": ["excited", "celebrate", "party", "dance", "festival"],
            "Sad": ["sad", "cry", "heartbroken", "lonely", "hurt", "miss"],
            "Melancholic": ["nostalgic", "memories", "rain", "empty", "bittersweet"],
            "Focus": ["study", "focus", "coding", "work", "concentrate", "productive"],
            "Intense": ["angry", "rage", "intense", "power", "aggressive", "heavy"],
            "Neutral": ["normal", "okay", "fine", "random", "anything"],
        }

        def normalize_text(text):
            return "".join(ch.lower() if ch.isalnum() else " " for ch in text)

        def detect_requested_mood(text, fallback="Calm"):
            normalized = normalize_text(text)
            scores = {}
            for mood, keywords in MOOD_INTENT_RULES.items():
                hits = [keyword for keyword in keywords if keyword in normalized]
                if hits:
                    scores[mood] = len(hits)
            if not scores:
                return fallback
            return sorted(scores.items(), key=lambda item: (item[1], item[0]), reverse=True)[0][0]

        detect_requested_mood("I am exhausted today and want soothing songs")
        """
    ),
    markdown("## 7. Recommendation Function"),
    code(
        """
        def recommend_songs(user_input, energy_level=30, n=10, prefer_popular=True):
            target_mood = detect_requested_mood(user_input)
            songs = model_df[model_df["mood"] == target_mood].copy()
            if songs.empty:
                songs = model_df.copy()

            desired_energy = energy_level / 100
            songs["energy_fit"] = 1 - (songs["energy"] - desired_energy).abs()
            songs["soothing_fit"] = (
                (1 - songs["energy"]) * 0.35
                + songs["acousticness"] * 0.25
                + (1 - songs["speechiness"]) * 0.15
                + songs["mood_score"].fillna(0.5) * 0.25
            )
            songs["popularity_fit"] = songs["popularity"].fillna(0) / 100
            popularity_weight = 0.18 if prefer_popular else 0.06
            songs["match_score"] = (
                songs["mood_score"].fillna(0.5) * 0.38
                + songs["energy_fit"].clip(0, 1) * 0.20
                + songs["soothing_fit"].clip(0, 1) * 0.25
                + songs["popularity_fit"].clip(0, 1) * popularity_weight
            )

            columns = ["track_name", "artists", "track_genre", "mood", "popularity", "energy", "valence", "acousticness", "match_score"]
            return target_mood, songs.sort_values(["match_score", "popularity"], ascending=False)[columns].head(n)

        mood, recs = recommend_songs("I am exhausted today. Please play soothing songs.", energy_level=25)
        print("Detected mood:", mood)
        recs
        """
    ),
    markdown("## 8. Save Pickle Files"),
    code(
        """
        model_payload = {
            "model": mood_model,
            "feature_columns": FEATURE_COLUMNS,
            "classes": sorted(model_df["mood"].unique().tolist()),
        }

        recommender_payload = {
            "songs": model_df[SONG_PAYLOAD_COLUMNS].drop_duplicates(subset=["track_id", "track_name", "artists"]).to_dict(orient="records"),
            "feature_columns": FEATURE_COLUMNS,
            "display_columns": DISPLAY_COLUMNS,
            "song_payload_columns": SONG_PAYLOAD_COLUMNS,
            "mood_intent_rules": MOOD_INTENT_RULES,
            "fallback_mood": "Calm",
        }

        joblib.dump(model_payload, PROJECT_DIR / "models" / "mood_model.pkl")
        joblib.dump(recommender_payload, PROJECT_DIR / "models" / "song_recommender.pkl")
        print("Saved mood_model.pkl and song_recommender.pkl")
        """
    ),
    markdown(
        """
        ## 9. Run the Streamlit App

        From this same folder, run:

        ```bash
        cd ..
        streamlit run app.py
        ```
        """
    ),
]

nbf.write(nb, NOTEBOOK_PATH)
print(f"Created {NOTEBOOK_PATH}")
