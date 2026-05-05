from __future__ import annotations

from pathlib import Path
from urllib.parse import quote_plus
import warnings

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "mood_model.pkl"
RECOMMENDER_PATH = BASE_DIR / "models" / "song_recommender.pkl"

warnings.filterwarnings("ignore", message="Could not find the number of physical cores.*")


st.set_page_config(
    page_title="MoodTune Recommender",
    page_icon="music",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
  --ink: #16202a;
  --muted: #5d6978;
  --line: #dfe6ee;
  --panel: rgba(255, 255, 255, 0.88);
  --green: #2f7d68;
  --rose: #c95f75;
  --amber: #c28422;
  --blue: #2f6f9f;
}

.stApp {
  background:
    linear-gradient(135deg, rgba(47, 125, 104, 0.13), rgba(201, 95, 117, 0.10) 44%, rgba(47, 111, 159, 0.12)),
    #f7f9fb;
  color: var(--ink);
  font-family: Inter, system-ui, sans-serif;
}

[data-testid="stHeader"] {
  background: transparent;
}

.block-container {
  padding-top: 2rem;
  padding-bottom: 3rem;
  max-width: 1240px;
}

.hero {
  border: 1px solid rgba(22, 32, 42, 0.08);
  background: linear-gradient(120deg, rgba(255,255,255,0.94), rgba(243,248,247,0.9));
  padding: 2rem;
  border-radius: 8px;
  box-shadow: 0 18px 60px rgba(24, 37, 51, 0.10);
}

.hero h1 {
  font-size: clamp(2.1rem, 5vw, 4.4rem);
  line-height: 0.96;
  margin: 0 0 0.8rem 0;
  letter-spacing: 0;
}

.hero p {
  max-width: 760px;
  color: var(--muted);
  font-size: 1.05rem;
  margin: 0;
}

.metric-card {
  border: 1px solid rgba(22, 32, 42, 0.08);
  background: var(--panel);
  padding: 1rem;
  border-radius: 8px;
  min-height: 112px;
}

.metric-card span {
  display: block;
  color: var(--muted);
  font-size: 0.85rem;
  font-weight: 600;
  text-transform: uppercase;
}

.metric-card strong {
  display: block;
  margin-top: 0.35rem;
  font-size: 1.6rem;
  color: var(--ink);
}

.song-card {
  border: 1px solid rgba(22, 32, 42, 0.08);
  background: rgba(255,255,255,0.9);
  border-radius: 8px;
  padding: 1rem;
  margin-bottom: 0.75rem;
}

.song-card h3 {
  margin: 0 0 0.25rem 0;
  font-size: 1.05rem;
}

.song-meta {
  color: var(--muted);
  margin-bottom: 0.65rem;
}

.tag {
  display: inline-block;
  border: 1px solid rgba(22, 32, 42, 0.12);
  border-radius: 999px;
  padding: 0.18rem 0.55rem;
  margin: 0 0.35rem 0.35rem 0;
  color: #334252;
  background: rgba(255,255,255,0.62);
  font-size: 0.82rem;
}

.small-note {
  color: var(--muted);
  font-size: 0.92rem;
}
</style>
"""


MOOD_COLORS = {
    "Calm": "#2f7d68",
    "Chill": "#2f6f9f",
    "Happy": "#c28422",
    "Energetic": "#b3532f",
    "Euphoric": "#c95f75",
    "Sad": "#51647a",
    "Melancholic": "#6f5a96",
    "Focus": "#455f5b",
    "Intense": "#8d3b35",
    "Neutral": "#68717d",
    "Live": "#6b7654",
    "Spoken": "#6b6470",
    "Party": "#b65e8d",
}


@st.cache_resource(show_spinner=False)
def load_artifacts() -> tuple[dict, dict]:
    model_bundle = joblib.load(MODEL_PATH)
    recommender_bundle = joblib.load(RECOMMENDER_PATH)
    return model_bundle, recommender_bundle


def normalize_text(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else " " for ch in text)


def detect_requested_mood(text: str, selected_mood: str, rules: dict, fallback_mood: str) -> tuple[str, list[str]]:
    if selected_mood != "Auto detect":
        return selected_mood, ["manual selection"]

    words = normalize_text(text).split()
    scores = {}
    matched = {}
    for mood, config in rules.items():
        keywords = config["keywords"] if isinstance(config, dict) else config
        hits = [keyword for keyword in keywords if keyword in words or keyword in normalize_text(text)]
        if hits:
            scores[mood] = len(hits)
            matched[mood] = hits

    if scores:
        mood = sorted(scores.items(), key=lambda item: (item[1], item[0]), reverse=True)[0][0]
        return mood, matched[mood]

    return fallback_mood, ["fallback"]


def score_recommendations(songs: pd.DataFrame, target_mood: str, intensity: int, prefer_popular: bool) -> pd.DataFrame:
    mood_songs = songs[songs["mood"] == target_mood].copy()
    if mood_songs.empty:
        mood_songs = songs.copy()

    desired_energy = intensity / 100
    mood_songs["energy_fit"] = 1 - (mood_songs["energy"] - desired_energy).abs()
    mood_songs["soothing_fit"] = (
        (1 - mood_songs["energy"]) * 0.35
        + mood_songs["acousticness"] * 0.25
        + (1 - mood_songs["speechiness"]) * 0.15
        + mood_songs["mood_score"].fillna(0.5) * 0.25
    )
    mood_songs["positive_fit"] = mood_songs["valence"] * 0.35 + mood_songs["danceability"] * 0.2
    mood_songs["popularity_fit"] = mood_songs["popularity"].fillna(0) / 100

    mood_weight = 0.38
    energy_weight = 0.20
    soothing_weight = 0.25 if target_mood in {"Calm", "Chill", "Sad", "Melancholic", "Focus"} else 0.10
    positive_weight = 0.16 if target_mood in {"Happy", "Energetic", "Euphoric", "Party"} else 0.07
    popularity_weight = 0.18 if prefer_popular else 0.06

    mood_songs["match_score"] = (
        mood_songs["mood_score"].fillna(0.5) * mood_weight
        + mood_songs["energy_fit"].clip(0, 1) * energy_weight
        + mood_songs["soothing_fit"].clip(0, 1) * soothing_weight
        + mood_songs["positive_fit"].clip(0, 1) * positive_weight
        + mood_songs["popularity_fit"].clip(0, 1) * popularity_weight
    )
    return mood_songs.sort_values(["match_score", "popularity"], ascending=False)


def spotify_search_url(track: str, artist: str) -> str:
    return f"https://open.spotify.com/search/{quote_plus(track + ' ' + artist)}"


def render_song_card(row: pd.Series, rank: int) -> None:
    title = row.get("track_name", "Unknown song")
    artist = row.get("artists", "Unknown artist")
    genre = row.get("track_genre", "music")
    score = row.get("match_score", 0)
    color = MOOD_COLORS.get(row.get("mood"), "#68717d")
    st.markdown(
        f"""
        <div class="song-card" style="border-left: 5px solid {color};">
          <div class="tag">#{rank}</div>
          <div class="tag">{row.get("mood", "Mood")}</div>
          <div class="tag">{genre}</div>
          <h3>{title}</h3>
          <div class="song-meta">{artist}</div>
          <span class="tag">{score * 100:.0f}% match</span>
          <span class="tag">Energy {row.get("energy", 0) * 100:.0f}%</span>
          <span class="tag">Valence {row.get("valence", 0) * 100:.0f}%</span>
          <span class="tag">Acoustic {row.get("acousticness", 0) * 100:.0f}%</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.link_button("Open on Spotify", spotify_search_url(str(title), str(artist)), use_container_width=True)


def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    if not MODEL_PATH.exists() or not RECOMMENDER_PATH.exists():
        st.error("Model files are missing. Run `python train_model.py` once, then start Streamlit again.")
        st.stop()

    model_bundle, recommender_bundle = load_artifacts()
    songs = pd.DataFrame(recommender_bundle["songs"])
    rules = recommender_bundle["mood_intent_rules"]
    available_moods = sorted(songs["mood"].dropna().unique().tolist())

    st.markdown(
        """
        <section class="hero">
          <h1>MoodTune Recommender</h1>
          <p>Type how you feel, and the app maps that feeling to a trained song-mood model before recommending tracks from your preprocessed music dataset.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Mood Controls")
        mood_text = st.text_area(
            "How are you feeling?",
            value="I am exhausted today and want something soothing.",
            height=130,
        )
        selected_mood = st.selectbox("Target mood", ["Auto detect"] + available_moods)
        intensity = st.slider("Energy level", 5, 95, 30)
        prefer_popular = st.toggle("Prefer popular songs", value=True)
        limit = st.slider("Number of songs", 5, 25, 10)
        recommend = st.button("Recommend Songs", type="primary", use_container_width=True)

    target_mood, matched_keywords = detect_requested_mood(
        mood_text,
        selected_mood,
        rules,
        recommender_bundle["fallback_mood"],
    )

    top_songs = score_recommendations(songs, target_mood, intensity, prefer_popular).head(limit)
    mood_profile = recommender_bundle["mood_profiles"].get(target_mood, {})
    mood_description = rules.get(target_mood, {}).get("description", "Recommended from the closest available mood.")

    metric_cols = st.columns(4)
    metric_cols[0].markdown(
        f'<div class="metric-card"><span>Detected Mood</span><strong>{target_mood}</strong></div>',
        unsafe_allow_html=True,
    )
    metric_cols[1].markdown(
        f'<div class="metric-card"><span>Matched Words</span><strong>{", ".join(matched_keywords[:2])}</strong></div>',
        unsafe_allow_html=True,
    )
    metric_cols[2].markdown(
        f'<div class="metric-card"><span>Dataset Tracks</span><strong>{len(songs):,}</strong></div>',
        unsafe_allow_html=True,
    )
    metric_cols[3].markdown(
        f'<div class="metric-card"><span>Mood Pool</span><strong>{int(mood_profile.get("count", 0)):,}</strong></div>',
        unsafe_allow_html=True,
    )

    st.write("")
    st.info(mood_description)

    chart_cols = st.columns([1, 1])
    with chart_cols[0]:
        st.subheader("Mood Audio Profile")
        if mood_profile:
            profile_df = pd.DataFrame(
                {
                    "feature": ["Energy", "Valence", "Acousticness", "Danceability"],
                    "value": [
                        mood_profile["avg_energy"],
                        mood_profile["avg_valence"],
                        mood_profile["avg_acousticness"],
                        mood_profile["avg_danceability"],
                    ],
                }
            )
            st.bar_chart(profile_df, x="feature", y="value", color="#2f7d68", height=280)
        st.caption("The model uses Spotify-style audio features such as energy, valence, acousticness, and danceability.")

    with chart_cols[1]:
        st.subheader("Try the Trained Classifier")
        sample = top_songs.iloc[[0]][model_bundle["feature_columns"]] if not top_songs.empty else None
        if sample is not None:
            predicted = model_bundle["model"].predict(sample)[0]
            probabilities = model_bundle["model"].predict_proba(sample)[0]
            class_names = model_bundle["model"].classes_
            probability = probabilities[list(class_names).index(predicted)]
            st.markdown(
                f"""
                <div class="metric-card">
                  <span>Top Song Model Prediction</span>
                  <strong>{predicted} ({probability * 100:.0f}%)</strong>
                  <p class="small-note">This verifies that the recommendation is aligned with the trained mood classifier.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.caption("User text chooses the target mood; the trained model validates song mood from audio features.")

    st.subheader("Recommended Songs")
    if recommend or True:
        if top_songs.empty:
            st.warning("No matching songs found. Try another mood or lower the filters.")
        else:
            for rank, (_, row) in enumerate(top_songs.iterrows(), start=1):
                render_song_card(row, rank)


if __name__ == "__main__":
    main()
