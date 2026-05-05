# MoodTune Recommender

MoodTune Recommender is a mood-based music recommendation ML project. It uses the preprocessed music dataset, trains a mood classifier from audio features, saves pickle files, and runs a Streamlit frontend where the user can type their mood and get song recommendations.

## Project Structure

```text
Mood_Music_Recommender_Project/
├── app.py
├── README.md
├── requirements.txt
├── data/
│   └── cleaned_mood_music_dataset.csv
├── docs/
│   └── PROJECT_STRUCTURE.txt
├── models/
│   ├── model_metrics.json
│   ├── mood_model.pkl
│   └── song_recommender.pkl
├── notebooks/
│   └── Mood_Based_Music_Model_Training_Recommender.ipynb
└── src/
    ├── create_notebook.py
    └── train_model.py
```

## Important Files

- `app.py` - Streamlit app for user mood input and song recommendations.
- `data/cleaned_mood_music_dataset.csv` - Cleaned/preprocessed dataset used by the project.
- `models/mood_model.pkl` - Trained ML model for mood prediction from audio features.
- `models/song_recommender.pkl` - Recommendation data bundle with song records and mood rules.
- `models/model_metrics.json` - Saved model metrics and dataset summary.
- `notebooks/Mood_Based_Music_Model_Training_Recommender.ipynb` - Notebook version of training and recommendation workflow.
- `src/train_model.py` - Script to retrain the model and recreate pickle files.
- `src/create_notebook.py` - Script to recreate the notebook file.
- `docs/PROJECT_STRUCTURE.txt` - One-line explanation of every file and the hierarchy.
- `requirements.txt` - Dependencies needed to run the project.

## Run the App

```bash
cd "/Users/divvyas/Documents/ML FINAL/Mood_Music_Recommender_Project"
/opt/anaconda3/bin/streamlit run app.py
```

Open the local URL printed by Streamlit, usually:

```text
http://localhost:8501
```

## Retrain the Model

```bash
cd "/Users/divvyas/Documents/ML FINAL/Mood_Music_Recommender_Project"
/opt/anaconda3/bin/python3 src/train_model.py
```

This recreates:

- `models/mood_model.pkl`
- `models/song_recommender.pkl`
- `models/model_metrics.json`

## Workflow

1. The cleaned CSV is loaded from `data/`.
2. Numeric audio features are used to train a mood classifier.
3. The trained classifier and recommendation bundle are saved in `models/`.
4. The Streamlit app reads the pickle files and accepts text such as `I am exhausted today`.
5. The app maps the input to a mood such as `Calm` and recommends suitable songs.
