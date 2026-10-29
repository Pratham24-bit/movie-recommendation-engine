# 🎬 CineMatch 2.0 — Movie Recommendation Engine

A content-based movie recommendation system built with Python that helps users discover movies based on their favourite films, genres, actors, directors, language, and mood.

## ✨ Features

- 🔍 **Smart Search:** Search by movie title, genre, actor, director, language, or streaming platform.
- 🎯 **Movie Recommendations:** Discover similar movies using TF-IDF and cosine similarity.
- 🎭 **Mood-Based Picks:** Find movies based on your mood, from feel-good comedies to thrillers.
- 🎬 **Multi-Movie Recommendations:** Select multiple favourite movies to discover recommendations matching your combined preferences.
- ⚖️ **Movie Comparison:** Compare two movies by genre, cast, director, language, rating, and similarity.
- 🎲 **Surprise Me:** Get a random movie recommendation.
- 📌 **Personal Watchlist:** Save movies you want to watch.
- 🎛️ **Advanced Filters:** Filter by language, platform, and minimum rating.
- 📊 **Interactive Dashboard:** Explore dataset statistics, ratings, genres, languages, and streaming platforms.
- ▶️ **Watch Options:** Find official streaming searches, legal viewing options, and movie trailers.

## 🛠️ Tech Stack

- **Language:** Python
- **Interface:** Streamlit
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Recommendation Algorithm:** TF-IDF Vectorization and Cosine Similarity
- **Data Storage:** CSV and JSON files

## 🧠 How It Works

1. Each movie is represented by its genre, director, cast, and language.
2. These features are combined into a text profile, with genres and directors given extra weight.
3. TF-IDF converts the movie profiles into numerical vectors.
4. Cosine similarity measures how closely the movie profiles match.
5. The system ranks similar movies and displays recommendations based on the selected movie or preferences.

## 📁 Project Structure

```text
movie-recommendation-engine/
├── app.py
├── movie.csv
├── requirements.txt
└── README.md
```

The application also creates `watchlist.json` when saving watchlist data.

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or a compatible Python version
- pip

### Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/YOUR-USERNAME/movie-recommendation-engine.git
   ```

2. Open the project directory:

   ```bash
   cd movie-recommendation-engine
   ```

3. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Run the application:

   ```bash
   python -m streamlit run app.py
   ```

5. Open the local URL displayed in your terminal to use CineMatch.

## 📊 Dataset

The `movie.csv` file contains the movie metadata used by the recommendation engine, including movie titles, genres, directors, cast, language, platform, ratings, and other descriptive information.

## 🔮 Future Improvements

- Integrate the TMDB API for updated movie information and posters.
- Add user accounts and personalized profiles.
- Explore collaborative filtering.
- Improve recommendation quality with user feedback.
- Deploy the application online for public access.

## 👨‍💻 Project Information

**Project:** CineMatch 2.0 — Movie Recommendation Engine  
**Category:** Python / Machine Learning / Recommendation Systems

Built as a project to explore content-based recommendation techniques and interactive movie discovery.

---

If you find this project useful, consider starring the repository ⭐
