# 🎬 Movie Recommender System

> **Movie Recommender System** is a full-stack, content-based recommendation platform built using **Python**, **Django**, **React.js**, **Scikit-Learn**, and **NLP (Natural Language Processing)**. Powered by the TMDB 5000 Movies Dataset, it features a vector similarity engine (`CountVectorizer` + `Cosine Similarity`) and automated poster/streaming link fetching via TMDB APIs with multi-level Django caching.

---

## 📌 Key Features

- **Full-Stack Architecture**: Responsive frontend integrated with Django REST API view controllers.
- **NLP Feature Extraction**:
  - Combined features (`overview`, `genres`, `keywords`, `cast`, `crew`) into unified metadata tags.
  - Text normalization, whitespace removal, and **Porter Stemmer** root-word reduction.
- **Vector Space Similarity Model**:
  - `CountVectorizer` (Bag-of-Words) transforming textual profiles into 5,000-dimensional vectors.
  - **Cosine Similarity** matrix calculation determining item distance across 4,800+ movies.
- **High-Performance Inference**: Precalculated recommendation dictionaries serialized via **Pickle** for instant (<200ms) recommendation generation.
- **Asynchronous TMDB API Integration**:
  - Parallelized poster image loading using `ThreadPoolExecutor`.
  - 30-day persistent caching via Django Cache framework.
  - Live watch provider streaming link lookup (`flatrate` US providers).

---

## 🏗️ System Workflow

```
[ TMDB 5000 Movies & Credits CSV ]
                 │
                 ▼
[ Data Preprocessing & Cleaning ] ── (AST literal evaluation of JSON fields)
                 │
                 ▼
[ Tag Generation & NLP Stemming ] ── (PorterStemmer: e.g., "loving", "loved" ──► "love")
                 │
                 ▼
[ Bag-of-Words Vectorization ]    ── (CountVectorizer max_features=5000)
                 │
                 ▼
[ Cosine Similarity Computation ] ── (Angle calculation across 5,000D space)
                 │
                 ▼
[ Pickle Serialization ]           ── (Save to artifacts/movie_dict.pkl & recommendations.pkl)
                 │
                 ▼
[ Django Web App Service Layer ]   ── (Parallel poster fetch + 30-day Django Cache)
```

---

## 🛠️ Technology Stack

| Layer | Technology | Usage |
| :--- | :--- | :--- |
| **Backend Framework** | Django 5.x | Web app serving, routing, API orchestration |
| **Frontend UI** | HTML5, CSS3, JavaScript / React.js | Dynamic UI with live search & poster rendering |
| **Machine Learning** | Scikit-Learn, NumPy, Pandas | Vectorization, Cosine similarity, Data wrangling |
| **NLP** | NLTK (PorterStemmer), AST | Text tokenization & JSON string parsing |
| **Caching & Async** | Django Cache, `concurrent.futures` | Parallel poster fetching & 30-day response caching |
| **External API** | TMDB REST API v3 | Movie poster artwork & streaming platform links |

---

## 📁 Directory Structure & File Map

```text
Movie-Recommender-System-Using-Machine-Learning/
├── data/                                    # TMDB CSV datasets
│   ├── tmdb_5000_movies.csv
│   └── tmdb_5000_credits.csv
├── artifacts/                               # Generated model artifacts (.pkl)
│   ├── movie_dict.pkl                       # Pickled movie metadata dictionary
│   ├── recommendations.pkl                  # Precalculated top-10 index mappings
│   └── similarity.pkl                       # Raw 4800x4800 similarity matrix
├── generate_models.py                       # ML pipeline script (Preprocessing & Vectorization)
├── web_app/                                 # Django web application
│   ├── manage.py                            # Django entry point
│   └── main/                                # Primary app directory
│       ├── views.py                         # Controller logic (TMDB fetching, caching, render)
│       └── templates/main/index.html        # Frontend template
├── Dockerfile                               # Containerization script
└── requirements.txt                         # Python dependencies
```

---

## ⚙️ Installation & Developer Guide

### 1. Prerequisites
- **Python 3.10+**
- **Git**

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/monudbg/Movie-Recommender-System-Using-Machine-Learning.git
cd Movie-Recommender-System-Using-Machine-Learning

# Create & activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Generate ML Models & Artifacts
Run the preprocessing and vectorization script to create the `.pkl` files inside `artifacts/`:
```bash
python generate_models.py
```

### 4. Run Django Web Server
```bash
cd web_app
python manage.py migrate
python manage.py runserver
```
Open `http://127.0.0.1:8000/` in your browser.

---

## 👨‍💻 Developer Notes & Code Documentation

- **`generate_models.py`**:
  - `convert()` & `convert3()` parse JSON strings using `ast.literal_eval`.
  - `PorterStemmer` reduces inflectional forms of words to common root stems.
  - Precomputes the top-10 recommendation index map to eliminate runtime matrix multiplication overhead.
- **`web_app/main/views.py`**:
  - `fetch_poster()` & `fetch_watch_provider()` interface with TMDB API with 30-day Django file-based caching.
  - `ThreadPoolExecutor` fetches posters in parallel to accelerate page rendering speeds.

---

## 📜 Author & License

- **Monu Manish** (IIIT Una)
- [GitHub](https://github.com/monudbg) | [LinkedIn](https://linkedin.com/in/monu-manish-64145428a)
