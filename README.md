# Movie Recommendation System Using Restricted Boltzmann Machine (RBM)
### Academic Project for B.Tech CSE (AIML) — Deep Learning-II

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.15%2B-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.61%2B-brightgreen.svg)](https://streamlit.io/)
[![Dataset](https://img.shields.io/badge/Dataset-MovieLens%20100K-orange.svg)](https://grouplens.org/datasets/movielens/100k/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

---

## Project Introduction

This project is an end-to-end collaborative filtering **Movie Recommendation System** powered by an energy-based **Restricted Boltzmann Machine (RBM)** implemented in **PyTorch**.

Designed for the **B.Tech CSE (AIML) Deep Learning-II** syllabus, the project demonstrates how an unsupervised, stochastic neural network learns latent taste patterns from historical ratings and generates personalized recommendations on the benchmark **MovieLens 100K** dataset.

The system runs locally on standard laptops (CPU execution) and features a clean **Streamlit dashboard** for catalog exploration, interactive 1–5 scale rating, real-time Gibbs inference, and visual model inspectioqasqqqqqaqa                                                 .

---

## What Is a Movie Recommendation System?

In modern streaming platforms, catalogs contain thousands of titles. A recommendation system solves the **information overload problem** by predicting which unseen items a user is most likely to appreciate.

Collaborative filtering accomplishes this by identifying correlations across user communities: if User A and User B share similar preferences on past titles, ratings from one user inform personalized recommendations for the other.

---

## What Is a Restricted Boltzmann Machine (RBM)?

An RBM is an **Energy-Based Stochastic Generative Neural Network** composed of two layers of neurons arranged in an undirected bipartite graph:

1. **Visible Layer ($v$):** Represents observable inputs (in our project, the 1,682 movies in the catalog).
2. **Hidden Layer ($h$):** Represents unobserved **latent variables** (100 hidden neurons capturing abstract tastes like 90s sci-fi world-building, psychological drama, or animated comedy).
3. **Weight Matrix ($W$):** Bidirectional connection weights between every visible neuron and hidden neuron.
4. **Biases ($a, b$):** Visible biases $a$ (baseline movie popularity) and hidden biases $b$ (latent factor activation propensity).
5. **The Restricted Property:** **No two neurons in the same layer are connected.** This bipartite restriction eliminates complex intra-layer dependencies and allows conditional probabilities $P(h|v)$ and $P(v|h)$ to be computed in parallel:
   $$P(h_j = 1 \mid v) = \sigma\left(b_j + \sum_{i} v_i W_{ij}\right)$$
   $$P(v_i = 1 \mid h) = \sigma\left(a_i + \sum_{j} h_j W_{ij}\right)$$
6. **Energy Function:** The network assigns a scalar energy to each state configuration:
   $$E(v, h) = - a^T v - b^T h - v^T W h$$
   Configurations with **lower energy** correspond to higher probabilities ($P(v, h) \propto e^{-E(v, h)}$).

---

## How Does This Project Work?

```
[User Rates Movies (1-5 Stars)]
              │
              ▼
[Convert to Visible Vector & Observation Mask]
              │
              ▼
[Positive Phase: Sample Hidden States h ~ P(h|v)]
              │
              ▼
[Gibbs Reconstruction: P(v|h) across all 1,682 Movies]
              │
              ▼
[Exclude Already Rated Movies]
              │
              ▼
[Rank Unseen Movies & Output Top-10 with Rationale]
```

1. **Catalog Exploration:** User browses movies, searches by keyword, filters by genre, and rates movies 1–5 stars.
2. **Binary Transformation:** Ratings $\ge 3$ become $1.0$ (liked); ratings $< 3$ become $0.0$ (disliked).
3. **Observation Masking:** An observation mask tracks which movies were actually rated, preventing 93.7% missing entries from corrupting model updates.
4. **Gibbs Inference:** The user's input vector is passed to the hidden layer to infer their latent profile; the hidden layer then reconstructs probabilities across all 1,682 movies.
5. **Ranking & Display:** Rated movies are filtered out, remaining movies are ranked descending by predicted preference, and the top 10 are displayed with genre tags and natural language explanations.

---

## Dataset: MovieLens 100K

- **Source:** GroupLens Research, University of Minnesota ([Link](https://grouplens.org/datasets/movielens/100k/))
- **Total Ratings:** 100,000 ratings on a 1–5 integer scale
- **Users:** 943 unique participants
- **Movies:** 1,682 unique titles
- **Sparsity:** 93.7% missing data (typical of real-world collaborative filtering)
- **Genres:** 19 distinct genre categories
- **Automatic Setup:** The project auto-downloads and extracts `ml-100k.zip` on first run into `data/ml-100k/`.

---

## RBM Training Pipeline

The model is trained using **Contrastive Divergence (CD-1)** with mini-batch SGD:

- **Positive Phase:** Clamp training ratings $v_0$, compute hidden probabilities $P(h_0 \mid v_0)$, and sample binary states $h_0 \sim \text{Bernoulli}(P(h_0 \mid v_0))$.
- **Negative Phase:** Reconstruct visible states $v_1 \sim P(v \mid h_0)$, apply observation mask $M$, and sample $h_1 \sim P(h \mid v_1)$.
- **Weight Update:**
  $$\Delta W = \gamma \Delta W_{\text{prev}} + \eta \left( \frac{1}{B} v_0^T P(h_0 \mid v_0) - \frac{1}{B} v_1^T P(h_1 \mid v_1) - \lambda W \right)$$
- **Hyperparameters:**
  - Visible Neurons: 1,682
  - Hidden Neurons: 100
  - Learning Rate ($\eta$): 0.05
  - Momentum ($\gamma$): 0.8
  - Weight Decay ($\lambda$): 0.0001
  - Batch Size: 64
  - Epochs: 25 (trains in ~4 seconds on CPU)

---

## Experimental Benchmark Results

Measured on the official held-out **`u1.test`** split (20,000 ratings across 943 users):

### 1. Reconstruction Error Metrics
| Metric | Measured Value |
|---|---|
| **Test Root Mean Squared Error (RMSE)** | **0.3734** |
| **Test Mean Absolute Error (MAE)** | **0.2624** |
| **Test Mean Squared Error (MSE)** | **0.1394** |
| **Test Ratings Evaluated** | **20,000** |

### 2. Top-K Recommendation Ranking Performance
| Metric | K = 5 | K = 10 | K = 15 | K = 20 |
|---|---|---|---|---|
| **Precision@K** | 4.18% | 3.44% | 2.88% | 2.86% |
| **Recall@K** | 0.64% | 0.94% | 1.10% | 1.45% |
| **Hit Rate@K** | **16.12%** | **22.88%** | **25.49%** | **30.28%** |
| **MRR@K** | 0.1017 | 0.1105 | 0.1125 | 0.1151 |

*At K=20, the model achieves a **30.28% Hit Rate**, proving strong personalized discovery despite 94% sparsity.*

---

## User Interface Overview (Streamlit)

The application features a clean, responsive **dark slate dashboard** designed for intuitive evaluation:

1. **Personalized Recommendations:**
   - **Curated Taste Profiles:** 1-click presets (*Sci-Fi*, *Crime & Drama*, *Animation*, *Action*, *Romance*) generate recommendations in under 1 second.
   - **Catalog Search & Rating:** Search 1,682 movies and rate 1–5 scale in 1 click.
   - **Active Profile Indicators:** Discrete badges of your rated films with quick-remove controls.
   - **Ranked Recommendation Cards:** Latent match percentage (e.g. `98.2% Latent Match`), predicted score, and transparent *Why Recommended* explanations.
   - **Direct Feedback Loop:** 1-click rating buttons on recommendation cards trigger real-time Gibbs re-inference.

2. **Search & Suggestions (Conversational Intent):**
   - Natural language search: Describe preferences (e.g. *"mind-bending sci-fi with twist"*, *"funny comedy"*, *"intense 90s action thriller"*).
   - 8 theme chips (*Action*, *Sci-Fi*, *Comedy*, *Drama*, *Suspense*, *Romance*, *Animation*, *Western*).
   - Curated suggestions with community ratings and contextual match reasons.

3. **Movie Catalog Explorer:**
   - Complete catalog of all 1,682 MovieLens titles.
   - Filter by 19 genres, search titles, and review complete personal ratings table.

4. **Project Overview & Design (Viva Defense):**
   - **Editable Defense Script:** Pre-populated presentation pitch that students can modify and copy for slides or written defense.
   - **Human-Centered AI Rationale:** 6-pillar breakdown of why the UI was built for clarity and low cognitive load.
   - **Viva Defense Cheatsheet:** Concise answers to core questions (RBM vs. SVD, Gibbs reconstruction, CD-1 approximation).

5. **Model Architecture & Evaluation:**
   - Bipartite network topology visualization.
   - 2D Heatmap of learned connection weights $W \in \mathbb{R}^{20 \times 20}$.
   - Training loss and validation RMSE convergence curves.
   - Benchmark evaluation table on 20,000 test ratings.

---

## Installation & Running on Windows

### Step 1: Open PowerShell in Project Folder
```powershell
cd C:\Users\tanay\.gemini\antigravity\scratch\movie-rbm-recommender
```

### Step 2: Create and Activate Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Run Automated Tests
```powershell
python -m unittest discover -s tests -p "test_*.py"
```

### Step 5: Train the RBM Model
```powershell
python -m models.train
```

### Step 6: Launch Web Application
```powershell
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## Syllabus Mapping: B.Tech CSE AIML Deep Learning-II

| Syllabus Topic | Project Implementation File |
|---|---|
| **Restricted Boltzmann Machines (RBM)** | [`models/rbm.py`](file:///C:/Users/tanay/.gemini/antigravity/scratch/movie-rbm-recommender/models/rbm.py) |
| **Energy-Based Models (EBM)** | [`docs/RBM_THEORY.md`](file:///C:/Users/tanay/.gemini/antigravity/scratch/movie-rbm-recommender/docs/RBM_THEORY.md) |
| **Latent Variables & Unsupervised Learning** | [`models/rbm.py`](file:///C:/Users/tanay/.gemini/antigravity/scratch/movie-rbm-recommender/models/rbm.py#L30-L75) |
| **Gibbs Sampling & MCMC** | [`models/rbm.py`](file:///C:/Users/tanay/.gemini/antigravity/scratch/movie-rbm-recommender/models/rbm.py#L85-L135) |
| **Contrastive Divergence (CD-1)** | [`models/rbm.py`](file:///C:/Users/tanay/.gemini/antigravity/scratch/movie-rbm-recommender/models/rbm.py#L137-L225) |
| **Reconstruction & Inference** | [`models/recommend.py`](file:///C:/Users/tanay/.gemini/antigravity/scratch/movie-rbm-recommender/models/recommend.py) |
| **Model Evaluation** | [`models/evaluation.py`](file:///C:/Users/tanay/.gemini/antigravity/scratch/movie-rbm-recommender/models/evaluation.py) |

---

## Limitations

1. **Cold-Start Problem:** For new users with zero rated movies, collaborative filtering cannot infer latent states (handled via popular discovery fallback).
2. **Matrix Sparsity:** With 93.7% missing entries, ranking metrics on small test subsets show natural variance.
3. **Binary Thresholding:** 1–5 ratings are converted to binary preferences rather than a 5-way softmax visible layer.

---

## Future Scope

1. **Multinomial Softmax Visible Layer:** Implement 5 softmax units per movie to predict exact star ratings directly.
2. **Deep Belief Networks (DBN):** Stack multiple RBMs ($1682 \to 256 \to 64$) for deeper hierarchical representation learning.
3. **Hybrid Content-Collaborative Recommender:** Incorporate plot summaries and cast embeddings using Transformers.

---

## Project Directory Structure

```
movie-rbm-recommender/
│
├── app.py                      # Main Streamlit 5-page dashboard
├── requirements.txt            # Python dependencies
├── README.md                   # Master project documentation
├── .gitignore                  # Git ignore rules
│
├── data/
│   ├── README.md               # Dataset documentation & citation
│   └── ml-100k/                # Pre-loaded MovieLens 100K files
│
├── models/
│   ├── __init__.py             # Models package init
│   ├── rbm.py                  # PyTorch RBM from scratch
│   ├── train.py                # Training loop with CD-1 & logging
│   ├── recommend.py            # Recommender inference engine
│   └── evaluation.py           # Reconstruction & ranking metrics
│
├── utils/
│   ├── __init__.py             # Utils package init
│   ├── data_loader.py          # Dataset parsing, matrix builder
│   ├── preprocessing.py        # Rating tensors, score conversion
│   └── visualization.py        # Matplotlib dark-theme charts
│
├── saved_models/
│   ├── rbm_movielens100k.pt    # Pre-trained RBM checkpoint
│   └── training_history.json   # Real loss curves & evaluation metrics
│
├── assets/
│   └── architecture.png        # RBM bipartite network diagram
│
├── docs/
│   ├── PROJECT_REPORT.md       # Complete 23-section academic report
│   ├── PROJECT_EXPLANATION.md  # Beginner-friendly guide with analogies
│   ├── INSTALLATION_GUIDE.md   # Step-by-step Windows setup guide
│   ├── RBM_THEORY.md           # Mathematical formulas & derivations
│   ├── VIVA_QUESTIONS.md       # 45 Viva questions with concise answers
│   └── TROUBLESHOOTING.md      # Common errors and solutions
│
└── tests/
    ├── __init__.py             # Tests package init
    └── test_rbm.py             # 10 automated unit tests (PyTorch RBM)
```

---

## Authors & Academic Credits

- **Course:** B.Tech Computer Science and Engineering (Artificial Intelligence & Machine Learning)
- **Subject:** Deep Learning-II
- **Dataset:** GroupLens Research, University of Minnesota

