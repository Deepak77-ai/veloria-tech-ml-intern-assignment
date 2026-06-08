# Veloria Tech ML Intern Assignment

This is my submission for the Veloria Tech AI/ML Engineering Internship assignment.
I completed all 3 tasks including the bonus task.

---

## What I Built

| Task | File | Description |
|------|------|-------------|
| Task 1 | `scraper.py` | Collects cricket match data using the Cricbuzz API and saves it as a CSV file |
| Task 2 | `model.py` | Trains a Random Forest model to predict which team will win a match |
| Task 3 (Bonus) | `rag_search.py` | Semantic search system using vector embeddings and FAISS |

---

## Project Structure

```
veloria-tech-ml-intern-assignment/
│
├── scraper.py          # Task 1 - Data collection
├── model.py            # Task 2 - ML prediction model
├── rag_search.py       # Task 3 - Semantic search (bonus)
├── match_data.csv      # Output from scraper.py (auto-generated)
├── requirements.txt    # All libraries to install
├── .env                # API key and base URL (not uploaded to GitHub)
├── .gitignore          # Tells Git to ignore the .env file
└── README.md           # This file
```

---

## How to Run

### Step 1 — Install all libraries

```bash
pip install -r requirements.txt
```

### Step 2 — Create your .env file

Create a file named `.env` in the project folder and add your RapidAPI credentials:

```
API_KEY=your_rapidapi_key_here
BASE_URL=https://cricbuzz-cricket.p.rapidapi.com
```

> You can get a free API key by signing up at [rapidapi.com](https://rapidapi.com) and subscribing to the Cricbuzz API.

### Step 3 — Run Task 1 (scraper)

This collects 10 recent cricket matches and saves them to `match_data.csv`

```bash
python scraper.py

```
<img width="1413" height="328" alt="image" src="https://github.com/user-attachments/assets/2fb1304c-419f-4e75-ad60-31d8e735c9d3" />

### Step 4 — Run Task 2 (ML model)

This trains a Random Forest model on the CSV and prints accuracy, F1 score, and confusion matrix

```bash
python model.py
```
<img width="488" height="436" alt="image" src="https://github.com/user-attachments/assets/ec7375cf-4234-45ee-9091-eafb43250f5a" />


### Step 5 — Run Task 3 (semantic search)

This builds a semantic search system over the match data. At the end it lets you type your own query.

```bash
python rag_search.py
```
<img width="1667" height="548" alt="image" src="https://github.com/user-attachments/assets/173885c3-2411-46a3-9d68-e9bb4b91d109" />


> **Note:** The first time you run `rag_search.py` it will download a ~80MB AI model automatically. Just wait for it to finish.

---

## Data Collected (match_data.csv)

I collected the last 10 completed international cricket matches using the Cricbuzz API via RapidAPI.

| Date | Team 1 | Team 2 | Venue | Winner | Top Scorer | Score |
|------|--------|--------|-------|--------|------------|-------|
| 2026-06-03 | Sri Lanka | West Indies | Sabina Park | Sri Lanka | Pathum Nissanka | 79 |
| 2026-05-30 | Rwanda | Botswana | Botswana Cricket Association Oval 1 | Rwanda | Karabo Motlhanka | 40 |
| 2026-06-05 | Hong Kong, China | Oman | Singapore National Cricket Ground | Hong Kong, China | Anshuman Rath | 140 |
| 2026-06-05 | Bahrain | Singapore | Singapore National Cricket Ground | Bahrain | Mahiyu Bhatia | 57 |
| 2026-06-04 | Nepal | Malaysia | Singapore National Cricket Ground | Nepal | Kushal Bhurtel | 126 |
| 2026-06-03 | Singapore | Hong Kong, China | Singapore National Cricket Ground | Hong Kong, China | Anshuman Rath | 32 |
| 2026-06-03 | Oman | Bahrain | Singapore National Cricket Ground | Oman | Mohammed Al Balushi | 81 |
| 2026-06-02 | China | Malaysia | Singapore National Cricket Ground | Malaysia | Muhammad Haziq Aiman | 44 |
| 2026-06-01 | Bahrain | Hong Kong, China | Singapore National Cricket Ground | Hong Kong, China | Babar Hayat | 93 |
| 2026-06-01 | Oman | Singapore | Singapore National Cricket Ground | Oman | Rezza Gaznavi | 49 |

---

## Task 2 — ML Model Details

**Algorithm used:** Algorithm used: I compared 6 algorithms (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, KNN, SVM) and automatically selected the best performing one based on accuracy.

**Why Random Forest?**
- It works well on small datasets like ours (only 10 rows)
- It does not need feature scaling
- It shows which features it found most useful

**Features I gave the model:**
- `is_home` — is team1 playing at their home venue? (1 or 0)
- `team1_win_rate` — how often team1 wins in our dataset
- `team2_win_rate` — how often team2 wins in our dataset
- `top_score_norm` — the top score normalized between 0 and 1
- `team1_num`, `team2_num`, `venue_num` — team and venue names converted to numbers

**Why Leave-One-Out Cross Validation?**

Because we only have 10 rows, a normal 80/20 train-test split would leave just 2 rows for testing which is too small to be reliable. With Leave-One-Out (LOO), the model trains on 9 rows and tests on 1, and repeats this for every row. This gives a more accurate picture of how the model performs on small datasets.

**Results:**
- Accuracy: ~70%
- F1 Score: ~0.65

---

## Task 3 — Semantic Search Details

**What is semantic search?**

Normal keyword search (like CTRL+F) only finds exact words. Semantic search understands the *meaning* of your query. For example, searching `"away team victory"` will find matches where a team won away from home, even if those exact words are not in the data.

**How it works:**
1. Each match is converted into a plain English sentence
2. The `sentence-transformers` library converts each sentence into a vector (a list of 384 numbers representing meaning)
3. All vectors are stored in a FAISS index (an in-memory vector search engine)
4. When you search, your query is also converted to a vector and FAISS finds the 3 most similar matches

**Example queries you can try:**
```
matches where the away team won
high scoring match with a century
match played in Singapore
low scoring match
```

---

## Libraries Used

| Library | Purpose |
|---------|---------|
| `requests` | Making API calls to fetch match data |
| `pandas` | Loading and cleaning the CSV data |
| `scikit-learn` | Building and evaluating the ML model |
| `sentence-transformers` | Converting text to vector embeddings |
| `faiss-cpu` | Storing and searching vectors efficiently |
| `python-dotenv` | Loading API keys securely from the .env file |

---

## Challenges I Faced

- Most cricket websites block automated scraping, so I used the Cricbuzz API via RapidAPI instead to collect real match data reliably.
- The dataset only has 10 rows which is very small for ML. I handled this by using Leave-One-Out cross validation instead of a normal train-test split.
- Understanding how FAISS works took some reading but once I understood that it just searches for similar vectors, it made sense.
- I stored the API key in a `.env` file instead of hardcoding it in the script, so it is not exposed when uploading to GitHub.

---

*Submitted by Deepak | M.Sc. Information Technology*
