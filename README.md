# AI Resume Screening & Candidate Ranking System — Frontend

A Streamlit frontend for the resume ranking pipeline described in the project:
**Document Parsing → Stopword Removal → TF-IDF → Feature Mapping → Cosine Similarity → KNN Ranking → Top N Resumes.**

## Features
- Paste a job description and upload multiple resumes (PDF, DOCX, or TXT)
- Automatic text cleaning (lowercasing, number/punctuation removal, stopword removal)
- TF-IDF vectorization + cosine similarity scoring
- Top-N candidate ranking via K-Nearest Neighbors (matches the System Architecture diagram)
- Match score, ranked candidate cards, matched-keyword highlights, similarity bar chart
- CSV export of the ranking results

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Then open the URL Streamlit prints (usually `http://localhost:8501`).

## Deploy to Streamlit Community Cloud (as specified in the slides)
1. Push this folder (`app.py` + `requirements.txt`) to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, select the repo/branch, and set the main file path to `app.py`.
4. Click **Deploy** — Streamlit Cloud installs `requirements.txt` automatically.

## Notes / where to plug in your backend
- `extract_text()` currently parses PDF/DOCX/TXT directly in the frontend for a self-contained demo.
- `rank_resumes()` contains the full TF-IDF + cosine similarity + KNN logic from the System Architecture slide.
- If you split this into a separate Flask/FastAPI backend later, replace the body of `rank_resumes()`
  with an API call and keep the same return shape (a DataFrame with Rank, Candidate/File, Match Score, Matched Keywords).
