
import re
import io
import base64
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text as sk_text
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

# Optional parsers for PDF / DOCX. App still runs (for .txt) if these aren't installed.
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    import docx  # python-docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False


# --------------------------------------------------------------------------------
# Page config & styling
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Resume Screening & Candidate Ranking",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #1a237e;
        margin-bottom: 0;
    }
    .subtitle {
        color: #5f6368;
        font-size: 1rem;
        margin-top: 0.2rem;
        margin-bottom: 1.5rem;
    }
    .stage-badge {
        display: inline-block;
        background: #e8eaf6;
        color: #1a237e;
        border-radius: 999px;
        padding: 2px 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .rank-card {
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        background: #ffffff;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .rank-number {
        font-size: 1.4rem;
        font-weight: 800;
        color: #1a237e;
    }
    .score-pill {
        float: right;
        background: #1a237e;
        color: white;
        border-radius: 999px;
        padding: 4px 14px;
        font-weight: 700;
        font-size: 0.95rem;
    }
    .keyword-chip {
        display: inline-block;
        background: #f1f3f4;
        border-radius: 6px;
        padding: 2px 8px;
        margin: 2px 4px 2px 0;
        font-size: 0.78rem;
        color: #3c4043;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------------
# Text extraction
# --------------------------------------------------------------------------------
def extract_text_from_pdf(file) -> str:
    if not PDF_AVAILABLE:
        return ""
    try:
        reader = PyPDF2.PdfReader(file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        return ""


def extract_text_from_docx(file) -> str:
    if not DOCX_AVAILABLE:
        return ""
    try:
        document = docx.Document(file)
        return "\n".join(p.text for p in document.paragraphs)
    except Exception:
        return ""


def extract_text_from_txt(file) -> str:
    try:
        return file.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""


def extract_text(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    elif name.endswith(".docx"):
        return extract_text_from_docx(uploaded_file)
    elif name.endswith(".txt"):
        return extract_text_from_txt(uploaded_file)
    return ""


# --------------------------------------------------------------------------------
# Preprocessing — mirrors "Document Parsing" + "Preparation" stages
# --------------------------------------------------------------------------------
def clean_text(raw_text: str) -> str:
    text_val = raw_text.lower()
    text_val = re.sub(r"[^a-z\s]", " ", text_val)   # remove numbers & punctuation
    text_val = re.sub(r"\s+", " ", text_val).strip()  # remove extra spaces
    return text_val


STOPWORDS = sk_text.ENGLISH_STOP_WORDS


def get_top_matched_keywords(vectorizer, job_vec, resume_vec, top_k=6):
    """Returns top overlapping TF-IDF terms between the job description and a resume."""
    feature_names = np.array(vectorizer.get_feature_names_out())
    job_arr = job_vec.toarray().flatten()
    res_arr = resume_vec.toarray().flatten()
    overlap_score = job_arr * res_arr
    top_idx = overlap_score.argsort()[::-1]
    keywords = [feature_names[i] for i in top_idx if overlap_score[i] > 0][:top_k]
    return keywords


# --------------------------------------------------------------------------------
# Core ranking pipeline
# --------------------------------------------------------------------------------
def rank_resumes(resume_texts: dict, job_description: str, top_n: int = 10):
    """
    resume_texts: {filename: raw_text}
    Returns a DataFrame ranked by similarity, plus vectorizer/vectors for keyword extraction.
    """
    filenames = list(resume_texts.keys())
    cleaned_resumes = [clean_text(t) for t in resume_texts.values()]
    cleaned_jd = clean_text(job_description)

    corpus = [cleaned_jd] + cleaned_resumes

    vectorizer = TfidfVectorizer(stop_words=list(STOPWORDS), max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(corpus)

    jd_vector = tfidf_matrix[0:1]
    resume_vectors = tfidf_matrix[1:]

    n_candidates = resume_vectors.shape[0]
    k = min(top_n, n_candidates)

    # KNN ranking using cosine distance (as per System Architecture: "Top closest resumes")
    knn = NearestNeighbors(n_neighbors=k, metric="cosine")
    knn.fit(resume_vectors)
    distances, indices = knn.kneighbors(jd_vector)

    distances = distances.flatten()
    indices = indices.flatten()
    similarities = 1 - distances  # cosine similarity = 1 - cosine distance

    rows = []
    for rank, (idx, sim) in enumerate(zip(indices, similarities), start=1):
        keywords = get_top_matched_keywords(vectorizer, jd_vector, resume_vectors[idx])
        rows.append({
            "Rank": rank,
            "Candidate / File": filenames[idx],
            "Match Score (%)": round(max(sim, 0) * 100, 2),
            "Matched Keywords": ", ".join(keywords) if keywords else "—",
        })

    result_df = pd.DataFrame(rows)
    return result_df


def to_csv_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# --------------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    top_n = st.slider("Number of top candidates to rank", min_value=3, max_value=20, value=10)
    st.markdown("---")
    st.markdown("### 📋 Pipeline")
    st.markdown(
        """
        <span class="stage-badge">1. Document Parsing</span>
        <span class="stage-badge">2. Stopword Removal</span>
        <span class="stage-badge">3. TF-IDF</span>
        <span class="stage-badge">4. Feature Mapping</span>
        <span class="stage-badge">5. Cosine Similarity</span>
        <span class="stage-badge">6. KNN Ranking</span>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("### ℹ️ Supported files")
    st.caption("PDF" + (" ✅" if PDF_AVAILABLE else " ⚠️ install PyPDF2"))
    st.caption("DOCX" + (" ✅" if DOCX_AVAILABLE else " ⚠️ install python-docx"))
    st.caption("TXT ✅")


# --------------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------------
st.markdown('<p class="main-title">🧠 AI Resume Screening & Candidate Ranking System</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Upload resumes, paste a job description, and instantly rank candidates by relevance using TF-IDF, cosine similarity, and KNN.</p>',
    unsafe_allow_html=True,
)

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("#### 📝 Job Description")
    job_description = st.text_area(
        "Paste the job description here",
        height=260,
        placeholder="e.g. We are looking for a Python developer with experience in machine learning, "
                    "NLP, and REST API development. Familiarity with pandas, scikit-learn, and SQL is required...",
        label_visibility="collapsed",
    )

with col_right:
    st.markdown("#### 📂 Upload Resumes")
    uploaded_files = st.file_uploader(
        "Upload resumes (PDF, DOCX, or TXT) — multiple files allowed",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded_files:
        st.caption(f"{len(uploaded_files)} file(s) ready for screening.")

st.markdown("")
run_btn = st.button("🚀 Rank Candidates", type="primary", use_container_width=True)

st.markdown("---")

# --------------------------------------------------------------------------------
# Run pipeline & display results
# --------------------------------------------------------------------------------
if run_btn:
    if not job_description.strip():
        st.warning("Please paste a job description before ranking candidates.")
    elif not uploaded_files:
        st.warning("Please upload at least one resume.")
    else:
        with st.spinner("Parsing resumes and computing candidate rankings..."):
            resume_texts = {}
            failed_files = []
            for f in uploaded_files:
                text_content = extract_text(f)
                if text_content.strip():
                    resume_texts[f.name] = text_content
                else:
                    failed_files.append(f.name)

            if failed_files:
                st.info(f"Could not extract text from: {', '.join(failed_files)} (skipped).")

            if len(resume_texts) == 0:
                st.error("No readable text found in the uploaded resumes.")
            else:
                results_df = rank_resumes(resume_texts, job_description, top_n=top_n)

                st.success(f"Ranked {len(resume_texts)} resume(s) against the job description.")

                # --- Metrics row ---
                m1, m2, m3 = st.columns(3)
                m1.metric("Resumes Processed", len(resume_texts))
                m2.metric("Top Match Score", f"{results_df['Match Score (%)'].iloc[0]}%")
                m3.metric("Candidates Shortlisted", len(results_df))

                st.markdown("### 📊 Similarity Scores")
                chart_df = results_df.set_index("Candidate / File")["Match Score (%)"]
                st.bar_chart(chart_df)

                st.markdown("### 🏆 Ranked Candidates")
                for _, row in results_df.iterrows():
                    keywords_html = "".join(
                        f'<span class="keyword-chip">{kw}</span>'
                        for kw in row["Matched Keywords"].split(", ") if kw != "—"
                    )
                    st.markdown(
                        f"""
                        <div class="rank-card">
                            <span class="rank-number">#{row['Rank']}</span>
                            <span class="score-pill">{row['Match Score (%)']}% match</span>
                            <div style="font-size:1.05rem; font-weight:600; margin-top:6px;">{row['Candidate / File']}</div>
                            <div style="margin-top:8px;">{keywords_html if keywords_html else '<span style="color:#9aa0a6;">No strong keyword overlap</span>'}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown("### 📥 Export")
                csv_bytes = to_csv_download(results_df)
                st.download_button(
                    label="Download results as CSV",
                    data=csv_bytes,
                    file_name=f"resume_ranking_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                )

                with st.expander("View raw ranking table"):
                    st.dataframe(results_df, use_container_width=True)
else:
    st.info("👆 Paste a job description and upload resumes, then click **Rank Candidates** to get started.")
