"""
Flask backend API for the AI Resume Screening & Candidate Ranking System.

Endpoints:
  GET  /health  -> service status
  POST /rank    -> multipart form: job_description (str), top_n (int, optional),
                    resumes (one or more files, PDF/DOCX/TXT)
                    returns JSON ranking results
"""

from flask import Flask, request, jsonify
from flask_cors import CORS

from ranking import extract_text, rank_resumes

app = Flask(__name__)
CORS(app)  # allow the Streamlit frontend (different origin/port) to call this API

MAX_CONTENT_LENGTH = 25 * 1024 * 1024  # 25 MB upload cap
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "resume-ranking-api"})


@app.route("/rank", methods=["POST"])
def rank():
    job_description = request.form.get("job_description", "").strip()
    top_n = request.form.get("top_n", 10)
    try:
        top_n = int(top_n)
    except (TypeError, ValueError):
        top_n = 10

    files = request.files.getlist("resumes")

    if not job_description:
        return jsonify({"error": "job_description is required"}), 400
    if not files:
        return jsonify({"error": "at least one resume file is required"}), 400

    resume_texts = {}
    failed_files = []
    for f in files:
        content = f.read()
        text_val = extract_text(f.filename, content)
        if text_val.strip():
            resume_texts[f.filename] = text_val
        else:
            failed_files.append(f.filename)

    if not resume_texts:
        return jsonify({
            "error": "no readable text could be extracted from the uploaded resumes",
            "failed_files": failed_files,
        }), 422

    results = rank_resumes(resume_texts, job_description, top_n=top_n)

    return jsonify({
        "count": len(resume_texts),
        "results": results,
        "failed_files": failed_files,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)