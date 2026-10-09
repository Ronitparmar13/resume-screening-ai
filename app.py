"""
Flask web application for Resume Screening AI.

Features:
- Accepts PDF/TXT resumes.
- Accepts a job description.
- Extracts resume information.
- Runs NLP and skill extraction.
- Calculates TF-IDF + skill-based match scores.
- Ranks candidates.
- Generates downloadable CSV reports.
"""

from __future__ import annotations

import csv
import io
import uuid
from pathlib import Path

from flask import (
    Flask,
    render_template,
    request,
    send_from_directory,
)

from matcher import rank_candidates
from nlp_processor import (
    analyze_job_description,
    analyze_resume,
)
from resume_parser import parse_resume


BASE_DIR = Path(__file__).resolve().parent
REPORT_DIR = BASE_DIR / "reports"


app = Flask(__name__)

# Maximum request size: 15 MB.
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


ALLOWED_EXTENSIONS = {
    ".pdf",
    ".txt",
}


def allowed_resume(filename: str) -> bool:
    """
    Check whether the uploaded file has a supported extension.
    """

    if not filename:
        return False

    return (
        Path(filename).suffix.lower()
        in ALLOWED_EXTENSIONS
    )


def parse_uploaded_resume(file_storage) -> dict:
    """
    Parse one uploaded resume directly from memory.

    The file is not permanently stored as an uploaded resume.
    """

    filename = (
        str(file_storage.filename or "").strip()
    )

    if not filename:
        raise ValueError(
            "One of the uploaded files has no filename."
        )

    if not allowed_resume(filename):
        raise ValueError(
            f"Unsupported file format for '{filename}'. "
            "Only PDF and TXT files are allowed."
        )

    data = file_storage.read()

    if not data:
        raise ValueError(
            f"'{filename}' is empty."
        )

    # Parse directly from an in-memory stream.
    parsed = parse_resume(
        io.BytesIO(data),
        filename=filename,
    )

    return analyze_resume(
        parsed
    )


def create_csv_report(
    ranked_candidates: list[dict],
) -> str:
    """
    Create a CSV report and return its filename.
    """

    filename = (
        f"candidate_ranking_"
        f"{uuid.uuid4().hex[:10]}.csv"
    )

    report_path = (
        REPORT_DIR / filename
    )

    fieldnames = [
        "Rank",
        "Candidate",
        "Email",
        "Phone",
        "Match Score (%)",
        "TF-IDF Score (%)",
        "Skill Match (%)",
        "Matched Skills",
        "Missing Skills",
        "Education",
        "Experience",
        "Resume File",
    ]

    with report_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for candidate in ranked_candidates:
            education = candidate.get(
                "education",
                {},
            )

            experience = candidate.get(
                "experience",
                {},
            )

            if not isinstance(
                education,
                dict,
            ):
                education = {}

            if not isinstance(
                experience,
                dict,
            ):
                experience = {}

            writer.writerow(
                {
                    "Rank": candidate.get(
                        "rank",
                        "",
                    ),
                    "Candidate": candidate.get(
                        "name",
                        "",
                    ),
                    "Email": candidate.get(
                        "email",
                        "",
                    ),
                    "Phone": candidate.get(
                        "phone",
                        "",
                    ),
                    "Match Score (%)": candidate.get(
                        "score",
                        0,
                    ),
                    "TF-IDF Score (%)": candidate.get(
                        "tfidf_score",
                        0,
                    ),
                    "Skill Match (%)": candidate.get(
                        "skill_score",
                        0,
                    ),
                    "Matched Skills": ", ".join(
                        candidate.get(
                            "matched_skills",
                            [],
                        )
                    ),
                    "Missing Skills": ", ".join(
                        candidate.get(
                            "missing_skills",
                            [],
                        )
                    ),
                    "Education": "; ".join(
                        education.get(
                            "degrees",
                            [],
                        )
                    ),
                    "Experience": "; ".join(
                        experience.get(
                            "roles",
                            [],
                        )
                    ),
                    "Resume File": candidate.get(
                        "filename",
                        "",
                    ),
                }
            )

    return filename


@app.route("/", methods=["GET"])
def index():
    """Render the main screening page."""

    return render_template(
        "index.html"
    )


@app.route("/screen", methods=["POST"])
def screen_candidates():
    """
    Screen uploaded resumes against a job description.
    """

    job_description = (
        request.form.get(
            "job_description",
            "",
        )
        .strip()
    )

    resume_files = request.files.getlist(
        "resumes"
    )

    errors: list[str] = []

    if not job_description:
        errors.append(
            "Please provide a job description."
        )

    if not resume_files or not any(
        file.filename
        for file in resume_files
    ):
        errors.append(
            "Please upload at least one PDF or TXT resume."
        )

    if errors:
        return render_template(
            "index.html",
            errors=errors,
            job_description=job_description,
        )

    try:
        analyzed_job = analyze_job_description(
            job_description
        )
    except Exception as exc:
        return render_template(
            "index.html",
            errors=[
                f"Could not process the job description: {exc}"
            ],
            job_description=job_description,
        )

    candidates = []

    for file_storage in resume_files:
        if not file_storage.filename:
            continue

        try:
            candidate = parse_uploaded_resume(
                file_storage
            )

            candidates.append(
                candidate
            )

        except Exception as exc:
            errors.append(
                f"{file_storage.filename}: {exc}"
            )

    if not candidates:
        errors.append(
            "No valid resumes could be processed."
        )

        return render_template(
            "index.html",
            errors=errors,
            job_description=job_description,
        )

    try:
        ranked_candidates = rank_candidates(
            candidates,
            analyzed_job,
        )

        report_filename = create_csv_report(
            ranked_candidates
        )

    except Exception as exc:
        return render_template(
            "index.html",
            errors=[
                f"Screening failed: {exc}"
            ],
            job_description=job_description,
        )

    return render_template(
        "index.html",
        job_description=job_description,
        ranked_candidates=ranked_candidates,
        report_filename=report_filename,
        errors=errors,
        job_skills=analyzed_job.get(
            "skills",
            [],
        ),
    )


@app.route(
    "/reports/<path:filename>",
    methods=["GET"],
)
def download_report(filename: str):
    """Download a generated ranking CSV."""

    return send_from_directory(
        REPORT_DIR,
        filename,
        as_attachment=True,
    )


@app.route("/health", methods=["GET"])
def health():
    """Simple application health endpoint."""

    return {
        "status": "ok",
        "service": "Resume Screening AI",
    }


@app.errorhandler(413)
def request_too_large(error):
    """Handle oversized uploads."""

    return render_template(
        "index.html",
        errors=[
            "Upload too large. Please keep the total request under 15 MB."
        ],
    ), 413


@app.errorhandler(500)
def internal_error(error):
    """Handle unexpected application errors."""

    return render_template(
        "index.html",
        errors=[
            "Something went wrong while processing the request. "
            "Please try again."
        ],
    ), 500


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
    )