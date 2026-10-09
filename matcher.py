"""
Candidate matching and ranking engine.

Responsibilities:
- Compare resumes against a job description.
- Calculate TF-IDF cosine similarity.
- Calculate skill overlap.
- Produce an explainable weighted match score.
- Rank multiple candidates.
"""

from __future__ import annotations

from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Scoring configuration
# ---------------------------------------------------------------------------

TFIDF_WEIGHT = 0.40
SKILL_WEIGHT = 0.60


# ---------------------------------------------------------------------------
# Text preparation
# ---------------------------------------------------------------------------


def _safe_text(value: Any) -> str:
    """Convert a value to clean text."""

    if value is None:
        return ""

    return str(value).strip()


def _get_processed_text(candidate: dict[str, Any]) -> str:
    """
    Get the NLP-processed resume text.

    Falls back to raw text when processed text is unavailable.
    """

    nlp = candidate.get("nlp", {})

    if isinstance(nlp, dict):
        processed = _safe_text(
            nlp.get("processed_text")
        )

        if processed:
            return processed

    return _safe_text(
        candidate.get("raw_text", "")
    )


# ---------------------------------------------------------------------------
# TF-IDF similarity
# ---------------------------------------------------------------------------


def calculate_tfidf_similarity(
    resume_text: str,
    job_description: str,
) -> float:
    """
    Calculate cosine similarity between a resume and job description.

    Returns a value between 0 and 100.
    """

    resume_text = _safe_text(resume_text)
    job_description = _safe_text(job_description)

    if not resume_text or not job_description:
        return 0.0

    try:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
        )

        matrix = vectorizer.fit_transform(
            [
                resume_text,
                job_description,
            ]
        )

        similarity = cosine_similarity(
            matrix[0:1],
            matrix[1:2],
        )[0][0]

    except ValueError:
        return 0.0

    similarity = max(
        0.0,
        min(1.0, float(similarity)),
    )

    return round(
        similarity * 100,
        2,
    )


# ---------------------------------------------------------------------------
# Skill matching
# ---------------------------------------------------------------------------


def calculate_skill_match(
    resume_skills: list[str],
    job_skills: list[str],
) -> dict[str, Any]:
    """
    Compare extracted resume skills against extracted JD skills.

    Returns:
        matched skills
        missing skills
        skill match percentage
    """

    resume_set = {
        _safe_text(skill).lower()
        for skill in resume_skills
        if _safe_text(skill)
    }

    job_set = {
        _safe_text(skill).lower()
        for skill in job_skills
        if _safe_text(skill)
    }

    if not job_set:
        return {
            "score": 0.0,
            "matched_skills": [],
            "missing_skills": [],
            "total_job_skills": 0,
        }

    matched = sorted(
        resume_set.intersection(job_set)
    )

    missing = sorted(
        job_set.difference(resume_set)
    )

    score = (
        len(matched) / len(job_set)
    ) * 100

    # Recover original display casing from resume/JD lists.
    display_lookup = {}

    for skill in resume_skills + job_skills:
        display_lookup[
            _safe_text(skill).lower()
        ] = _safe_text(skill)

    matched_display = [
        display_lookup.get(
            skill,
            skill,
        )
        for skill in matched
    ]

    missing_display = [
        display_lookup.get(
            skill,
            skill,
        )
        for skill in missing
    ]

    return {
        "score": round(
            score,
            2,
        ),
        "matched_skills": matched_display,
        "missing_skills": missing_display,
        "total_job_skills": len(job_set),
    }


# ---------------------------------------------------------------------------
# Final candidate score
# ---------------------------------------------------------------------------


def calculate_match_score(
    candidate: dict[str, Any],
    job_description: dict[str, Any],
) -> dict[str, Any]:
    """
    Calculate the complete match score for one candidate.

    Final score:

        70% TF-IDF similarity
        30% skill overlap
    """

    resume_text = _get_processed_text(
        candidate
    )

    jd_nlp = job_description.get(
        "nlp",
        {},
    )

    if isinstance(jd_nlp, dict):
        jd_text = _safe_text(
            jd_nlp.get(
                "processed_text",
                "",
            )
        )
    else:
        jd_text = ""

    if not jd_text:
        jd_text = _safe_text(
            job_description.get(
                "raw_text",
                "",
            )
        )

    tfidf_score = calculate_tfidf_similarity(
        resume_text,
        jd_text,
    )

    skill_result = calculate_skill_match(
        candidate.get(
            "skills",
            [],
        ),
        job_description.get(
            "skills",
            [],
        ),
    )

    skill_score = skill_result["score"]

    final_score = (
        TFIDF_WEIGHT * tfidf_score
        + SKILL_WEIGHT * skill_score
    )

    return {
        "tfidf_score": round(
            tfidf_score,
            2,
        ),
        "skill_score": round(
            skill_score,
            2,
        ),
        "final_score": round(
            final_score,
            2,
        ),
        "matched_skills": skill_result[
            "matched_skills"
        ],
        "missing_skills": skill_result[
            "missing_skills"
        ],
        "total_job_skills": skill_result[
            "total_job_skills"
        ],
    }


# ---------------------------------------------------------------------------
# Candidate summary
# ---------------------------------------------------------------------------


def build_candidate_summary(
    candidate: dict[str, Any],
    score_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Build a clean summary object for the UI/report.
    """

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

    return {
        "name": _safe_text(
            candidate.get(
                "name",
                "",
            )
        )
        or "Unknown Candidate",
        "email": _safe_text(
            candidate.get(
                "email",
                "",
            )
        ),
        "phone": _safe_text(
            candidate.get(
                "phone",
                "",
            )
        ),
        "filename": _safe_text(
            candidate.get(
                "filename",
                "",
            )
        ),
        "skills": candidate.get(
            "skills",
            [],
        ),
        "education": education,
        "experience": experience,
        "score": score_result[
            "final_score"
        ],
        "tfidf_score": score_result[
            "tfidf_score"
        ],
        "skill_score": score_result[
            "skill_score"
        ],
        "matched_skills": score_result[
            "matched_skills"
        ],
        "missing_skills": score_result[
            "missing_skills"
        ],
    }


# ---------------------------------------------------------------------------
# Rank candidates
# ---------------------------------------------------------------------------


def rank_candidates(
    candidates: list[dict[str, Any]],
    job_description: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Calculate scores for all candidates and rank them
    from highest to lowest.
    """

    ranked: list[dict[str, Any]] = []

    for candidate in candidates:
        score_result = calculate_match_score(
            candidate,
            job_description,
        )

        summary = build_candidate_summary(
            candidate,
            score_result,
        )

        ranked.append(summary)

    ranked.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    for index, candidate in enumerate(
        ranked,
        start=1,
    ):
        candidate["rank"] = index

    return ranked


# ---------------------------------------------------------------------------
# Job-description analysis helper
# ---------------------------------------------------------------------------


def build_job_description_for_matching(
    raw_text: str,
    skills: list[str],
    processed_text: str,
) -> dict[str, Any]:
    """
    Create a standard job-description object.
    """

    return {
        "raw_text": _safe_text(
            raw_text
        ),
        "skills": list(skills),
        "nlp": {
            "processed_text": _safe_text(
                processed_text
            )
        },
    }


# ---------------------------------------------------------------------------
# Human-readable score explanation
# ---------------------------------------------------------------------------


def explain_score(
    score_result: dict[str, Any],
) -> str:
    """
    Return a concise explanation of the final score.
    """

    return (
        f"Final score: {score_result['final_score']:.2f}% "
        f"(TF-IDF similarity: "
        f"{score_result['tfidf_score']:.2f}%, "
        f"skill match: "
        f"{score_result['skill_score']:.2f}%)."
    )


# ---------------------------------------------------------------------------
# Command-line demonstration
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import json

    from nlp_processor import (
        analyze_job_description,
        analyze_resume,
    )
    from resume_parser import parse_resume

    resume_path = (
        "data/sample_resumes/test_resume.txt"
    )

    sample_jd = """
    We are looking for a Python developer who has experience with
    Flask, REST APIs, SQL, JavaScript, Git, and machine learning.

    The candidate should have a Computer Science degree and practical
    software development experience.
    """

    parsed_resume = parse_resume(
        resume_path
    )

    candidate = analyze_resume(
        parsed_resume
    )

    job_description = analyze_job_description(
        sample_jd
    )

    score = calculate_match_score(
        candidate,
        job_description,
    )

    print(
        json.dumps(
            {
                "candidate": candidate[
                    "name"
                ],
                "resume_skills": candidate[
                    "skills"
                ],
                "job_skills": job_description[
                    "skills"
                ],
                "score": score,
                "explanation": explain_score(
                    score
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
    )