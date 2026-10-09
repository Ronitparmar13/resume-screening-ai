"""
NLP Processor for Resume Screening AI.

Responsibilities:
- Tokenize resume/job-description text.
- Remove common stop words.
- Apply lightweight lemmatization.
- Extract skills using a configurable skill dictionary.
- Extract education information.
- Extract experience information.
- Produce structured candidate NLP data.

The module is intentionally lightweight and deterministic so that
the screening pipeline is explainable and easy to reproduce.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
SKILLS_PATH = BASE_DIR / "data" / "skills.json"


# ---------------------------------------------------------------------------
# Basic English stop-word list
# ---------------------------------------------------------------------------

STOP_WORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
}


# ---------------------------------------------------------------------------
# Loading skill dictionary
# ---------------------------------------------------------------------------


def load_skill_dictionary(
    path: Path = SKILLS_PATH,
) -> list[dict[str, Any]]:
    """
    Load the skill dictionary from JSON.
    """

    try:
        data = json.loads(
            path.read_text(encoding="utf-8")
        )
    except (
        FileNotFoundError,
        json.JSONDecodeError,
        OSError,
    ) as exc:
        raise ValueError(
            f"Unable to load skills dictionary: {exc}"
        ) from exc

    skills = data.get("skills")

    if not isinstance(skills, list):
        raise ValueError(
            "data/skills.json must contain a 'skills' list."
        )

    valid_skills = []

    for item in skills:
        if not isinstance(item, dict):
            continue

        name = item.get("name")
        aliases = item.get("aliases", [])

        if not isinstance(name, str):
            continue

        if not isinstance(aliases, list):
            aliases = []

        clean_aliases = [
            str(alias).strip()
            for alias in aliases
            if str(alias).strip()
        ]

        # Always include canonical skill name.
        if name.strip() not in clean_aliases:
            clean_aliases.append(name.strip())

        valid_skills.append(
            {
                "name": name.strip(),
                "aliases": clean_aliases,
            }
        )

    return valid_skills


SKILL_DICTIONARY = load_skill_dictionary()


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """
    Normalize arbitrary text for NLP processing.
    """

    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Keep letters/numbers and useful separators.
    text = text.replace("&", " and ")

    text = re.sub(
        r"[^a-z0-9+#./\-\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------


def tokenize(text: str) -> list[str]:
    """
    Tokenize normalized text.

    Handles technical tokens such as:
    - c++
    - node.js
    - next.js
    - scikit-learn
    """

    normalized = normalize_text(text)

    if not normalized:
        return []

    return re.findall(
        r"[a-z0-9]+(?:[+#.-][a-z0-9+#.-]+)*",
        normalized,
    )


# ---------------------------------------------------------------------------
# Stop-word removal
# ---------------------------------------------------------------------------


def remove_stop_words(
    tokens: list[str],
) -> list[str]:
    """
    Remove common English stop words.
    """

    return [
        token
        for token in tokens
        if token not in STOP_WORDS
    ]


# ---------------------------------------------------------------------------
# Lightweight lemmatization
# ---------------------------------------------------------------------------


def lemmatize_token(token: str) -> str:
    """
    Apply lightweight rule-based lemmatization.

    This avoids requiring external NLTK corpora while still reducing
    common plural/verb forms to simpler representations.
    """

    if len(token) <= 3:
        return token

    # Preserve important technical words.
    protected = {
        "python",
        "flask",
        "django",
        "fastapi",
        "javascript",
        "typescript",
        "react",
        "next.js",
        "node.js",
        "sql",
        "mysql",
        "postgresql",
        "mongodb",
        "redis",
        "docker",
        "kubernetes",
        "linux",
        "aws",
        "azure",
        "tensorflow",
        "pytorch",
        "pandas",
        "numpy",
    }

    if token in protected:
        return token

    # Common plural forms.
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"

    if token.endswith("sses"):
        return token[:-2]

    if token.endswith("s") and not token.endswith(
        ("ss", "us", "is")
    ):
        return token[:-1]

    # Common verb forms.
    if token.endswith("ing") and len(token) > 5:
        stem = token[:-3]

        if len(stem) > 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]

        return stem

    if token.endswith("ed") and len(token) > 4:
        stem = token[:-2]

        if len(stem) > 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]

        return stem

    return token


def lemmatize_tokens(
    tokens: list[str],
) -> list[str]:
    """
    Lemmatize a list of tokens.
    """

    return [
        lemmatize_token(token)
        for token in tokens
    ]


# ---------------------------------------------------------------------------
# Complete preprocessing
# ---------------------------------------------------------------------------


def preprocess_text(text: str) -> dict[str, Any]:
    """
    Run the complete NLP preprocessing pipeline.

    Returns:
        original text
        normalized text
        tokens
        filtered tokens
        lemmatized tokens
        processed text
    """

    tokens = tokenize(text)
    filtered = remove_stop_words(tokens)
    lemmatized = lemmatize_tokens(filtered)

    return {
        "original_text": text or "",
        "normalized_text": normalize_text(text),
        "tokens": tokens,
        "filtered_tokens": filtered,
        "lemmatized_tokens": lemmatized,
        "processed_text": " ".join(lemmatized),
    }


# ---------------------------------------------------------------------------
# Skill extraction
# ---------------------------------------------------------------------------


def _alias_pattern(alias: str) -> re.Pattern:
    """
    Build a case-insensitive pattern for a skill alias.

    Boundary handling makes sure:
        "c" does not incorrectly match "css"
        "ml" does not match arbitrary words containing "ml"
    """

    normalized = normalize_text(alias)

    escaped = re.escape(normalized)

    return re.compile(
        rf"(?<![a-z0-9]){escaped}(?![a-z0-9])",
        re.IGNORECASE,
    )


def extract_skills(
    text: str,
    skill_dictionary: list[dict[str, Any]] | None = None,
) -> list[str]:
    """
    Extract canonical skills from arbitrary text.

    Longer aliases are checked first so phrases such as
    'machine learning' are preferred over short aliases such as 'ml'.
    """

    if not text:
        return []

    dictionary = (
        skill_dictionary
        if skill_dictionary is not None
        else SKILL_DICTIONARY
    )

    matches: list[tuple[int, str]] = []

    for skill in dictionary:
        canonical = str(
            skill.get("name", "")
        ).strip()

        aliases = skill.get(
            "aliases",
            [],
        )

        if not canonical:
            continue

        all_aliases = list(aliases)

        if canonical not in all_aliases:
            all_aliases.append(canonical)

        # Longest phrases first.
        all_aliases = sorted(
            {
                str(alias).strip()
                for alias in all_aliases
                if str(alias).strip()
            },
            key=len,
            reverse=True,
        )

        for alias in all_aliases:
            pattern = _alias_pattern(alias)
            match = pattern.search(
                text
            )

            if match:
                matches.append(
                    (
                        match.start(),
                        canonical,
                    )
                )

                # One match per canonical skill is enough.
                break

    # Sort by position in the document.
    matches.sort(
        key=lambda item: item[0]
    )

    skills: list[str] = []

    for _, skill in matches:
        if skill not in skills:
            skills.append(skill)

    return skills


# ---------------------------------------------------------------------------
# Education extraction
# ---------------------------------------------------------------------------


DEGREE_PATTERNS = [
    r"\bb\.?\s*tech\b",
    r"\bbtech\b",
    r"\bbachelor(?:'s)?\b",
    r"\bb\.?\s*e\b",
    r"\bbe\b",
    r"\bb\.?\s*sc\b",
    r"\bbsc\b",
    r"\bm\.?\s*tech\b",
    r"\bmtech\b",
    r"\bmaster(?:'s)?\b",
    r"\bm\.?\s*e\b",
    r"\bme\b",
    r"\bm\.?\s*sc\b",
    r"\bmsc\b",
    r"\bmca\b",
    r"\bmba\b",
    r"\bphd\b",
    r"\bdoctorate\b",
    r"\bdiploma\b",
]


def extract_education_details(
    education_text: str,
) -> dict[str, Any]:
    """
    Extract useful education signals.
    """

    if not education_text:
        return {
            "raw": "",
            "degrees": [],
            "institutions": [],
            "years": [],
        }

    lines = [
        line.strip()
        for line in education_text.splitlines()
        if line.strip()
    ]

    degrees: list[str] = []

    for line in lines:
        lowered = line.lower()

        if any(
            re.search(
                pattern,
                lowered,
                flags=re.IGNORECASE,
            )
            for pattern in DEGREE_PATTERNS
        ):
            if line not in degrees:
                degrees.append(line)

    years = sorted(
        set(
            re.findall(
                r"\b(?:19|20)\d{2}\b",
                education_text,
            )
        )
    )

    # Heuristic institution extraction:
    # lines that contain university/college/institute/school.
    institutions = []

    for line in lines:
        lowered = line.lower()

        if any(
            keyword in lowered
            for keyword in [
                "university",
                "college",
                "institute",
                "school",
            ]
        ):
            if line not in institutions:
                institutions.append(line)

    return {
        "raw": education_text.strip(),
        "degrees": degrees,
        "institutions": institutions,
        "years": years,
    }


# ---------------------------------------------------------------------------
# Experience extraction
# ---------------------------------------------------------------------------


def extract_experience_details(
    experience_text: str,
) -> dict[str, Any]:
    """
    Extract useful experience signals from the experience section.
    """

    if not experience_text:
        return {
            "raw": "",
            "roles": [],
            "companies": [],
            "years": [],
        }

    lines = [
        line.strip()
        for line in experience_text.splitlines()
        if line.strip()
    ]

    years = sorted(
        set(
            re.findall(
                r"\b(?:19|20)\d{2}\b",
                experience_text,
            )
        )
    )

    roles: list[str] = []
    companies: list[str] = []

    role_keywords = [
        "intern",
        "developer",
        "engineer",
        "analyst",
        "manager",
        "designer",
        "consultant",
        "scientist",
        "administrator",
        "specialist",
        "lead",
    ]

    company_keywords = [
        "pvt",
        "ltd",
        "llp",
        "inc",
        "technologies",
        "technology",
        "solutions",
        "systems",
        "services",
        "software",
        "company",
        "labs",
        "lab",
        "corp",
    ]

    for line in lines:
        lowered = line.lower()

        if any(
            keyword in lowered
            for keyword in role_keywords
        ):
            if line not in roles:
                roles.append(line)

        if any(
            keyword in lowered
            for keyword in company_keywords
        ):
            if line not in companies:
                companies.append(line)

    return {
        "raw": experience_text.strip(),
        "roles": roles,
        "companies": companies,
        "years": years,
    }


# ---------------------------------------------------------------------------
# Resume-level NLP analysis
# ---------------------------------------------------------------------------


def analyze_resume(parsed_resume: dict[str, Any]) -> dict[str, Any]:
    """
    Convert output from resume_parser.parse_resume()
    into an NLP-ready candidate representation.
    """

    raw_text = str(
        parsed_resume.get(
            "raw_text",
            "",
        )
        or ""
    )

    skills_section = str(
        parsed_resume.get(
            "skills_section",
            "",
        )
        or ""
    )

    education = str(
        parsed_resume.get(
            "education",
            "",
        )
        or ""
    )

    experience = str(
        parsed_resume.get(
            "experience",
            "",
        )
        or ""
    )

    # Skill extraction from the complete resume is more robust than
    # extracting only from the skills section.
    detected_skills = extract_skills(
        raw_text
    )

    skills_section_skills = extract_skills(
        skills_section
    )

    education_details = extract_education_details(
        education
    )

    experience_details = extract_experience_details(
        experience
    )

    processed = preprocess_text(
        raw_text
    )

    return {
        "filename": parsed_resume.get(
            "filename",
            "",
        ),
        "name": parsed_resume.get(
            "name",
            "",
        ),
        "email": parsed_resume.get(
            "email",
            "",
        ),
        "phone": parsed_resume.get(
            "phone",
            "",
        ),
        "skills": detected_skills,
        "skills_from_skills_section": skills_section_skills,
        "education": education_details,
        "experience": experience_details,
        "nlp": processed,
        "raw_text": raw_text,
    }


# ---------------------------------------------------------------------------
# Utility for job descriptions
# ---------------------------------------------------------------------------


def analyze_job_description(
    job_description: str,
) -> dict[str, Any]:
    """
    Process a job description using the same NLP pipeline.

    This ensures the resume and JD are normalized consistently
    before TF-IDF similarity is calculated.
    """

    processed = preprocess_text(
        job_description
    )

    skills = extract_skills(
        job_description
    )

    return {
        "raw_text": job_description or "",
        "skills": skills,
        "nlp": processed,
    }


# ---------------------------------------------------------------------------
# Command-line demo
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print(
            "Usage: python nlp_processor.py <resume.txt>"
        )
        raise SystemExit(1)

    resume_path = Path(sys.argv[1])

    try:
        from resume_parser import parse_resume

        parsed = parse_resume(
            resume_path
        )

        analyzed = analyze_resume(
            parsed
        )

        output = {
            "filename": analyzed["filename"],
            "name": analyzed["name"],
            "email": analyzed["email"],
            "phone": analyzed["phone"],
            "skills": analyzed["skills"],
            "skills_from_skills_section": analyzed[
                "skills_from_skills_section"
            ],
            "education": analyzed["education"],
            "experience": analyzed["experience"],
            "processed_text_preview": analyzed[
                "nlp"
            ]["processed_text"][:500],
        }

        print(
            json.dumps(
                output,
                indent=2,
                ensure_ascii=False,
            )
        )

    except Exception as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)