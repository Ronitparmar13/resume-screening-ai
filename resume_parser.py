"""
Resume Parser for Resume Screening AI.

Responsibilities:
- Extract text from PDF and TXT resumes.
- Clean and normalize extracted text.
- Detect common resume sections.
- Extract basic candidate information.
- Extract education and experience sections.
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import BinaryIO


try:
    import pdfplumber
except ImportError as exc:
    raise ImportError(
        "pdfplumber is required. Install it with: pip install pdfplumber"
    ) from exc


# ---------------------------------------------------------------------------
# Resume section headings
# ---------------------------------------------------------------------------

SECTION_ALIASES = {
    "summary": {
        "summary",
        "professional summary",
        "profile",
        "career objective",
        "objective",
    },
    "skills": {
        "skills",
        "technical skills",
        "core skills",
        "key skills",
        "skills & technologies",
        "technical expertise",
        "technologies",
    },
    "education": {
        "education",
        "academic background",
        "academic qualifications",
        "qualifications",
        "educational background",
    },
    "experience": {
        "experience",
        "work experience",
        "professional experience",
        "employment history",
        "work history",
    },
    "projects": {
        "projects",
        "academic projects",
        "personal projects",
        "key projects",
    },
    "certifications": {
        "certifications",
        "certificates",
        "licenses & certifications",
    },
    "achievements": {
        "achievements",
        "accomplishments",
        "awards",
    },
}


# Flatten aliases for fast matching.
HEADING_TO_SECTION = {
    alias.lower(): section
    for section, aliases in SECTION_ALIASES.items()
    for alias in aliases
}


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------


def clean_text(text: str) -> str:
    """
    Normalize extracted resume text.

    Keeps useful punctuation while removing:
    - excessive whitespace
    - repeated blank lines
    - common PDF extraction artifacts
    """

    if not isinstance(text, str):
        return ""

    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Normalize common bullet characters.
    text = text.replace("•", "\n- ")
    text = text.replace("▪", "\n- ")
    text = text.replace("◦", "\n- ")
    text = text.replace("●", "\n- ")

    # Remove repeated spaces/tabs while preserving line breaks.
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Clean whitespace around newlines.
    text = re.sub(r" *\n *", "\n", text)

    return text.strip()


# ---------------------------------------------------------------------------
# File type detection
# ---------------------------------------------------------------------------


def _get_filename(source: object, filename: str | None = None) -> str:
    """Return a best-effort filename from the source."""

    if filename:
        return str(filename)

    # Handle filesystem paths passed as strings or Path objects.
    if isinstance(source, (str, Path)):
        return Path(source).name

    source_name = getattr(source, "name", None)

    if source_name:
        return Path(str(source_name)).name

    source_filename = getattr(source, "filename", None)

    if source_filename:
        return Path(str(source_filename)).name

    return ""


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------


def extract_text_from_pdf(source: str | Path | BinaryIO) -> str:
    """
    Extract text from all pages of a PDF.

    Supports:
    - filesystem path
    - file-like binary object
    """

    pages_text: list[str] = []

    try:
        with pdfplumber.open(source) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""

                if page_text.strip():
                    pages_text.append(page_text)

    except Exception as exc:
        raise ValueError(
            f"Unable to read PDF resume: {exc}"
        ) from exc

    return clean_text("\n\n".join(pages_text))


# ---------------------------------------------------------------------------
# TXT extraction
# ---------------------------------------------------------------------------


def extract_text_from_txt(
    source: str | Path | BinaryIO,
) -> str:
    """
    Extract text from a plain-text resume.
    """

    try:
        if isinstance(source, (str, Path)):
            return clean_text(
                Path(source).read_text(
                    encoding="utf-8",
                    errors="ignore",
                )
            )

        raw = source.read()

        if isinstance(raw, bytes):
            raw = raw.decode(
                "utf-8",
                errors="ignore",
            )

        return clean_text(str(raw))

    except Exception as exc:
        raise ValueError(
            f"Unable to read TXT resume: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------


def extract_text(
    source: str | Path | BinaryIO,
    filename: str | None = None,
) -> str:
    """
    Extract text from a supported resume file.

    Supported formats:
    - PDF
    - TXT
    """

    name = _get_filename(
        source,
        filename,
    ).lower()

    if name.endswith(".pdf"):
        return extract_text_from_pdf(source)

    if name.endswith(".txt"):
        return extract_text_from_txt(source)

    raise ValueError(
        "Unsupported resume format. "
        "Please upload a PDF or TXT file."
    )


# ---------------------------------------------------------------------------
# Section heading detection
# ---------------------------------------------------------------------------


def _normalize_heading(line: str) -> str:
    """
    Normalize a candidate section heading for matching.
    """

    line = line.strip().lower()

    # Replace common heading separators.
    line = re.sub(r"[:|]+$", "", line)
    line = re.sub(r"\s+", " ", line)

    return line.strip()


def detect_section_heading(line: str) -> str | None:
    """
    Return the internal section name if a line looks like
    a recognized resume heading.
    """

    normalized = _normalize_heading(line)

    if not normalized:
        return None

    # Direct heading match.
    if normalized in HEADING_TO_SECTION:
        return HEADING_TO_SECTION[normalized]

    # Handle headings such as:
    # "TECHNICAL SKILLS:"
    # "WORK EXPERIENCE"
    without_punctuation = re.sub(
        r"[^a-z& ]",
        "",
        normalized,
    ).strip()

    return HEADING_TO_SECTION.get(
        without_punctuation
    )


# ---------------------------------------------------------------------------
# Section extraction
# ---------------------------------------------------------------------------


def extract_sections(text: str) -> dict[str, str]:
    """
    Split a resume into recognized sections.

    Returns a dictionary where each key is a section name.
    """

    text = clean_text(text)

    if not text:
        return {}

    lines = text.splitlines()

    sections: dict[str, list[str]] = {}
    current_section = "header"
    sections[current_section] = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            continue

        detected = detect_section_heading(stripped)

        if detected:
            current_section = detected

            if current_section not in sections:
                sections[current_section] = []

            continue

        sections.setdefault(
            current_section,
            [],
        ).append(stripped)

    return {
        section: "\n".join(content).strip()
        for section, content in sections.items()
        if content
    }


# ---------------------------------------------------------------------------
# Candidate information extraction
# ---------------------------------------------------------------------------


EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@"
    r"[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)


PHONE_PATTERN = re.compile(
    r"(?<!\d)"
    r"(?:\+\d{1,3}[\s.-]?)?"
    r"(?:\d{5}[\s.-]\d{5}|\d{10})"
    r"(?!\d)"
)


def extract_email(text: str) -> str:
    """Extract the first email address."""

    if not text:
        return ""

    match = EMAIL_PATTERN.search(text)

    return match.group(0) if match else ""


def extract_phone(text: str) -> str:
    """Extract the first likely phone number."""

    if not text:
        return ""

    match = PHONE_PATTERN.search(text)

    if not match:
        return ""

    value = match.group(0).strip()

    # Avoid returning very short accidental matches.
    digits = re.sub(r"\D", "", value)

    if len(digits) < 10:
        return ""

    return value


def extract_name(text: str) -> str:
    """
    Best-effort candidate name extraction.

    The parser first inspects the beginning of the resume and
    ignores obvious contact/header lines.
    """

    if not text:
        return ""

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    for line in lines[:8]:
        lowered = line.lower()

        if "@" in line:
            continue

        if any(
            keyword in lowered
            for keyword in [
                "resume",
                "curriculum vitae",
                "cv",
                "github",
                "linkedin",
                "http",
                "www.",
            ]
        ):
            continue

        if EMAIL_PATTERN.search(line):
            continue

        if PHONE_PATTERN.search(line):
            continue

        # A name is usually a short alphabetic line.
        words = line.split()

        if 2 <= len(words) <= 5:
            if all(
                re.fullmatch(
                    r"[A-Za-z][A-Za-z.'-]*",
                    word,
                )
                for word in words
            ):
                return line

    return ""


# ---------------------------------------------------------------------------
# Section-specific helpers
# ---------------------------------------------------------------------------


def extract_education(text: str) -> str:
    """
    Extract education information.

    Primarily uses the detected Education section.
    """

    sections = extract_sections(text)

    return sections.get(
        "education",
        "",
    )


def extract_experience(text: str) -> str:
    """
    Extract professional/work experience information.
    """

    sections = extract_sections(text)

    return sections.get(
        "experience",
        "",
    )


def extract_skills_section(text: str) -> str:
    """
    Extract the skills section.

    Actual skill keyword matching will be handled later by
    nlp_processor.py.
    """

    sections = extract_sections(text)

    return sections.get(
        "skills",
        "",
    )


# ---------------------------------------------------------------------------
# Complete candidate parsing
# ---------------------------------------------------------------------------


def parse_resume(
    source: str | Path | BinaryIO,
    filename: str | None = None,
) -> dict:
    """
    Parse a resume and return structured information.
    """

    resolved_filename = _get_filename(
        source,
        filename,
    )

    raw_text = extract_text(
        source,
        filename=resolved_filename,
    )

    if not raw_text:
        raise ValueError(
            "No readable text was found in the resume."
        )

    sections = extract_sections(raw_text)

    result = {
        "filename": resolved_filename,
        "name": extract_name(raw_text),
        "email": extract_email(raw_text),
        "phone": extract_phone(raw_text),
        "skills_section": extract_skills_section(raw_text),
        "education": extract_education(raw_text),
        "experience": extract_experience(raw_text),
        "sections": sections,
        "raw_text": raw_text,
    }

    return result


# ---------------------------------------------------------------------------
# Utility for Flask uploads
# ---------------------------------------------------------------------------


def parse_uploaded_file(
    file_storage,
) -> dict:
    """
    Parse a Flask/Werkzeug uploaded file.

    Example:
        result = parse_uploaded_file(request.files["resume"])
    """

    if file_storage is None:
        raise ValueError(
            "No resume file was provided."
        )

    filename = str(
        getattr(file_storage, "filename", "")
        or ""
    )

    if not filename:
        raise ValueError(
            "The uploaded resume does not have a filename."
        )

    extension = Path(filename).suffix.lower()

    # Read into memory so the same upload can safely be processed.
    data = file_storage.read()

    if not data:
        raise ValueError(
            "The uploaded resume is empty."
        )

    if extension == ".pdf":
        source = io.BytesIO(data)
    elif extension == ".txt":
        source = io.BytesIO(data)
    else:
        raise ValueError(
            "Unsupported resume format. "
            "Only PDF and TXT files are accepted."
        )

    source.name = filename

    return parse_resume(
        source,
        filename=filename,
    )


# ---------------------------------------------------------------------------
# Command-line test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Parse a PDF or TXT resume."
    )

    parser.add_argument(
        "resume",
        help="Path to a PDF or TXT resume.",
    )

    args = parser.parse_args()

    try:
        result = parse_resume(args.resume)

        preview = {
            "filename": result["filename"],
            "name": result["name"],
            "email": result["email"],
            "phone": result["phone"],
            "skills_section": result["skills_section"],
            "education": result["education"],
            "experience": result["experience"],
            "detected_sections": list(
                result["sections"].keys()
            ),
            "text_preview": result["raw_text"][:500],
        }

        print(
            json.dumps(
                preview,
                indent=2,
                ensure_ascii=False,
            )
        )

    except Exception as exc:
        print(f"ERROR: {exc}")