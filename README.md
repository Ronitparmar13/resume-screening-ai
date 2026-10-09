# Resume Screening AI

An explainable AI-powered resume screening application built with Python, Flask, NLP, and Scikit-learn. The application extracts candidate information from PDF and TXT resumes, compares candidates against a job description, calculates match scores, ranks candidates, and generates a downloadable CSV report.

## Features

- **Multiple resume uploads:** Screen multiple PDF and TXT resumes in one submission.
- **Resume parsing:** Extract candidate names, email addresses, phone numbers, education, experience, and resume text.
- **Skill extraction:** Identify technical skills using a configurable JSON skill dictionary.
- **NLP preprocessing:** Tokenization, stop-word removal, and lightweight rule-based lemmatization.
- **Job-description matching:** Compare candidate resumes with a supplied job description.
- **Explainable scoring:** Combine skill coverage and TF-IDF cosine similarity.
- **Candidate ranking:** Order candidates by their final match score.
- **Candidate summaries:** View extracted information, matched skills, missing skills, and score breakdowns.
- **CSV export:** Download a candidate ranking report.

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Web framework | Flask |
| PDF extraction | pdfplumber |
| NLP preprocessing | Custom Python processing |
| Skill matching | JSON-based skill dictionary |
| Text similarity | Scikit-learn TF-IDF and cosine similarity |
| Frontend | HTML, CSS, JavaScript |
| Report export | Python CSV |

## Scoring Algorithm

The application uses a weighted score:

**Final Match Score = (0.60 × Skill Match) + (0.40 × TF-IDF Similarity)**

Both components are expressed as percentages.

- **Skill Match (60%):** Percentage of distinct skills extracted from the job description that are also detected in the candidate's resume.
- **TF-IDF Similarity (40%):** Cosine similarity between the processed resume text and job-description text.

For example, if a candidate has a 90% skill match and 70% TF-IDF similarity:

`Final Score = (0.60 × 90) + (0.40 × 70) = 82%`

Candidates are ranked from highest to lowest final score.

The score is a screening aid, not a definitive measure of candidate suitability. Extraction and matching results depend on resume formatting, wording, and the skill dictionary.

## Project Structure

```text
resume-screening-ai/
├── app.py
├── resume_parser.py
├── nlp_processor.py
├── matcher.py
├── config.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── skills.json
├── templates/
│   └── index.html
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
├── reports/
└── tests/
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Ronitparmar13/resume-screening-ai.git
cd resume-screening-ai
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows, activate it using:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Run the Application

Start the Flask server:

```bash
python app.py
```

Open the application in your browser:

http://127.0.0.1:5000

## How to Use

1. Paste the job description into the text area.
2. Upload one or more PDF or TXT resumes.
3. Start screening.
4. Review the ranked candidates and their score breakdowns.
5. Inspect extracted education, experience, matched skills, and missing skills.
6. Download the candidate ranking CSV report.

## Limitations

- Resume parsing relies on extracted text; scanned image-only PDFs may require OCR.
- Skill extraction depends on the configured skill dictionary and its aliases.
- Education, experience, candidate names, and other fields are detected using heuristics and may need manual verification.
- TF-IDF measures textual similarity rather than deep semantic understanding.
- Results should support, not replace, human review and fair hiring decisions.

## Security and Privacy

Use synthetic or appropriately authorized resumes when demonstrating this project. Candidate contact details are personal information and should not be published in a public repository. Uploaded resumes are processed in memory by the application, while generated CSV reports are saved locally.

Keep virtual environments, environment secrets, and reports containing personal data out of Git.

## Author

**Ronit Parmar**

WeIntern — Week 2 Artificial Intelligence Internship Project

---

*This project demonstrates an explainable resume-screening workflow using NLP, skill matching, and TF-IDF similarity.*
