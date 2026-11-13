from pathlib import Path
import re
import logging

from flask import Flask, render_template, render_template_string, request, redirect
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# 1. FIND THE PROJECT AND HTML TEMPLATE FOLDER
# =========================================================

BASE_DIR = Path(__file__).resolve().parent


def find_template_dir():
    candidates = [
        BASE_DIR / "templates",
        BASE_DIR / "Templates",
        BASE_DIR,
        BASE_DIR.parent / "templates",
        BASE_DIR.parent / "Templates",
    ]

    for folder in candidates:
        if folder.is_dir():
            try:
                names = {
                    item.name.lower()
                    for item in folder.iterdir()
                    if item.is_file()
                }
                if "index.html" in names:
                    return folder
            except OSError:
                pass

    for folder in candidates:
        if folder.is_dir():
            try:
                names = {
                    item.name.lower()
                    for item in folder.iterdir()
                    if item.is_file()
                }
                if "resume.html" in names:
                    return folder
            except OSError:
                pass

    return BASE_DIR / "templates"


TEMPLATE_DIR = find_template_dir()

app = Flask(
    __name__,
    template_folder=str(TEMPLATE_DIR)
)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
app.logger.setLevel(logging.INFO)


def find_template_file(wanted_name):
    if not TEMPLATE_DIR.is_dir():
        return None

    try:
        for item in TEMPLATE_DIR.iterdir():
            if (
                item.is_file()
                and item.name.lower() == wanted_name.lower()
            ):
                return item.name
    except OSError:
        return None

    return None


# =========================================================
# 2. LOAD THE AI MODEL
# =========================================================

print("Loading CareerFlix AI...")

try:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("CareerFlix AI loaded successfully!")

except Exception:
    model = None
    app.logger.exception(
        "Could not load SentenceTransformer. "
        "The website can open, but resume analysis needs the model."
    )


# =========================================================
# 3. SKILLS AND EXTRA KEYWORDS
# =========================================================

SKILLS = [
    "Python", "Java", "C++", "C",
    "Machine Learning", "Artificial Intelligence",
    "NLP", "Deep Learning", "Computer Vision",
    "SQL", "DBMS", "MongoDB", "Git", "GitHub",
    "Flask", "Django", "OpenCV", "TensorFlow",
    "PyTorch", "Pandas", "NumPy", "Scikit-learn",
    "Data Structures", "Algorithms", "REST API",
    "APIs", "HTML", "CSS", "JavaScript",
    "React", "AWS", "Docker", "Linux"
]

EXTRA_TERMS = [
    "software development",
    "problem solving",
    "debugging",
    "testing",
    "teamwork",
    "communication",
    "data analysis",
    "model training",
    "model evaluation",
    "data preprocessing",
    "object oriented programming",
    "version control",
    "cloud",
    "deployment",
    "internship",
    "project"
]


# =========================================================
# 4. BUILT-IN HOMEPAGE
# Used only if index.html is missing or cannot be loaded.
# =========================================================

HOME_FALLBACK = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CareerFlix AI</title>

<style>
* {
    box-sizing: border-box;
}

body {
    margin: 0;
    background: #111;
    color: #fff;
    font-family: Arial, sans-serif;
}

nav {
    padding: 22px 6%;
    background: #090909;
    border-bottom: 1px solid #333;
    font-size: 25px;
    font-weight: bold;
}

nav span {
    color: #e50914;
}

.hero {
    padding: 60px 6% 35px;
    background: linear-gradient(110deg, #181818, #090909);
}

.hero h1 {
    font-size: clamp(32px, 5vw, 56px);
    margin: 0 0 14px;
}

.hero p {
    color: #ccc;
    font-size: 18px;
    max-width: 700px;
    line-height: 1.6;
}

main {
    padding: 25px 6% 60px;
}

.cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 18px;
}

.card {
    display: block;
    text-decoration: none;
    color: white;
    background: #202020;
    border: 1px solid #333;
    border-radius: 12px;
    padding: 25px;
    min-height: 150px;
    transition: 0.2s;
}

.card:hover {
    border-color: #e50914;
    transform: translateY(-3px);
}

.card h2 {
    margin-top: 0;
    color: #fff;
}

.card p {
    color: #ccc;
    line-height: 1.5;
}

.tag {
    color: #e50914;
    font-weight: bold;
}

footer {
    padding: 24px 6%;
    color: #aaa;
    border-top: 1px solid #333;
}
</style>
</head>

<body>

<nav>
    <span>Career</span>Flix
    <small style="font-size:14px;color:#aaa">AI</small>
</nav>

<section class="hero">
    <p class="tag">YOUR CAREER, YOUR NEXT CHAPTER</p>

    <h1>Build skills. Crack interviews. Get hired.</h1>

    <p>
        Analyse your resume, understand your skills,
        and prepare for your career in one place.
    </p>
</section>

<main>
<div class="cards">

    <a class="card" href="{{ url_for('resume') }}">
        <h2>📄 Resume Analyzer</h2>
        <p>Upload a PDF resume and compare it with a target job description.</p>
    </a>

    <a class="card" href="{{ url_for('interview') }}">
        <h2>🎙️ Interview Practice</h2>
        <p>Explore technical and HR interview preparation.</p>
    </a>

    <a class="card" href="{{ url_for('jobs') }}">
        <h2>💼 Job Recommendations</h2>
        <p>Explore job recommendations.</p>
    </a>

    <a class="card" href="{{ url_for('progress') }}">
        <h2>📊 Progress Dashboard</h2>
        <p>Progress tracking is planned for a future version.</p>
    </a>

    <a class="card" href="#internships-section">
        <h2>🎓 Online Internships</h2>
        <p>Explore online and remote internship opportunities for students.</p>
    </a>

</div>
</main>

<footer>CareerFlix AI · Academic Project</footer>

</body>
</html>
"""


# =========================================================
# 5. BUILT-IN RESUME PAGE
# =========================================================

RESUME_FALLBACK = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Resume Analyzer - CareerFlix</title>

<style>
* {
    box-sizing: border-box;
}

body {
    margin: 0;
    background: #111;
    color: #fff;
    font-family: Arial, sans-serif;
    padding: 30px 15px;
}

.box {
    max-width: 760px;
    margin: 25px auto;
    background: #1d1d1d;
    border: 1px solid #333;
    border-radius: 14px;
    padding: 28px;
}

h1 {
    color: #e50914;
}

label {
    display: block;
    margin-top: 18px;
    margin-bottom: 7px;
    font-weight: bold;
}

input, textarea {
    width: 100%;
    padding: 12px;
    border-radius: 7px;
    border: 1px solid #555;
    background: #292929;
    color: #fff;
}

textarea {
    min-height: 150px;
    resize: vertical;
}

button {
    margin-top: 22px;
    background: #e50914;
    color: white;
    border: 0;
    padding: 13px 20px;
    border-radius: 7px;
    font-weight: bold;
    cursor: pointer;
}

a {
    color: #ff5b63;
}

.help {
    color: #bbb;
    line-height: 1.5;
}
</style>
</head>

<body>
<div class="box">

<h1>Resume Analyzer</h1>

<p class="help">
Upload a text-based PDF resume and paste the job description
to calculate an estimated match.
</p>

<form action="{{ url_for('analyze_resume') }}"
      method="POST"
      enctype="multipart/form-data">

    <label for="resume">Resume PDF</label>
    <input id="resume"
           type="file"
           name="resume"
           accept=".pdf,application/pdf"
           required>

    <label for="company">Company name</label>
    <input id="company"
           type="text"
           name="company"
           placeholder="e.g. Target Company">

    <label for="job_role">Target job role</label>
    <input id="job_role"
           type="text"
           name="job_role"
           placeholder="e.g. Python Developer">

    <label for="job_description">Job description</label>
    <textarea id="job_description"
              name="job_description"
              placeholder="Paste the complete job description here"
              required></textarea>

    <button type="submit">Analyse Resume</button>
</form>

<p>
    <a href="{{ url_for('home') }}">← Back to CareerFlix</a>
</p>

</div>
</body>
</html>
"""


# =========================================================
# 6. ERROR PAGE
# =========================================================

ERROR_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">

<title>CareerFlix - Message</title>

<style>
body {
    font-family: Arial, sans-serif;
    background: #111;
    color: white;
    padding: 40px 15px;
}

.box {
    max-width: 700px;
    margin: auto;
    background: #202020;
    border: 1px solid #444;
    border-radius: 12px;
    padding: 25px;
}

h1 {
    color: #e50914;
}

a {
    color: #ff6b73;
    margin-right: 14px;
}
</style>
</head>

<body>
<div class="box">
    <h1>CareerFlix</h1>
    <p>{{ message }}</p>
    <a href="{{ url_for('resume') }}">Try Again</a>
    <a href="{{ url_for('home') }}">Home</a>
</div>
</body>
</html>
"""


# =========================================================
# 7. RESUME ANALYSIS REPORT
# =========================================================

REPORT_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">

<title>CareerFlix - Resume Report</title>

<style>
* {
    box-sizing: border-box;
}

body {
    margin: 0;
    background: #111;
    color: #eee;
    font-family: Arial, sans-serif;
    padding: 25px 14px;
}

.container {
    max-width: 950px;
    margin: auto;
}

.top {
    background: #202020;
    padding: 25px;
    border-radius: 12px;
    border: 1px solid #393939;
}

h1, h2 {
    color: #ff343f;
}

.score {
    font-size: 34px;
    font-weight: bold;
}

.grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    gap: 12px;
}

.card {
    background: #202020;
    border: 1px solid #393939;
    border-radius: 10px;
    padding: 18px;
    margin-top: 15px;
    overflow-wrap: anywhere;
}

li {
    margin: 7px 0;
    line-height: 1.45;
}

.muted {
    color: #bbb;
}

a {
    color: #ff737b;
}
</style>
</head>

<body>
<div class="container">

<div class="top">
    <h1>CareerFlix Resume Analysis Report</h1>

    <p><b>Company:</b> {{ company }}</p>
    <p><b>Target role:</b> {{ role }}</p>

    <p class="score">{{ final }}% — {{ label }}</p>

    <p class="muted">
        This is an estimated text-based match score,
        not a guarantee of hiring.
    </p>
</div>

<div class="grid">

    <div class="card">
        <b>Semantic similarity</b>
        <p>{{ semantic }}%</p>
    </div>

    <div class="card">
        <b>Skill score</b>
        <p>{{ skill }}%</p>
    </div>

    <div class="card">
        <b>Keyword score</b>
        <p>{{ keyword }}%</p>
    </div>

    <div class="card">
        <b>Words analysed</b>
        <p>{{ words }}</p>
    </div>

</div>

<div class="card">
    <h2>Skills found in resume</h2>

    {% if found %}
        <ul>
        {% for item in found %}
            <li>{{ item }}</li>
        {% endfor %}
        </ul>
    {% else %}
        <p>No recognised skills were detected.</p>
    {% endif %}
</div>

<div class="card">
    <h2>Missing or undetected target skills</h2>

    {% if missing %}
        <ul>
        {% for item in missing %}
            <li>{{ item }}</li>
        {% endfor %}
        </ul>
    {% else %}
        <p>No missing target skills were identified.</p>
    {% endif %}

    <p class="muted">
        A skill not detected in the PDF may still be one you know.
        Check your resume manually.
    </p>
</div>

<div class="card">
    <h2>Job keywords found</h2>

    {% if present %}
        <p>{{ present | join(', ') }}</p>
    {% else %}
        <p>No matching keywords were found.</p>
    {% endif %}
</div>

<div class="card">
    <h2>Keywords not found</h2>

    {% if missing_terms %}
        <p>{{ missing_terms | join(', ') }}</p>
    {% else %}
        <p>No missing keywords were identified.</p>
    {% endif %}
</div>

<div class="card">
    <h2>Resume improvement suggestions</h2>

    <ul>
    {% for item in suggestions %}
        <li>{{ item }}</li>
    {% endfor %}
    </ul>
</div>

<div class="card">
    <h2>Resume sections detected</h2>

    {% if sections %}
        <p>{{ sections | join(', ') }}</p>
    {% else %}
        <p>No common section headings were detected.</p>
    {% endif %}
</div>

<p style="margin:22px 0">
    <a href="{{ url_for('resume') }}">Analyse another resume</a>
    &nbsp; | &nbsp;
    <a href="{{ url_for('home') }}">Back to home</a>
</p>

</div>
</body>
</html>
"""


# =========================================================
# 8. PLACEHOLDER PAGES
# =========================================================

PLACEHOLDER_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">

<title>{{ title }} - CareerFlix</title>

<style>
body {
    font-family: Arial, sans-serif;
    background: #111;
    color: white;
    padding: 45px 15px;
}

.box {
    max-width: 650px;
    margin: auto;
    background: #202020;
    border: 1px solid #333;
    border-radius: 12px;
    padding: 30px;
}

h1 {
    color: #e50914;
}

a {
    color: #ff737b;
}
</style>
</head>

<body>
<div class="box">
    <h1>{{ icon }} {{ title }}</h1>
    <p>{{ message }}</p>
    <a href="{{ url_for('home') }}">Back to CareerFlix Home</a>
</div>
</body>
</html>
"""


# =========================================================
# 9. INTERNSHIPS SECTION
# Integrated into the SAME homepage.
# No separate internships.html file is required.
# =========================================================

INTERNSHIPS_SECTION = r"""
<section id="internships-section" aria-labelledby="internships-heading">

<style>
#internships-section {
    margin: 34px auto 0;
    padding: 32px 6% 42px;
    max-width: 1500px;
    background: #151515;
    color: #f5f5f5;
    border-top: 1px solid #333;
    border-radius: 14px;
    font-family: Arial, sans-serif;
    box-sizing: border-box;
}

#internships-section * {
    box-sizing: border-box;
}

#internships-section .cf-internship-eyebrow {
    color: #ff555d;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-size: 12px;
    font-weight: bold;
}

#internships-section h2 {
    font-size: clamp(27px, 3.3vw, 38px);
    margin: 10px 0;
    color: inherit;
}

#internships-section .cf-internship-intro {
    max-width: 800px;
    line-height: 1.65;
    color: #ccc;
    margin-bottom: 22px;
}

#internships-section .cf-internship-toolbar {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin: 18px 0 22px;
}

#internships-section input,
#internships-section select {
    padding: 12px 14px;
    border: 1px solid #444;
    border-radius: 8px;
    color: #fff;
    background: #202020;
    font-size: 15px;
    max-width: 100%;
}

#internships-section input {
    flex: 1;
    min-width: 230px;
}

#internships-section select {
    min-width: 185px;
}

#internships-section .cf-internship-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(225px, 1fr));
    gap: 16px;
}

#internships-section .cf-internship-card {
    background: #202020;
    border: 1px solid #363636;
    border-radius: 12px;
    padding: 20px;
    display: flex;
    flex-direction: column;
    min-height: 270px;
    transition: border-color 0.2s, transform 0.2s;
}

#internships-section .cf-internship-card:hover {
    border-color: #e50914;
    transform: translateY(-3px);
}

#internships-section .cf-internship-icon {
    font-size: 29px;
}

#internships-section .cf-internship-card h3 {
    font-size: 20px;
    line-height: 1.3;
    margin: 12px 0 8px;
    color: #fff;
}

#internships-section .cf-internship-card p {
    line-height: 1.55;
    color: #c9c9c9;
    flex: 1;
    margin: 0 0 15px;
}

#internships-section .cf-internship-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 7px;
    margin: 0 0 16px;
}

#internships-section .cf-internship-tag {
    background: #302022;
    color: #ffb7ba;
    border: 1px solid #573034;
    border-radius: 18px;
    padding: 5px 9px;
    font-size: 12px;
}

#internships-section .cf-internship-btn {
    display: inline-block;
    text-align: center;
    text-decoration: none;
    background: #e50914;
    color: #fff;
    border-radius: 7px;
    padding: 11px 13px;
    font-weight: bold;
}

#internships-section .cf-internship-btn:hover {
    background: #bb0810;
}

#internships-section .cf-internship-no-results {
    display: none;
    padding: 18px;
    color: #ddd;
    background: #202020;
    border-radius: 9px;
    margin-top: 12px;
}

#internships-section .cf-internship-safety {
    margin-top: 20px;
    border: 1px solid #454545;
    background: #191919;
    padding: 15px;
    border-radius: 10px;
    color: #c9c9c9;
    line-height: 1.6;
}

#internships-section .cf-internship-safety strong {
    color: #fff;
}

@media (max-width: 600px) {
    #internships-section {
        padding-left: 4%;
        padding-right: 4%;
    }
}
</style>

<div class="cf-internship-eyebrow">
    CareerFlix · Student Opportunities
</div>

<h2 id="internships-heading">
    🎓 Online Internships for Students
</h2>

<p class="cf-internship-intro">
    Explore platforms where you can search for online or remote internships.
    Check each original listing for work mode, eligibility, duration, stipend,
    and deadline. These links open external websites; CareerFlix does not
    submit applications on your behalf.
</p>

<div class="cf-internship-toolbar">

    <input
        type="search"
        id="cfInternshipSearch"
        placeholder="Search: AI, Python, web development, design..."
        aria-label="Search internship platforms"
    >

    <select
        id="cfInternshipType"
        aria-label="Filter internship type"
    >
        <option value="all">All opportunities</option>
        <option value="remote">Online / remote search</option>
        <option value="student">Student portal</option>
        <option value="simulation">Virtual work simulation</option>
    </select>

</div>

<div class="cf-internship-grid" id="cfInternshipGrid">

    <article
        class="cf-internship-card"
        data-type="student remote"
        data-search="aicte government student internships india engineering technology online remote"
    >
        <div class="cf-internship-icon">🏛️</div>

        <h3>AICTE National Internship Portal</h3>

        <p>
            Explore student-focused internship listings and check each
            opportunity for work mode, eligibility, dates, and requirements.
        </p>

        <div class="cf-internship-tags">
            <span class="cf-internship-tag">Student-focused</span>
            <span class="cf-internship-tag">Portal</span>
        </div>

        <a
            class="cf-internship-btn"
            href="https://internship.aicte-india.org/"
            target="_blank"
            rel="noopener noreferrer"
        >
            Browse opportunities ↗
        </a>
    </article>


    <article
        class="cf-internship-card"
        data-type="remote"
        data-search="internshala online work from home remote python web development data science ai design students stipend"
    >
        <div class="cf-internship-icon">💻</div>

        <h3>Internshala</h3>

        <p>
            Search work-from-home internships in technology, data,
            creative fields, business, and more.
        </p>

        <div class="cf-internship-tags">
            <span class="cf-internship-tag">Online / remote</span>
            <span class="cf-internship-tag">Multiple fields</span>
        </div>

        <a
            class="cf-internship-btn"
            href="https://internshala.com/internships/work-from-home-internships/"
            target="_blank"
            rel="noopener noreferrer"
        >
            Find online internships ↗
        </a>
    </article>


    <article
        class="cf-internship-card"
        data-type="remote"
        data-search="linkedin remote internships india software engineering ai machine learning web development design data analyst"
    >
        <div class="cf-internship-icon">🔎</div>

        <h3>LinkedIn Remote Internships</h3>

        <p>
            Explore remote listings and verify the employer, work arrangement,
            eligibility, deadline, and stipend on each listing.
        </p>

        <div class="cf-internship-tags">
            <span class="cf-internship-tag">Remote search</span>
            <span class="cf-internship-tag">Multiple industries</span>
        </div>

        <a
            class="cf-internship-btn"
            href="https://in.linkedin.com/jobs/remote-internships-jobs"
            target="_blank"
            rel="noopener noreferrer"
        >
            Search remote roles ↗
        </a>
    </article>


    <article
        class="cf-internship-card"
        data-type="simulation"
        data-search="forage virtual job simulation online experience software engineering ai data analytics students not formal internship"
    >
        <div class="cf-internship-icon">🧩</div>

        <h3>Forage Virtual Work Experience</h3>

        <p>
            Practise industry-style tasks through online simulations.
            These can help you learn, but they are not formal internships.
        </p>

        <div class="cf-internship-tags">
            <span class="cf-internship-tag">Virtual experience</span>
            <span class="cf-internship-tag">Not a formal internship</span>
        </div>

        <a
            class="cf-internship-btn"
            href="https://www.theforage.com/"
            target="_blank"
            rel="noopener noreferrer"
        >
            Explore simulations ↗
        </a>
    </article>

</div>

<div class="cf-internship-no-results" id="cfInternshipNoResults">
    No opportunities match your search. Try a different keyword or filter.
</div>

<div class="cf-internship-safety">
    <strong>Safety tip:</strong>
    Check the company, job description, eligibility, duration, stipend,
    and deadline on the original platform. Be cautious if anyone asks
    you to pay money to secure an internship.
</div>

<script>
(function () {
    const section = document.getElementById("internships-section");

    if (!section) {
        return;
    }

    const searchBox = section.querySelector("#cfInternshipSearch");
    const typeFilter = section.querySelector("#cfInternshipType");

    const cards = Array.from(
        section.querySelectorAll(".cf-internship-card")
    );

    const noResults = section.querySelector("#cfInternshipNoResults");

    function filterInternships() {
        const query = searchBox.value.trim().toLowerCase();
        const type = typeFilter.value;
        let visible = 0;

        cards.forEach(function (card) {
            const searchable = (
                card.dataset.search + " " + card.innerText
            ).toLowerCase();

            const types = (card.dataset.type || "").split(/\s+/);

            const matchesSearch =
                !query || searchable.includes(query);

            const matchesType =
                type === "all" || types.includes(type);

            const show = matchesSearch && matchesType;

            card.style.display = show ? "flex" : "none";

            if (show) {
                visible++;
            }
        });

        noResults.style.display = visible ? "none" : "block";
    }

    searchBox.addEventListener("input", filterInternships);
    typeFilter.addEventListener("change", filterInternships);
})();
</script>

</section>
"""


# =========================================================
# 10. INSERT INTERNSHIPS INTO THE SAME HOMEPAGE
# =========================================================

def integrate_internships_section(page):
    """Add the internship section directly to the homepage HTML."""

    if not isinstance(page, str):
        return page

    # Change old links to /internships into same-page links.
    page = re.sub(
        r'href=(["\'])/internships\1',
        'href="#internships-section"',
        page,
        flags=re.IGNORECASE
    )

    page = re.sub(
        r'href=(["\'])/internships/\1',
        'href="#internships-section"',
        page,
        flags=re.IGNORECASE
    )

    lower_page = page.lower()

    # Add the navigation link if it does not already exist.
    if 'href="#internships-section"' not in lower_page:
        internship_link = (
            '<a href="#internships-section" '
            'style="color:inherit;text-decoration:none;margin-left:14px;" '
            'aria-label="Jump to online internships">'
            '🎓 Internships</a>'
        )

        nav_end = lower_page.find("</nav>")

        if nav_end != -1:
            page = (
                page[:nav_end]
                + internship_link
                + page[nav_end:]
            )
        else:
            body_start = re.search(
                r"<body[^>]*>",
                page,
                flags=re.IGNORECASE
            )

            if body_start:
                insertion = (
                    '<div style="padding:12px 6%;background:#111;">'
                    + internship_link
                    + "</div>"
                )

                page = (
                    page[:body_start.end()]
                    + insertion
                    + page[body_start.end():]
                )

        lower_page = page.lower()

    # Insert the actual cards only once.
    if 'id="internships-section"' not in lower_page:
        body_end = lower_page.rfind("</body>")

        if body_end != -1:
            page = (
                page[:body_end]
                + INTERNSHIPS_SECTION
                + page[body_end:]
            )
        else:
            page += INTERNSHIPS_SECTION

    return page


# =========================================================
# 11. SAFE TEMPLATE RENDERING
# =========================================================

def render_page(template_name, fallback_html, **context):
    actual_name = find_template_file(template_name)

    if actual_name:
        try:
            return render_template(actual_name, **context)

        except Exception:
            app.logger.exception(
                "Could not render %s from %s; using built-in fallback.",
                actual_name,
                TEMPLATE_DIR
            )

    else:
        app.logger.warning(
            "Template %s was not found in %s; using built-in fallback.",
            template_name,
            TEMPLATE_DIR
        )

    return render_template_string(fallback_html, **context)


# =========================================================
# 12. RESUME HELPER FUNCTIONS
# =========================================================

def pdf_text(uploaded_file):
    reader = PdfReader(uploaded_file)

    return "\n".join(
        (page.extract_text() or "")
        for page in reader.pages
    ).strip()


def skills(text):
    low = text.lower()
    found = []
    missing = []

    for skill in SKILLS:
        if skill.lower() in low:
            found.append(skill)
        else:
            missing.append(skill)

    return found, missing


def terms(job_description):
    low = job_description.lower()

    found, _ = skills(job_description)
    output = list(found)

    for term in EXTRA_TERMS:
        if term.lower() in low and term not in output:
            output.append(term)

    return output


def score_label(score):
    if score >= 80:
        return "Excellent Match"
    elif score >= 65:
        return "Strong Match"
    elif score >= 50:
        return "Moderate Match"
    elif score >= 35:
        return "Needs Improvement"
    else:
        return "Low Match"


def suggestions(resume_text, job_description, found, missing):
    resume_lower = resume_text.lower()
    output = []

    job_skills, _ = skills(job_description)

    gaps = [
        skill for skill in job_skills
        if skill not in found
    ]

    if gaps:
        output.append(
            "If you genuinely have them, highlight these relevant skills: "
            + ", ".join(gaps[:8]) + "."
        )

    if "project" not in resume_lower:
        output.append(
            "Add 2–3 relevant projects, the technologies used, "
            "your contribution, and results."
        )

    measurable = re.search(
        r"\b\d+%|\b\d+\+|\b\d+\s*"
        r"(users|records|projects|models|datasets|months|years)\b",
        resume_lower
    )

    if not measurable:
        output.append(
            "Add measurable achievements where possible, such as accuracy, "
            "performance improvement, or dataset size."
        )

    if "git" not in resume_lower:
        output.append(
            "Mention Git/GitHub if you have used version control."
        )

    if (
        "experience" not in resume_lower
        and "intern" not in resume_lower
    ):
        output.append(
            "Include internships, training, or practical experience if applicable."
        )

    return output[:6] or [
        "Your resume covers the main detected requirements. "
        "Tailor projects and achievements to the target job."
    ]


def show_error(message):
    return render_template_string(
        ERROR_TEMPLATE,
        message=message
    )


# =========================================================
# 13. HOMEPAGE
# =========================================================

@app.route("/")
def home():
    page = render_page("index.html", HOME_FALLBACK)
    return integrate_internships_section(page)


# =========================================================
# 14. RESUME PAGE
# =========================================================

@app.route("/resume")
def resume():
    return render_page("resume.html", RESUME_FALLBACK)


# =========================================================
# 15. RESUME ANALYSIS
# =========================================================

@app.route("/analyze-resume", methods=["POST"])
def analyze_resume():
    uploaded_file = request.files.get("resume")

    company = (
        request.form.get("company") or "Target Company"
    ).strip()

    role = (
        request.form.get("job_role") or "Target Role"
    ).strip()

    job_description = (
        request.form.get("job_description") or ""
    ).strip()

    if not uploaded_file or not uploaded_file.filename:
        return show_error("Please upload your resume PDF.")

    if not uploaded_file.filename.lower().endswith(".pdf"):
        return show_error("Please upload a PDF resume.")

    if not job_description:
        return show_error("Please enter the job description.")

    if model is None:
        return show_error(
            "The AI similarity model could not be loaded. "
            "Check the Thonny Shell and internet connection, then restart."
        )

    try:
        resume_text = pdf_text(uploaded_file)

        if not resume_text:
            return show_error(
                "No readable text was found in this PDF. "
                "Please upload a text-based PDF rather than a scanned image PDF."
            )

        # Semantic similarity
        resume_embedding = model.encode(
            [resume_text],
            normalize_embeddings=True
        )

        job_embedding = model.encode(
            [job_description],
            normalize_embeddings=True
        )

        semantic_score = float(
            cosine_similarity(
                resume_embedding,
                job_embedding
            )[0][0]
        ) * 100

        semantic_score = max(
            0.0,
            min(100.0, semantic_score)
        )

        # Skill comparison
        found, all_missing = skills(resume_text)
        job_skills, _ = skills(job_description)

        matched_skills = [
            skill for skill in job_skills
            if skill in found
        ]

        if job_skills:
            skill_score = (
                len(matched_skills) / len(job_skills)
            ) * 100
        else:
            skill_score = (
                len(found) / len(SKILLS)
            ) * 100

        # Keyword comparison
        job_terms = terms(job_description)
        resume_lower = resume_text.lower()

        present_terms = [
            term for term in job_terms
            if term.lower() in resume_lower
        ]

        missing_terms = [
            term for term in job_terms
            if term.lower() not in resume_lower
        ]

        if job_terms:
            keyword_score = (
                len(present_terms) / len(job_terms)
            ) * 100
        else:
            keyword_score = 0.0

        # Weighted score: 60% semantic, 30% skills, 10% keywords
        final_score = (
            semantic_score * 0.60
            + skill_score * 0.30
            + keyword_score * 0.10
        )

        # Detect common resume sections
        section_names = [
            "education",
            "experience",
            "projects",
            "skills",
            "certifications",
            "summary"
        ]

        sections = [
            section.title()
            for section in section_names
            if section in resume_lower
        ]

        # Missing skills relevant to the selected job
        relevant_missing = [
            skill for skill in job_skills
            if skill not in found
        ]

        if not relevant_missing:
            relevant_missing = all_missing[:10]

        return render_template_string(
            REPORT_TEMPLATE,
            company=company,
            role=role,
            final=round(final_score, 2),
            semantic=round(semantic_score, 2),
            skill=round(skill_score, 2),
            keyword=round(keyword_score, 2),
            label=score_label(final_score),
            found=found,
            missing=relevant_missing,
            present=present_terms,
            missing_terms=missing_terms[:10],
            suggestions=suggestions(
                resume_text,
                job_description,
                found,
                all_missing
            ),
            words=len(resume_text.split()),
            sections=sections
        )

    except Exception:
        app.logger.exception("Resume analysis failed")

        return show_error(
            "Resume analysis failed. Check that the PDF is readable. "
            "The exact technical error is printed in the Thonny Shell."
        )


# =========================================================
# 16. INTERVIEW PAGE
# =========================================================

@app.route("/interview")
def interview():
    return render_page(
        "interview.html",
        PLACEHOLDER_TEMPLATE,
        title="AI Interview",
        icon="🎙️",
        message=(
            "Interview practice is available as this "
            "project module is developed."
        )
    )


# =========================================================
# 17. PROGRESS PAGE
# =========================================================

@app.route("/progress")
def progress():
    # Interview scoring is not yet connected to persistent storage.
    # Show only recorded results; do not invent performance statistics.
    scores = []
    overall = round(sum(scores) / len(scores)) if scores else 0
    interviews = len(scores)
    questions = 0
    improvement = (
        round(scores[-1] - scores[0])
        if len(scores) > 1 else 0
    )

    return render_page(
        "progress.html",
        PLACEHOLDER_TEMPLATE,
        title="Progress Dashboard",
        icon="📊",
        message="Your CareerFlix learning progress.",
        overall=overall,
        interviews=interviews,
        questions=questions,
        improvement=improvement,
        scores=scores,
        skill_areas=[
            "Programming Fundamentals",
            "Data Structures & Algorithms",
            "SQL & DBMS",
            "Projects & Communication"
        ]
    )


# =========================================================
# 18. JOBS PAGE
# =========================================================

@app.route("/jobs")
def jobs():
    return render_page(
        "jobs.html",
        PLACEHOLDER_TEMPLATE,
        title="Job Recommendations",
        icon="💼",
        message=(
            "Personalized job recommendations are planned "
            "for this module."
        )
    )


# =========================================================
# 19. INTERNSHIP ROUTE
# Redirects to the section on the SAME homepage.
# =========================================================

@app.route("/internships")
def internships():
    return redirect("/#internships-section")


# =========================================================
# 20. HANDLE SERVER ERRORS
# =========================================================

@app.errorhandler(500)
def internal_server_error(error):
    app.logger.exception(
        "Unhandled server error: %s",
        error
    )

    return show_error(
        "The page encountered an error. "
        "Check the Thonny Shell for the detailed traceback."
    ), 500


# =========================================================
# 21. RUN CAREERFLIX
# =========================================================

if __name__ == "__main__":
    print("CareerFlix project folder:", BASE_DIR)
    print("HTML template folder:", TEMPLATE_DIR)
    print(
        "Internships are on the homepage: "
        "http://127.0.0.1:5001/#internships-section"
    )

    app.run(
        host="127.0.0.1",
        port=5001,
        debug=False
    )