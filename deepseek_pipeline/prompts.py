# -*- coding: utf-8 -*-
"""
prompts.py
----------
Exact prompts used to convert raw CV / JMP text into structured variables.

OA1_PROMPT is transcribed verbatim from Table OA1 of Ke & Long (2026),
"Prompting Using GPT-4". We feed it to DeepSeek (deepseek-chat) instead of
GPT-4 to extract the same JSON structure, then engineer the model features
from that JSON.

Two auxiliary prompts (gender, research area/method) cover the few dataset
columns that the paper derives separately — they are NOT part of Table OA1
but are required to build the full Set C feature vector.
"""

# ── Table OA1 — verbatim from the paper ──────────────────────────────────────
# The candidate CV text is appended AFTER this prompt (the prompt ends with
# "Here is the text of the candidate CV:").
OA1_PROMPT = """As a recruitment member of the accounting department of a university. Your task is to process the CV texts of candidates, extract the information I want to structural output. Here is the list of the information I want from their CVs.
1. Name
2. Education experience
3. Employment experience
4. Research interest
5. Publication
6. Working paper
7. Conference presentation
8. Teaching experience
9. Awards
10. Professional service and membership
11. Reference
12. Language
Present your findings as a single JSON object, conforming to the following structure:
```{
 "name": "name of the candidate",
 "Bachelor degree": "the name of the university where the candidate got the bachelor degree (Do not include the country. Do not include the school name or department name, just the name of the university.)",
 "has Bachelor honor": 1 if the candidate has an honor in bachelor stage, 0 otherwise,
 "Master degree":  "the name of the university where the candidate got the master degree (do not include the country. Do not include the school name or department name, just the name of the university.)",
 "has Master honor": 1 if the candidate has an honor in Master stage, 0 otherwise,
 "PhD degree": "the name of the university where the candidate will get the PhD degree (do not include the country. Do not include the school name or department name, just the name of the university.)",
 "has PhD honor": 1 if the candidate has an honor (e.g., Summa Cum Laude) in PhD stage, 0 otherwise,
 "visiting experience": the list of universities (note: not school names) the candidate visited as a visiting PhD or visiting scholar(do not include the country), in the form of a list ["university 1","university 2",...],
 "research interest": the text describing the research interests of the candidate,
 "papers": the list of paper including the following attributes: title,coauthor (if no coauthor, say "solo"), journal (if not under review, leave this blank), status (including under review, revise and resubmit(R&R), published),
 "number of published papers": the number of published papers, 0 if no publication,
 "published journal list": the list of the journals if the candidate has published papers like ["journal 1", "journal 2"], blank list if no publication,
 "number of R&R papers": the number of papers that get revise and resubmit (R&R) by journals (note that under review is not R&R),
 "R&R journal list": the list of the journals gave an R&R to the candidate like ["journal 1", "journal 2"],
 "number of papers in progress": the number of papers that are not published or get an R&R,
 "coauthor list": the list of all the coauthors of all the papers of the candidate like ["coauthor 1","coauther 2",...],
 "number of coauthors": the number of coauthors of all the papers mentioned in this CV,
 "conference presentation": the list of conferences the candidate participated (including academic conferences and workshops),
 "number of presentations": the total number of presentations, including academic conferences, workshops, etc.
 "number of presentations at top conferences": the total number of presentations at top conferences, including American Accounting Association (AAA) annual conference, Journal of Accounting and Economics (JAE) conference, Contemporary Accounting Research (CAR) conference, Journal of Accounting Research (JAR) conference, Review of Accounting Studies (RAS) conference, Financial Accounting and Reporting Midyear meeting, Managerial Accounting Midyear Meeting, Journal of the American Taxation Association (JATA) conference.
 "Teaching experiences": the list of teaching experiences in the form of ["experience 1","experience 2",......],
 "number of teaching experiences": the number of teaching experiences,
 "academic awards": the list of academic awards,
 "number of awards": the number of academic awards,
 "reviewer": the list of journals or associations where the candidate serves as ad hoc reviewer,
 "number of reviewers": the number of journals or associations where the candidate serves as ad hoc reviewer,
 "membership": the associations that the candidate join as a member,
 "number of membership": the number of associations that the candidate join as a member,
 "number of working experiences": the number of working experiences,
 "had academic work": 1 if the candidate has worked in academic institutions, 0 otherwise,
 "had non-academic work": 1 if the candidate has worked outside academic institutions, 0 otherwise,
 "references": the list of names of the references like ["name 1","name 2"] (remove the titles like professor, prof, Dr, etc.),
 "provide abstract": 1 if the candidate presents the abstracts of the papers in the CV file, 0 otherwise,
 "abstract list": the list of the abstracts in the form of ["abstract 1", "abstract 2",......]
 "language list": the list of languages that the candidate speak in the form of ["language 1","language 2",...], blank if not mentioned.
}```
If any of the above information is not mentioned in the CV profile, just fill the item with a blank. Your output should be a single valid JSON object and should not contain any additional or explanatory text. Here is the text of the candidate CV:
"""


# ── Auxiliary prompt: gender (column `gender`) ───────────────────────────────
# Table OA1 does not output gender; the paper infers it separately. We use a
# lightweight DeepSeek classification on the candidate name.
GENDER_PROMPT = """Based on the following name of an academic job market candidate, infer the most likely gender.
Respond with a single valid JSON object only: {{"gender": 1}} for male or {{"gender": 0}} for female.
If genuinely ambiguous, make your best guess. Name: {name}"""


# ── Auxiliary prompt: primary research area + method ─────────────────────────
# Produces the columns PrimaryResearchArea_* and PrimaryResearchMethod_*.
# Driven by the candidate's research interest text + paper titles/abstracts.
RESEARCH_CLASSIFY_PROMPT = """You are classifying an accounting PhD candidate's research profile.
Given their research interest text and paper titles below, output a single valid JSON object only, with exactly these keys:
{{
 "primary_area": one of "financial", "auditing", "managerial", "tax",
 "primary_method": one of "archival", "experiment", "analytical"
}}
Pick the single best primary area and the single best primary method. Do not output anything else.

Research interest: {research_interest}
Papers: {papers}"""
