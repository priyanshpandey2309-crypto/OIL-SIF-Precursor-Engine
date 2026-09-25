# SIF Precursor Detection Engine (SIH 2026, PS 26165, Oil India Limited)

Reads free-text UA/UC, near-miss and incident reports and
1. flags reports with Serious Injury & Fatality (SIF) potential,
2. tags each to an IOGP Life-Saving Rule (Energy Isolation, Hot Work, Confined Space, Line of Fire, Working at Height, Safe Mechanical Lifting, Work Authorisation, Bypassing Safety Controls, Driving),
3. extracts activity, location and failed barrier, and ranks sites and activities by SIF-precursor density on an interactive dashboard.

## Run
    pip install -r requirements.txt
    streamlit run app.py
Use "Demo data" in the sidebar, or upload `sample_reports.csv` (columns: text, site, date; report_id and report_type optional).

## How it works
- `engine.py`: TF-IDF (1-2 grams) + logistic regression for SIF (class-balanced) and for LSR tagging; keyword fallback for low-confidence LSR tags; regex lexicons for activity, location and barrier failure; word-level explanation of each SIF score; a confidence label (High/Medium/Low) alongside the probability; a reviewer-feedback store and a `retrain()` method that folds confirmed corrections back into training.
- `app.py`: seven tabs — Where to intervene, Life-Saving Rules, Precursor patterns, Flagged reports, Try a report, **Reviewer feedback** (confirm/correct a flagged report's tags, see the correction rate, retrain live), and **Data & validation** (the honest data story and the plan to move to OIL's real reports).
- SIF density = flagged reports / total reports, shown per site and per activity (with a minimum-report filter so small samples do not dominate).
- `architecture_diagram.png`: one-page pipeline diagram (ingestion → pre-processing → NLP engine → dashboard → HSE action, with the reviewer-feedback loop) for the PPT.

## Limits to state honestly
- OIL's real reports are not public. Models are trained on synthetic reports written to follow the SIF-precursor research the problem statement cites (DEKRA Martin & Black 2015, EEI SIF Precursor model, VelocityEHS 2024 PSIF classifier), so the reported F1 (~0.9) is only a pipeline check, not real-world accuracy. The in-app Data & validation tab explains this and the retraining plan — use it, don't hide it.
- The Reviewer feedback tab is a working demo of the human-in-the-loop mechanism, not a claim that the model has learned from real HSE reviewers yet.
- Check the LSR names against IOGP Report 459 before final submission.
- Add Hindi/Assamese-English mixed text handling once real report examples are available.

## Next steps
Transformer model (e.g. fine-tuned DistilBERT/IndicBERT), a persistent database for the feedback log, role-based login for HSE reviewers, automated alerts to site HSE heads, API for the HSSE platform.
