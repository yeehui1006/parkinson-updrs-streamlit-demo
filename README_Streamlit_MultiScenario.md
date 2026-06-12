# WQD7003 Multi-Scenario Streamlit App

This Streamlit app demonstrates the final source-aware and scenario-aware Parkinson UPDRS project workflow.

## Product positioning

The app is a **known-patient longitudinal telemonitoring prototype**. It is designed for an enrolled Parkinson's patient with clinician-assessed baseline information and historical voice records.

It is **not**:
- a universal Parkinson's screening app;
- a diagnostic tool;
- a medical device;
- a tool for a new user to record one voice sample and receive a reliable UPDRS score.

## Included tabs

1. Project Overview
2. Baseline & Monitoring Concept
3. Scenario A Monitoring Prediction
4. Delta UPDRS Monitoring
5. New-Patient Generalisation Boundary
6. Supplementary & Merged Analyses
7. Technical Diagnostics

## How to run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Interpretation

The main prediction tab implements the Scenario A known-patient monitoring prototype model saved from the final notebook. Scenario A2 supports change-from-baseline monitoring. Scenario B is included as a boundary analysis showing why blind prediction for unseen patients is not the deployment target.

Real-world use would require external validation, clinician-approved thresholds, data drift monitoring, uncertainty reporting, and regulatory review.
