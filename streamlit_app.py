
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from PIL import Image

APP_DIR = Path(__file__).parent
st.set_page_config(
    page_title="Parkinson UPDRS Source-Aware Monitoring Prototype",
    page_icon="🧠",
    layout="wide"
)

@st.cache_data
def load_csv(name):
    path = APP_DIR / name
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_resource
def load_model():
    path = APP_DIR / "final_scenario_A_monitoring_prototype_model.joblib"
    if not path.exists():
        return None
    return joblib.load(path)

def show_table(df, max_rows=20):
    if df is None or df.empty:
        st.info("No file available for this table in the app package.")
    else:
        st.dataframe(df.head(max_rows), use_container_width=True)

def metric_from_row(row, key, default=np.nan):
    return row.get(key, default) if isinstance(row, (pd.Series, dict)) else default

model_bundle = load_model()
df1 = load_csv("df1_main_telemonitoring.csv")
df_delta = load_csv("df1_delta_monitoring_table.csv")
all_results = load_csv("all_model_summary_results.csv")
best_results = load_csv("best_models_by_analysis_target.csv")
scenario_a = load_csv("scenario_a_known_patient_monitoring_results.csv")
scenario_a2 = load_csv("scenario_a2_delta_updrs_results.csv")
scenario_b = load_csv("scenario_b_new_patient_groupkfold_summary.csv")
scenario_b_folds = load_csv("scenario_b_new_patient_groupkfold_folds.csv")
dataset2_row = load_csv("supplementary_dataset2_pd_rowlevel_summary.csv")
dataset2_subject = load_csv("supplementary_dataset2_subject_aggregation_summary.csv")
merged_summary = load_csv("exploratory_merged_summary.csv")
importance = load_csv("final_model_permutation_importance.csv")
visual_metrics = load_csv("actual_vs_predicted_visual_diagnostic_metrics.csv")
high_corr = load_csv("dataset1_high_correlation_pairs.csv")
pca_design = load_csv("pca_feature_engineering_design.csv")
validation_design = load_csv("validation_design.csv")
prep_audit = load_csv("data_preparation_audit.csv")
raw_overview = load_csv("raw_dataset_overview.csv")

st.title("Source-Aware Prediction of Parkinson's Disease Severity")
st.caption("Known-patient longitudinal telemonitoring prototype based on the final WQD7003 notebook outputs")

st.warning(
    "This app is a coursework research prototype for known-patient longitudinal telemonitoring. "
    "It is not a universal screening tool, not a diagnostic tool, not a medical device, "
    "and not a substitute for clinician assessment. It assumes an enrolled patient with baseline/history; "
    "new-patient generalisation is shown only as a boundary analysis."
)

tabs = st.tabs([
    "1. Project Overview",
    "2. Baseline & Monitoring Concept",
    "3. Scenario A Monitoring Prediction",
    "4. Delta UPDRS Monitoring",
    "5. New-Patient Generalisation Boundary",
    "6. Supplementary & Merged Analyses",
    "7. Technical Diagnostics"
])

with tabs[0]:
    st.header("Project Overview")
    st.markdown(
        """
        This project evaluates whether biomedical voice features can support Parkinson's disease severity monitoring using UPDRS-related targets.
        The final design is **source-aware** and **scenario-aware** rather than a blind merge of heterogeneous datasets.

        **Product positioning:** this app is a **longitudinal telemonitoring prototype for known Parkinson's patients**. It is not a universal screening app. It should not be used to diagnose a new user from a single voice recording.
        """
    )
    source_table = pd.DataFrame({
        "Item": ["Dataset Name", "Source", "Link", "File Used", "Main Information", "Role in Project"],
        "Dataset 1": [
            "Parkinson's UPDRS Telemonitoring Dataset",
            "Kaggle",
            "https://www.kaggle.com/datasets/mannatpruthi/parkinsons-disease/data",
            "parkinsons_updrs.data",
            "Voice features, age, sex, test_time, motor_UPDRS, total_UPDRS",
            "Main longitudinal dataset for known-patient monitoring and new-patient generalisation testing"
        ],
        "Dataset 2": [
            "Parkinson Speech Dataset with Multiple Types of Sound Recordings",
            "UCI Machine Learning Repository",
            "https://archive.ics.uci.edu/dataset/301/parkinson+speech+dataset+with+multiple+types+of+audio+recordings",
            "train_data.txt",
            "Voice features, UPDRS_score, status, subject ID",
            "Supplementary heterogeneous dataset for source-aware comparison and exploratory integration"
        ]
    })
    st.subheader("Dataset Source Overview")
    st.dataframe(source_table, use_container_width=True, hide_index=True)

    st.subheader("Scenario Design")
    st.markdown(
        """
        - **Scenario A: Known-patient monitoring** uses historical records from enrolled patients and supports the app's main prediction workflow.
        - **Scenario A2: Delta UPDRS monitoring** evaluates change-from-baseline prediction for tracking within-patient deterioration.
        - **Scenario B: New-patient generalisation** tests prediction for completely unseen subjects using GroupKFold. It is a boundary analysis, not the app's deployment target.
        - **Supplementary Dataset 2 and exploratory merged analysis** examine source robustness under a different recording design.
        """
    )
    st.subheader("Raw Dataset Overview")
    show_table(raw_overview)
    st.subheader("Validation Design")
    show_table(validation_design)

with tabs[1]:
    st.header("Baseline Calibration and Monitoring Concept")
    st.markdown(
        """
        This app is designed for **known patients**, not blind screening. The intended workflow is:

        1. **Baseline calibration:** after clinical diagnosis or enrolment, the patient has an initial clinician-assessed UPDRS score and historical voice records.
        2. **Repeated home monitoring:** the patient records voice samples over time, and the system compares new records with the patient's own history.
        3. **Absolute severity trend:** Scenario A estimates later UPDRS values within the known-patient monitoring setting.
        4. **Change-from-baseline tracking:** Scenario A2 focuses on delta UPDRS to show whether the patient's condition changes relative to their own baseline.
        5. **Early-warning concept:** repeated worsening patterns could trigger a non-clinical recommendation to consult a clinician. Real use would require clinically validated thresholds.

        The app should **not** be interpreted as: "new user records voice for a few seconds and receives a reliable diagnosis or UPDRS score." Scenario B in this project specifically shows why that blind generalisation claim is not supported.
        """
    )
    if not raw_overview.empty:
        st.subheader("Dataset Overview")
        show_table(raw_overview)
    if not validation_design.empty:
        st.subheader("Validation Design")
        show_table(validation_design)

with tabs[2]:
    st.header("Scenario A: Known-Patient Monitoring Prediction")
    st.markdown(
        """
        This tab uses the final Scenario A monitoring prototype model saved from the notebook. The model is trained for a
        **patient-history-assisted longitudinal monitoring** setting. It assumes the patient is already enrolled and has baseline/history information.
        Its high performance should not be interpreted as voice-only prediction, diagnostic screening, or evidence of reliable prediction for unseen patients.
        """
    )
    if model_bundle is None or df1.empty:
        st.error("Model or Dataset 1 monitoring table is missing from the app package.")
    else:
        model = model_bundle["model"] if isinstance(model_bundle, dict) else model_bundle
        target = model_bundle.get("target", "motor_UPDRS") if isinstance(model_bundle, dict) else "motor_UPDRS"
        features = model_bundle.get("features", []) if isinstance(model_bundle, dict) else []
        st.info(f"Prototype target: **{target}** | Feature set: **{model_bundle.get('feature_set_name', 'unknown')}** | Model: **{model_bundle.get('model_name', 'unknown')}**")

        col1, col2 = st.columns([1, 2])
        with col1:
            subjects = sorted(df1["global_subject_id"].unique()) if "global_subject_id" in df1.columns else []
            chosen_subject = st.selectbox("Select patient / global subject ID", subjects)
            subject_df = df1[df1["global_subject_id"] == chosen_subject].sort_values("test_time")
            if subject_df.empty:
                st.stop()
            record_labels = [f"index={idx} | test_time={row['test_time']:.2f} | actual {target}={row[target]:.2f}" for idx, row in subject_df.iterrows()]
            chosen_label = st.selectbox("Select prepared record", record_labels)
            chosen_index = int(chosen_label.split("|")[0].split("=")[1].strip())
            selected = df1.loc[[chosen_index]].copy()

        with col2:
            pred = float(model.predict(selected[features])[0])
            actual = float(selected[target].iloc[0]) if target in selected.columns else np.nan
            error = pred - actual if not np.isnan(actual) else np.nan
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Predicted UPDRS", f"{pred:.2f}")
            m2.metric("Actual UPDRS", f"{actual:.2f}" if not np.isnan(actual) else "N/A")
            m3.metric("Prediction Error", f"{error:.2f}" if not np.isnan(error) else "N/A")
            m4.metric("Test Time", f"{float(selected['test_time'].iloc[0]):.2f}")
            st.subheader("Selected Record")
            display_cols = [c for c in ["global_subject_id", "subject_id", "age", "sex", "test_time", "motor_UPDRS", "total_UPDRS"] if c in selected.columns]
            st.dataframe(selected[display_cols], use_container_width=True, hide_index=True)

        st.subheader("Scenario A Result Table")
        show_table(scenario_a, max_rows=50)
        img_path = APP_DIR / "actual_vs_predicted_scenario_A.png"
        if img_path.exists():
            st.image(Image.open(img_path), caption="Scenario A actual vs predicted diagnostic plot", use_container_width=True)

with tabs[3]:
    st.header("Scenario A2: Change-from-Baseline UPDRS Monitoring")
    st.markdown(
        """
        Delta UPDRS analysis evaluates whether the modelling workflow can predict **change from each patient's own baseline**.
        This supports the monitoring concept because patients and clinicians often care about deterioration relative to the patient's own prior state, not only the absolute score.
        This remains a research prototype. Any early-warning threshold would require clinical validation before real use.
        """
    )
    st.subheader("Delta Model Results")
    show_table(scenario_a2, max_rows=50)
    if not df_delta.empty:
        st.subheader("Delta Record Explorer")
        target_options = [c for c in ["delta_motor_UPDRS", "delta_total_UPDRS"] if c in df_delta.columns]
        target_delta = st.selectbox("Select delta target to inspect", target_options) if target_options else None
        if target_delta:
            subjects_delta = sorted(df_delta["global_subject_id"].unique()) if "global_subject_id" in df_delta.columns else []
            chosen_sub_delta = st.selectbox("Select subject for delta trajectory", subjects_delta, key="delta_subject")
            sub_delta = df_delta[df_delta["global_subject_id"] == chosen_sub_delta].sort_values("test_time")
            st.line_chart(sub_delta.set_index("test_time")[[target_delta]])
            cols = [c for c in ["global_subject_id", "test_time", "motor_UPDRS", "total_UPDRS", "baseline_motor_UPDRS", "baseline_total_UPDRS", "delta_motor_UPDRS", "delta_total_UPDRS"] if c in sub_delta.columns]
            st.dataframe(sub_delta[cols], use_container_width=True)
    else:
        st.info("Delta monitoring table is not available in the package.")

with tabs[4]:
    st.header("Scenario B: New-Patient Generalisation Boundary")
    st.markdown(
        """
        Scenario B uses subject-grouped validation to test whether the model generalises to completely unseen patients.
        This scenario is deliberately stricter than Scenario A. The final interpretation is that new-patient generalisation is limited,
        and the Streamlit app should therefore not be used as a blind prediction tool for unseen patients.
        """
    )
    st.subheader("Scenario B Summary")
    show_table(scenario_b, max_rows=50)
    st.subheader("Scenario B Fold-Level Results")
    show_table(scenario_b_folds, max_rows=50)
    img_path = APP_DIR / "actual_vs_predicted_scenario_B.png"
    if img_path.exists():
        st.image(Image.open(img_path), caption="Scenario B actual vs predicted diagnostic plot", use_container_width=True)
    st.subheader("Visual Diagnostic Metrics")
    show_table(visual_metrics)
    st.info(
        "If Scenario B has negative R² or fails to beat the mean baseline, this is not a code error. It indicates that the model has limited generalisation to completely unseen patients under the tested feature sets and validation design."
    )

with tabs[5]:
    st.header("Supplementary and Exploratory Analyses")
    st.markdown(
        """
        These analyses support the source-aware project design. Dataset 2 is used as supplementary evidence under a different recording design,
        while the merged analysis is exploratory and should not be interpreted as the final deployment model.
        """
    )
    st.subheader("Dataset 2 PD-only Row-Level GroupKFold")
    show_table(dataset2_row, max_rows=50)
    st.subheader("Dataset 2 Subject-Level Aggregation")
    show_table(dataset2_subject, max_rows=50)
    st.subheader("Exploratory Merged Analysis")
    show_table(merged_summary, max_rows=50)
    st.subheader("Within-Source Standardisation and PCA Design")
    show_table(pca_design, max_rows=50)

with tabs[6]:
    st.header("Technical Diagnostics")
    st.subheader("Final Model Permutation Importance")
    show_table(importance, max_rows=50)
    st.caption("High Scenario A performance is patient-history-assisted. Patient/time/demographic variables may contribute strongly alongside voice features.")
    st.subheader("High-Correlation Audit for Dataset 1 Acoustic Features")
    show_table(high_corr, max_rows=50)
    st.subheader("Data Preparation Audit")
    show_table(prep_audit, max_rows=50)
    st.subheader("Auto-Generated Result Notes")
    notes_path = APP_DIR / "auto_generated_result_notes.txt"
    if notes_path.exists():
        st.text(notes_path.read_text(encoding="utf-8"))
    else:
        st.info("No auto-generated result notes found.")

st.divider()
st.caption("WQD7003 Group 16 | Source-aware and scenario-aware Parkinson voice severity analysis")
