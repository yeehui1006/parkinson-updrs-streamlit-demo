import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "Jitter(%)",
    "Jitter(Abs)",
    "Jitter:RAP",
    "Jitter:PPQ5",
    "Shimmer",
    "Shimmer(dB)",
    "Shimmer:APQ3",
    "Shimmer:APQ5",
    "Shimmer:APQ11",
]


@st.cache_data
def load_data():
    return pd.read_csv("merged_pd_voice_prepared.csv")


@st.cache_resource
def train_demo_model(df):
    X = df[FEATURES].values
    y = df["UPDRS_target"].values
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(X, y)
    return model


def interpret_updrs(score):
    if score < 20:
        return "Lower severity", "Below 20"
    if score < 35:
        return "Moderate severity", "20 to below 35"
    return "Higher severity", "35 and above"


st.set_page_config(
    page_title="Parkinson UPDRS Prediction Demo",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0.75rem;
        max-width: 1180px;
    }
    h1 {
        font-size: 1.65rem !important;
        margin-bottom: 0.15rem !important;
    }
    h2, h3 {
        font-size: 1rem !important;
        margin-top: 0.35rem !important;
        margin-bottom: 0.35rem !important;
    }
    p, li, table, div {
        font-size: 0.88rem;
    }
    [data-testid="stMetric"] {
        background: linear-gradient(180deg, #fafafa 0%, #f1f3f5 100%);
        border: 1px solid #d7dce0;
        border-radius: 8px;
        padding: 0.55rem 0.65rem;
        box-shadow: 0 1px 2px rgba(31, 41, 55, 0.08);
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.78rem;
        color: #5f6b76;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.35rem;
        color: #1f2933;
    }
    [data-testid="stMetricDelta"] {
        color: #5f6b76;
    }
    .stButton > button {
        width: 100%;
        height: 2.35rem;
        border-radius: 8px;
        border: 1px solid #d1d5db;
    }
    .small-note {
        color: #667985;
        font-size: 0.78rem;
        line-height: 1.25;
    }
    .score-band {
        background: #f6f7f8;
        border: 1px solid #dde1e5;
        border-radius: 8px;
        padding: 0.75rem 0.85rem;
    }
    @media (prefers-color-scheme: dark) {
        [data-testid="stMetric"] {
            background: linear-gradient(180deg, #fafafa 0%, #f1f3f5 100%);
            border-color: #d7dce0;
        }
        [data-testid="stMetricLabel"] {
            color: #5f6b76;
        }
        [data-testid="stMetricValue"] {
            color: #1f2933;
        }
        .small-note {
            color: #667985;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

df = load_data()
model = train_demo_model(df)

if "sample_seed" not in st.session_state:
    st.session_state.sample_seed = 42

if st.session_state.get("randomize_now", False):
    st.session_state.sample_seed = int(np.random.randint(0, 1_000_000))
    st.session_state.randomize_now = False

sample = df.sample(n=1, random_state=st.session_state.sample_seed).iloc[0]
X_sample = sample[FEATURES].to_frame().T
prediction = float(model.predict(X_sample.values)[0])
actual = float(sample["UPDRS_target"])
error = actual - prediction
actual_level, actual_range = interpret_updrs(actual)
predicted_level, predicted_range = interpret_updrs(prediction)

st.title("Parkinson Disease Severity Prediction Demo")
st.caption(
    "Randomly selects one prepared patient voice record and predicts its UPDRS "
    "severity score using the final voice features from the notebook."
)

left, right = st.columns([1.05, 1.15], gap="medium")

with left:
    st.subheader("Demo Record")
    if st.button("Randomly Select Patient Record"):
        st.session_state.randomize_now = True
        st.rerun()

    id_col, source_col = st.columns([0.55, 1.45])
    id_col.metric("Subject ID", str(sample["subject_id"]))
    source_col.metric("Dataset Source", str(sample["source"]))

    score1, score2, score3 = st.columns(3)
    score1.metric("Actual UPDRS", f"{actual:.2f}")
    score2.metric("Predicted UPDRS", f"{prediction:.2f}")
    score3.metric("Error", f"{error:.2f}")

    sev1, sev2 = st.columns(2)
    sev1.metric("Actual Severity", actual_level)
    sev1.caption(actual_range)
    sev2.metric("Predicted Severity", predicted_level)
    sev2.caption(predicted_range)

with right:
    st.subheader("UPDRS Meaning")
    st.markdown(
        """
        **UPDRS** means **Unified Parkinson's Disease Rating Scale**.
        A higher score generally indicates more severe Parkinson symptoms.

        | Score range | Demo interpretation |
        |---|---|
        | Below 20 | Lower severity |
        | 20 to below 35 | Moderate severity |
        | 35 and above | Higher severity |
        """
    )
    st.markdown(
        "<p class='small-note'>This severity guide is for project-demo understanding "
        "only, not an official clinical diagnosis scale.</p>",
        unsafe_allow_html=True,
    )

st.subheader("Input Voice Features")
feature_table = X_sample.round(5).T.rename(columns={X_sample.index[0]: "Value"})
st.dataframe(feature_table, use_container_width=True, height=255)

st.markdown(
    "<p class='small-note'>Academic demonstration only. This model is a decision-support "
    "prototype and should be externally validated before any real clinical use.</p>",
    unsafe_allow_html=True,
)
