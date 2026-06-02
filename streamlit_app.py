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


st.set_page_config(
    page_title="Parkinson UPDRS Prediction Demo",
    page_icon="",
    layout="centered",
)

st.title("Parkinson Disease Severity Prediction Demo")
st.write(
    "This demo randomly selects one prepared patient voice record and predicts "
    "its UPDRS severity score using the same final voice features used in the notebook."
)

df = load_data()
model = train_demo_model(df)

st.subheader("Dataset Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Records", f"{len(df):,}")
col2.metric("Subjects", f"{df['subject_id'].nunique():,}")
col3.metric("Features", len(FEATURES))

st.subheader("Random Patient Recording")

if "sample_seed" not in st.session_state:
    st.session_state.sample_seed = 42

if st.button("Randomly Select Patient Record"):
    st.session_state.sample_seed = int(np.random.randint(0, 1_000_000))

sample = df.sample(n=1, random_state=st.session_state.sample_seed).iloc[0]
X_sample = sample[FEATURES].to_frame().T
prediction = float(model.predict(X_sample.values)[0])
actual = float(sample["UPDRS_target"])
error = actual - prediction

info1, info2 = st.columns(2)
info1.metric("Subject ID", str(sample["subject_id"]))
info2.metric("Dataset Source", str(sample["source"]))

score1, score2, score3 = st.columns(3)
score1.metric("Actual UPDRS", f"{actual:.2f}")
score2.metric("Predicted UPDRS", f"{prediction:.2f}")
score3.metric("Error", f"{error:.2f}")

st.subheader("Input Voice Features")
st.dataframe(X_sample.round(5), use_container_width=True)

st.info(
    "This is a demonstration decision-support workflow only. It is not a "
    "clinical diagnosis tool and should be externally validated before real use."
)

