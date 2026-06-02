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
        return "Lower severity", "Score is below 20 in this project demo range."
    if score < 35:
        return "Moderate severity", "Score is between 20 and 35 in this project demo range."
    return "Higher severity", "Score is 35 or above in this project demo range."


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

with st.expander("What does the UPDRS score mean?"):
    st.write(
        "UPDRS stands for Unified Parkinson's Disease Rating Scale. It is used to "
        "measure Parkinson Disease symptoms and disease severity. In general, a "
        "higher UPDRS score means more severe symptoms."
    )
    st.write(
        "For this project demo, the predicted score is interpreted using a simple "
        "three-level guide based on the score range observed in the prepared dataset:"
    )
    st.markdown(
        """
        | UPDRS score | Demo interpretation |
        |---|---|
        | Below 20 | Lower severity |
        | 20 to below 35 | Moderate severity |
        | 35 and above | Higher severity |
        """
    )
    st.caption(
        "This guide is for presentation and model-demo understanding only. It is not "
        "an official clinical diagnosis scale."
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
actual_level, actual_note = interpret_updrs(actual)
predicted_level, predicted_note = interpret_updrs(prediction)

info1, info2 = st.columns(2)
info1.metric("Subject ID", str(sample["subject_id"]))
info2.metric("Dataset Source", str(sample["source"]))

score1, score2, score3 = st.columns(3)
score1.metric("Actual UPDRS", f"{actual:.2f}")
score2.metric("Predicted UPDRS", f"{prediction:.2f}")
score3.metric("Error", f"{error:.2f}")

level1, level2 = st.columns(2)
level1.metric("Actual Severity", actual_level)
level1.caption(actual_note)
level2.metric("Predicted Severity", predicted_level)
level2.caption(predicted_note)

st.subheader("Input Voice Features")
st.dataframe(X_sample.round(5), use_container_width=True)

st.info(
    "This is a demonstration decision-support workflow only. It is not a "
    "clinical diagnosis tool and should be externally validated before real use."
)
