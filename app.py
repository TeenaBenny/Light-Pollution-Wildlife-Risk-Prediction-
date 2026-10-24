
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="Wildlife Risk Predictor",
    page_icon="🌿",
    layout="wide",
)

# ---------------- PATHS ----------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "wildlife_model.joblib"
CLEAN_PATH = BASE_DIR / "data" / "processed" / "wildlife_cleaned.csv"
RAW_PATH = BASE_DIR / "data" / "raw" / "Sanders_et_al_datafile.csv"

# ---------------- WEBSITE DESIGN ----------------
st.markdown("""
<style>
.stApp {
    background-color: #f7faf7;
    color: #26352c;
}
.block-container {
    max-width: 1150px;
    padding-top: 2rem;
}
h1, h2, h3 {
    color: #234b35;
}
.hero {
    background: #e6f3e9;
    padding: 28px;
    border-radius: 14px;
    border: 1px solid #d0e5d5;
    margin-bottom: 24px;
}
.hero h1 {
    color: #234b35;
    margin-bottom: 8px;
}
.hero p {
    color: #496353;
    font-size: 16px;
}
div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #dce8df;
    border-radius: 10px;
    padding: 15px;
}
.stButton button, .stFormSubmitButton button {
    background: #2f7548;
    color: white;
    border-radius: 8px;
    border: none;
    min-height: 42px;
}
.stButton button:hover, .stFormSubmitButton button:hover {
    background: #245b38;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL AND DATA ----------------
if not MODEL_PATH.exists():
    st.error("Model not found. Run `py train_model.py` first.")
    st.stop()

DATA_PATH = CLEAN_PATH if CLEAN_PATH.exists() else RAW_PATH

if not DATA_PATH.exists():
    st.error("Dataset not found. Check your data folder.")
    st.stop()

try:
    saved = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH, encoding="latin1")
    df.columns = df.columns.str.strip()

    model = saved["model"]
    features = saved["features"]
    target = saved["target"]

except Exception as error:
    st.error(f"Could not load model or dataset: {error}")
    st.stop()

missing = [c for c in features if c not in df.columns]
if missing:
    st.error(f"Missing model features in dataset: {missing}")
    st.stop()

numeric_features = df[features].select_dtypes(
    include=["number", "bool"]
).columns.tolist()

categorical_features = [
    c for c in features if c not in numeric_features
]

# ---------------- CHART HELPER ----------------
def show_chart_explanation(title, explanation):
    st.subheader(title)
    st.caption(explanation)


# ---------------- NAVIGATION ----------------
st.sidebar.title("🌿 Wildlife Predictor")
page = st.sidebar.radio(
    "Menu",
    ["Home", "Make Prediction", "About"],
)

# ---------------- HOME PAGE ----------------
if page == "Home":

    st.markdown("""
    <div class="hero">
        <h1>🌿 Wildlife Risk Prediction</h1>
        <p>
            Explore research findings about artificial light at night
            and its potential effects on wildlife.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Research Records", f"{len(df):,}")
    col2.metric("Input Features", len(features))
    col3.metric("Prediction Target", target)

    st.subheader("About this project")
    st.write(
        "This machine-learning project learns patterns from wildlife "
        "research data and predicts a recorded research-effect category."
    )

    st.info(
        "The model predicts the dataset's recorded target category. "
        "This is not automatically a validated Low, Medium, or High "
        "wildlife-risk score."
    )

    # ---------------- CHART 1 ----------------
    st.divider()
    show_chart_explanation(
        "Chart 1: Research Records by Effect Category",
        "Each bar represents the number of records assigned to a "
        "recorded outcome category. Taller bars mean that category "
        "appears more often in the dataset."
    )

    if target in df.columns:
        category_counts = (
            df[target]
            .dropna()
            .astype(str)
            .value_counts()
            .sort_values(ascending=False)
        )

        if not category_counts.empty:
            fig, ax = plt.subplots(figsize=(9, 4.5))
            category_counts.plot(
                kind="bar",
                ax=ax,
                color="#43865b",
                edgecolor="white",
            )

            ax.set_title("Number of Research Records in Each Category")
            ax.set_xlabel("Recorded effect category")
            ax.set_ylabel("Number of records")
            ax.tick_params(axis="x", rotation=30)
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.caption(
                "How to read it: compare the bar heights to see which "
                "recorded categories are most and least common. "
                "Frequency does not tell us how harmful an effect is."
            )
        else:
            st.warning("No recorded target categories are available.")

    # ---------------- CHART 2 ----------------
    st.divider()
    show_chart_explanation(
        "Chart 2: Distribution of Night-time Light Exposure",
        "This histogram groups recorded Night_Lux measurements into "
        "ranges. It helps show whether the dataset contains mostly "
        "lower or higher recorded light levels."
    )

    if "Night_Lux" in df.columns:
        lux = pd.to_numeric(df["Night_Lux"], errors="coerce").dropna()

        if not lux.empty:
            fig, ax = plt.subplots(figsize=(9, 4.5))
            ax.hist(
                lux,
                bins=20,
                color="#4a91ad",
                edgecolor="white",
            )

            ax.set_title("How Often Different Night-time Light Levels Occur")
            ax.set_xlabel("Night-time light level (Night_Lux; check source units)")
            ax.set_ylabel("Number of records")
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.caption(
                "How to read it: the tallest bar shows the range of light "
                "levels with the most records. This is a distribution of "
                "recorded measurements, not a measure of wildlife damage."
            )
        else:
            st.info("No usable numeric Night_Lux values were found.")
    else:
        st.info("The dataset does not contain a Night_Lux column.")

    # ---------------- DATA PREVIEW ----------------
    st.divider()
    st.subheader("Dataset Preview")
    st.write("These are the first five records in the dataset.")
    st.dataframe(df.head(5), use_container_width=True)

# ---------------- PREDICTION PAGE ----------------
elif page == "Make Prediction":

    st.title("Make a Prediction")
    st.write(
        "Enter research information below. The model will predict "
        "the recorded outcome category."
    )

    with st.form("prediction_form"):
        values = {}
        left, right = st.columns(2)

        with left:
            st.subheader("Numeric Information")

            for feature in numeric_features:
                series = pd.to_numeric(
                    df[feature], errors="coerce"
                ).dropna()

                if series.empty:
                    values[feature] = None
                    continue

                low = float(series.min())
                high = float(series.max())
                default = float(series.median())

                if low == high:
                    values[feature] = low
                    st.number_input(
                        feature,
                        value=low,
                        disabled=True,
                        key=f"number_{feature}",
                    )
                else:
                    values[feature] = st.number_input(
                        feature,
                        min_value=low,
                        max_value=high,
                        value=default,
                        key=f"number_{feature}",
                        help=f"Observed dataset range: {low:g} to {high:g}",
                    )

        with right:
            st.subheader("Study Information")

            for feature in categorical_features:
                options = (
                    df[feature]
                    .dropna()
                    .astype(str)
                    .value_counts()
                    .index.tolist()
                )

                if not options:
                    values[feature] = None
                elif len(options) <= 40:
                    values[feature] = st.selectbox(
                        feature,
                        options,
                        key=f"category_{feature}",
                    )
                else:
                    values[feature] = st.text_input(
                        feature,
                        value=options[0],
                        key=f"text_{feature}",
                    )

        submitted = st.form_submit_button(
            "Predict",
            use_container_width=True,
        )

    if submitted:
        input_df = pd.DataFrame([values], columns=features)

        for feature in numeric_features:
            input_df[feature] = pd.to_numeric(
                input_df[feature], errors="coerce"
            )

        try:
            result = model.predict(input_df)[0]

            st.success(f"Predicted research-effect category: {result}")

            # ---------------- LABELED PREDICTION CHART ----------------
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_df)[0]

                classifier = model.named_steps.get("classifier")
                if classifier is not None:
                    classes = classifier.classes_
                else:
                    classes = saved.get("classes", [])

                if len(classes) == len(probabilities):
                    scores = pd.DataFrame({
                        "Recorded effect category": [
                            str(c) for c in classes
                        ],
                        "Model-estimated score (%)": [
                            round(float(p) * 100, 2)
                            for p in probabilities
                        ],
                    }).sort_values(
                        "Model-estimated score (%)",
                        ascending=False,
                    )

                    show_chart_explanation(
                        "Chart 3: Model Scores for Each Category",
                        "Each bar shows the model's estimated score for "
                        "a possible category. A longer bar means a higher "
                        "model score for this input—not a higher chance "
                        "of wildlife harm."
                    )

                    fig, ax = plt.subplots(figsize=(9, 4.5))
                    bars = ax.barh(
                        scores["Recorded effect category"],
                        scores["Model-estimated score (%)"],
                        color="#43865b",
                    )

                    ax.set_title("Estimated Score by Predicted Category")
                    ax.set_xlabel("Model-estimated score (%)")
                    ax.set_ylabel("Recorded effect category")
                    ax.set_xlim(0, 100)
                    ax.grid(axis="x", linestyle="--", alpha=0.3)
                    ax.invert_yaxis()

                    for bar in bars:
                        width = bar.get_width()
                        ax.text(
                            min(width + 1, 96),
                            bar.get_y() + bar.get_height() / 2,
                            f"{width:.1f}%",
                            va="center",
                            fontsize=9,
                        )

                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    st.caption(
                        "Important: these are model-estimated scores, "
                        "not probabilities that wildlife will be harmed."
                    )

        except Exception as error:
            st.error(f"Prediction failed: {error}")

# ---------------- ABOUT PAGE ----------------
else:
    st.title("About the Project")

    st.write("""
    **Project:** Light Pollution Wildlife Risk Prediction

    **Technologies used**
    - Python
    - Pandas
    - Scikit-learn
    - Random Forest
    - Streamlit
    - Matplotlib

    **Goal:** Explore research data about artificial light at night
    and wildlife-related outcomes.
    """)

    st.warning(
        "A research-effect prediction should not be treated as a "
        "confirmed wildlife-risk assessment without scientific validation."
    )
