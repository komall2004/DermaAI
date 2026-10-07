
import csv
import io
from datetime import datetime

import pandas as pd
import requests
import streamlit as st
from PIL import Image, UnidentifiedImageError


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "http://127.0.0.1:8000"

CLASSES = [
    "acne",
    "blackheades",
    "dark spots",
    "pigmentation",
    "pores",
    "redness",
    "wrinkles"
]

CLASS_LABELS = {
    "blackheades": "Blackheads"
}

st.set_page_config(
    page_title="DermaAI | Skin Condition Analyzer",
    page_icon="🧬",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
.stApp {
    background: #f7fafc;
}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
}

[data-testid="stSidebar"] {
    background: #ffffff;
}

.hero {
    background: linear-gradient(110deg, #e6f7f2, #edf5ff);
    border: 1px solid #dcebe8;
    border-radius: 18px;
    padding: 32px;
    margin-bottom: 18px;
}

.hero h1 {
    color: #12313c;
    margin: 0 0 10px 0;
}

.hero p {
    color: #435563;
    font-size: 1.05rem;
    margin: 0;
}

.note {
    background: #fff8e8;
    border: 1px solid #f1dfaa;
    border-radius: 12px;
    padding: 14px;
    color: #654d17;
    margin-top: 18px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

for key, default in {
    "history": [],
    "latest": None,
    "page": "🏠 Dashboard"
}.items():

    if key not in st.session_state:
        st.session_state[key] = default


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def friendly(name):
    return CLASS_LABELS.get(name, name.title())


def history_csv(records):

    if not records:
        return ""

    output = io.StringIO()

    writer = csv.DictWriter(
        output,
        fieldnames=[
            "time",
            "filename",
            "prediction",
            "confidence_percent",
            "second_prediction",
            "third_prediction"
        ]
    )

    writer.writeheader()
    writer.writerows(records)

    return output.getvalue()


def go_to_analyze():
    # Callback runs before Streamlit recreates the radio widget.
    st.session_state.page = "🔬 Analyze Skin"


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🧬 DermaAI")

    st.caption("EfficientNetB0 Skin Condition Analyzer")

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "🔬 Analyze Skin",
            "📊 Model Performance",
            "ℹ️ About"
        ],
        key="page"
    )

    st.divider()

    st.caption("Model: EfficientNetB0")
    st.caption("Task: 7-class image classification")
    st.caption("For educational use, not medical diagnosis.")


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown("""
    <div class="hero">
        <h1>DermaAI Dashboard</h1>
        <p>
            Analyze skin images, view your latest result,
            and review your prediction history.
        </p>
    </div>
    """, unsafe_allow_html=True)

    count = len(st.session_state.history)

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Images analyzed",
        count
    )

    latest = st.session_state.latest

    col2.metric(
        "Latest prediction",
        friendly(latest["predictions"][0])
        if latest else "—"
    )

    col3.metric(
        "Latest confidence",
        f'{latest["confidence"][0] * 100:.2f}%'
        if latest else "—"
    )

    # Navigation callback fixes the session-state error.
    st.button(
        "🔬 Analyze an image",
        type="primary",
        on_click=go_to_analyze
    )

    st.subheader("Recent analyses")

    if st.session_state.history:

        st.dataframe(
            pd.DataFrame(st.session_state.history[::-1]),
            width="stretch",
            hide_index=True
        )

        st.download_button(
            "⬇️ Download session history (CSV)",
            data=history_csv(st.session_state.history),
            file_name="dermaai_session_history.csv",
            mime="text/csv"
        )

        if st.button("Clear session history"):

            st.session_state.history = []
            st.session_state.latest = None

            st.rerun()

    else:

        st.info(
            "No analyses yet. Upload an image on the "
            "Analyze Skin page to see results here."
        )

    st.caption(
        "History is stored only in this Streamlit session."
    )


# ============================================================
# ANALYZE SKIN
# ============================================================

elif page == "🔬 Analyze Skin":

    st.title("🔬 Analyze Skin")

    st.write(
        "Upload a clear JPG or PNG image. "
        "EfficientNetB0 will return the top three predictions."
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image_bytes = uploaded_file.getvalue()

        try:

            preview = Image.open(
                io.BytesIO(image_bytes)
            ).convert("RGB")

        except (UnidentifiedImageError, OSError):

            st.error(
                "This file could not be opened as an image."
            )

            st.stop()

        left, right = st.columns(2)

        with left:

            st.image(
                preview,
                caption=uploaded_file.name,
                width="stretch"
            )

        with right:

            st.subheader("AI Analysis")

            st.write(
                "The backend resizes the image to "
                "224 × 224 RGB pixels before prediction."
            )

            if st.button(
                "🔍 Analyze image",
                type="primary"
            ):

                try:

                    with st.spinner(
                        "Running EfficientNetB0..."
                    ):

                        response = requests.post(
                            f"{API_URL}/predict",
                            files={
                                "file": (
                                    uploaded_file.name,
                                    image_bytes,
                                    uploaded_file.type
                                    or "image/jpeg"
                                )
                            },
                            timeout=120
                        )

                    response.raise_for_status()

                    result = response.json()

                    predictions = result["predictions"]
                    confidence = result["confidence"]

                    if (
                        len(predictions) < 3
                        or len(confidence) < 3
                    ):

                        raise ValueError(
                            "Missing top-three predictions"
                        )

                    st.session_state.latest = {
                        "predictions": predictions,
                        "confidence": confidence,
                        "filename": uploaded_file.name,
                        "upload_id": (
                            uploaded_file.name,
                            len(image_bytes),
                            hash(image_bytes)
                        )
                    }

                    st.session_state.history.append({
                        "time": datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "filename": uploaded_file.name,
                        "prediction": friendly(
                            predictions[0]
                        ),
                        "confidence_percent": round(
                            float(confidence[0]) * 100,
                            2
                        ),
                        "second_prediction": friendly(
                            predictions[1]
                        ),
                        "third_prediction": friendly(
                            predictions[2]
                        )
                    })

                    st.success(
                        "Analysis complete! Dashboard updated."
                    )

                except requests.exceptions.ConnectionError:

                    st.error(
                        "Cannot reach FastAPI. "
                        "Start your backend first."
                    )

                except requests.exceptions.Timeout:

                    st.error(
                        "The API took too long to respond."
                    )

                except requests.exceptions.HTTPError as exc:

                    st.error(
                        f"API error: {exc}. "
                        f"{response.text[:250]}"
                    )

                except (
                    ValueError,
                    KeyError,
                    TypeError
                ) as exc:

                    st.error(
                        f"Could not read prediction results: {exc}"
                    )

        # ====================================================
        # DISPLAY PREDICTION RESULTS
        # ====================================================

        latest = st.session_state.latest

        current_id = (
            uploaded_file.name,
            len(image_bytes),
            hash(image_bytes)
        )

        if (
            latest
            and latest["upload_id"] == current_id
        ):

            st.divider()

            st.subheader("📊 Prediction Results")

            col1, col2 = st.columns(2)

            col1.metric(
                "Top prediction",
                friendly(latest["predictions"][0])
            )

            col2.metric(
                "Model confidence",
                f'{latest["confidence"][0] * 100:.2f}%'
            )

            st.write("**Top three predictions**")

            for name, score in zip(
                latest["predictions"][:3],
                latest["confidence"][:3]
            ):

                st.write(
                    f"{friendly(name)} — {score * 100:.2f}%"
                )

                st.progress(
                    min(
                        1.0,
                        max(0.0, float(score))
                    )
                )

            st.download_button(
                "⬇️ Download this result (CSV)",
                data=history_csv(
                    st.session_state.history[-1:]
                ),
                file_name="dermaai_prediction.csv",
                mime="text/csv"
            )

    st.markdown("""
    <div class="note">
        ⚠️ Model confidence is not medical certainty.
        This application cannot diagnose skin conditions
        or reliably identify every condition present
        in an image. Consult a dermatologist for
        medical advice.
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "📊 Model Performance":

    st.title("📊 EfficientNetB0 Model Performance")

    st.write(
        "Evaluation on the held-out test dataset. "
        "These results do not guarantee performance "
        "on unfamiliar real-world images."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Test accuracy", "92.85%")
    col2.metric("Test images", "1,803")
    col3.metric("Classes", "7")
    col4.metric("Macro F1", "0.91")

    st.subheader("Per-class evaluation")

    scores = pd.DataFrame({
        "Class": [
            "Acne",
            "Blackheads",
            "Dark spots",
            "Pigmentation",
            "Pores",
            "Redness",
            "Wrinkles"
        ],
        "Precision": [
            0.93, 0.90, 0.90, 0.82,
            0.98, 0.97, 0.95
        ],
        "Recall": [
            0.91, 0.93, 0.92, 0.72,
            1.00, 0.81, 0.99
        ],
        "F1": [
            0.92, 0.91, 0.91, 0.77,
            0.99, 0.88, 0.97
        ],
        "Test images": [
            406, 288, 326, 89,
            277, 89, 328
        ]
    })

    st.dataframe(
        scores,
        width="stretch",
        hide_index=True
    )

    st.subheader("Confusion Matrix")

    cm = [
        [370, 20, 11, 2, 0, 1, 2],
        [12, 267, 4, 3, 2, 0, 0],
        [5, 5, 300, 7, 0, 1, 8],
        [4, 2, 13, 64, 2, 0, 4],
        [0, 0, 0, 0, 277, 0, 0],
        [5, 2, 3, 2, 1, 72, 4],
        [1, 0, 2, 0, 1, 0, 324]
    ]

    matrix = pd.DataFrame(
        cm,
        index=scores["Class"],
        columns=scores["Class"]
    )

    st.dataframe(
        matrix,
        width="stretch"
    )

    st.caption(
        "Rows represent actual labels. "
        "Columns represent predicted labels."
    )

    st.subheader("Inference Pipeline")

    st.write(
        "Image upload → RGB conversion → "
        "224 × 224 resize → EfficientNetB0 → "
        "Softmax → Top-three predictions"
    )

    st.info(
        "This is single-label multiclass classification. "
        "The three displayed predictions are alternatives, "
        "not three simultaneous diagnoses."
    )


# ============================================================
# ABOUT
# ============================================================

elif page == "ℹ️ About":

    st.title("ℹ️ About DermaAI")

    st.write(
        "DermaAI is an educational skin-image "
        "classification project built using "
        "transfer learning with EfficientNetB0."
    )

    st.subheader("Technology Stack")

    st.write(
        "**Frontend:** Streamlit\n\n"
        "**Backend:** FastAPI\n\n"
        "**Deep Learning:** TensorFlow / Keras\n\n"
        "**Model:** EfficientNetB0\n\n"
        "**Image Processing:** Pillow and NumPy"
    )

    st.subheader("Supported Categories")

    st.write(
        ", ".join(
            friendly(category)
            for category in CLASSES
        )
    )

    st.subheader("Limitations")

    st.write(
        "The model assigns one primary category "
        "per image and does not have a normal "
        "or unknown class. It is not a medical "
        "diagnostic system."
    )

    st.subheader("API Status")

    if st.button("Check Backend Connection"):

        try:

            response = requests.get(
                f"{API_URL}/health",
                timeout=5
            )

            response.raise_for_status()

            st.success("Backend is reachable!")

            st.json(response.json())

        except requests.RequestException:

            st.error(
                "Backend is unavailable. "
                "Start FastAPI and try again."
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "DermaAI • Educational and research use only "
    "• Not a medical diagnosis"
)