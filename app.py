
import streamlit as st
from PIL import Image
import numpy as np

from predict import BoneAgePredictor


st.set_page_config(
    page_title="BoneAge AI",
    page_icon="🦴",
    layout="centered"
)

st.title("🦴 BoneAge AI")
st.subheader("Pediatric Bone Age Prediction")

st.write(
    "Upload a pediatric hand X-ray to estimate bone age "
    "using a deep learning model."
)

st.warning(
    "Experimental research project. "
    "Predictions are not intended for clinical diagnosis."
)


@st.cache_resource
def load_predictor():
    return BoneAgePredictor()


uploaded_file = st.file_uploader(
    "Upload Hand X-ray",
    type=["png", "jpg", "jpeg"]
)

sex_option = st.selectbox(
    "Select sex",
    ["Female", "Male"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("L")
    image_array = np.array(image)

    st.image(
        image,
        caption="Uploaded Hand X-ray",
        use_container_width=True
    )

    if st.button("Predict Bone Age", type="primary"):

        with st.spinner("Analyzing X-ray..."):

            try:
                predictor = load_predictor()

                sex = 1 if sex_option == "Male" else 0

                predicted_age = predictor.predict(
                    image_array,
                    sex
                )

                years = int(predicted_age // 12)
                months = int(round(predicted_age % 12))

                if months == 12:
                    years += 1
                    months = 0

                st.success("Prediction completed!")

                st.metric(
                    "Predicted Bone Age",
                    f"{predicted_age:.2f} months"
                )

                st.write(
                    f"**Equivalent age:** {years} years "
                    f"and {months} months"
                )

            except Exception as e:
                st.error(f"Prediction failed: {e}")
