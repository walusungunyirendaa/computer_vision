import streamlit as st
from PIL import Image

from model_utils import (
    load_trained_model, preprocess, predict,
    make_gradcam_heatmap, overlay_heatmap, CLASS_NAMES,
)

LOW_CONFIDENCE_THRESHOLD = 0.5

st.set_page_config(page_title="Diabetic Retinopathy Grading", layout="centered")
st.title("Diabetic Retinopathy Grading") 

st.markdown(
    "**This is a student coursework prototype, not a medical device.** "
    "It has not been clinically validated. Do not use it to make, support, "
    "or delay any real medical decision."
)

with st.expander("About this app"):
    st.markdown(
        """
        **What this is:** grades a fundus photograph on the 5-point ICDR
        diabetic retinopathy severity scale (No DR, Mild, Moderate, Severe,
        Proliferative DR).
        **What this isn't:** a diagnostic tool.

        **Run locally:**
```bash
        pip install -r requirements.txt
        streamlit run app.py
```
        Requires `retinopathy_model.keras` in this folder.
        """
    )


@st.cache_resource
def get_model():
    return load_trained_model()


uploaded_file = st.file_uploader("Upload a fundus photograph", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    model = get_model()
    input_array = preprocess(image)
    predicted_class, confidence, probabilities = predict(model, input_array)

    st.subheader("Prediction")
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        st.warning(
            f"{predicted_class} ({confidence:.0%} confidence) — the model is "
            "not confident about this image. Treat this result as unreliable."
        )
    else:
        st.success(f"{predicted_class} ({confidence:.0%} confidence)")

    st.write(
        f"The model predicts this fundus photo shows "
        f"{'no visible signs of' if predicted_class == 'No DR' else predicted_class.lower() + ' signs of'} "
        "diabetic retinopathy."
    )

    if st.checkbox("Show what the model focused on (Grad-CAM)"):
        predicted_idx = CLASS_NAMES.index(predicted_class)
        heatmap = make_gradcam_heatmap(model, input_array, predicted_idx)
        overlay = overlay_heatmap(image, heatmap)
        st.image(overlay, caption=f"Model focus for predicted class: {predicted_class}",
                  use_container_width=True)