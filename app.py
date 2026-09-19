import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page setup
st.set_page_config(page_title="BanaVey AI", page_icon="🍌", layout="centered")

# Load the model once, cached so it doesn't reload on every interaction
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('banavey_model.keras')

model = load_model()

class_names = ['overripe', 'ripe', 'rotten', 'unripe']

recommendation_rules = {
    "unripe": {
        "grade": "A", "status": "Early Stage", "urgency": "LOW",
        "action": "Long-distance transport / controlled distribution",
        "reason": "Firm and unripe; generally more suitable for transport before full ripening.",
        "impact": "Low waste risk right now — routing early helps this batch reach full value before spoilage becomes a concern."
    },
    "ripe": {
        "grade": "B", "status": "Market Ready", "urgency": "HIGH",
        "action": "Prioritize regional or local retail",
        "reason": "Already ripe, so it should reach the market relatively quickly.",
        "impact": "Peak market value right now — quick routing to local retail captures full value before quality declines."
    },
    "overripe": {
        "grade": "C", "status": "Processing Priority", "urgency": "VERY HIGH",
        "action": "Prioritize processing / value addition",
        "reason": "Fresh-market suitability is declining; processing helps recover value.",
        "impact": "Without action, this batch risks becoming a total loss. Redirecting to processing (chips, jam) recovers meaningful value instead of discarding it."
    },
    "rotten": {
        "grade": "D", "status": "Spoiled", "urgency": "IMMEDIATE",
        "action": "Reject from food-market route; use appropriate waste management",
        "reason": "Severe visible spoilage makes it unsuitable for normal fresh sale.",
        "impact": "No recoverable market value remains. Early, correct identification prevents spoiled stock from contaminating or being mixed with sellable batches."
    }
}



def get_recommendation(predicted_class, confidence):
    if confidence < 70:  # raised from 60
        return {"grade": "RECHECK", "status": "Low Confidence", "urgency": "MANUAL CHECK",
                "action": "Manual inspection recommended", "reason": "AI confidence is low; result should be verified."}
    return recommendation_rules[predicted_class]


def predict(img):
    img = img.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array, verbose=0)
    predicted_index = np.argmax(prediction[0])
    predicted_class = class_names[predicted_index]
    confidence = float(prediction[0][predicted_index]) * 100
    return predicted_class, confidence

# ---- UI ----
st.title("🍌 BanaVey AI")
st.write("Upload a banana photo to check its ripeness and get a routing recommendation.")

uploaded_file = st.file_uploader("Choose a banana photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded photo", use_container_width=True)

    with st.spinner("Analyzing..."):
        predicted_class, confidence = predict(image)
        rec = get_recommendation(predicted_class, confidence)

    st.subheader(f"Ripeness: {predicted_class.upper()}")
    st.write(f"**AI Confidence:** {confidence:.1f}%")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Grade", rec["grade"])
    with col2:
        st.metric("Urgency", rec["urgency"])

    st.write(f"**Status:** {rec['status']}")
    st.write(f"**Recommended Action:** {rec['action']}")
    st.write(f"**Reason:** {rec['reason']}")
    st.markdown("---")
    st.subheader("🌍 Estimated Impact")
    st.info(rec.get("impact", "No impact data available."))
    st.caption("Note: Impact statements are illustrative estimates based on ripeness stage, not measured data.")
