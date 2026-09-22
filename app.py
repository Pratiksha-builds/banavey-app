import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page setup
st.set_page_config(page_title="BanaVey AI", page_icon="🍌", layout="centered")
st.markdown("""
<style>
    .stApp {
        background-color: #1a1a2e;
    }
    h1 {
        color: #f9d71c;
    }
    .stButton>button {
        background-color: #f9d71c;
        color: #1a1a2e;
        font-weight: bold;
    }
    div[data-testid="stMetricValue"] {
        color: #f9d71c;
    }
    .stInfo {
        border-left: 4px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)



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
st.caption("🌍 Built for Jalgaon's banana supply chain — reducing post-harvest waste through AI")

mode = st.radio("Choose mode:", ["📷 Single Scan", "📦 Batch Scan"], horizontal=True)

urgency_colors = {
    "LOW": "🟢", "HIGH": "🟡", "VERY HIGH": "🟠",
    "IMMEDIATE": "🔴", "MANUAL CHECK": "⚪"
}

if mode == "📷 Single Scan":
    uploaded_file = st.file_uploader("Choose a banana photo", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded photo", width="stretch")

        with st.spinner("Analyzing..."):
            predicted_class, confidence = predict(image)
            rec = get_recommendation(predicted_class, confidence)

        st.subheader(f"Ripeness: {predicted_class.upper()}")
        st.write(f"**AI Confidence:** {confidence:.1f}%")

        emoji = urgency_colors.get(rec["urgency"], "")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Decision Class", rec["grade"])
        with col2:
            st.metric("Urgency", f"{emoji} {rec['urgency']}")

        st.write(f"**Status:** {rec['status']}")
        st.write(f"**Recommended Action:** {rec['action']}")
        st.write(f"**Reason:** {rec['reason']}")
        st.markdown("---")
        st.subheader("🧠 BanaVey Decision Insight")
        st.info(rec.get("impact", "Impact assessment pending manual verification."))
        st.caption("Note: Insight statements are based on ripeness stage classification, not measured shelf-life data.")

else:  # Batch Scan
    st.subheader("📦 Batch Scanner")
    st.write("Upload several banana photos at once to see the whole batch's condition and priority actions.")

    uploaded_files = st.file_uploader(
        "Choose banana photos",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        results = []
        with st.spinner(f"Analyzing {len(uploaded_files)} bananas..."):
            for f in uploaded_files:
                img = Image.open(f).convert("RGB")
                predicted_class, confidence = predict(img)
                rec = get_recommendation(predicted_class, confidence)
                results.append({
                    "filename": f.name, "image": img,
                    "ripeness": predicted_class, "confidence": confidence,
                    **rec
                })

        st.markdown(f"### Batch Summary — {len(results)} bananas scanned")

        from collections import Counter
        counts = Counter()
        for r in results:
            if r["grade"] == "RECHECK":
                counts["Needs Recheck"] += 1
            else:
                counts[r["ripeness"]] += 1

        stage_emoji = {"unripe": "🟢", "ripe": "🟡", "overripe": "🟠", "rotten": "🔴", "Needs Recheck": "⚪"}
        cols = st.columns(len(counts))
        for col, (k, v) in zip(cols, counts.items()):
            with col:
                st.metric(f"{stage_emoji.get(k, '')} {k.capitalize()}", v)

        st.markdown("### 🚦 Batch Priority Actions")
        priority_order = ["rotten", "overripe", "ripe", "unripe"]
        any_action = False
        for stage in priority_order:
            items = [r for r in results if r["ripeness"] == stage and r["grade"] != "RECHECK"]
            if items:
                info = recommendation_rules[stage]
                st.write(f"**{len(items)} {stage} banana(s) → {info['action']}**")
                any_action = True

        recheck_items = [r for r in results if r["grade"] == "RECHECK"]
        if recheck_items:
            st.write(f"**{len(recheck_items)} banana(s) need manual recheck** (low AI confidence)")

        with st.expander("See individual results"):
            grid_cols = st.columns(4)
            for i, r in enumerate(results):
                with grid_cols[i % 4]:
                    st.image(r["image"], width="stretch")
                    st.caption(f"{r['ripeness'].upper()} ({r['confidence']:.0f}%)")
