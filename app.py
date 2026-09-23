import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

import qrcode
from io import BytesIO
import urllib.parse
from datetime import datetime

def generate_passport_qr(data_dict):
    query_string = urllib.parse.urlencode(data_dict)
    app_url = "https://banavey-app-etesbszsteozcwmqcy9pjc.streamlit.app"  # your actual app URL
    full_url = f"{app_url}/?{query_string}"

    qr = qrcode.QRCode(box_size=8, border=2)
    qr.add_data(full_url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="#1a1a2e", back_color="#f9d71c")

    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), full_url

def show_passport_view(params):
    st.markdown("""
    <div style="background: linear-gradient(135deg, #2d2d5f, #1a1a2e); padding: 30px; border-radius: 16px; border: 2px solid #f9d71c;">
        <h2 style="margin-top:0;">🍌 BanaVey Digital Passport</h2>
    """, unsafe_allow_html=True)

    st.write(f"**Batch ID:** {params.get('batch_id', 'N/A')}")
    st.write(f"**Detected Stage:** {params.get('stage', 'N/A').upper()}")
    st.write(f"**AI Confidence:** {params.get('confidence', 'N/A')}%")
    st.write(f"**Decision Class:** {params.get('grade', 'N/A')}")
    st.write(f"**Urgency:** {params.get('urgency', 'N/A')}")
    st.write(f"**Recommended Route:** {params.get('action', 'N/A')}")
    st.write(f"**Scan Date:** {params.get('date', 'N/A')}")
    st.markdown("</div>", unsafe_allow_html=True)
    st.caption("This passport was generated from a real BanaVey AI scan.")


# Page setup
st.set_page_config(page_title="BanaVey AI", page_icon="🍌", layout="centered")
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    h1 {
        color: #f9d71c;
        font-weight: 800;
    }
    h2, h3 {
        color: #f9d71c;
    }
    .stButton>button {
        background-color: #f9d71c;
        color: #1a1a2e;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1.5rem;
    }
    div[data-testid="stMetricValue"] {
        color: #f9d71c;
        font-size: 2rem;
    }
    div[data-testid="stMetric"] {
        background-color: rgba(249, 215, 28, 0.08);
        border-radius: 12px;
        padding: 10px;
        border: 1px solid rgba(249, 215, 28, 0.2);
    }
    .stInfo {
        border-left: 4px solid #4CAF50;
        border-radius: 8px;
    }
    div[data-testid="stExpander"] {
        border: 1px solid rgba(249, 215, 28, 0.3);
        border-radius: 10px;
    }
    .stRadio > label {
        font-weight: 600;
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

st.markdown("""
<div style="background: linear-gradient(90deg, #2d2d5f, #1a1a2e); padding: 20px; border-radius: 12px; margin-bottom: 20px; border: 1px solid rgba(249,215,28,0.3);">
    <p style="color: #ccc; margin: 0; font-size: 0.95rem;">🌍 <b>From Banana Image → Post-Harvest Action</b><br>
    AI-powered ripeness detection with routing, value-recovery, and traceability for Jalgaon's banana supply chain.</p>
</div>
""", unsafe_allow_html=True)


mode = st.radio("Choose mode:", ["📷 Single Scan", "📦 Batch Scan", "🎪 Exhibition Demo"], horizontal=True)

query_params = st.query_params
if "batch_id" in query_params:
    show_passport_view(query_params)
    st.stop()


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

        st.markdown("---")
        with st.expander("🔮 What Happens Next? (Scenario Simulation)"):
            st.caption("This shows a typical ripening progression scenario — not a re-analysis of this exact banana over time.")
            stage_progression = ["unripe", "ripe", "overripe", "rotten"]
            if predicted_class in stage_progression:
                current_index = stage_progression.index(predicted_class)
                for days_ahead, label in [(1, "In ~1 day"), (2, "In ~2 days"), (3, "In ~3 days")]:
                    future_index = min(current_index + days_ahead, len(stage_progression) - 1)
                    future_stage = stage_progression[future_index]
                    future_info = recommendation_rules[future_stage]
                    stage_emoji = urgency_colors.get(future_info["urgency"], "")
                    st.write(f"**{label}:** {stage_emoji} Likely stage: **{future_stage.upper()}** → {future_info['action']}")
            else:
                st.write("Scenario simulation isn't available for a low-confidence result — please recheck manually first.")

        st.markdown("---")
        if st.button("📱 Generate Digital Passport"):
            batch_id = f"BV-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            passport_data = {
                "batch_id": batch_id,
                "stage": predicted_class,
                "confidence": f"{confidence:.1f}",
                "grade": rec["grade"],
                "urgency": rec["urgency"],
                "action": rec["action"],
                "date": datetime.now().strftime("%d %b %Y")
            }
            qr_bytes, passport_url = generate_passport_qr(passport_data)
            st.image(qr_bytes, caption="Scan to view this banana's Digital Passport", width=250)
            st.caption(f"Or visit: {passport_url}")


elif mode == "📦 Batch Scan":  # Batch Scan
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



elif mode == "🎪 Exhibition Demo":   # Demo section
    st.subheader("🎪 Exhibition Demo")
    st.write("Choose a prepared sample to instantly see BanaVey's full analysis.")

    sample_choice = st.radio(
        "Choose a sample:",
        ["🟢 Unripe Banana", "🟡 Ripe Banana", "🟠 Overripe Banana", "🔴 Rotten Banana", "📦 Mixed Batch"]
    )

    sample_map = {
        "🟢 Unripe Banana": "samples/unripe.jpg",
        "🟡 Ripe Banana": "samples/ripe.jpg",
        "🟠 Overripe Banana": "samples/overripe.jpg",
        "🔴 Rotten Banana": "samples/rotten.jpg",
    }

    if sample_choice == "📦 Mixed Batch":
        results = []
        for path in sample_map.values():
            img = Image.open(path).convert("RGB")
            predicted_class, confidence = predict(img)
            rec = get_recommendation(predicted_class, confidence)
            results.append({"image": img, "ripeness": predicted_class, "confidence": confidence, **rec})

        st.markdown(f"### Batch Summary — {len(results)} bananas scanned")
        from collections import Counter
        counts = Counter(r["ripeness"] for r in results)
        cols = st.columns(len(counts))
        stage_emoji = {"unripe": "🟢", "ripe": "🟡", "overripe": "🟠", "rotten": "🔴"}
        for col, (k, v) in zip(cols, counts.items()):
            with col:
                st.metric(f"{stage_emoji.get(k, '')} {k.capitalize()}", v)

        st.markdown("### 🚦 Batch Priority Actions")
        for stage in ["rotten", "overripe", "ripe", "unripe"]:
            items = [r for r in results if r["ripeness"] == stage]
            if items:
                st.write(f"**{len(items)} {stage} banana(s) → {recommendation_rules[stage]['action']}**")

        grid_cols = st.columns(4)
        for i, r in enumerate(results):
            with grid_cols[i]:
                st.image(r["image"], width="stretch")
                st.caption(f"{r['ripeness'].upper()} ({r['confidence']:.0f}%)")

    else:
        img_path = sample_map[sample_choice]
        image = Image.open(img_path).convert("RGB")
        st.image(image, caption="Sample photo", width="stretch")

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
