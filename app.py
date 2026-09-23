import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

import qrcode
from io import BytesIO
import urllib.parse
from datetime import datetime
import csv
from collections import Counter


def generate_passport_qr(data_dict):
    query_string = urllib.parse.urlencode(data_dict)
    app_url = "https://banavey-app-etesbszsteozcwmqcy9pjc.streamlit.app"
    full_url = f"{app_url}/?{query_string}"

    qr = qrcode.QRCode(box_size=8, border=2)
    qr.add_data(full_url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="#1a1a2e", back_color="#f9d71c")

    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), full_url


def load_image(file_or_path):
    """Open an image and fix phone-camera EXIF rotation."""
    img = Image.open(file_or_path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def show_passport_view(params):
    st.markdown("""
    <div class="bv-card" style="border: 2px solid #f9d71c;">
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

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("← Back to Scanner"):
        st.query_params.clear()
        st.rerun()


# ---------------- Page setup ----------------
st.set_page_config(page_title="BanaVey AI", page_icon="🍌", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at top left, #22224a 0%, #1a1a2e 45%, #16213e 100%);
    }

    h1 {
        color: #f9d71c;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    h2, h3 {
        color: #f9d71c;
        font-weight: 700;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #ffe066, #f9d71c);
        color: #1a1a2e;
        font-weight: 700;
        border-radius: 10px;
        border: none;
        padding: 0.6rem 1.6rem;
        box-shadow: 0 4px 14px rgba(249, 215, 28, 0.25);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(249, 215, 28, 0.4);
        color: #1a1a2e;
    }

    /* Metrics */
    div[data-testid="stMetricValue"] {
        color: #f9d71c;
        font-size: 2rem;
        font-weight: 800;
    }
    div[data-testid="stMetric"] {
        background: rgba(249, 215, 28, 0.08);
        border-radius: 14px;
        padding: 12px;
        border: 1px solid rgba(249, 215, 28, 0.25);
    }

    .stInfo {
        border-left: 4px solid #4CAF50;
        border-radius: 10px;
    }

    div[data-testid="stExpander"] {
        border: 1px solid rgba(249, 215, 28, 0.3);
        border-radius: 12px;
        background: rgba(255,255,255,0.02);
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        font-weight: 600;
        border-radius: 10px 10px 0 0;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #f9d71c !important;
        border-bottom: 3px solid #f9d71c !important;
    }

    /* Uploaded / sample image */
    div[data-testid="stImage"] img {
        border-radius: 14px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.35);
    }

    /* Custom card */
    .bv-card {
        background: linear-gradient(135deg, #2d2d5f, #1a1a2e);
        padding: 26px;
        border-radius: 16px;
        border: 1px solid rgba(249,215,28,0.25);
        margin-bottom: 20px;
    }
    .bv-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .bv-footer {
        text-align: center;
        color: #777;
        font-size: 0.8rem;
        padding: 30px 0 10px 0;
    }

    /* Confidence bar */
    .bv-conf-track {
        background: rgba(255,255,255,0.08);
        border-radius: 999px;
        height: 10px;
        width: 100%;
        overflow: hidden;
        margin: 6px 0 2px 0;
    }
    .bv-conf-fill {
        background: linear-gradient(90deg, #f9d71c, #ffe066);
        height: 100%;
        border-radius: 999px;
    }
</style>
""", unsafe_allow_html=True)


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

urgency_colors = {
    "LOW": "🟢", "HIGH": "🟡", "VERY HIGH": "🟠",
    "IMMEDIATE": "🔴", "MANUAL CHECK": "⚪"
}

urgency_hex = {
    "LOW": "#4CAF50", "HIGH": "#f9d71c", "VERY HIGH": "#ff9800",
    "IMMEDIATE": "#e53935", "MANUAL CHECK": "#9e9e9e"
}


def get_recommendation(predicted_class, confidence):
    if confidence < 70:
        return {"grade": "RECHECK", "status": "Low Confidence", "urgency": "MANUAL CHECK",
                "action": "Manual inspection recommended", "reason": "AI confidence is low; result should be verified."}
    return recommendation_rules[predicted_class]


def predict(img):
    img_r = img.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img_r)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array, verbose=0)
    predicted_index = np.argmax(prediction[0])
    predicted_class = class_names[predicted_index]
    confidence = float(prediction[0][predicted_index]) * 100
    return predicted_class, confidence


def render_confidence_bar(confidence):
    st.markdown(f"""
    <div class="bv-conf-track">
        <div class="bv-conf-fill" style="width:{confidence:.1f}%;"></div>
    </div>
    <p style="color:#aaa; font-size:0.85rem; margin-top:0;">AI Confidence: <b style="color:#f9d71c;">{confidence:.1f}%</b></p>
    """, unsafe_allow_html=True)


def render_result_card(predicted_class, confidence, rec):
    emoji = urgency_colors.get(rec["urgency"], "")
    hexcol = urgency_hex.get(rec["urgency"], "#f9d71c")

    st.subheader(f"Ripeness: {predicted_class.upper()}")
    render_confidence_bar(confidence)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Decision Class", rec["grade"])
    with col2:
        st.metric("Urgency", f"{emoji} {rec['urgency']}")

    st.markdown(f"""
    <div class="bv-card">
        <p style="margin:4px 0;"><b>Status:</b> {rec['status']}</p>
        <p style="margin:4px 0;"><b>Recommended Action:</b> {rec['action']}</p>
        <p style="margin:4px 0;"><b>Reason:</b> {rec['reason']}</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("🧠 BanaVey Decision Insight")
    st.info(rec.get("impact", "Impact assessment pending manual verification."))
    st.caption("Note: Insight statements are based on ripeness stage classification, not measured shelf-life data.")


# ---------------- Landing screen gating ----------------
if "entered_app" not in st.session_state:
    st.session_state.entered_app = False

query_params = st.query_params
if "batch_id" in query_params:
    show_passport_view(query_params)
    st.stop()


if not st.session_state.entered_app:
    st.markdown("""
    <div style="text-align: center; padding: 50px 20px 20px 20px;">
        <div style="font-size: 4.5rem;">🍌</div>
        <h1 style="color: #f9d71c; margin-bottom: 0; font-size: 2.6rem;">BanaVey AI</h1>
        <p style="font-size: 1.2rem; color: #ddd; margin-top: 8px;">From Banana Image → Post-Harvest Action</p>
        <p style="color: #999; max-width: 520px; margin: 20px auto; line-height: 1.6;">
            An AI-powered decision support system helping Jalgaon's banana supply chain
            reduce waste through smart ripeness detection, routing, and traceability.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🚀 Enter BanaVey", width="stretch"):
            st.session_state.entered_app = True
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    feature_cards = [
        ("🤖", "AI Vision", "Identifies banana ripeness from a photo"),
        ("🚦", "Smart Routing", "Suggests the right next step"),
        ("📱", "Digital Passport", "QR-based traceable record"),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3], feature_cards):
        with col:
            st.markdown(f"""
            <div class="bv-card" style="text-align:center; padding: 20px 14px;">
                <div style="font-size:1.8rem;">{icon}</div>
                <div style="color:#f9d71c; font-weight:700; margin-top:6px;">{title}</div>
                <div style="color:#999; font-size:0.85rem; margin-top:4px;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="bv-footer">Built for Mind Spring Exhibition · 5 October 2026</div>', unsafe_allow_html=True)

else:
    # ---------------- Main app ----------------
    st.markdown("""
    <div style="text-align:center; margin-bottom: 4px;">
        <span style="font-size:2.4rem;">🍌</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<h1 style='text-align:center; margin-top:0;'>BanaVey AI</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center; color:#ccc;'>Upload a banana photo to check its ripeness and get a routing recommendation.</p>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align:center; color:#888; font-size:0.9rem;'>🌍 Built for Jalgaon's banana supply chain — reducing post-harvest waste through AI</p>",
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="bv-card">
        <p style="color: #ccc; margin: 0; font-size: 0.95rem;">🌍 <b>From Banana Image → Post-Harvest Action</b><br>
        AI-powered ripeness detection with routing, value-recovery, and traceability for Jalgaon's banana supply chain.</p>
    </div>
    """, unsafe_allow_html=True)

    tab_single, tab_batch, tab_demo, tab_impact, tab_learn = st.tabs(
        ["📷 Single Scan", "📦 Batch Scan", "🎪 Exhibition Demo", "💰 Impact Calculator", "📚 Learn"]
    )

    # ---------- Single Scan ----------
    with tab_single:
        input_mode = st.radio(
            "How do you want to provide the photo?",
            ["📁 Upload Photo", "📷 Use Camera"],
            horizontal=True,
            key="single_input_mode"
        )

        if input_mode == "📁 Upload Photo":
            uploaded_file = st.file_uploader("Choose a banana photo", type=["jpg", "jpeg", "png"], key="single")
        else:
            uploaded_file = st.camera_input("Point your camera at a banana and capture", key="single_camera")

        if uploaded_file is not None:
            image = load_image(uploaded_file)
            st.image(image, caption="Uploaded photo", width="stretch")

            with st.spinner("Analyzing..."):
                predicted_class, confidence = predict(image)
                rec = get_recommendation(predicted_class, confidence)

            render_result_card(predicted_class, confidence, rec)

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

    # ---------- Batch Scan ----------
    with tab_batch:
        st.subheader("📦 Batch Scanner")
        st.write("Upload several banana photos at once to see the whole batch's condition and priority actions.")

        uploaded_files = st.file_uploader(
            "Choose banana photos",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="batch"
        )

        if uploaded_files:
            results = []
            with st.spinner(f"Analyzing {len(uploaded_files)} bananas..."):
                for f in uploaded_files:
                    img = load_image(f)
                    predicted_class, confidence = predict(img)
                    rec = get_recommendation(predicted_class, confidence)
                    results.append({
                        "filename": f.name, "image": img,
                        "ripeness": predicted_class, "confidence": confidence,
                        **rec
                    })

            st.markdown(f"### Batch Summary — {len(results)} bananas scanned")

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
            for stage in priority_order:
                items = [r for r in results if r["ripeness"] == stage and r["grade"] != "RECHECK"]
                if items:
                    info = recommendation_rules[stage]
                    st.write(f"**{len(items)} {stage} banana(s) → {info['action']}**")

            recheck_items = [r for r in results if r["grade"] == "RECHECK"]
            if recheck_items:
                st.write(f"**{len(recheck_items)} banana(s) need manual recheck** (low AI confidence)")

            # CSV export
            import io as _io
            text_buf = _io.StringIO()
            writer = csv.writer(text_buf)
            writer.writerow(["Filename", "Ripeness", "Confidence (%)", "Decision Class", "Urgency", "Recommended Action"])
            for r in results:
                writer.writerow([r["filename"], r["ripeness"], f"{r['confidence']:.1f}", r["grade"], r["urgency"], r["action"]])
            st.download_button(
                "⬇️ Download Batch Report (CSV)",
                data=text_buf.getvalue(),
                file_name=f"banavey_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

            with st.expander("See individual results"):
                grid_cols = st.columns(4)
                for i, r in enumerate(results):
                    with grid_cols[i % 4]:
                        st.image(r["image"], width="stretch")
                        st.caption(f"{r['ripeness'].upper()} ({r['confidence']:.0f}%)")

    # ---------- Exhibition Demo ----------
    with tab_demo:
        st.subheader("🎪 Exhibition Demo")
        st.write("Choose a prepared sample to instantly see BanaVey's full analysis.")

        sample_choice = st.radio(
            "Choose a sample:",
            ["🟢 Unripe Banana", "🟡 Ripe Banana", "🟠 Overripe Banana", "🔴 Rotten Banana", "📦 Mixed Batch"],
            horizontal=True
        )

        sample_map = {
            "🟢 Unripe Banana": "samples/unripe.jpg",
            "🟡 Ripe Banana": "samples/ripe.jpg",
            "🟠 Overripe Banana": "samples/overripe.jpg",
            "🔴 Rotten Banana": "samples/rotten.jpg",
        }

        try:
            if sample_choice == "📦 Mixed Batch":
                results = []
                for path in sample_map.values():
                    img = load_image(path)
                    predicted_class, confidence = predict(img)
                    rec = get_recommendation(predicted_class, confidence)
                    results.append({"image": img, "ripeness": predicted_class, "confidence": confidence, **rec})

                st.markdown(f"### Batch Summary — {len(results)} bananas scanned")
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
                image = load_image(img_path)
                st.image(image, caption="Sample photo", width="stretch")

                predicted_class, confidence = predict(image)
                rec = get_recommendation(predicted_class, confidence)
                render_result_card(predicted_class, confidence, rec)

        except FileNotFoundError:
            st.error(
                "⚠️ Demo sample images weren't found on this deployment. "
                "Make sure the `samples/` folder (unripe.jpg, ripe.jpg, overripe.jpg, rotten.jpg) "
                "is uploaded alongside app.py."
            )

    # ---------- Impact Calculator ----------
    with tab_impact:
        st.subheader("💰 Impact Calculator")
        st.write(
            "Estimate the value BanaVey-style routing could recover for a banana operation — "
            "a mandi, a trader, or a cooperative. Adjust the numbers to match a real or hypothetical scale."
        )
        st.caption(
            "⚠️ This is an illustrative estimate built from the numbers you enter below — "
            "not a measured or published statistic. Edit any field to match your own scenario."
        )

        with st.form("impact_form"):
            col_a, col_b = st.columns(2)
            with col_a:
                daily_volume_kg = st.number_input(
                    "Bananas handled per day (kg)", min_value=0.0, value=1000.0, step=50.0
                )
                price_per_kg = st.number_input(
                    "Average market price (₹ / kg)", min_value=0.0, value=15.0, step=1.0
                )
            with col_b:
                current_waste_pct = st.slider(
                    "Estimated waste WITHOUT AI-assisted routing (%)", 0, 60, 20
                )
                waste_reduction_pct = st.slider(
                    "Expected waste reduction WITH BanaVey (percentage points)", 0, current_waste_pct, 8
                )
            submitted = st.form_submit_button("📊 Calculate Impact")

        if submitted or "impact_calculated" in st.session_state:
            st.session_state.impact_calculated = True

            new_waste_pct = max(current_waste_pct - waste_reduction_pct, 0)
            waste_before_kg = daily_volume_kg * (current_waste_pct / 100)
            waste_after_kg = daily_volume_kg * (new_waste_pct / 100)
            kg_saved_per_day = max(waste_before_kg - waste_after_kg, 0)
            value_saved_per_day = kg_saved_per_day * price_per_kg
            value_saved_per_year = value_saved_per_day * 365

            st.markdown("### 📈 Estimated Results")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Waste Reduced / Day", f"{kg_saved_per_day:,.0f} kg")
            with c2:
                st.metric("Value Recovered / Day", f"₹{value_saved_per_day:,.0f}")
            with c3:
                st.metric("Value Recovered / Year", f"₹{value_saved_per_year:,.0f}")

            st.markdown(f"""
            <div class="bv-card">
                <p style="margin:4px 0;">Without AI-assisted routing, an estimated <b>{current_waste_pct}%</b>
                of a {daily_volume_kg:,.0f} kg/day operation goes to waste — about
                <b>{waste_before_kg:,.0f} kg/day</b>.</p>
                <p style="margin:4px 0;">With BanaVey-style ripeness detection and routing, waste could drop to
                roughly <b>{new_waste_pct}%</b>, recovering an estimated
                <b>{kg_saved_per_day:,.0f} kg/day (₹{value_saved_per_day:,.0f}/day)</b> that would otherwise
                have been lost.</p>
            </div>
            """, unsafe_allow_html=True)

            st.caption(
                "Note: These figures are calculated directly from the inputs above using a simple assumption "
                "(waste % × volume × price). They illustrate the scale of impact possible, not a verified field measurement."
            )

    # ---------- Learn ----------
    with tab_learn:
        st.subheader("📚 Learn About BanaVey")

        st.markdown("### 🍌 Why Bananas?")
        st.write("Bananas are a **climacteric fruit** — meaning they continue ripening after harvest, unlike some other fruits. This makes timing and handling decisions critical: a banana that's fine today may need a completely different action tomorrow. Jalgaon produces roughly two-thirds of Maharashtra's bananas, making this a locally significant problem, not just a technical exercise.")

        st.markdown("---")
        st.markdown("### 🤖 What Does BanaVey Do?")
        st.markdown("""
        <div class="bv-card" style="text-align: center; font-size: 1.1rem; line-height: 2.2;">
        📷 <b>Photo of a banana</b><br>↓<br>
        🤖 <b>AI Classification</b> (ripeness stage)<br>↓<br>
        🏷️ <b>Decision Class</b> (A/B/C/D)<br>↓<br>
        🚦 <b>Recommended Route</b> (transport / sell / process / discard)<br>↓<br>
        📱 <b>QR Traceability</b> (digital passport)
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🍌 The Four Stages")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("🟢 **Unripe** — Firm, green. Best suited for long-distance transport since it ripens on the way.")
            st.markdown("🟡 **Ripe** — Peak quality. Should reach local/regional markets quickly.")
        with col2:
            st.markdown("🟠 **Overripe** — Declining fresh-market appeal, but still recoverable through processing (chips, jam).")
            st.markdown("🔴 **Rotten** — No fresh-market or processing value; safely routed to waste management.")

        st.markdown("---")
        st.caption("BanaVey was built to reduce post-harvest banana waste in Jalgaon's supply chain using AI-based decision support.")

    st.markdown('<div class="bv-footer">🍌 BanaVey AI · Mind Spring Exhibition · 5 October 2026</div>', unsafe_allow_html=True)
