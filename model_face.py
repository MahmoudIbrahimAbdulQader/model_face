import streamlit as st
from transformers import pipeline

# ---------------------------------------------------------------------------
# 1. Page Config & Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Toxicity & Roast Meter | Hugging Face + Streamlit",
    page_icon="🔥",
    layout="centered",
)

st.markdown(
    """
    <style>
        .stApp {
            background-color: #0B1120;
            color: #F1F5F9;
        }
        .hero-box {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.95));
            border: 1px solid rgba(56, 189, 248, 0.25);
            border-radius: 18px;
            padding: 24px 28px;
            margin-bottom: 20px;
        }
        .badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 700;
            background: rgba(56, 189, 248, 0.12);
            color: #38BDF8;
            border: 1px solid rgba(56, 189, 248, 0.3);
            margin-bottom: 10px;
        }
        .verdict-card {
            background: rgba(30, 41, 59, 0.75);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 20px 24px;
            margin: 14px 0 20px 0;
        }
        .stTextArea textarea {
            background-color: #131B2E !important;
            color: #F8FAFC !important;
            border: 1px solid #26354D !important;
            border-radius: 14px !important;
        }
        div.stButton > button {
            border-radius: 10px !important;
            font-weight: 600 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 2. Load Hugging Face Model (Cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading unitary/toxic-bert model...")
def load_toxicity_classifier():
    return pipeline(
        "text-classification",
        model="unitary/toxic-bert",
        top_k=None,
    )


classifier = load_toxicity_classifier()

# ---------------------------------------------------------------------------
# 3. Header & Quick Examples
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-box">
        <div class="badge">Hugging Face • unitary/toxic-bert</div>
        <h1 style="margin: 0 0 8px 0; font-size: 30px; color: #FFFFFF;">🔥 Toxicity &amp; Roast Meter</h1>
        <p style="margin: 0; color: #94A3B8; font-size: 15px; line-height: 1.6;">
            Enter a sentence or comment to inspect how toxic, insulting, or chill it looks according to the model.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "sample_text" not in st.session_state:
    st.session_state.sample_text = "Your code actually works on the first try, impressive!"

st.caption("Quick examples to try:")
c1, c2, c3 = st.columns(3)
with c1:
    if st.button("😊 Friendly / Chill", use_container_width=True):
        st.session_state.sample_text = "Thanks for the help, that made the pipeline much cleaner."
with c2:
    if st.button("😏 Sarcastic / Spicy", use_container_width=True):
        st.session_state.sample_text = "Bold of you to push to main without testing anything."
with c3:
    if st.button("⚡ Harsh / Toxic", use_container_width=True):
        st.session_state.sample_text = "You are completely clueless and your solution is garbage."

text_input = st.text_area(
    "Enter text to analyze:",
    value=st.session_state.sample_text,
    height=115,
    placeholder="Type a sentence or comment to analyze...",
)

analyze_clicked = st.button("🚀 Analyze Text", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# 4. Run Prediction & Display Results
# ---------------------------------------------------------------------------
LABEL_NAMES = {
    "toxic": "Toxic",
    "severe_toxic": "Severe Toxic",
    "obscene": "Obscene",
    "threat": "Threat",
    "insult": "Insult",
    "identity_hate": "Identity Hate",
}

if analyze_clicked:
    cleaned = text_input.strip()
    if not cleaned:
        st.warning("Please enter a sentence before clicking Analyze Text.")
    else:
        with st.spinner("Analyzing..."):
            raw_scores = classifier(cleaned)[0]

        scores_map = {item["label"]: float(item["score"]) for item in raw_scores}
        top_score = max(scores_map.values()) if scores_map else 0.0

        if top_score >= 0.70:
            verdict_title = "🚨 High Heat / Toxic Tone"
            verdict_desc = "The text shows a strong toxic or confrontational tone according to the model."
        elif top_score >= 0.35:
            verdict_title = "⚠️ Borderline / Spicy"
            verdict_desc = "The text is borderline — sarcasm, banter, or sharp phrasing may trigger partial scores."
        else:
            verdict_title = "✅ Chill / Safe Tone"
            verdict_desc = "The text appears mostly harmless and non-toxic."

        st.markdown(
            f"""
            <div class="verdict-card">
                <h3 style="margin: 0 0 6px 0; color: #38BDF8;">{verdict_title}</h3>
                <p style="margin: 0; color: #CBD5E1; font-size: 14px;">{verdict_desc}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("Category Breakdown")
        for label_key, display_label in LABEL_NAMES.items():
            prob = scores_map.get(label_key, 0.0)
            pct = round(prob * 100, 1)
            st.markdown(f"**{display_label}** — `{pct}%`")
            st.progress(min(max(prob, 0.0), 1.0))