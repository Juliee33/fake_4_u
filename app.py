"""
app.py

Streamlit web application for the Fake News Detection System.

Author: Nimco
Final Year Project
"""

# ---------------------------------------------------------
# Imports
# ---------------------------------------------------------

import streamlit as st
import os
import sys
import time
import csv
import re
import pandas as pd
import random
import uuid
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from predict import predict_news
from predict_distilbert import predict_fake_news


# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)


# ---------------------------------------------------------
# Anonymous User ID (for research logging)
# ---------------------------------------------------------

if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())[:8]


# ---------------------------------------------------------
# Dataset Loader (cached)
# ---------------------------------------------------------

@st.cache_data
def load_dataset():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(BASE_DIR, "data", "processed", "test_data.csv")
    return pd.read_csv(path)

dataset = load_dataset()


# ---------------------------------------------------------
# Build Question Bank
# ---------------------------------------------------------

def build_question_bank():

    questions = []

    for _, row in dataset.iterrows():

        label = "Authentic News" if row["label"] == 1 else "Disinformation"

        questions.append({
            "text": row["text"],
            "answer": label
        })

    return questions


# ---------------------------------------------------------
# Load CSS
# ---------------------------------------------------------

def load_css():

    css_path = os.path.join("styles","design.css")

    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()


# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------

page = st.sidebar.selectbox(
    "Navigate Protocol",
    ["Forensic Scan","Spot the Fake Game"]
)


# ---------------------------------------------------------
# Normalize Model Labels
# ---------------------------------------------------------

def normalize_prediction(pred):

    if "real" in pred.lower():
        return "Authentic News"
    else:
        return "Disinformation"


# ---------------------------------------------------------
# Hint Generator
# ---------------------------------------------------------

def generate_hint(text):

    t = text.lower()

    if any(word in t for word in ["shocking","exclusive","breaking","scandal"]):
        return "This article uses sensational wording."

    if text.count("!") > 1:
        return "Multiple exclamation marks often signal emotional manipulation."

    if "according to" in t or "reported by" in t:
        return "The article references sources, which is typical in credible journalism."

    if text.isupper():
        return "ALL CAPS text can indicate emotional persuasion."

    return "Look carefully at the tone. Does it sound objective or opinionated?"


# ---------------------------------------------------------
# Boot Sequence
# ---------------------------------------------------------

def boot_sequence():

    if "boot_complete" not in st.session_state:

        boot = st.empty()

        boot.markdown("### INITIALIZING TRUTH ENGINE...")
        time.sleep(0.6)

        boot.markdown("### DECRYPTING METADATA...")
        time.sleep(0.6)

        boot.markdown("### ACCESSING FORENSIC DATABASE...")
        time.sleep(0.6)

        boot.markdown("### LOADING DETECTION MODEL...")
        time.sleep(0.6)

        boot.markdown("### SYSTEM READY")

        st.session_state.boot_complete = True
        boot.empty()


# ---------------------------------------------------------
# Suspicious Word Scanner
# ---------------------------------------------------------

def highlight_suspicious_words(text):

    suspicious_words = [
        "secret","conspiracy","shocking","breaking",
        "urgent","ballistic","leaked","exposed",
        "exclusive","scandal"
    ]

    highlighted = text

    for word in suspicious_words:

        pattern = re.compile(rf"\b{word}\b", re.IGNORECASE)

        highlighted = pattern.sub(
            f"<span style='background-color:#ff4b4b;color:white;padding:2px 4px;border-radius:4px'>{word}</span>",
            highlighted
        )

    return highlighted


# ---------------------------------------------------------
# Forensic Scan Page
# ---------------------------------------------------------

def render_forensic_scan():

    boot_sequence()

    st.title("📰 DECEPTION DETECTION PROTOCOL v2.0")

    st.write("""
    This system classifies news articles as **Real** or **Fake**
    using classical machine learning and transformer models.
    """)

    st.divider()

    model_choice = st.selectbox(
        "Choose AI Model:",
        ["Linear SVM","DistilBERT"]
    )

    user_input = st.text_area(
        "INPUT NEWS TEXT:",
        height=200
    )

    if st.button("EXECUTE FORENSIC SCAN"):

        if user_input.strip() == "":
            st.warning("Please enter some text.")
            return

        highlighted = highlight_suspicious_words(user_input)

        st.markdown(highlighted, unsafe_allow_html=True)

        if model_choice == "Linear SVM":
            prediction,confidence = predict_news(user_input)
        else:
            prediction,confidence = predict_fake_news(user_input)

        st.divider()

        if prediction == "Real News":
            st.success("✔ AUTHENTIC NEWS SIGNAL DETECTED")
        else:
            st.error("⚠ DISINFORMATION DETECTED")

      
        
        confidence =max(0.0, min(confidence, 1.0))
        st.progress(confidence)
        st.caption(f"Model confidence:{confidence:.2f}")
        
        
                


# ---------------------------------------------------------
# Spot the Fake Game
# ---------------------------------------------------------

def render_game_page():

    st.title("🎮 Human vs AI: Spot the Fake")

    st.caption("Anonymous research data is collected for academic analysis only.")

    if "game_started" not in st.session_state:

        st.session_state.questions = random.sample(build_question_bank(),3)
        st.session_state.index = 0
        st.session_state.human_score = 0
        st.session_state.svm_score = 0
        st.session_state.bert_score = 0
        st.session_state.answered = False
        st.session_state.game_finished = False
        st.session_state.game_started = True

    q = st.session_state.questions[st.session_state.index]

    st.markdown(f"### Question {st.session_state.index+1} of 3")

    snippet = q["text"][:350]
    if len(q["text"]) > 350:
        snippet += "..."

    st.info(snippet)

    if st.button("Hint"):
        st.info(generate_hint(q["text"]))

    guess = st.radio(
        "What is your verdict?",
        ["Authentic News","Disinformation"],
        index=None,
        key=f"guess_{st.session_state.index}"
    )

    if not st.session_state.answered:

        if st.button("Submit Answer"):

            if guess is None:
                st.warning("Select an answer first.")
                return

            if guess == q["answer"]:
                st.success("Correct!")
                st.session_state.human_score += 1
            else:
                st.error(f"Incorrect. Correct answer: {q['answer']}")

            svm_pred,_ = predict_news(q["text"])
            bert_pred,_ = predict_fake_news(q["text"])

            svm_pred = normalize_prediction(svm_pred)
            bert_pred = normalize_prediction(bert_pred)

            st.subheader("AI Comparison")

            col1,col2 = st.columns(2)

            with col1:
                st.write("Linear SVM")
                st.write(svm_pred)

            with col2:
                st.write("DistilBERT")
                st.write(bert_pred)

            if svm_pred == q["answer"]:
                st.session_state.svm_score += 1

            if bert_pred == q["answer"]:
                st.session_state.bert_score += 1

            st.session_state.answered = True

    if st.session_state.answered and st.session_state.index < 2:

        if st.button("Next Question"):
            st.session_state.index += 1
            st.session_state.answered = False
            st.rerun()

    if st.session_state.index == 2 and st.session_state.answered:
        st.session_state.game_finished = True

    if st.session_state.game_finished:

        st.subheader("🏁 Final Results")

        human = st.session_state.human_score
        svm = st.session_state.svm_score
        bert = st.session_state.bert_score

        st.write(f"Human Accuracy: {human} / 3")
        st.write(f"SVM Accuracy: {svm} / 3")
        st.write(f"DistilBERT Accuracy: {bert} / 3")

        chart_data = pd.DataFrame({
            "Model":["Human","SVM","DistilBERT"],
            "Score":[human,svm,bert]
        })

        st.bar_chart(chart_data.set_index("Model"))

        st.balloons()

        st.subheader("Research Feedback")

        difficulty = st.slider("How difficult was it to determine whether the articles were real or fake",1,10,5)
        reason = st.text_area("What clues or reasoning helped you to decide your answers?")

        if st.button("Submit Response"):

            os.makedirs("results", exist_ok=True)
            results_path = os.path.join("results","game_results.csv")
            file_exists = os.path.isfile(results_path)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            stories_seen = [
                q["text"][:50].replace(",", " ").replace(" 's", "'s").replace(" n't", "n't")
                for q in st.session_state.questions
            ]

            stories_string = " | ".join(stories_seen)

            with open(results_path,"a",newline="",encoding="utf-8") as f:

                writer = csv.writer(f)

                if not file_exists:
                    writer.writerow([
                        "user_id",
                        "timestamp",
                        "human_score",
                        "svm_score",
                        "bert_score",
                        "difficulty",
                        "reason",
                        "articles_preview"
                    ])

                writer.writerow([
                    st.session_state.user_id,
                    timestamp,
                    human,
                    svm,
                    bert,
                    difficulty,
                    reason,
                    stories_string
                ])

            st.success("Response recorded! Data saved for analysis.")

        if st.button("Restart Game"):

            for key in list(st.session_state.keys()):
                del st.session_state[key]

            st.rerun()


# ---------------------------------------------------------
# Page Routing
# ---------------------------------------------------------

if page == "Forensic Scan":
    render_forensic_scan()

elif page == "Spot the Fake Game":

    if "consent_given" not in st.session_state:

        consent = st.checkbox("I agree to participate in this anonymous research study.")

        if consent:
            st.session_state.consent_given = True
            st.rerun()

    else:
        render_game_page()
