import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from fuzzywuzzy import process
import os

# ==============================
# 🧠 Load & Train Chatbot Model
# ==============================
if os.path.exists("Ananth.csv"):
    chat_df = pd.read_csv("Ananth.csv")
    chat_X = chat_df['input']
    chat_y = chat_df['chatbot']

    chat_vectorizer = TfidfVectorizer()
    chat_X_vec = chat_vectorizer.fit_transform(chat_X)

    chat_model = LogisticRegression()
    chat_model.fit(chat_X_vec, chat_y)

    chat_dict = dict(zip(chat_df['input'].str.lower(), chat_df['chatbot']))
else:
    st.error("❌ 'Ananth.csv' file not found. Please upload the chatbot dataset.")
    st.stop()

# ===============================
# 🧠 Load & Train Sentiment Model
# ===============================
if os.path.exists("friendly_emotion_chatbot.csv"):
    emotion_df = pd.read_csv("friendly_emotion_chatbot.csv")
    emo_X = emotion_df['input']
    emo_y = emotion_df['emotion']

    emo_vectorizer = TfidfVectorizer()
    emo_X_vec = emo_vectorizer.fit_transform(emo_X)

    emo_model = LogisticRegression()
    emo_model.fit(emo_X_vec, emo_y)

    emo_dict = dict(zip(emotion_df['input'].str.lower(), emotion_df['emotion']))
else:
    st.error("❌ 'friendly_emotion_chatbot.csv' file not found. Please upload the emotion dataset.")
    st.stop()

# =========================
# 🤖 Prediction Functions
# =========================
def get_chat_response(user_input):
    user_vec = chat_vectorizer.transform([user_input])
    pred = chat_model.predict(user_vec)[0]

    match = process.extractOne(user_input.lower(), chat_dict.keys())
    if match and match[1] >= 70:
        return chat_dict[match[0]]
    else:
        return pred

def get_emotion(user_input):
    user_vec = emo_vectorizer.transform([user_input])
    pred = emo_model.predict(user_vec)[0]

    match = process.extractOne(user_input.lower(), emo_dict.keys())
    if match and match[1] >= 70:
        return emo_dict[match[0]]
    else:
        return pred

# ===================
# 🎨 Page UI Layout
# ===================
st.set_page_config("Tanglish Chatbot with Mood", layout="wide")
st.markdown("<h2 style='text-align: center;'>🤖 Friendly Chatbot + Mood Detector</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Talk like a friend. I reply & feel your emotion too 💬❤️</p>", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

# ===================
# 🔄 Layout with Sidebar Chat
# ===================
col1, col2 = st.columns([1, 3])

with col1:
    st.markdown("<h4>🕘 Chat History</h4>", unsafe_allow_html=True)
    chat_history_box = ""
    for speaker, message in st.session_state.history:
        bubble_color = "#1E88E5" if speaker == "You" else "#43A047"
        chat_history_box += f"""
            <div style='
                ba
