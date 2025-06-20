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
st.set_page_config("Fun Chatbot ", layout="wide")
st.markdown("<h2 style='text-align: center;'>🤖 Friendly Chatbot + Sentiment Detector</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Talk like a friend. I reply & feel your emotion too 💬❤️</p>", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

if "user_questions" not in st.session_state:
    st.session_state.user_questions = []

# ===================
# 🔄 Layout with Sidebar Chat
# ===================
col1, col2 = st.columns([1, 3])

with col1:
    st.markdown("<h4>📜 Your Asked Questions</h4>", unsafe_allow_html=True)
    for msg in st.session_state.user_questions:
        st.markdown(f"""
            <div style="
                background-color:#263238;
                color:#FFFFFF;
                padding:10px;
                border-radius:8px;
                margin:6px 0;
                font-size:13px;
                font-family:Segoe UI;
            ">
            👉 {msg}
            </div>
        """, unsafe_allow_html=True)

with col2:
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("💬 Type your message:", placeholder="Ex: enna panra da...")
        submitted = st.form_submit_button("Send")

    if submitted and user_input:
        bot_reply = get_chat_response(user_input)
        emotion = get_emotion(user_input)

        st.session_state.history.append(("You", user_input))
        st.session_state.history.append(("Bot", bot_reply))
        st.session_state.user_questions.append(user_input)

        emotion_color_map = {
            "happy": "#3949AB",
            "sad": "#D32F2F",
            "stress": "#F57C00",
            "emotional": "#7B1FA2",
        }

        st.markdown(f"""
            <div style="
                background-color:{emotion_color_map.get(emotion, '#616161')};
                color:#FFFFFF;
                padding:12px;
                border-left:5px solid #fff;
                border-radius:8px;
                font-size:16px;
                margin-top:15px;
            ">
            🔔 <b>Detected Emotion:</b> {emotion.upper()}
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
            <div style="
                background-color:#43A047;
                color:#FFFFFF;
                padding:12px;
                border-radius:15px;
                margin-top:15px;
                font-family:'Segoe UI',sans-serif;
                font-size:16px;
            ">
            <b>Bot:</b> {bot_reply}
            </div>
        """, unsafe_allow_html=True)

# =============
# 🔚 Footer
# =============
st.markdown("---")
st.markdown("<center><small>Made with ❤️ using Streamlit • Chat & Mood Aware</small></center>", unsafe_allow_html=True)
