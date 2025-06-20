import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from fuzzywuzzy import process
import os

# ============================
# 🧠 Load & Train Chatbot Model
# ============================
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

# ========================
# 🤖 Prediction Functions
# ========================
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
st.set_page_config("Chatbot", layout="wide")

st.markdown("""
    <style>
    .main-container {
        background: linear-gradient(to bottom right, #e3f2fd, #f3e5f5);
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    .question-box {
        background-color:#1976D2;
        color:#FFFFFF;
        padding:10px;
        border-radius:10px;
        margin:5px 0;
        font-size:14px;
    }
    .response-box {
        background-color:#2E7D32;
        color:#FFFFFF;
        padding:12px;
        border-radius:15px;
        margin-top:15px;
        font-size:16px;
    }
    .emotion-box {
        color:#FFFFFF;
        padding:12px;
        border-left:5px solid #fff;
        border-radius:8px;
        font-size:16px;
        margin-top:15px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<h2 style='text-align: center; color: #3F51B5;'>🤖 Friendly Chatbot & Sentiment Detector</h2>
<p style='text-align: center; font-size:18px;'>Talk like a friend. I reply & feel your emotion too 💬❤️</p>
""", unsafe_allow_html=True)

# Session state init
if "history" not in st.session_state:
    st.session_state.history = []
if "user_questions" not in st.session_state:
    st.session_state.user_questions = []

# Emotion colors
emotion_color_map = {
    "happy": "#4CAF50",
    "sad": "#E53935",
    "stress": "#FF9800",
    "emotional": "#9C27B0",
    "angry": "#F44336",
    "love": "#EC407A",
    "depression": "#455A64"
}

# 🎧 Spotify playlist embed links
spotify_embed_links = {
    "happy": "https://open.spotify.com/embed/playlist/6MZtZRdQUSaIO6eELQxAcy?utm_source=generator",
    "sad": "https://open.spotify.com/embed/playlist/37i9dQZF1DWVrtsSlLKzro",
    "stress": "https://open.spotify.com/embed/playlist/37i9dQZF1DWUvHZA1zLcjW",
    "depression": "https://open.spotify.com/embed/playlist/37i9dQZF1DX4sWSpwq3LiO",
    "love": "https://open.spotify.com/embed/playlist/37i9dQZF1DWYNSm3Z3MxiM",
    "angry": "https://open.spotify.com/embed/playlist/37i9dQZF1DX1tyCD9QhIWF",
    "emotional": "https://open.spotify.com/embed/playlist/37i9dQZF1DX7gIoKXt0gmx"
}

# Layout
col1, col2 = st.columns([1, 3])

with col1:
    st.markdown("<h4>📜 Your Asked Questions</h4>", unsafe_allow_html=True)

    if st.button("🧹 Clear Chat"):
        st.session_state.history = []
        st.session_state.user_questions = []
        st.rerun()

    for question in st.session_state.user_questions:
        st.markdown(f"<div class='question-box'><b>You:</b> {question}</div>", unsafe_allow_html=True)

with col2:
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("💬 Type your message:", placeholder="Yedhachum Pesu Daa")
        submitted = st.form_submit_button("Send")

    if submitted and user_input:
        bot_reply = get_chat_response(user_input)
        emotion = get_emotion(user_input)

        st.session_state.history.append(("You", user_input))
        st.session_state.history.append(("Bot", bot_reply))
        st.session_state.user_questions.append(user_input)

        st.markdown(f"""
            <div class='emotion-box' style="background-color:{emotion_color_map.get(emotion, '#616161')};">
            🔔 <b>Detected Emotion:</b> {emotion.upper()}
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
            <div class='response-box'>
                <b>Bot:</b> {bot_reply}
            </div>
        """, unsafe_allow_html=True)

        # 🎧 Embed Spotify based on detected emotion
        spotify_url = spotify_embed_links.get(emotion)
        if spotify_url:
            st.markdown(f"""
                <iframe style="border-radius:12px; margin-top:10px;" 
                        src="{spotify_url}" 
                        width="100%" 
                        height="152" 
                        frameborder="0" 
                        allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" 
                        loading="lazy">
                </iframe>
            """, unsafe_allow_html=True)

# =================
# 🔚 Footer
# =================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<center><small style='color:#555;'>Made with ❤️ using Streamlit • Chat + Mood Aware</small></center>", unsafe_allow_html=True)
