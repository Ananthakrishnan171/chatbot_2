import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from fuzzywuzzy import process

# ===== Load Friendly Chatbot Dataset =====
chat_df = pd.read_csv("Ananth.csv")
chat_X = chat_df['input']
chat_y = chat_df['chatbot']

chat_vectorizer = TfidfVectorizer()
chat_X_vec = chat_vectorizer.fit_transform(chat_X)

chat_model = LogisticRegression()
chat_model.fit(chat_X_vec, chat_y)

chat_dict = dict(zip(chat_df['input'].str.lower(), chat_df['chatbot']))

# ===== Load Sentiment Dataset =====
emotion_df = pd.read_csv("emotion_chat_dataset.csv")
emo_X = emotion_df['input']
emo_y = emotion_df['emotion']

emo_vectorizer = TfidfVectorizer()
emo_X_vec = emo_vectorizer.fit_transform(emo_X)

emo_model = LogisticRegression()
emo_model.fit(emo_X_vec, emo_y)

emo_dict = dict(zip(emotion_df['input'].str.lower(), emotion_df['emotion']))

# ===== Chatbot Prediction Function =====
def get_chat_response(user_input):
    user_vec = chat_vectorizer.transform([user_input])
    pred = chat_model.predict(user_vec)[0]

    matches = process.extract(user_input.lower(), chat_dict.keys(), limit=1)
    if matches and matches[0][1] >= 70:
        return chat_dict[matches[0][0]]
    else:
        return pred

# ===== Emotion Detection Function =====
def get_emotion(user_input):
    user_vec = emo_vectorizer.transform([user_input])
    pred = emo_model.predict(user_vec)[0]

    matches = process.extract(user_input.lower(), emo_dict.keys(), limit=1)
    if matches and matches[0][1] >= 70:
        return emo_dict[matches[0][0]]
    else:
        return pred

# ===== UI Setup =====
st.set_page_config("Friendly Chatbot", layout="centered")
st.markdown("<h2 style='text-align: center;'>🤖 Friendly chatbot kuda Sentiment Chatbot</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Talk to your virtual friend — I'll also feel your mood 😄😭😤</p>", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

# ===== Input Form =====
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("💬 Type your message:", placeholder="Ex: hi da, enna panra...")
    submitted = st.form_submit_button("Send")

# ===== Response + Emotion Alert =====
if submitted and user_input:
    bot_reply = get_chat_response(user_input)
    emotion = get_emotion(user_input)

    st.session_state.history.append(("You", user_input))
    st.session_state.history.append(("Bot", bot_reply))

    # ===== Alert Box =====
    color_map = {
        "happy": "#D4EDDA",
        "sad": "#F8D7DA",
        "stress": "#FFF3CD",
        "emotional": "#D1ECF1"
    }
    st.markdown(
        f"""
        <div style="
            background-color:{color_map.get(emotion,'#E2E3E5')};
            padding:12px;
            border-left:5px solid #999;
            border-radius:8px;
            font-size:16px;
            margin-bottom:15px;">
        🔔 Detected Emotion: <b>{emotion.upper()}</b>
        </div>
        """, unsafe_allow_html=True
    )

# ===== Chat History =====
for speaker, message in st.session_state.history:
    bg = "#DCF8C6" if speaker == "You" else "#F1F0F0"
    align = "5% 30%" if speaker == "You" else "30% 5%"
    st.markdown(
        f"""
        <div style="
            background-color:{bg};
            padding:12px;
            border-radius:15px;
            margin:10px {align};
            font-family:'Segoe UI',sans-serif;
            font-size:16px;">
            <b>{speaker}:</b> {message}
        </div>
        """, unsafe_allow_html=True
    )

# ===== Footer =====
st.markdown("---")
st.markdown("<center><small>Created with ❤️ using Streamlit •  Chat & Mood Aware</small></center>", unsafe_allow_html=True)
