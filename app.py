import json
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page configuration
st.set_page_config(
    page_title="College FAQ Chatbot",
    page_icon="🎓",
    layout="centered"
)

# Custom CSS styling
st.markdown("""
<style>
/* Main background */
.stApp {
    background: linear-gradient(
        135deg,
        #12002f,
        #30105e,
        #542080
    );
    color: white;
}

/* Main heading */
h1 {
    color: #e9d5ff !important;
    text-align: center;
    font-size: 42px !important;
    font-weight: 800;
}

/* Subheadings */
h2, h3 {
    color: #e9d5ff !important;
}

/* Normal text */
p, label, .stCaption {
    color: #f3e8ff !important;
}

/* Input boxes */
.stTextInput input {
    background-color: #ffffff !important;
    color: #24113b !important;
    border: 2px solid #c084fc !important;
    border-radius: 12px !important;
}

/* Dropdown */
.stSelectbox div[data-baseweb="select"] {
    background-color: #ffffff;
    border-radius: 12px;
}

/* Buttons */
.stButton button {
    background: linear-gradient(
        90deg,
        #9333ea,
        #c026d3
    );
    color: white !important;
    border: none;
    border-radius: 12px;
    padding: 10px 22px;
    font-weight: bold;
    transition: 0.3s;
}

/* Button hover effect */
.stButton button:hover {
    background: linear-gradient(
        90deg,
        #c026d3,
        #9333ea
    );
    transform: scale(1.03);
    border: 1px solid #e9d5ff;
}

/* Chat message containers */
.stChatMessage {
    background-color: rgba(255, 255, 255, 0.10);
    border: 1px solid rgba(233, 213, 255, 0.25);
    border-radius: 16px;
    padding: 12px;
}

/* Answer boxes */
.stAlert {
    border-radius: 14px;
}

/* Welcome card */
.welcome-card {
    background: rgba(255, 255, 255, 0.10);
    border: 1px solid rgba(233, 213, 255, 0.30);
    border-radius: 20px;
    padding: 25px;
    text-align: center;
    margin-bottom: 25px;
}

/* Footer */
footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# Load FAQ data
with open("faq_data.json", "r", encoding="utf-8") as file:
    faqs = json.load(file)

questions = [faq["question"] for faq in faqs]

# NLP processing
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

# Page heading
st.markdown(
    '<div class="main-title">🎓 College FAQ Chatbot</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Your smart assistant for college information</div>',
    unsafe_allow_html=True
)

# Welcome card
st.markdown("""
<div class="welcome-card">
    <h3 style="color:#312e81;">Welcome, Student! 👋</h3>
    <p style="color:#475569;">
        Have questions about college admissions, courses, fees,
        or placements? Ask me anything!
    </p>
</div>
""", unsafe_allow_html=True)

# Question input
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Question input
# Suggested questions
st.markdown("### 💡 Suggested Questions")

suggested_questions = [
    "What courses are available?",
    "How can I apply for admission?",
    "What is the fee structure?",
    "Does the college have a library?",
    "Does the college provide placements?"
]

selected_question = st.selectbox(
    "Choose a sample question:",
    ["Select a question..."] + suggested_questions,
    key="selected_question"
)

if selected_question != "Select a question...":
    default_question = selected_question
else:
    default_question = ""

user_question = st.text_input(
    "💬 Type your question here",
    placeholder="Example: How can I apply for admission?",
    key="user_question"
)

if st.button("🔍 Find Answer"):

    if user_question.strip():

        user_vector = vectorizer.transform([user_question])

        similarities = cosine_similarity(
            user_vector, question_vectors
        )[0]

        best_match = similarities.argmax()
        best_score = similarities[best_match]

        if best_score >= 0.20:
            answer = faqs[best_match]["answer"]
        else:
            answer = (
                "Sorry, I couldn't find a suitable answer. "
                "Please try asking another question."
            )

        st.session_state.chat_history.append({
            "question": user_question,
            "answer": answer
        })

    else:
        st.warning("Please type a question first.")

# Display conversation history
st.markdown("## 💬 Chat History")

if st.button("🗑️ Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()

for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(chat["question"])

    with st.chat_message("assistant"):
        st.write(chat["answer"])

if not st.session_state.chat_history:
    st.caption("Your questions and answers will appear here.")

# About Project
st.markdown("---")

with st.expander("📘 About This Project"):
    st.write("""
    ### College FAQ Chatbot

    *Objective:*
    To help students find answers to common
    college-related questions.

    *Technologies Used:*
    - Python
    - Streamlit
    - Natural Language Processing (NLP)
    - TF-IDF Vectorization
    - Cosine Similarity
    - JSON

    *How It Works:*
    1. The chatbot loads college FAQs.
    2. It converts questions into numerical vectors.
    3. It compares the student's question with stored questions.
    4. It displays the most relevant answer.
    """)

# Footer
st.markdown("""
<div class="footer">
    Developed as an NLP Mini Project 💜<br>
    College FAQ Chatbot | Powered by Python
</div>
""", unsafe_allow_html=True)