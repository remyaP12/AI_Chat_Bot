import streamlit as st
from chatbot import get_response

st.title("🤖 AI-Powered Chatbot")
st.write("Built with Python, NLP & TF-IDF")

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history
for sender, msg in st.session_state.messages:
    if sender == "You":
        st.chat_message("user").write(msg)
    else:
        st.chat_message("assistant").write(msg)

# User input
user_input = st.chat_input("Type your message here...")

if user_input:
    response = get_response(user_input)
    st.session_state.messages.append(("You", user_input))
    st.session_state.messages.append(("Bot", response))
    st.rerun()