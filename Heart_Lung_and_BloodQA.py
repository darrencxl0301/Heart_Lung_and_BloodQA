import streamlit as st
import pandas as pd
import numpy as np
import ast
import openai
from openai.embeddings_utils import cosine_similarity

# Initialize OpenAI API key
openai.api_key =  st.secrets["mykey"]

# Load the dataset with caching
@st.cache_data
def load_data():
    df = pd.read_csv("qa_dataset_with_embeddings.csv")
    df['Question_Embedding'] = df['Question_Embedding'].apply(ast.literal_eval)
    return df

# Function to get embedding for a user's question (cache this if it's expensive)
@st.cache_data
def get_embedding(text, model="text-embedding-ada-002"):
    response = openai.Embedding.create(input=[text], model=model)
    return response['data'][0]['embedding']

# Function to find the best answer (can be cached based on input)
@st.cache_data
def find_best_answer(user_question, df):
    user_question_embedding = get_embedding(user_question)
    df['Similarity'] = df['Question_Embedding'].apply(lambda x: cosine_similarity(x, user_question_embedding))
    most_similar_index = df['Similarity'].idxmax()
    max_similarity = df['Similarity'].max()

    similarity_threshold = 0.6
    if max_similarity >= similarity_threshold:
        best_answer = df.loc[most_similar_index, 'Answer']
        return best_answer, max_similarity
    else:
        return "I apologize, but I don't have information on that topic yet. Could you please ask other questions?", max_similarity

# Load data once using caching
df = load_data()

# Streamlit interface
st.title("Heart, Lung, and Blood Health QA")

# Manage session state for input field and other states
if 'user_question' not in st.session_state:
    st.session_state.user_question = ""

if 'rating_feedback' not in st.session_state:
    st.session_state.rating_feedback = ""

# Text input for user's question
st.session_state.user_question = st.text_input("Ask your health-related question:", value=st.session_state.user_question)

# Button to trigger the answer search
if st.button("Find Answer"):
    if st.session_state.user_question:
        answer, similarity = find_best_answer(st.session_state.user_question, df)
        st.session_state.answer = answer
        st.session_state.similarity = similarity
        st.session_state.rating_feedback = ""

# Display the answer and similarity score only if the user has asked a question
if 'answer' in st.session_state:
    st.markdown(f"**Answer:** {st.session_state.answer}")
    st.markdown(f"**Similarity Score:** {st.session_state.similarity:.2f}")

    # Rating system with initial unselected state
    rating = st.radio("Was this answer helpful?", ('Select an option', 'Yes', 'No'), index=0)
    if rating == 'Yes':
        st.session_state.rating_feedback = "Thank you for your feedback!"
    elif rating == 'No':
        st.session_state.rating_feedback = "Sorry to hear that. We'll strive to improve."

# Display the feedback without causing a rerun
if st.session_state.rating_feedback:
    st.markdown(f"**{st.session_state.rating_feedback}**")

# Clear button to reset input and session state
clear_button_key = "clear_button"
clear_button = st.button("Clear", key=clear_button_key)
if clear_button:
    st.session_state.user_question = ""
    st.session_state.answer = None
    st.session_state.rating_feedback = ""

# Optional: Add a section for common FAQs
st.subheader("Frequently Asked Questions")
st.write("Here are some common questions users ask:")
for question in df['Question'].head(5):
    st.write(f"- {question}")
