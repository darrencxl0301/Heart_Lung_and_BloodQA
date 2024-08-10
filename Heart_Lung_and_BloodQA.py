import streamlit as st
import pandas as pd
import numpy as np
import ast
import openai
from openai.embeddings_utils import cosine_similarity

# Initialize OpenAI API key
openai.api_key =  st.secrets["mykey"]

# Load the dataset
df = pd.read_csv("qa_dataset_with_embeddings.csv")

# Convert the string embeddings back to lists (if needed)
df['Question_Embedding'] = df['Question_Embedding'].apply(ast.literal_eval)

# Function to get embedding for a user's question
def get_embedding(text, model="text-embedding-ada-002"):
    response = openai.Embedding.create(input=[text], model=model)
    return response['data'][0]['embedding']

# Function to find the best answer
def find_best_answer(user_question):
    # Get embedding for the user's question
    user_question_embedding = get_embedding(user_question)

    # Calculate cosine similarities for all questions in the dataset
    df['Similarity'] = df['Question_Embedding'].apply(lambda x: cosine_similarity(x, user_question_embedding))

    # Find the most similar question and get its corresponding answer
    most_similar_index = df['Similarity'].idxmax()
    max_similarity = df['Similarity'].max()

    # Set a similarity threshold to determine if a question is relevant enough
    similarity_threshold = 0.6  # Adjust this value as needed

    if max_similarity >= similarity_threshold:
        best_answer = df.loc[most_similar_index, 'Answer']
        return best_answer, max_similarity
    else:
        return "I apologize, but I don't have information on that topic yet. Could you please ask other questions?", max_similarity

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
        # Find the best answer based on the user's question
        answer, similarity = find_best_answer(st.session_state.user_question)
        st.session_state.answer = answer
        st.session_state.similarity = similarity
        st.session_state.rating_feedback = ""  # Reset rating feedback when a new question is asked

        st.write(f"**Answer:** {answer}")
        st.write(f"**Similarity Score:** {similarity:.2f}")

        # Rating system with initial unselected state
        rating = st.radio("Was this answer helpful?", ('Select an option', 'Yes', 'No'), index=0)
        if rating == 'Yes':
            st.session_state.rating_feedback = "Thank you for your feedback!"
        elif rating == 'No':
            st.session_state.rating_feedback = "Sorry to hear that. We'll strive to improve."
    else:
        st.session_state.answer = None

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
for question in df['Question'].head(5):  # Display top 5 questions as an example
    st.write(f"- {question}")
