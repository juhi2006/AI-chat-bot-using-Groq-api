import os
import fitz
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from textblob import TextBlob
from PIL import Image
import re
import time

load_dotenv()
key = os.getenv("GORQ_API_KEY")
if not key:
    st.error("GORQ_API_KEY is not set. Please ensure it is defined in your .env file.")
    st.stop()
chat_model = ChatGroq(api_key=key)
st.title("ChatGroq PDF Assistant")

if 'logged_in_user' not in st.session_state:
    st.session_state.logged_in_user = None

USER_DATA_FILE = 'users.csv'

user_data = pd.read_csv(USER_DATA_FILE) if os.path.exists(USER_DATA_FILE) else pd.DataFrame(
    columns=["email", "password"])

# Initialize session state for responses
if 'responses' not in st.session_state:
    st.session_state.responses = []

def classify_sentiment(score):
    if score > 0.5:
        return "Happy"
    elif score < -0.5:
        return "Wrong"
    else:
        return "Neutral"

def retry_request(prompt, model, max_retries=5):
    wait_time = 5  # Initial wait time in seconds
    for attempt in range(max_retries):
        try:
            response = model.invoke(prompt)
            return response.content if hasattr(response, 'content') else response
        except Exception as e:
            error_message = str(e)
            # Extract the wait time from the error message
            match = re.search(r'Try again in (\d+\.\d+)s', error_message)
            if match:
                wait_time = float(match.group(1))
                st.error(f"Rate limit exceeded. Waiting for {wait_time} seconds before retrying...")
            else:
                st.error(f"An error occurred: {e}")
                break
            time.sleep(wait_time)
            wait_time *= 2  # Exponential backoff
    return "Unable to generate response."

if st.session_state.logged_in_user is None:
    st.sidebar.header("Options")
    page_option = st.sidebar.selectbox("Choose an option:", ["Sign Up", "Login"])

    if page_option == "Sign Up":
        st.header("Create an Account")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if st.button("Sign Up"):
            with st.spinner("Creating account..."):
                if email in user_data['email'].values:
                    st.error("Email already exists. Please choose a different email.")
                elif email == "":
                    st.error("Email cannot be empty.")
                elif password == "":
                    st.error("Password cannot be empty.")
                elif len(password) < 8 or not re.search("[a-z]", password) or not re.search("[A-Z]", password) or not re.search("[0-9]", password):
                    st.error("Password must be at least 8 characters long and contain a mix of uppercase, lowercase, and digits.")
                else:
                    new_entry = pd.DataFrame({"email": [email], "password": [password]})
                    user_data = pd.concat([user_data, new_entry], ignore_index=True)
                    user_data.to_csv(USER_DATA_FILE, index=False)
                    st.success("Account created successfully! You can now log in.")

    elif page_option == "Login":
        st.header("Log In")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if st.button("Log In"):
            with st.spinner("Logging in..."):
                if email not in user_data['email'].values:
                    st.error("Email not found. Please sign up.")
                else:
                    stored_password = user_data.loc[user_data['email'] == email, 'password'].values
                    if stored_password.size == 0:
                        st.error("Email not found. Please sign up.")
                    elif stored_password[0] != password:
                        st.error("Incorrect password. Please try again.")
                    else:
                        st.session_state.logged_in_user = email
                        st.success("Logged in successfully!")

else:
    st.sidebar.header("Options")
    option = st.sidebar.selectbox("Choose an option:", ["Upload PDF", "Ask Question", "Chat with Bot", "Upload Image"])

    if 'pdf_text' not in st.session_state:
        st.session_state.pdf_text = ""

    if option == "Upload PDF":
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        if uploaded_file is not None:
            with st.spinner("Processing PDF..."):
                pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                pdf_text = ""
                for page_num in range(len(pdf_document)):
                    page = pdf_document.load_page(page_num)
                    pdf_text += page.get_text()
                st.session_state.pdf_text = pdf_text
            st.success("PDF uploaded successfully!")
            st.text_area("PDF Content", st.session_state.pdf_text, height=300)

    elif option == "Ask Question":
        if not st.session_state.pdf_text:
            st.warning("Please upload a PDF first.")
        else:
            st.sidebar.subheader("Additional Options")
            if st.sidebar.button("Summarize PDF"):
                with st.spinner("Summarizing PDF..."):
                    prompt = f"Please summarize the following text:\n\n{st.session_state.pdf_text}"
                    summary = retry_request(prompt, chat_model)
                    st.text_area("PDF Summary", summary, height=300)

            prompt = st.chat_input("What question do you have about the PDF?")
            if prompt:
                st.write(f"User has sent the following prompt: {prompt}")

                blob_prompt = TextBlob(prompt)
                sentiment_score_prompt = blob_prompt.sentiment.polarity
                sentiment_category_prompt = classify_sentiment(sentiment_score_prompt)
                st.write(f"Prompt Sentiment Score: {sentiment_score_prompt:.2f}, Category: {sentiment_category_prompt}")

                # Extract relevant sections of the PDF based on the user's question
                relevant_sections = []
                for paragraph in st.session_state.pdf_text.split("\n"):
                    if any(keyword.lower() in paragraph.lower() for keyword in prompt.split()):
                        relevant_sections.append(paragraph)

                relevant_text = " ".join(relevant_sections)

                # Check if the relevant text is too long and chunk if necessary
                if len(relevant_text) > 2000:
                    chunk_size = 2000
                    chunks = [relevant_text[i:i + chunk_size] for i in range(0, len(relevant_text), chunk_size)]
                    with st.spinner("Generating response..."):
                        responses = [retry_request(
                            f"Based on the following text, answer the question: {prompt}\n\nText: {chunk}", chat_model) for chunk in
                            chunks]
                        full_response = "\n\n".join(responses)
                else:
                    prompt = f"Based on the following text, answer the question: {prompt}\n\nText: {relevant_text}"
                    with st.spinner("Generating response..."):
                        full_response = retry_request(prompt, chat_model)

                # Analyze sentiment of the response
                blob_response = TextBlob(full_response)
                sentiment_score_response = blob_response.sentiment.polarity
                sentiment_category_response = classify_sentiment(sentiment_score_response)
                st.write(f"Response Sentiment Score: {sentiment_score_response:.2f}, Category: {sentiment_category_response}")

                # Store and display the response
                st.session_state.responses.append(full_response)  # Store response
                st.text_area("ChatGroq Response", full_response, height=300)

    elif option == "Chat with Bot":
        st.header("Chat with Bot")
        prompt = st.chat_input("Say something")
        if prompt:
            st.write(f"User has sent the following prompt: {prompt}")

            blob_prompt = TextBlob(prompt)
            sentiment_score_prompt = blob_prompt.sentiment.polarity
            sentiment_category_prompt = classify_sentiment(sentiment_score_prompt)
            st.write(f"Prompt Sentiment Score: {sentiment_score_prompt:.2f}, Category: {sentiment_category_prompt}")

            with st.spinner("Generating response..."):
                response_text = retry_request(prompt, chat_model)

            blob_response = TextBlob(response_text)
            sentiment_score_response = blob_response.sentiment.polarity
            sentiment_category_response = classify_sentiment(sentiment_score_response)
            st.write(f"Response Sentiment Score: {sentiment_score_response:.2f}, Category: {sentiment_category_response}")

            st.session_state.responses.append(response_text)  # Store response
            st.text_area("ChatGroq Response", response_text, height=300)

    elif option == "Upload Image":
        uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            with st.spinner("Processing image..."):
                image = Image.open(uploaded_image)
                st.image(image, caption='Uploaded Image.', use_column_width=True)

                gray_image = image.convert("L")
                st.image(gray_image, caption='Grayscale Image.', use_column_width=True)

                color_stats = image.getcolors(maxcolors=256)
                st.write("Image Color Statistics:", color_stats)

    # Display previous responses
    st.sidebar.header("Previous Responses")
    if st.session_state.responses:
        for idx, resp in enumerate(st.session_state.responses):
            st.sidebar.text_area(f"Response {idx + 1}", resp, height=100)
