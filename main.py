# # Move API key configuration after streamlit import
# 
# # os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]


from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Together

import os
import time
import re

# API Keys


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

tweet_template = """
🎯 **Your Task**: You are a social media manager for a company. Generate exactly {number} unique tweets on the topic provided in {language}. Ensure each tweet is distinct and written in {language}.🚀

📝 **Topic**: {topic} 🌟

✨ Get creative and make sure the tweets are fun, engaging, and relevant to the audience! 😎💬

Let's get tweeting! 📱
"""

tweet_prompt = PromptTemplate(template=tweet_template, input_variables=["topic", "number"])

# Initialize models
gemini_model = GoogleGenerativeAI(model="gemini-1.0-pro")

# Together AI Models
mistral_model = Together(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    max_tokens=2048
)

llama_model = Together(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    max_tokens=2048
)

qwen_model = Together(
    model="Qwen/Qwen2.5-Coder-32B-Instruct",
    max_tokens=2048
)

# Create chains for each model
tweet_chains = {
    "Gemini Pro": tweet_prompt | gemini_model,
    "Mixtral 8x7B": tweet_prompt | mistral_model,
    "LLaMA 2 70B": tweet_prompt | llama_model,
    "Qwen 32B": tweet_prompt | qwen_model
}

import streamlit as st

# Define a comprehensive list of languages
languages = [
    "Afrikaans", "Amharic", "Bulgarian", "Catalan", "Chinese (Hong Kong)", "Chinese (PRC)", 
    "Chinese (Taiwan)", "Croatian", "Czech", "Danish", "Dutch", "English (UK)", "English (US)", 
    "Estonian", "Filipino", "Finnish", "French (Canada)", "French (France)", "German", "Greek", 
    "Hebrew", "Hindi", "Hungarian", "Icelandic", "Indonesian", "Italian", "Japanese", "Korean", 
    "Latvian", "Lithuanian", "Malay", "Norwegian", "Polish", "Portuguese (Brazil)", 
    "Portuguese (Portugal)", "Romanian", "Russian", "Serbian", "Slovak", "Slovenian", 
    "Spanish (Latin America)", "Spanish (Spain)", "Swahili", "Swedish", "Thai", "Turkish", 
    "Ukrainian", "Vietnamese", "Zulu"
]

st.title("Tweet Generator")
st.subheader("Generate tweets using Generative AI")

# Model selection
selected_model = st.selectbox(
    "Select AI Model",
    options=list(tweet_chains.keys()),
    help="Choose the AI model you want to use for generating tweets"
)

# Language selection
selected_language = st.selectbox(
    "Select Language",
    options=languages,
    help="Choose the language for the generated tweets"
)

topic = st.text_input("Enter a topic", placeholder="Enter a topic")
number = st.number_input("Number of tweets", min_value=1, max_value=50, value=1, step=1)

def clean_redundant_text(text):
    # Remove common prefixes like "Tweet 1:", "1.", etc.
    cleaned = re.sub(r'^(?:\d+\.|Tweet \d+:|\*\*Tweet \d+:\*\*|\d+\))\s*', '', text, flags=re.MULTILINE)
    # Remove empty lines and strip whitespace
    cleaned = '\n'.join(line.strip() for line in cleaned.splitlines() if line.strip())
    return cleaned

if st.button("Generate tweets"):
    if topic == "":
        st.error("Please enter a topic.")
    else:
        with st.spinner(f'🐦 Generating your tweets using {selected_model} in {selected_language}...'):
            raw_tweets = tweet_chains[selected_model].invoke({
                "topic": topic,
                "number": number,
                "language": selected_language
            })
            tweets = clean_redundant_text(raw_tweets)
        success_message = st.success("✨ Tweets generated successfully!")
        time.sleep(1.5)
        success_message.empty()
        st.write(tweets)

