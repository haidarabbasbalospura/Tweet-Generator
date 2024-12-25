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


# tweet_template = """
# 🎯 **Your Task**: You are a social media manager for a company. Generate exactly {number} unique tweets on the topic provided in {language}. Ensure each tweet is distinct and written in {language}.🚀

# 📝 **Topic**: {topic} 🌟

# ✨ Get creative and make sure the tweets are fun, engaging, and relevant to the audience! 😎💬

# Let's get tweeting! 📱
# """

tweet_template = """
🎯 **Your Task**: Generate exactly {number} tweets about {topic} in {language}. Follow the EXACT format shown in the examples below.

📝 Format Requirements:
1. Each tweet MUST start with a number and an emoji
2. Each tweet MUST include at least one relevant hashtag
4. Each tweet MUST maintain the topic focus

Example tweets about YouTube:
1. 📺 Ready to level up your content game? Join YouTube's creator community today! From editing tips to growth strategies - we've got you covered! #YouTubeCreator
2. 🎬 Transform your ideas into amazing videos! YouTube is your platform to shine and share your passion with the world. #YouTubeCommunity
3. 🌟 Discover endless entertainment on YouTube! Gaming, music, education - your next favorite channel is just one click away! #YouTubeLife

Now, generate {number} tweets following this EXACT format for the topic: {topic}

Remember:
- Start each tweet with a number and emoji
- do not add any other unnecessary text like here is the tweets only tweets
- Include relevant hashtags
- Keep tweets focused on {topic}
- Maintain consistent tone and style
"""

tweet_prompt = PromptTemplate(template=tweet_template, input_variables=["topic", "number"])

# Initialize models
gemini_model = GoogleGenerativeAI(model="gemini-1.0-pro")

# Together AI Models
mistral_model = Together(
    model="mistralai/Mistral-7B-Instruct-v0.3",

)

llama_model = Together(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
)

qwen_model = Together(
    model="Qwen/Qwen2.5-Coder-32B-Instruct",
)

# Create chains for each model
tweet_chains = {
    "Gemini Pro": tweet_prompt | gemini_model,
    "Mixtral 8x7B": tweet_prompt | mistral_model,
    "LLaMA 2 70B": tweet_prompt | llama_model,
    "Qwen 32B": tweet_prompt | qwen_model
}

import streamlit as st

# Define a comprehensive list of languages with English (US) first
languages = [
    "English (US)", "English (UK)",  # English options first
    "Afrikaans", "Amharic", "Bulgarian", "Catalan", "Chinese (Hong Kong)", "Chinese (PRC)", 
    "Chinese (Taiwan)", "Croatian", "Czech", "Danish", "Dutch", "Estonian", "Filipino", 
    "Finnish", "French (Canada)", "French (France)", "German", "Greek", "Hebrew", "Hindi", 
    "Hungarian", "Icelandic", "Indonesian", "Italian", "Japanese", "Korean", "Latvian", 
    "Lithuanian", "Malay", "Norwegian", "Polish", "Portuguese (Brazil)", 
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

# Language selection with default set to English (US)
selected_language = st.selectbox(
    "Select Language",
    options=languages,
    index=0,  # Set default to first option (English US)
    help="Choose the language for the generated tweets"
)

topic = st.text_input("Enter a topic", placeholder="Enter a topic")
number = st.number_input("Number of tweets", min_value=1, max_value=50, value=1, step=1)


# if st.button("Generate tweets"):
#     if topic == "":
#         st.error("Please enter a topic.")
#     else:
#         with st.spinner(f'🐦 Generating your tweets using {selected_model} in {selected_language}...'):
#             raw_tweets = tweet_chains[selected_model].invoke({
#                 "topic": topic,
#                 "number": number,
#                 "language": selected_language
#             })
#             tweets = raw_tweets
#         success_message = st.success("✨ Tweets generated successfully!")
#         time.sleep(1.5)
#         success_message.empty()
#         st.write(tweets)
    
# if st.button("Generate Tweets"):
#     if topic == "":
#         st.error("Please enter a topic.")
#     else:
#         with st.spinner(f'🐦 Generating tweets using {selected_model} in {selected_language}...'):
#             raw_tweets = tweet_chains[selected_model].invoke({
#                 "topic": topic,
#                 "number": number,
#                 "language": selected_language
#             })
#             tweets = raw_tweets.split("\n")  # Assuming tweets are newline-separated

#         success_message = st.success("✨ Tweets generated successfully!")
#         time.sleep(1.5)
#         success_message.empty()

#         # Display each tweet with a copy button
#         for i, tweet in enumerate(tweets):
#             with st.container():
#                 st.code(tweet, language="text")
#                 # st.markdown(
#                 #     f"""
#                 #     <button onclick="navigator.clipboard.writeText(`{tweet}`)" 
#                 #     style="background-color: #4CAF50; color: white; border: none; 
#                 #     padding: 5px 10px; text-align: center; text-decoration: none;
#                 #     display: inline-block; font-size: 12px; margin-top: 5px; cursor: pointer;">
#                 #     📋 Copy
#                 #     </button>
#                 #     """,
#                 #     unsafe_allow_html=True
#                 # )


if st.button("Generate Tweets"):
    if topic == "":
        st.error("Please enter a topic.")
    else:
        with st.spinner(f'🐦 Generating tweets using {selected_model} in {selected_language}...'):
            raw_tweets = tweet_chains[selected_model].invoke({
                "topic": topic,
                "number": number,
                "language": selected_language
            })
            tweets = [tweet.strip() for tweet in raw_tweets.split("\n") if tweet.strip()]

        # Create a success message and store it in a variable
        success_placeholder = st.empty()
        success_placeholder.success("✨ Tweets generated successfully!")
        time.sleep(0.2)
        success_placeholder.empty()  # Clear the success message
        
        # Display each tweet with a copy button
        for i, tweet in enumerate(tweets):
            with st.container():
                st.code(tweet, language="text")