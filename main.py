# Import streamlit first
import streamlit as st
from langchain_google_genai import GoogleGenerativeAI
from langchain import LLMChain, PromptTemplate
import os
import time

# Move API key configuration after streamlit import
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]


tweet_template = """
🎯 **Your Task**: You are a social media manager for a company. Your goal is to craft {number} catchy tweets based on the given topic. 🚀

📝 **Topic**: {topic} 🌟

✨ Get creative and make sure the tweets are fun, engaging, and relevant to the audience! 😎💬

Let's get tweeting! 📱
"""


tweet_prompt = PromptTemplate(template=tweet_template, input_variables=["topic", "number"])

gemini_model = GoogleGenerativeAI(model="gemini-1.0-pro")


tweet_chain = tweet_prompt | gemini_model

# Frontend Code

st.title("Tweet Generator")


st.subheader("Generate tweets using Generative AI")

topic = st.text_input("Enter a topic" , placeholder="Enter a topic")



number = st.number_input("Number of tweets", min_value=1, max_value=50, value=1, step=1)

if st.button("Generate tweets"):
    if topic == "":
        st.error("Please enter a topic.")  # Display an error if no input is provided
    else:
        with st.spinner('🐦 Generating your tweets...'):  # Shows a spinner with custom message
            tweets = tweet_chain.invoke({"topic": topic, "number": number})
        success_message = st.success("✨ Tweets generated successfully!")  # Show success after generation
        time.sleep(1.5)  # Wait for 1.5 seconds
        success_message.empty()  # Remove the success message
        st.write(tweets)
