import logging
import asyncio
import aiohttp
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)

# Function for input validation
def validate_input(text):
    if not text or not isinstance(text, str):
        raise ValueError("Input text must be a non-empty string.")

# Async function for fetching sentiment analysis
async def fetch_sentiment(session, text):
    try:
        async with session.post('http://sentiment-api/parse', json={'text': text}) as response:
            return await response.json()
    except Exception as e:
        logging.error(f"Error fetching sentiment: {e}")
        return None

# Function for batch processing and sentiment analysis
async def analyze_sentiments(texts):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_sentiment(session, text) for text in texts]
        return await asyncio.gather(*tasks)

# Daily aggregation function
def aggregate_daily_data(data):
    return data.groupby(data['date']).agg({'sentiment': 'mean', 'count': 'sum'})

# Streamlit UI
st.title('Sentiment Analysis Dashboard')
text_input = st.text_input('Enter your text for analysis')

if st.button('Analyze'):
    try:
        validate_input(text_input)
        st.info(f'Analyzing sentiment for: {text_input}')
        analysis_result = asyncio.run(analyze_sentiments([text_input]))
        st.write(analysis_result)
    except ValueError as ve:
        st.error(f'Validation error: {ve}')
    except Exception as e:
        st.error(f'An error occurred: {e}')