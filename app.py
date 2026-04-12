import logging
import time
import asyncio
from telethon import TelegramClient
from google_news import GoogleNews
from nitter import NitterSearch
from sentiment_analysis import SentimentAnalyzer
from streamlit import st

# Set up logging
logging.basicConfig(level=logging.INFO)

# Constants
API_ID = 'your_api_id'
API_HASH = 'your_api_hash'

# Function for exponential backoff
async def retry_with_backoff(func, *args, retries=5, delay=1):
    for i in range(retries):
        try:
            return await func(*args)
        except Exception as e:
            logging.error(f"Error: {e}")
            await asyncio.sleep(delay)
            delay *= 2  # Exponential backoff
    logging.error("Max retries exceeded")
    return None

# Input validation function
def validate_input(input_data):
    if not isinstance(input_data, str) or len(input_data.strip()) == 0:
        raise ValueError("Input must be a non-empty string")

# Telegram client setup
client = TelegramClient('session_name', API_ID, API_HASH)

# Main processing function
async def main(input_text):
    validate_input(input_text)

    # Asynchronously fetching data
    google_news = GoogleNews()
    news_data = await retry_with_backoff(google_news.search, input_text)
    nitter_data = await retry_with_backoff(NitterSearch().search, input_text)

    # Sentiment analysis
    analyzer = SentimentAnalyzer()
    sentiment_results = await asyncio.gather(*[analyzer.analyze(news) for news in news_data])

    # Visualization in Streamlit
    st.title(f'Sentiment Analysis for: {input_text}')
    for result in sentiment_results:
        st.write(f'Sentiment: {result}')  # Display results

    # Telegram message fetching
    async with client:
        messages = await client.get_messages('channel_name')
        for message in messages:
            st.write(message.text)  # Display messages

# Streamlit app
if __name__ == '__main__':
    input_text = st.text_input('Enter text for sentiment analysis:')
    if st.button('Analyze'):
        asyncio.run(main(input_text))
        
        # Daily aggregation and visualization code goes here
