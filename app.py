import logging
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TelegramBot:
    def __init__(self, token):
        self.token = token
        self.api_url = f'https://api.telegram.org/bot{self.token}/sendMessage'

    def send_message(self, chat_id, text):
        payload = {'chat_id': chat_id, 'text': text}
        try:
            response = requests.post(self.api_url, data=payload)
            response.raise_for_status()
            logging.info("Message sent to Telegram successfully.")
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to send message: {e}")

class InputValidator:
    @staticmethod
    def validate_input(user_input):
        if not user_input:
            logging.error("Invalid input: Input cannot be empty.")
            raise ValueError("Input cannot be empty.")
        logging.info("Input validated successfully.")

class SentimentAnalyzer:
    def analyze_sentiment(self, text):
        # Placeholder for sentiment analysis logic
        logging.info("Analyzing sentiment...")
        # Assume it returns some sentiment score
        return "positive"

def main():
    bot_token = 'YOUR_TELEGRAM_BOT_TOKEN'
    chat_id = 'YOUR_CHAT_ID'
    user_input = input("Enter your input: ")
    InputValidator.validate_input(user_input)

    # Retry logic
    for attempt in range(3):
        try:
            sentiment_analyzer = SentimentAnalyzer()
            sentiment = sentiment_analyzer.analyze_sentiment(user_input)
            logging.info(f"Sentiment: {sentiment}")
            # Send message to Telegram
            bot = TelegramBot(bot_token)
            bot.send_message(chat_id, f"Sentiment analysis result: {sentiment}")
            break
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed: {e}")
            if attempt == 2:
                logging.critical("Max retries reached. Exiting.")

if __name__ == '__main__':
    main()