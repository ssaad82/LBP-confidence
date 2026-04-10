# Lebanese Lira Sentiment Streamlit App

This app runs **daily sentiment analysis** for Lebanese lira discussions from:
- Telegram
- Google News (RSS)
- Nitter (RSS)

It uses a multilingual **BERT sentiment model** and aggregates results by day.

## Context window
The app is anchored to **February 28, 2026**, the start date of the Middle East conflict timeline used by this project.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Telegram notes
Telegram Bot API access is limited by channel permissions and API capabilities.
You can provide:
- `TELEGRAM_BOT_TOKEN` environment variable, or
- token in the Streamlit sidebar.

Then pass channel handles in the sidebar (comma-separated).

## Output
- Daily positive / neutral / negative counts
- Daily sentiment index: `(positive - negative) / total`
- Table of latest scored items
- CSV export for daily aggregates
