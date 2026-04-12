import asyncio
import concurrent.futures
from datetime import datetime, timezone

import feedparser
import pandas as pd
import requests
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from telethon.sync import TelegramClient


# ================= CONFIG =================

WAR_START_DATE = datetime(2026, 2, 28, tzinfo=timezone.utc)

DEFAULT_QUERY = '"الليرة اللبنانية" OR "Lebanese lira" OR "سعر الصرف"'

# 🎯 Lira + finance focused channels
DEFAULT_CHANNELS = [
    "@lbprate",
    "@lebanonfx",
    "@lirarate",
    "@usd_lbp",
    "@lebfinance",
    "@lebanoneconomy",
    "@arabmarkets",
    "@lebanon24",
    "@naharnet",
]

# 🔍 Strong LBP filtering
KEYWORDS = [
    "ليرة", "دولار", "سعر الصرف",
    "usd", "lbp", "exchange", "parallel",
]

API_ID = 34069133
API_HASH = "f8ec2f82d1eb6df4bdfd004fbd7fea48"
SESSION_NAME = "telegram_session"

MAX_BERT_ITEMS = 40


# ================= MODEL =================

@st.cache_resource(show_spinner=False)
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        truncation=True
    )


def normalize_label(label):
    label = label.upper()
    if "1" in label or "2" in label or "NEG" in label:
        return "negative"
    if "4" in label or "5" in label or "POS" in label:
        return "positive"
    return "neutral"


# ================= FILTER =================

def filter_relevant(rows):
    filtered = []
    for r in rows:
        text = r["text"].lower()
        if any(k in text for k in KEYWORDS):
            filtered.append(r)
    return filtered


# ================= GOOGLE =================

def search_google_news(query, start_date, end_date):
    try:
        encoded = requests.utils.quote(query)
        url = f"https://news.google.com/rss/search?q={encoded}%20when:30d"

        feed = feedparser.parse(url)

        rows = []
        for entry in feed.entries[:30]:
            published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)

            if start_date <= published <= end_date:
                rows.append({
                    "source": "google",
                    "published_at": published,
                    "text": entry.title,
                    "url": entry.link,
                })

        return rows
    except:
        return []


# ================= NITTER =================

def search_nitter(query, start_date, end_date):
    try:
        encoded = requests.utils.quote(query)
        url = f"https://nitter.net/search/rss?f=tweets&q={encoded}"

        response = requests.get(url, timeout=3)

        if response.status_code != 200:
            return []

        feed = feedparser.parse(response.text)

        rows = []
        for entry in feed.entries[:10]:
            if not getattr(entry, "published_parsed", None):
                continue

            published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)

            if start_date <= published <= end_date:
                rows.append({
                    "source": "nitter",
                    "published_at": published,
                    "text": entry.title,
                    "url": entry.link,
                })

        return rows

    except:
        return []


# ================= TELEGRAM =================

def search_telegram(channel_ids, start_date, end_date):

    def run():
        rows = []

        async def fetch():
            async with TelegramClient(SESSION_NAME, API_ID, API_HASH) as client:
                for channel in channel_ids:
                    try:
                        entity = await client.get_entity(channel)
                        messages = await client.get_messages(entity, limit=10)

                        for msg in messages:
                            if not msg.text or not msg.date:
                                continue

                            published = msg.date.replace(tzinfo=timezone.utc)

                            if start_date <= published <= end_date:
                                rows.append({
                                    "source": "telegram",
                                    "published_at": published,
                                    "text": msg.text,
                                    "url": f"https://t.me/{channel.lstrip('@')}/{msg.id}",
                                })

                    except:
                        continue

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(fetch())
        loop.close()

        return rows

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run)
            return future.result(timeout=5)
    except concurrent.futures.TimeoutError:
        return []


# ================= SENTIMENT =================

def score_sentiment(items, classifier):
    if not items:
        return pd.DataFrame()

    items = items[:MAX_BERT_ITEMS]

    texts = [x["text"][:512] for x in items]
    preds = classifier(texts, batch_size=8)

    rows = []
    for item, pred in zip(items, preds):
        rows.append({
            **item,
            "sentiment": normalize_label(pred["label"]),
            "score": float(pred["score"]),
            "day": item["published_at"].date(),
        })

    return pd.DataFrame(rows)


def build_daily(df):
    daily = (
        df.groupby(["day", "sentiment"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for col in ["positive", "neutral", "negative"]:
        if col not in daily:
            daily[col] = 0

    total = daily[["positive", "neutral", "negative"]].sum(axis=1)

    # 🎯 ONE FINAL SENTIMENT INDEX
    daily["sentiment_index"] = (
        (daily["positive"] - daily["negative"]) / total.replace(0, 1)
    )

    return daily


# ================= MAIN =================

def main():
    st.title("🇱🇧 Lebanese Lira Sentiment Index")

    model_name = st.text_input(
        "Model",
        "nlptown/bert-base-multilingual-uncased-sentiment"
    )

    query = st.text_input("Query", DEFAULT_QUERY)

    channels_input = st.text_area(
        "Telegram channels",
        ",".join(DEFAULT_CHANNELS)
    )

    start_date = WAR_START_DATE
    end_date = datetime.now(timezone.utc)

    if st.button("Run analysis"):

        classifier = load_model(model_name)

        channels = [c.strip() for c in channels_input.split(",") if c.strip()]

        # 🚀 PARALLEL SOURCES
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                "google": executor.submit(search_google_news, query, start_date, end_date),
                "nitter": executor.submit(search_nitter, query, start_date, end_date),
                "telegram": executor.submit(search_telegram, channels, start_date, end_date),
            }

            google = futures["google"].result()
            nitter = futures["nitter"].result()
            telegram = futures["telegram"].result()

        # 🔥 FILTER ONLY LBP-RELEVANT CONTENT
        data = filter_relevant(google + nitter + telegram)

        df = score_sentiment(data, classifier)

        if df.empty:
            st.warning("No relevant LBP data found")
            return

        daily = build_daily(df)

        # 📊 OUTPUT
        st.subheader("Sentiment Index")
        st.line_chart(daily.set_index("day")[["sentiment_index"]])

        st.subheader("Sentiment Breakdown")
        st.area_chart(daily.set_index("day")[["positive", "neutral", "negative"]])

        st.subheader("Raw Data")
        st.dataframe(df.head(200))


if __name__ == "__main__":
    main()
