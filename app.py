import os
import random
import time
import urllib.request
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, List

import feedparser
import pandas as pd
import requests
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from telethon.sync import TelegramClient


# ================= CONFIG =================

WAR_START_DATE = datetime(2026, 2, 28, tzinfo=timezone.utc)
DEFAULT_QUERY = '"الليرة اللبنانية" OR "Lebanese lira" OR "سعر الصرف"'
DEFAULT_DAYS = 7

# Telegram (Telethon)
API_ID = 34069133
API_HASH = "f8ec2f82d1eb6df4bdfd004fbd7fea48"
SESSION_NAME = "telegram_session"

# Nitter fallback instances
NITTER_INSTANCES = [
    "https://nitter.net",
    "https://nitter.it",
    "https://nitter.cz",
    "https://nitter.unixfox.eu"
]


# ================= MODEL =================

@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, truncation=True)


def normalize_label(label: str) -> str:
    normalized = label.upper()
    if "1" in normalized or "2" in normalized or "NEG" in normalized:
        return "negative"
    if "4" in normalized or "5" in normalized or "POS" in normalized:
        return "positive"
    return "neutral"


# ================= GOOGLE NEWS =================

def search_google_news(query: str, start_date: datetime, end_date: datetime) -> List[Dict]:
    encoded = requests.utils.quote(query)
    rss_url = (
        "https://news.google.com/rss/search?"
        f"q={encoded}%20when:{(end_date - start_date).days}d&hl=en-US&gl=US&ceid=US:en"
    )

    feed = feedparser.parse(rss_url)
    rows: List[Dict] = []

    for entry in feed.entries:
        published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)

        if start_date <= published <= end_date:
            rows.append(
                {
                    "source": "google_news",
                    "published_at": published,
                    "text": f"{entry.title}. {entry.get('summary', '')}",
                    "url": entry.link,
                }
            )

    return rows


# ================= NITTER (FIXED) =================

def search_nitter(query: str, start_date: datetime, end_date: datetime) -> List[Dict]:
    encoded = requests.utils.quote(query)

    instances = NITTER_INSTANCES.copy()
    random.shuffle(instances)

    for instance in instances:
        try:
            rss_url = f"{instance.rstrip('/')}/search/rss?f=tweets&q={encoded}"

            headers = {"User-Agent": "Mozilla/5.0"}
            req = urllib.request.Request(rss_url, headers=headers)

            feed = feedparser.parse(req)

            if not feed.entries:
                continue

            rows: List[Dict] = []

            for entry in feed.entries:
                if not getattr(entry, "published_parsed", None):
                    continue

                published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)

                if start_date <= published <= end_date:
                    rows.append(
                        {
                            "source": "nitter",
                            "published_at": published,
                            "text": f"{entry.title}. {entry.get('summary', '')}",
                            "url": entry.link,
                        }
                    )

            return rows

        except Exception as e:
            print(f"Nitter failed on {instance}: {e}")
            time.sleep(1)

    return []


# ================= TELEGRAM (TELETHON) =================

def search_telegram(channel_ids: List[str], start_date: datetime, end_date: datetime) -> List[Dict]:
    rows: List[Dict] = []

    async def fetch():
        async with TelegramClient(SESSION_NAME, API_ID, API_HASH) as client:
            for channel in channel_ids:
                try:
                    entity = await client.get_entity(channel)

                    async for message in client.iter_messages(entity, limit=500):
                        if not message.date or not message.text:
                            continue

                        published = message.date.replace(tzinfo=timezone.utc)

                        if published < start_date:
                            break

                        if start_date <= published <= end_date:
                            rows.append(
                                {
                                    "source": "telegram",
                                    "published_at": published,
                                    "text": message.text,
                                    "url": f"https://t.me/{channel.lstrip('@')}/{message.id}",
                                }
                            )

                except Exception as e:
                    print(f"Telethon error on {channel}: {e}")

    asyncio.run(fetch())
    return rows


# ================= SENTIMENT =================

def score_sentiment(items: List[Dict], classifier):
    if not items:
        return pd.DataFrame()

    texts = [x["text"][:512] for x in items]
    preds = classifier(texts)

    records = []
    for item, pred in zip(items, preds):
        label = normalize_label(pred["label"])

        records.append(
            {
                **item,
                "sentiment": label,
                "score": float(pred["score"]),
                "day": item["published_at"].date(),
            }
        )

    return pd.DataFrame(records)


def build_daily(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    daily = (
        df.groupby(["day", "sentiment"])
        .size()
        .rename("count")
        .reset_index()
        .pivot(index="day", columns="sentiment", values="count")
        .fillna(0)
        .reset_index()
    )

    for col in ["positive", "neutral", "negative"]:
        if col not in daily:
            daily[col] = 0

    total = daily[["positive", "neutral", "negative"]].sum(axis=1)
    daily["sentiment_index"] = ((daily["positive"] - daily["negative"]) / total.replace(0, 1)).round(4)

    return daily.sort_values("day")


# ================= MAIN APP =================

def main():
    st.set_page_config(page_title="Lebanese Lira Sentiment", layout="wide")

    st.title("🇱🇧 Daily Lebanese Lira Sentiment (Telegram + Google + Nitter)")

    with st.sidebar:
        st.header("Settings")

        model_name = st.text_input(
            "BERT model",
            value="nlptown/bert-base-multilingual-uncased-sentiment",
        )

        query = st.text_input("Search query", value=DEFAULT_QUERY)

        lookback_days = st.number_input("Window (days)", min_value=1, max_value=365, value=DEFAULT_DAYS)

        telegram_channels = st.text_input(
            "Telegram channels (comma separated @handles)",
            value="@mtvlebanon,@lbci_news,@lebanon24"
        )

    end_date = datetime.now(timezone.utc)
    start_date = max(WAR_START_DATE, end_date - timedelta(days=int(lookback_days)))

    st.write(f"Date range: **{start_date.date()} → {end_date.date()}**")

    if st.button("Run daily sentiment analysis", type="primary"):

        with st.spinner("Loading model..."):
            classifier = load_model(model_name)

        with st.spinner("Collecting data..."):
            google_rows = search_google_news(query, start_date, end_date)
            nitter_rows = search_nitter(query, start_date, end_date)

            channel_list = [c.strip() for c in telegram_channels.split(",") if c.strip()]
            telegram_rows = search_telegram(channel_list, start_date, end_date)

            rows = google_rows + nitter_rows + telegram_rows

        st.info(
            f"Collected {len(rows)} items "
            f"(Google: {len(google_rows)}, Nitter: {len(nitter_rows)}, Telegram: {len(telegram_rows)})"
        )

        df = score_sentiment(rows, classifier)

        if df.empty:
            st.warning("No data found.")
            return

        daily = build_daily(df)

        st.line_chart(daily.set_index("day")[["sentiment_index"]])
        st.area_chart(daily.set_index("day")[["positive", "neutral", "negative"]])

        st.dataframe(df.sort_values("published_at", ascending=False).head(200))


if __name__ == "__main__":
    main()
