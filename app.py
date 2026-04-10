import os
import random
import time
import urllib.request
import asyncio
import socket
from datetime import datetime, timezone
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

# Telegram (Telethon)
API_ID = 34069133
API_HASH = "f8ec2f82d1eb6df4bdfd004fbd7fea48"
SESSION_NAME = "telegram_session"

# Nitter instances
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
    label = label.upper()
    if "1" in label or "2" in label or "NEG" in label:
        return "negative"
    if "4" in label or "5" in label or "POS" in label:
        return "positive"
    return "neutral"


# ================= GOOGLE NEWS =================

def search_google_news(query: str, start_date: datetime, end_date: datetime) -> List[Dict]:
    encoded = requests.utils.quote(query)
    rss_url = (
        "https://news.google.com/rss/search?"
        f"q={encoded}%20when:30d&hl=en-US&gl=US&ceid=US:en"
    )

    feed = feedparser.parse(rss_url)
    rows = []

    for entry in feed.entries:
        published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)

        if start_date <= published <= end_date:
            rows.append({
                "source": "google_news",
                "published_at": published,
                "text": f"{entry.title}. {entry.get('summary', '')}",
                "url": entry.link,
            })

    return rows


# ================= NITTER (FAST + SAFE) =================

def search_nitter(query: str, start_date: datetime, end_date: datetime) -> List[Dict]:
    encoded = requests.utils.quote(query)
    instances = NITTER_INSTANCES.copy()
    random.shuffle(instances)

    socket.setdefaulttimeout(5)

    for instance in instances:
        try:
            rss_url = f"{instance.rstrip('/')}/search/rss?f=tweets&q={encoded}"

            headers = {"User-Agent": "Mozilla/5.0"}
            req = urllib.request.Request(rss_url, headers=headers)

            feed = feedparser.parse(req)

            if not feed.entries:
                continue

            rows = []

            for entry in feed.entries:
                if not getattr(entry, "published_parsed", None):
                    continue

                published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)

                if start_date <= published <= end_date:
                    rows.append({
                        "source": "nitter",
                        "published_at": published,
                        "text": f"{entry.title}. {entry.get('summary', '')}",
                        "url": entry.link,
                    })

            return rows

        except Exception as e:
            print(f"Nitter failed: {e}")
            time.sleep(1)

    return []


# ================= TELEGRAM (POSTS + COMMENTS) =================

def search_telegram(channel_ids: List[str], start_date: datetime, end_date: datetime) -> List[Dict]:
    rows = []

    async def fetch():
        async with TelegramClient(SESSION_NAME, API_ID, API_HASH) as client:
            for channel in channel_ids:
                try:
                    entity = await client.get_entity(channel)

                    async for message in client.iter_messages(entity, limit=100):

                        if not message.date:
                            continue

                        published = message.date.replace(tzinfo=timezone.utc)

                        if published < start_date:
                            break

                        # MAIN POST
                        if message.text:
                            rows.append({
                                "source": "telegram_post",
                                "published_at": published,
                                "text": message.text,
                                "url": f"https://t.me/{channel.lstrip('@')}/{message.id}",
                            })

                        # COMMENTS (REPLIES)
                        if message.replies and message.replies.comments:
                            try:
                                async for reply in client.iter_messages(entity, reply_to=message.id, limit=50):
                                    if not reply.text:
                                        continue

                                    reply_date = reply.date.replace(tzinfo=timezone.utc)

                                    if start_date <= reply_date <= end_date:
                                        rows.append({
                                            "source": "telegram_comment",
                                            "published_at": reply_date,
                                            "text": reply.text,
                                            "url": f"https://t.me/{channel.lstrip('@')}/{message.id}",
                                        })
                            except:
                                pass

                except Exception as e:
                    print(f"Telethon error: {e}")

    asyncio.run(fetch())
    return rows


# ================= SENTIMENT =================

def score_sentiment(items: List[Dict], classifier):
    if not items:
        return pd.DataFrame()

    items = items[:200]  # speed limit
    texts = [x["text"][:512] for x in items]

    preds = classifier(texts)

    records = []
    for item, pred in zip(items, preds):
        records.append({
            **item,
            "sentiment": normalize_label(pred["label"]),
            "score": float(pred["score"]),
            "day": item["published_at"].date(),
        })

    return pd.DataFrame(records)


def build_daily(df: pd.DataFrame) -> pd.DataFrame:
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


# ================= MAIN =================

def main():
    st.set_page_config(layout="wide")
    st.title("🇱🇧 Lebanese Lira Sentiment (FULL Timeline + Comments)")

    model_name = st.text_input("Model", value="nlptown/bert-base-multilingual-uncased-sentiment")
    query = st.text_input("Query", value=DEFAULT_QUERY)
    channels = st.text_input("Telegram channels", value="@mtvlebanon,@lbci_news")

    start_date = WAR_START_DATE
    end_date = datetime.now(timezone.utc)

    st.write(f"Tracking from {start_date.date()} → {end_date.date()}")

    if st.button("Run analysis"):

        classifier = load_model(model_name)

        progress = st.progress(0)

        google_rows = search_google_news(query, start_date, end_date)
        progress.progress(30)

        nitter_rows = search_nitter(query, start_date, end_date)
        progress.progress(60)

        channel_list = [c.strip() for c in channels.split(",") if c.strip()]
        telegram_rows = search_telegram(channel_list, start_date, end_date)
        progress.progress(100)

        rows = google_rows + nitter_rows + telegram_rows

        df = score_sentiment(rows, classifier)

        if df.empty:
            st.warning("No data")
            return

        daily = build_daily(df)

        st.line_chart(daily.set_index("day")[["sentiment_index"]])
        st.area_chart(daily.set_index("day")[["positive", "neutral", "negative"]])

        st.dataframe(df.sort_values("published_at", ascending=False).head(200))


if __name__ == "__main__":
    main()
