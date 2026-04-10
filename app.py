import random
import asyncio
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

API_ID = 34069133
API_HASH = "f8ec2f82d1eb6df4bdfd004fbd7fea48"
SESSION_NAME = "telegram_session"

MAX_TELEGRAM_POSTS = 20
MAX_TELEGRAM_COMMENTS = 40
MAX_NITTER_ITEMS = 10
MAX_BERT_ITEMS = 120


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


# ================= GOOGLE =================

def search_google_news(query: str, start_date: datetime, end_date: datetime) -> List[Dict]:
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


# ================= NITTER (SAFE) =================

def search_nitter(query: str, start_date: datetime, end_date: datetime) -> List[Dict]:
    try:
        encoded = requests.utils.quote(query)
        url = f"https://nitter.net/search/rss?f=tweets&q={encoded}"

        response = requests.get(url, timeout=3)

        if response.status_code != 200:
            return []

        feed = feedparser.parse(response.text)

        rows = []
        for entry in feed.entries[:MAX_NITTER_ITEMS]:
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


# ================= TELEGRAM (FIXED + FAST) =================

def search_telegram(channel_ids: List[str], start_date: datetime, end_date: datetime) -> List[Dict]:
    rows = []

    async def fetch():
        async with TelegramClient(SESSION_NAME, API_ID, API_HASH) as client:
            for channel in channel_ids:
                try:
                    entity = await client.get_entity(channel)

                    messages = await client.get_messages(entity, limit=MAX_TELEGRAM_POSTS)

                    comment_count = 0

                    for message in messages:

                        if not message.date:
                            continue

                        published = message.date.replace(tzinfo=timezone.utc)

                        if start_date <= published <= end_date and message.text:
                            rows.append({
                                "source": "telegram_post",
                                "published_at": published,
                                "text": message.text,
                                "url": f"https://t.me/{channel.lstrip('@')}/{message.id}",
                            })

                        # COMMENTS (VERY LIMITED)
                        if message.replies and message.replies.comments and comment_count < MAX_TELEGRAM_COMMENTS:
                            try:
                                replies = await client.get_messages(entity, reply_to=message.id, limit=2)

                                for reply in replies:
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
                                        comment_count += 1

                            except:
                                pass

                except Exception as e:
                    print("Telegram error:", e)

    # ✅ FIXED event loop (no freezing)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(fetch())
    loop.close()

    return rows


# ================= SENTIMENT =================

def score_sentiment(items: List[Dict], classifier):
    if not items:
        return pd.DataFrame()

    items = items[:MAX_BERT_ITEMS]

    texts = [x["text"][:512] for x in items]
    preds = classifier(texts)

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
    daily["sentiment_index"] = ((daily["positive"] - daily["negative"]) / total.replace(0, 1))

    return daily


# ================= MAIN =================

def main():
    st.title("🇱🇧 LBP Sentiment (FAST & STABLE)")

    model_name = st.text_input("Model", "nlptown/bert-base-multilingual-uncased-sentiment")
    query = st.text_input("Query", DEFAULT_QUERY)
    channels = st.text_input("Telegram channels", "@mtvlebanon,@lbci_news")

    start_date = WAR_START_DATE
    end_date = datetime.now(timezone.utc)

    if st.button("Run analysis"):

        classifier = load_model(model_name)

        progress = st.progress(0)

        google = search_google_news(query, start_date, end_date)
        progress.progress(30, "Google done")

        nitter = search_nitter(query, start_date, end_date)
        progress.progress(60, "Nitter done")

        ch = [c.strip() for c in channels.split(",") if c.strip()]
        telegram = search_telegram(ch, start_date, end_date)
        progress.progress(90, "Telegram done")

        data = google + nitter + telegram

        df = score_sentiment(data, classifier)

        if df.empty:
            st.warning("No data found")
            return

        daily = build_daily(df)

        progress.progress(100, "Done")

        st.line_chart(daily.set_index("day")[["sentiment_index"]])
        st.area_chart(daily.set_index("day")[["positive", "neutral", "negative"]])

        st.dataframe(df.head(200))


if __name__ == "__main__":
    main()
