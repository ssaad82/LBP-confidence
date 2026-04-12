import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List

import feedparser
import pandas as pd
import requests
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Telegram (MTProto)
from telethon import TelegramClient
import asyncio


WAR_START_DATE = datetime(2026, 3, 2, tzinfo=timezone.utc)
DEFAULT_QUERY = '"الليرة اللبنانية" OR "Lebanese lira" OR "سعر الصرف"'
DEFAULT_DAYS = 7
DEFAULT_NITTER_INSTANCE = "https://nitter.poast.org"


# ---------------- MODEL ----------------
@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, truncation=True)


def normalize_label(label: str) -> str:
    label = label.lower()
    if label in ["1 star", "2 stars", "negative"]:
        return "negative"
    elif label in ["4 stars", "5 stars", "positive"]:
        return "positive"
    return "neutral"


# ---------------- DATA SOURCES ----------------
@st.cache_data(ttl=600)
def search_google_news(query: str, start_date: datetime, end_date: datetime) -> List[Dict]:
    encoded = requests.utils.quote(query)
    rss_url = (
        "https://news.google.com/rss/search?"
        f"q={encoded}%20when:{(end_date - start_date).days}d&hl=en-US&gl=US&ceid=US:en"
    )
    feed = feedparser.parse(rss_url)

    rows = []
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


@st.cache_data(ttl=600)
def search_nitter(query: str, start_date: datetime, end_date: datetime, nitter_base: str) -> List[Dict]:
    encoded = requests.utils.quote(query)
    rss_url = f"{nitter_base.rstrip('/')}/search/rss?f=tweets&q={encoded}"
    feed = feedparser.parse(rss_url)

    rows = []
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


# ---------------- TELEGRAM (FIXED) ----------------
async def fetch_telegram_messages(api_id, api_hash, channels, start_date, end_date):
    client = TelegramClient("session", api_id, api_hash)
    await client.start()

    rows = []

    for channel in channels:
        try:
            async for msg in client.iter_messages(channel, limit=100):
                if not msg.text:
                    continue

                # Optional keyword filter (faster + relevant)
                if not any(k in msg.text.lower() for k in ["lira", "ليرة", "صرف"]):
                    continue

                published = msg.date
                if published.tzinfo is None:
                    published = published.replace(tzinfo=timezone.utc)

                if start_date <= published <= end_date:
                    rows.append(
                        {
                            "source": "telegram",
                            "published_at": published,
                            "text": msg.text,
                            "url": f"https://t.me/{channel.lstrip('@')}",
                        }
                    )
        except Exception as e:
            print(f"Error with {channel}: {e}")

    await client.disconnect()
    return rows


def search_telegram(api_id, api_hash, channel_ids, start_date, end_date):
    if not api_id or not api_hash or not channel_ids:
        return []

    return asyncio.run(
        fetch_telegram_messages(api_id, api_hash, channel_ids, start_date, end_date)
    )


# ---------------- SENTIMENT ----------------
def score_sentiment(items: List[Dict], classifier):
    if not items:
        return pd.DataFrame()

    texts = [x["text"][:512] for x in items]

    # batching for speed
    batch_size = 16
    preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        preds.extend(classifier(batch))

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


# ---------------- AGGREGATION ----------------
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


# ---------------- MAIN APP ----------------
def main():
    st.set_page_config(page_title="Lebanese Lira Sentiment", layout="wide")

    st.title("🇱🇧 Lebanese Lira Sentiment Dashboard (Fixed)")

    with st.sidebar:
        st.header("Settings")

        model_name = st.text_input(
            "Model",
            value="cardiffnlp/twitter-xlm-roberta-base-sentiment"
        )

        query = st.text_input("Search query", value=DEFAULT_QUERY)
        lookback_days = st.number_input("Days", min_value=1, max_value=365, value=DEFAULT_DAYS)

        nitter_instance = st.text_input("Nitter", value=DEFAULT_NITTER_INSTANCE)

        st.subheader("Telegram")
        telegram_api_id = st.text_input("API ID", value="")
        telegram_api_hash = st.text_input("API HASH", type="password")
        telegram_channels = st.text_input("Channels (@...,...)", value="")

    end_date = datetime.now(timezone.utc)
    start_date = max(WAR_START_DATE, end_date - timedelta(days=int(lookback_days)))

    st.write(f"Range: {start_date.date()} → {end_date.date()}")

    if st.button("Run", type="primary"):
        with st.spinner("Loading model..."):
            classifier = load_model(model_name)

        with st.spinner("Fetching data..."):
            google_rows = search_google_news(query, start_date, end_date)
            nitter_rows = search_nitter(query, start_date, end_date, nitter_instance)

            channel_list = [c.strip() for c in telegram_channels.split(",") if c.strip()]

            telegram_rows = search_telegram(
                int(telegram_api_id) if telegram_api_id else None,
                telegram_api_hash,
                channel_list,
                start_date,
                end_date
            )

            rows = google_rows + nitter_rows + telegram_rows

        st.info(f"Collected {len(rows)} items")

        df = score_sentiment(rows, classifier)

        if df.empty:
            st.warning("No data found")
            return

        daily = build_daily(df)

        c1, c2, c3 = st.columns(3)
        c1.metric("Docs", len(df))
        c2.metric("Avg score", f"{df['score'].mean():.2f}")
        c3.metric("Last index", f"{daily['sentiment_index'].iloc[-1]:.2f}")

        st.line_chart(daily.set_index("day")["sentiment_index"])
        st.area_chart(daily.set_index("day")[["positive", "neutral", "negative"]])

        st.dataframe(df.sort_values("published_at", ascending=False).head(200))


if __name__ == "__main__":
    main()
