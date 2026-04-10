import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List

import feedparser
import pandas as pd
import requests
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


WAR_START_DATE = datetime(2026, 2, 28, tzinfo=timezone.utc)
DEFAULT_QUERY = '"الليرة اللبنانية" OR "Lebanese lira" OR "سعر الصرف"'
DEFAULT_DAYS = 7
DEFAULT_NITTER_INSTANCE = "https://nitter.net"


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


def search_nitter(query: str, start_date: datetime, end_date: datetime, nitter_base: str) -> List[Dict]:
    encoded = requests.utils.quote(query)
    rss_url = f"{nitter_base.rstrip('/')}/search/rss?f=tweets&q={encoded}"
    feed = feedparser.parse(rss_url)
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


def search_telegram(bot_token: str, channel_ids: List[str], start_date: datetime, end_date: datetime) -> List[Dict]:
    if not bot_token or not channel_ids:
        return []

    rows: List[Dict] = []
    base_url = f"https://api.telegram.org/bot{bot_token}"

    for channel in channel_ids:
        endpoint = f"{base_url}/getChatHistory"
        response = requests.get(endpoint, params={"chat_id": channel, "limit": 100}, timeout=30)
        if response.status_code != 200:
            continue
        payload = response.json()
        for msg in payload.get("result", []):
            date_unix = msg.get("date")
            text = msg.get("text") or msg.get("caption")
            if not date_unix or not text:
                continue
            published = datetime.fromtimestamp(date_unix, tz=timezone.utc)
            if start_date <= published <= end_date:
                rows.append(
                    {
                        "source": "telegram",
                        "published_at": published,
                        "text": str(text),
                        "url": f"https://t.me/{channel.lstrip('@')}",
                    }
                )

    return rows


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


def main():
    st.set_page_config(page_title="Lebanese Lira Sentiment", layout="wide")
    st.title("🇱🇧 Daily Lebanese Lira Sentiment (Telegram + Google + Nitter)")
    st.caption("Tracking from February 28, 2026 (start of the conflict in the middle east) to today.")

    with st.sidebar:
        st.header("Settings")
        model_name = st.text_input(
            "BERT model",
            value="nlptown/bert-base-multilingual-uncased-sentiment",
            help="You can swap this with another multilingual BERT sentiment model.",
        )
        query = st.text_input("Search query", value=DEFAULT_QUERY)
        lookback_days = st.number_input("Window (days)", min_value=1, max_value=365, value=DEFAULT_DAYS)
        nitter_instance = st.text_input("Nitter instance", value=DEFAULT_NITTER_INSTANCE)
        telegram_bot_token = st.text_input("Telegram Bot Token", value=os.getenv("TELEGRAM_BOT_TOKEN", ""), type="password")
        telegram_channels = st.text_input("Telegram channels (comma separated @handles)", value="")

    end_date = datetime.now(timezone.utc)
    start_date = max(WAR_START_DATE, end_date - timedelta(days=int(lookback_days)))

    st.write(f"Date range: **{start_date.date()} → {end_date.date()}**")

    if st.button("Run daily sentiment analysis", type="primary"):
        with st.spinner("Loading BERT model..."):
            classifier = load_model(model_name)

        with st.spinner("Collecting data from Google News, Nitter, and Telegram..."):
            google_rows = search_google_news(query, start_date, end_date)
            nitter_rows = search_nitter(query, start_date, end_date, nitter_instance)
            channel_list = [c.strip() for c in telegram_channels.split(",") if c.strip()]
            telegram_rows = search_telegram(telegram_bot_token, channel_list, start_date, end_date)
            rows = google_rows + nitter_rows + telegram_rows

        st.info(
            f"Collected {len(rows)} posts/articles "
            f"(Google: {len(google_rows)}, Nitter: {len(nitter_rows)}, Telegram: {len(telegram_rows)})."
        )

        df = score_sentiment(rows, classifier)
        if df.empty:
            st.warning("No data found in selected range/sources. Try broader query, more days, or different channels.")
            return

        daily = build_daily(df)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total documents", len(df))
        c2.metric("Avg confidence", f"{df['score'].mean():.2f}")
        c3.metric("Last day sentiment index", f"{daily['sentiment_index'].iloc[-1]:.2f}")

        st.subheader("Daily sentiment trend")
        st.line_chart(daily.set_index("day")[["sentiment_index"]])

        st.subheader("Daily sentiment mix")
        st.area_chart(daily.set_index("day")[["positive", "neutral", "negative"]])

        st.subheader("Latest scored items")
        st.dataframe(
            df.sort_values("published_at", ascending=False)[
                ["published_at", "source", "sentiment", "score", "text", "url"]
            ].head(200),
            use_container_width=True,
        )

        csv = daily.to_csv(index=False).encode("utf-8")
        st.download_button("Download daily sentiment CSV", data=csv, file_name="lira_sentiment_daily.csv", mime="text/csv")


if __name__ == "__main__":
    main()
