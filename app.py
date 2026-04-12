import asyncio
import concurrent.futures
from datetime import datetime, timezone

import feedparser
import pandas as pd
import requests
import streamlit as st
from telethon.sync import TelegramClient
from transformers import pipeline


# ================= CONFIG =================

WAR_START_DATE = datetime(2026, 2, 28, tzinfo=timezone.utc)

DEFAULT_QUERY = '"الليرة اللبنانية" OR "Lebanese lira" OR "سعر الصرف"'

DEFAULT_CHANNELS = [
    "@lbprate",
    "@lebanonfx",
    "@usd_lbp",
    "@lebanoneconomy",
]

KEYWORDS = [
    "ليرة", "دولار", "سعر الصرف",
    "usd", "lbp", "exchange",
]

POSITIVE_WORDS = [
    "تحسن", "استقرار", "انخفاض",
    "improve", "stable", "recovery"
]

NEGATIVE_WORDS = [
    "ارتفاع", "انهيار", "تدهور",
    "crisis", "collapse", "devaluation"
]

MAX_BERT_ITEMS = 12  # 🔥 small number


API_ID = 34069133
API_HASH = "f8ec2f82d1eb6df4bdfd004fbd7fea48"
SESSION_NAME = "telegram_session"


# ================= FAST SENTIMENT =================

def fast_sentiment(text):
    text = text.lower()
    score = 0

    for w in POSITIVE_WORDS:
        if w in text:
            score += 1

    for w in NEGATIVE_WORDS:
        if w in text:
            score -= 1

    if score > 0:
        return "positive", score
    elif score < 0:
        return "negative", score
    else:
        return "neutral", 0


def is_relevant(text):
    return any(k in text.lower() for k in KEYWORDS)


# ================= MODEL =================

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment")


# ================= GOOGLE =================

@st.cache_data(ttl=300)
def search_google_news(query, start_date, end_date):
    rows = []
    try:
        encoded = requests.utils.quote(query)
        url = f"https://news.google.com/rss/search?q={encoded}%20when:30d"
        feed = feedparser.parse(url)

        for entry in feed.entries[:20]:
            text = entry.title

            if not is_relevant(text):
                continue

            published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)

            if start_date <= published <= end_date:
                rows.append({
                    "source": "google",
                    "published_at": published,
                    "text": text,
                })

    except:
        pass

    return rows


# ================= NITTER =================

@st.cache_data(ttl=300)
def search_nitter(query, start_date, end_date):
    rows = []
    try:
        encoded = requests.utils.quote(query)
        url = f"https://nitter.net/search/rss?f=tweets&q={encoded}"

        response = requests.get(url, timeout=2)
        if response.status_code != 200:
            return []

        feed = feedparser.parse(response.text)

        for entry in feed.entries[:10]:
            text = entry.title

            if not is_relevant(text):
                continue

            published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)

            if start_date <= published <= end_date:
                rows.append({
                    "source": "nitter",
                    "published_at": published,
                    "text": text,
                })

    except:
        pass

    return rows


# ================= TELEGRAM =================

@st.cache_data(ttl=300)
def search_telegram(channel_ids, start_date, end_date):

    def run():
        rows = []

        async def fetch():
            async with TelegramClient(SESSION_NAME, API_ID, API_HASH) as client:
                for channel in channel_ids:
                    try:
                        entity = await client.get_entity(channel)
                        messages = await client.get_messages(entity, limit=5)

                        for msg in messages:
                            if not msg.text:
                                continue

                            if not is_relevant(msg.text):
                                continue

                            published = msg.date.replace(tzinfo=timezone.utc)

                            if start_date <= published <= end_date:
                                rows.append({
                                    "source": "telegram",
                                    "published_at": published,
                                    "text": msg.text,
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
            return executor.submit(run).result(timeout=4)
    except:
        return []


# ================= HYBRID SENTIMENT =================

def hybrid_sentiment(data, classifier):

    # Step 1: fast scoring
    rows = []
    for item in data:
        sentiment, score = fast_sentiment(item["text"])
        rows.append({**item, "fast_score": score})

    df = pd.DataFrame(rows)

    # Step 2: pick most important texts
    df = df.sort_values("fast_score", key=abs, ascending=False)

    top_df = df.head(MAX_BERT_ITEMS).copy()

    # Step 3: run BERT only on top items
    texts = top_df["text"].tolist()
    preds = classifier(texts)

    bert_scores = []
    for p in preds:
        label = p["label"].lower()

        if "positive" in label:
            bert_scores.append(1)
        elif "negative" in label:
            bert_scores.append(-1)
        else:
            bert_scores.append(0)

    top_df["bert_score"] = bert_scores

    # Step 4: merge scores
    df["final_score"] = df["fast_score"]
    df.loc[top_df.index, "final_score"] = top_df["bert_score"]

    df["day"] = df["published_at"].dt.date

    return df


def build_daily(df):
    daily = df.groupby("day")["final_score"].mean().reset_index()
    daily.rename(columns={"final_score": "sentiment_index"}, inplace=True)
    return daily


# ================= MAIN =================

def main():
    st.title("⚡ Hybrid Lebanese Lira Sentiment")

    query = st.text_input("Query", DEFAULT_QUERY)

    channels_input = st.text_area(
        "Telegram channels",
        ",".join(DEFAULT_CHANNELS)
    )

    start_date = WAR_START_DATE
    end_date = datetime.now(timezone.utc)

    if st.button("Run analysis"):

        classifier = load_model()

        channels = [c.strip() for c in channels_input.split(",") if c.strip()]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                "google": executor.submit(search_google_news, query, start_date, end_date),
                "nitter": executor.submit(search_nitter, query, start_date, end_date),
                "telegram": executor.submit(search_telegram, channels, start_date, end_date),
            }

            google = futures["google"].result()
            nitter = futures["nitter"].result()
            telegram = futures["telegram"].result()

        data = google + nitter + telegram

        if not data:
            st.warning("No data found")
            return

        df = hybrid_sentiment(pd.DataFrame(data), classifier)
        daily = build_daily(df)

        st.line_chart(daily.set_index("day"))
        st.dataframe(df.head(100))


if __name__ == "__main__":
    main()
