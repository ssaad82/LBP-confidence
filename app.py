import logging
import time
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import feedparser
import pandas as pd
import requests
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from telethon import TelegramClient

# ============== CONFIGURATION ==============
WAR_START_DATE = datetime(2026, 3, 2, tzinfo=timezone.utc)
DEFAULT_QUERY = '"الليرة اللبنانية" OR "Lebanese lira" OR "سعر الصرف"'
DEFAULT_DAYS = 7
DEFAULT_NITTER_INSTANCE = "https://nitter.poast.org"
DEFAULT_TELEGRAM_KEYWORDS = ["lira", "ليرة", "صرف"]
MAX_RETRIES = 3
RETRY_DELAY = 2
REQUEST_TIMEOUT = 10

# ============== LOGGING ==============
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============== MODEL ==============
@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    """Load sentiment analysis model with error handling."""
    try:
        logger.info(f"Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, truncation=True)
        logger.info("Model loaded successfully")
        return classifier
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        st.error(f"⚠️ Model loading failed: {str(e)}")
        raise

def normalize_label(label: str) -> str:
    """Normalize sentiment labels."""
    try:
        label = str(label).lower().strip()
        if label in ["1 star", "2 stars", "negative"]:
            return "negative"
        elif label in ["4 stars", "5 stars", "positive"]:
            return "positive"
        return "neutral"
    except Exception as e:
        logger.warning(f"Error normalizing label: {e}")
        return "neutral"

# ============== DATA SOURCES ==============
def _parse_rss_feed(rss_url: str, timeout: int = REQUEST_TIMEOUT) -> Optional[feedparser.FeedParserDict]:
    """Fetch and parse RSS feed with retry logic."""
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(rss_url, timeout=timeout)
            response.raise_for_status()
            feed = feedparser.parse(response.content)
            if feed.bozo:
                logger.warning(f"RSS parsing warning: {feed.bozo_exception}")
            return feed
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout (attempt {attempt + 1}/{MAX_RETRIES})")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request error (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return None
    logger.error(f"Failed after {MAX_RETRIES} attempts")
    return None

@st.cache_data(ttl=600)
def search_google_news(query: str, start_date: datetime, end_date: datetime) -> List[Dict]:
    """Search Google News with error handling."""
    if not query or not query.strip():
        logger.warning("Empty query provided")
        return []
    try:
        encoded = requests.utils.quote(query)
        rss_url = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"
        feed = _parse_rss_feed(rss_url)
        if not feed or not feed.entries:
            return []
        rows = []
        for entry in feed.entries:
            try:
                published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                if start_date <= published <= end_date:
                    rows.append({
                        "source": "google_news",
                        "published_at": published,
                        "text": f"{entry.get('title', '')}. {entry.get('summary', '')}",
                        "url": entry.get('link', ''),
                    })
            except Exception:
                continue
        logger.info(f"Collected {len(rows)} from Google News")
        return rows
    except Exception as e:
        logger.error(f"Google News error: {str(e)}")
        return []

@st.cache_data(ttl=600)
def search_nitter(query: str, start_date: datetime, end_date: datetime, nitter_base: str) -> List[Dict]:
    """Search Nitter with error handling."""
    if not query or not query.strip() or not nitter_base:
        return []
    try:
        encoded = requests.utils.quote(query)
        rss_url = f"{nitter_base.rstrip('/')}/search/rss?f=tweets&q={encoded}"
        feed = _parse_rss_feed(rss_url)
        if not feed or not feed.entries:
            return []
        rows = []
        for entry in feed.entries:
            try:
                published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                if start_date <= published <= end_date:
                    rows.append({
                        "source": "nitter",
                        "published_at": published,
                        "text": f"{entry.get('title', '')}. {entry.get('summary', '')}",
                        "url": entry.get('link', ''),
                    })
            except Exception:
                continue
        logger.info(f"Collected {len(rows)} from Nitter")
        return rows
    except Exception as e:
        logger.error(f"Nitter error: {str(e)}")
        return []

# ============== TELEGRAM ==============
async def fetch_telegram_messages(api_id: int, api_hash: str, channels: List[str], start_date: datetime, end_date: datetime) -> List[Dict]:
    """Fetch Telegram messages with error handling."""
    client = TelegramClient("session", api_id, api_hash)
    rows = []
    try:
        await client.start()
        for channel in channels:
            try:
                async for msg in client.iter_messages(channel, limit=100):
                    if msg.text and any(k in msg.text.lower() for k in DEFAULT_TELEGRAM_KEYWORDS):
                        published = msg.date if msg.date.tzinfo else msg.date.replace(tzinfo=timezone.utc)
                        if start_date <= published <= end_date:
                            rows.append({
                                "source": "telegram",
                                "published_at": published,
                                "text": msg.text,
                                "url": f"https://t.me/{channel.lstrip('@')}",
                            })
            except Exception as e:
                logger.error(f"Telegram channel error: {str(e)}")
    except Exception as e:
        logger.error(f"Telegram client error: {str(e)}")
    finally:
        await client.disconnect()
    return rows

def search_telegram(api_id: Optional[int], api_hash: Optional[str], channels: List[str], start_date: datetime, end_date: datetime) -> List[Dict]:
    """Telegram search wrapper."""
    if not all([api_id, api_hash, channels]):
        return []
    try:
        return asyncio.run(fetch_telegram_messages(api_id, api_hash, channels, start_date, end_date))
    except Exception as e:
        logger.error(f"Telegram search failed: {str(e)}")
        return []

# ============== SENTIMENT ==============
def score_sentiment(items: List[Dict], classifier) -> pd.DataFrame:
    """Score sentiment with batch processing."""
    if not items:
        return pd.DataFrame()
    try:
        texts = [str(x.get("text", ""))[:512] for x in items]
        batch_size = 16
        preds = []
        for i in range(0, len(texts), batch_size):
            try:
                preds.extend(classifier(texts[i:i + batch_size]))
            except Exception as e:
                logger.error(f"Batch error: {e}")
                preds.extend([{"label": "neutral", "score": 0.0}] * (min(batch_size, len(texts) - i)))
        records = []
        for item, pred in zip(items, preds):
            try:
                records.append({
                    **item,
                    "sentiment": normalize_label(pred.get("label", "neutral")),
                    "score": float(pred.get("score", 0.0)),
                    "day": item["published_at"].date(),
                })
            except Exception:
                continue
        logger.info(f"Scored {len(records)} items")
        return pd.DataFrame(records)
    except Exception as e:
        logger.error(f"Sentiment scoring error: {str(e)}")
        return pd.DataFrame()

# ============== AGGREGATION ==============
def build_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Build daily aggregation."""
    if df.empty:
        return df
    try:
        daily = df.groupby(["day", "sentiment"]).size().rename("count").reset_index().pivot(index="day", columns="sentiment", values="count").fillna(0).astype(int).reset_index()
        for col in ["positive", "neutral", "negative"]:
            if col not in daily:
                daily[col] = 0
        total = daily[["positive", "neutral", "negative"]].sum(axis=1)
        daily["sentiment_index"] = ((daily["positive"] - daily["negative"]) / total.replace(0, 1)).round(4)
        return daily.sort_values("day")
    except Exception as e:
        logger.error(f"Aggregation error: {str(e)}")
        return df

# ============== UI ==============
def validate_inputs(query: str, model_name: str, nitter_instance: str) -> bool:
    """Validate inputs."""
    if not query or not query.strip():
        st.error("❌ Query empty")
        return False
    if not model_name or not model_name.strip():
        st.error("❌ Model empty")
        return False
    if not nitter_instance or not nitter_instance.strip():
        st.error("❌ Nitter instance empty")
        return False
    return True

def main():
    st.set_page_config(page_title="Lebanese Lira Sentiment", layout="wide")
    st.title("🇱🇧 Lebanese Lira Sentiment Dashboard")
    
    with st.sidebar:
        st.header("⚙️ Settings")
        model_name = st.text_input("Model", value="cardiffnlp/twitter-xlm-roberta-base-sentiment")
        query = st.text_area("Search query", value=DEFAULT_QUERY, height=3)
        lookback_days = st.slider("Days", 1, 365, DEFAULT_DAYS)
        nitter_instance = st.text_input("Nitter", value=DEFAULT_NITTER_INSTANCE)
        st.subheader("📱 Telegram")
        use_telegram = st.checkbox("Enable")
        telegram_api_id = st.text_input("API ID") if use_telegram else ""
        telegram_api_hash = st.text_input("API HASH", type="password") if use_telegram else ""
        telegram_channels = st.text_input("Channels") if use_telegram else ""
    
    end_date = datetime.now(timezone.utc)
    start_date = max(WAR_START_DATE, end_date - timedelta(days=int(lookback_days)))
    st.write(f"📅 Range: **{start_date.date()}** → **{end_date.date()}**")
    
    if st.button("🔍 Run Analysis", type="primary"):
        if not validate_inputs(query, model_name, nitter_instance):
            return
        try:
            with st.spinner("📦 Loading model..."):
                classifier = load_model(model_name)
        except:
            return
        
        with st.spinner("🌐 Fetching data..."):
            google_rows = search_google_news(query, start_date, end_date)
            nitter_rows = search_nitter(query, start_date, end_date, nitter_instance)
            telegram_rows = []
            if use_telegram and telegram_api_id and telegram_api_hash:
                try:
                    channels = [c.strip() for c in telegram_channels.split(",") if c.strip()]
                    telegram_rows = search_telegram(int(telegram_api_id), telegram_api_hash, channels, start_date, end_date)
                except ValueError:
                    st.error("❌ Invalid Telegram API ID")
            rows = google_rows + nitter_rows + telegram_rows
        
        if not rows:
            st.warning("⚠️ No data found")
            return
        st.success(f"✅ Collected **{len(rows)}** items")
        
        with st.spinner("💭 Analyzing..."):
            df = score_sentiment(rows, classifier)
        
        if df.empty:
            st.warning("⚠️ No results")
            return
        
        daily = build_daily(df)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📊 Total", len(df))
        col2.metric("⭐ Avg Score", f"{df['score'].mean():.2f}")
        col3.metric("📈 Positive %", f"{(df['sentiment']=='positive').sum()/len(df)*100:.1f}%")
        col4.metric("📉 Latest Index", f"{daily['sentiment_index'].iloc[-1]:.2f}")
        
        st.subheader("📈 Trend")
        st.line_chart(daily.set_index("day")["sentiment_index"], use_container_width=True)
        st.subheader("📊 Distribution")
        st.area_chart(daily.set_index("day")[["positive", "neutral", "negative"]], use_container_width=True)
        st.subheader("📋 Recent")
        st.dataframe(df.sort_values("published_at", ascending=False).head(200)[["source", "published_at", "sentiment", "score", "text"]], use_container_width=True, height=400)

if __name__ == "__main__":
    main()
