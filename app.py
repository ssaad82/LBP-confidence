import streamlit as st
from transformers import pipeline

# -----------------------------
# CACHED MODEL LOADING
# -----------------------------
@st.cache_resource
def load_model():
    # Using a stable model that does NOT require tiktoken
    return pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment"
    )

# -----------------------------
# MAIN APP
# -----------------------------
def main():
    st.set_page_config(page_title="LBP Confidence Tracker", layout="wide")

    st.title("💰 LBP Sentiment / Confidence Tracker")
    st.write("Analyze sentiment related to the Lebanese Lira from text input.")

    # Load model
    classifier = load_model()

    # User input
    user_input = st.text_area("Enter text (news, tweet, etc.):")

    if st.button("Analyze"):
        if user_input.strip() == "":
            st.warning("Please enter some text.")
        else:
            result = classifier(user_input)[0]

            label = result["label"]
            score = result["score"]

            # Convert label to confidence score (simple mapping)
            # Model outputs 1 to 5 stars
            stars = int(label.split()[0])
            confidence = (stars - 3) / 2  # normalize between -1 and +1

            st.subheader("Result")
            st.write(f"**Raw Sentiment:** {label}")
            st.write(f"**Model Confidence:** {score:.2f}")
            st.write(f"**LBP Confidence Score:** {confidence:.2f}")

            # Visual indicator
            if confidence > 0:
                st.success("Positive sentiment toward LBP")
            elif confidence < 0:
                st.error("Negative sentiment toward LBP")
            else:
                st.info("Neutral sentiment")

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    main()
