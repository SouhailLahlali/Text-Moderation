import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ========================
# Load model & tokenizer
# ========================
@st.cache_resource
def load_model():
    file_path = './saved_toxic_bert'
    model = AutoModelForSequenceClassification.from_pretrained(file_path)
    tokenizer = AutoTokenizer.from_pretrained(file_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, device

model, tokenizer, device = load_model()
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# ========================
# Prediction Function
# ========================
def predict_toxicity(text, threshold=0.5, max_length=128):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.sigmoid(logits)

    probs = probs.cpu().numpy()[0]
    preds = (probs >= threshold).astype(int)
    nsfw_flag = "⚠️ NSFW" if preds.sum() > 0 else "✅ SFW"

    return nsfw_flag, preds, probs

# ========================
# Streamlit App UI
# ========================
st.set_page_config(page_title="Toxicity Classifier", page_icon="🛡️", layout="wide")

st.title("🛡️ Toxicity Classification App")
st.markdown("""
Type any text below and this app will analyze its toxicity across multiple categories.
The model outputs probabilities and predicts whether it's **Safe for Work (SFW)** or **Not Safe for Work (NSFW)**.
""")

# User Input
text = st.text_area("✍️ Enter your text here:", height=150, placeholder="Type something toxic or safe...")
threshold = st.slider("⚖️ Classification Threshold", 0.0, 1.0, 0.5, 0.05)

if st.button("🔍 Analyze"):
    if text.strip():
        nsfw_flag, preds, probs = predict_toxicity(text, threshold)

        # Make result more prominent
        st.markdown(f"<h1 style='color:red; font-size:48px;'>Classification: {nsfw_flag}</h1>", unsafe_allow_html=True)

        # DataFrame for visualization
        df = pd.DataFrame({
            "Category": label_cols,
            "Probability": probs,
            "Prediction": ["Yes" if p else "No" for p in preds]
        })

        # Use columns to place chart and table side by side
        col1, col2 = st.columns([2, 1])  # chart bigger than table

        # Bar chart with coherent colors (Streamlit neutral palette)
        with col1:
            fig = px.bar(
                df, x="Category", y="Probability", color="Prediction",
                color_discrete_map={"Yes": "#FF4B4B", "No": "#4CAF50"},  # red/green for toxic/clean
                text=df["Probability"].round(2),
                title="Toxicity Probability per Category"
            )
            fig.update_traces(textposition="outside")
            fig.update_yaxes(range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

        # Table with coherent colors
        with col2:
            # Prepare cell colors
            cell_colors = []
            for pred in df.Prediction:
                if pred == "Yes":
                    cell_colors.append(['#FFCDD2']*3)  # soft red row
                else:
                    cell_colors.append(['#C8E6C9']*3)  # soft green row

            # Transpose colors to match Plotly Table format
            cell_colors = list(map(list, zip(*cell_colors)))

            fig_table = go.Figure(data=[go.Table(
                header=dict(
                    values=["Category", "Probability", "Prediction"],
                    fill_color='#333333',  # dark header for Streamlit dark mode
                    font=dict(color='white', size=14, family="Arial"),
                    align='center'
                ),
                cells=dict(
                    values=[df.Category, df.Probability.round(4), df.Prediction],
                    fill_color=cell_colors,
                    font=dict(color='black', size=12, family="Arial"),
                    align='center',
                    height=30
                )
            )])
            st.subheader("📊 Detailed Scores")
            st.plotly_chart(fig_table, use_container_width=True)

    else:
        st.warning("⚠️ Please enter some text before analyzing.")


# ========================
# Sidebar
# ========================
st.sidebar.header("📌 About")
st.sidebar.info("This app uses a fine-tuned BERT model for multi-label toxicity classification. Categories include toxic, severe toxic, obscene, threat, insult, and identity hate.")
st.sidebar.markdown("---")
