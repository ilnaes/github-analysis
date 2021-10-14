import torch
import streamlit as st
import logging

from src.cfg import MODEL_NAME
from src.model import MyModel

from transformers import AutoModelForSequenceClassification, AutoTokenizer

import matplotlib.pyplot as plt
import numpy as np

classes = ["C", "Java", "Javascript", "Python", "R"]


@st.cache(allow_output_mutation=True)
def load_model(path):
    checkpoint = torch.load(path, map_location="cpu")
    config = {"model_name": MODEL_NAME}

    logging.info("Loading model")

    model = MyModel(config, pretrained=False)
    model.load_state_dict(checkpoint)

    return model, AutoTokenizer.from_pretrained(MODEL_NAME)


model, tokenizer = load_model("outputs/model_1.pt")


def predict(s):
    feed = [" ".join(s.split())]
    data = tokenizer(feed, return_tensors="pt", padding="max_length", max_length=256)

    return model(input_ids=data["input_ids"], attention_mask=data["attention_mask"])


"""
# What language should your project written in?
"""

x = st.text_area("Input project description (English ASCII descriptions only)")

if x:
    with torch.no_grad():
        logits = predict(x)["logits"].numpy()[0]
        choice = np.argmax(logits)
        probs = np.exp(logits) / np.exp(logits).sum()

        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        ax.barh(classes, probs)

        st.write(f"Your project should be written in {classes[choice]}.")
        st.pyplot(fig)
