import torch
import streamlit as st
import logging

from src.cfg import MODEL_NAME
from src.model import MyModel

from transformers import AutoModelForSequenceClassification, AutoTokenizer

import matplotlib.pyplot as plt
import numpy as np

classes = ["c", "java", "js", "python", "r"]


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
# What language is/should your project written in?
"""

x = st.text_area("Input project description")

if x:
    with torch.no_grad():
        logits = predict(x)["logits"].numpy()[0]
        probs = np.exp(logits) / np.exp(logits).sum()

        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        ax.barh(classes, probs)

        st.pyplot(fig)
