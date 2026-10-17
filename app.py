import pandas as pd
import streamlit as st
import numpy as np
import json
from pathlib import Path
from PIL import Image
from tensorflow.keras.models import load_model

st.title("Fruit & Vegetable Classification - CNN")
st.write("Meyve veya sebze görseli yükleyerek türünü tahmin edin.")

folder = Path(__file__).parent
model = load_model(folder / "fruit_cnn.keras")

with open(folder / "fruit_classes.json", encoding="utf-8") as f:
    classes = json.load(f)

file = st.file_uploader("Görüntü yükle", type=["jpg", "jpeg", "png"])

if file is not None:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Yüklenen görüntü")

    x = np.array(img.resize((128, 128)), dtype="float32")
    x = np.expand_dims(x / 255.0, axis=0)

    pred = model.predict(x, verbose=0)[0]

    st.success(f"Tahmin: {classes[np.argmax(pred)]}")
    st.write(f"Güven skoru: {np.max(pred):.2%}")
    
top5 = pd.Series(pred, index=classes).nlargest(5)
st.bar_chart(top5)