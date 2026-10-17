import streamlit as st
import numpy as np
import json
from pathlib import Path
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.title("Fruit & Vegetable Classification - MobileNetV2")
st.write("Transfer Learning ile meyve ve sebze sınıflandırma.")

folder = Path(__file__).parent
model = load_model(folder / "fruit_mobilenet.keras")

with open(folder / "fruit_classes.json", encoding="utf-8") as f:
    classes = json.load(f)

file = st.file_uploader("Görüntü yükle", type=["jpg", "jpeg", "png"])

if file is not None:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Yüklenen görüntü")

    x = np.array(img.resize((128, 128)), dtype="float32")
    x = np.expand_dims(x, axis=0)

    pred = model.predict(preprocess_input(x), verbose=0)[0]

    st.success(f"Tahmin: {classes[np.argmax(pred)]}")
    st.write(f"Güven skoru: {np.max(pred):.2%}")
