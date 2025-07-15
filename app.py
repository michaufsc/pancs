import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from huggingface_hub import hf_hub_download

# Configurações do modelo no Hugging Face Hub
REPO_ID = "michaufsc27/pancs_modelo"   # Troque pelo seu repositório HF
MODEL_FILENAME = "modelo_pancs.h5"

@st.cache_resource(show_spinner=False)
def baixar_modelo():
    if not os.path.exists(MODEL_FILENAME):
        st.info("📥 Baixando modelo do Hugging Face Hub...")
        hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME, cache_dir=".")
    model = tf.keras.models.load_model(MODEL_FILENAME)
    return model

model = baixar_modelo()

# Lista de classes na mesma ordem do treino do seu modelo
def carregar_classes(arquivo="classes.txt"):
    with open(arquivo, "r") as f:
        classes = [linha.strip() for linha in f.readlines()]
    return classes

class_names = carregar_classes()


st.title("🌿 PancsID - Identificador de Plantas PANC")

uploaded_file = st.file_uploader("Envie uma imagem da planta para classificar", type=["jpg", "jpeg", "png"])

if uploaded_file and model:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagem enviada", use_column_width=True)

    # Pré-processamento da imagem
    img = image.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predição
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_label = class_names[predicted_index]
    confidence = 100 * np.max(prediction)

    st.success(f"🌱 Previsão: **{predicted_label}**")
    st.write(f"Confiança: {confidence:.2f}%")
