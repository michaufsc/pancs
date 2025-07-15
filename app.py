import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from huggingface_hub import hf_hub_download

# Configurações
REPO_ID = "michaufsc27/pancs_modelo"
MODEL_FILENAME = "modelo_pancs.h5"
CLASSES_FILE = "classes.txt"

@st.cache_resource(show_spinner="Baixando modelo...")
def load_model():
    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=MODEL_FILENAME,
        local_dir=".",
        force_filename=MODEL_FILENAME
    )
    return tf.keras.models.load_model(model_path)

def load_classes():
    with open(CLASSES_FILE, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

# Interface
st.title("🌿 PancsID - Identificador de PANC")

model = load_model()
class_names = load_classes()

uploaded_file = st.file_uploader(
    "Envie uma imagem da planta",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    try:
        # Processamento da imagem
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Imagem enviada", use_column_width=True)
        
        img = img.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) / 255.0

        # Predição
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        
        st.success(f"🌱 Identificado como: **{class_names[np.argmax(score)]}**")
        st.write(f"Confiança: {100 * np.max(score):.2f}%")
        
    except Exception as e:
        st.error(f"Erro ao processar imagem: {str(e)}")
