import tensorflow as tf
import numpy as np
import requests
import streamlit as st
from PIL import Image
from huggingface_hub import hf_hub_download
import os
import traceback

# --- CONFIGURAÇÃO ---
API_TOKEN = st.secrets.get("HF_TOKEN", "")  # Pegando token do secrets
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"} if API_TOKEN else {}

REPO_ID = "michaufsc27/pancs_modelo"
MODEL_FILENAME = "modelo_pancs.h5"
CLASSES_FILENAME = "labels.txt"
API_URL = f"https://api-inference.huggingface.co/models/{REPO_ID}"

# --- FUNÇÕES ---

@st.cache_resource(show_spinner="🔄 Carregando modelo...")
def carregar_modelo():
    try:
        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=MODEL_FILENAME,
            cache_dir=".",
            token=API_TOKEN if API_TOKEN else None
        )
        if not os.path.exists(model_path) or os.path.getsize(model_path) < 1024:
            raise ValueError("Arquivo do modelo inválido ou corrompido")
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile()
        return model
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar modelo: {e}")

@st.cache_resource
def carregar_classes():
    try:
        classes_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=CLASSES_FILENAME,
            cache_dir=".",
            token=API_TOKEN if API_TOKEN else None
        )
        with open(classes_path, "r", encoding="utf-8") as f:
            classes = [linha.strip() for linha in f]
            if not classes:
                raise ValueError("Arquivo de classes está vazio.")
            return classes
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar classes: {e}")

def preprocess_image(image, target_size=(224, 224)):
    try:
        img = image.convert("RGB").resize(target_size)  # garante que a imagem está no modo correto
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        raise RuntimeError(f"Erro ao processar a imagem: {e}")

# --- INTERFACE PRINCIPAL ---
def main():
    st.set_page_config(page_title="Identificador de PANCs", layout="centered")
    st.title("🌿 Identificador de Plantas Alimentícias Não Convencionais (PANCs)")
    st.write("Envie uma imagem de uma planta para identificar a espécie.")

    uploaded_file = st.file_uploader("📷 Escolha uma imagem...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Imagem enviada", use_container_width=True)

            if st.button("🔍 Identificar"):
                with st.spinner("Processando imagem..."):
                    model = carregar_modelo()
                    classes = carregar_classes()
                    input_data = preprocess_image(image)
                    predictions = model.predict(input_data)[0]

                    top_index = int(np.argmax(predictions))
                    confidence = float(predictions[top_index])
                    predicted_label = classes[top_index]

                    st.success(f"🌱 Previsão: **{predicted_label}** com confiança de {confidence:.2%}")

        except Exception:
            st.error("⚠️ Erro ao processar a imagem ou o modelo.")
            st.code(traceback.format_exc())

# --- EXECUÇÃO ---
if __name__ == "__main__":
    try:
        main()
    except Exception:
        st.error("⚠️ Erro geral no aplicativo:")
        st.code(traceback.format_exc())
