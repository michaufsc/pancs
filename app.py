import tensorflow as tf
import numpy as np
import requests
import streamlit as st
from PIL import Image
from huggingface_hub import hf_hub_download
import os

# --- CONFIGURAÇÃO ---
API_TOKEN = st.secrets.get("HF_TOKEN", "")  # Pegando token do secrets
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"} if API_TOKEN else {}

REPO_ID = "michaufsc27/pancs_modelo"
MODEL_FILENAME = "modelo_pancs.keras"
CLASSES_FILENAME = "labels.txt"
API_URL = f"https://api-inference.huggingface.co/models/{REPO_ID}"

# --- FUNÇÕES ---
@st.cache_resource(show_spinner="Carregando modelo...")
def carregar_modelo():
    try:
        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=MODEL_FILENAME,
            cache_dir=".",
            token=API_TOKEN if API_TOKEN else None
        )
        # Verifica se o arquivo do modelo existe e tem tamanho razoável
        if not os.path.exists(model_path) or os.path.getsize(model_path) < 1024:
            raise ValueError("Arquivo do modelo inválido ou corrompido")
            
        model = tf.keras.models.load_model(model_path, compile=False)
        if model is None:
            raise ValueError("Falha ao carregar o modelo - retornou None")
        model.compile()
        return model
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {str(e)}")
        st.stop()

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
                raise ValueError("Nenhuma classe carregada")
            return classes
    except Exception as e:
        st.error(f"Erro ao carregar classes: {str(e)}")
        st.stop()

def query_huggingface_api(image_bytes):
    if not API_TOKEN:
        st.error("Token do Hugging Face não configurado.")
        return None

    try:
        response = requests.post(API_URL, headers=HEADERS, data=image_bytes, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Erro na API: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Erro na conexão com a API: {str(e)}")
        return None

def preprocess_image(image, target_size=(224, 224)):
    img = image.resize(target_size)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)  # Ajuste conforme seu modelo
    return np.expand_dims(img_array, axis=0)

# ... (o resto do código permanece igual com as melhorias sugeridas)
