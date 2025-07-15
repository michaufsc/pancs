# app.py
import streamlit as st
import requests
from PIL import Image

# --- CONFIG ---
API_TOKEN = st.secrets["HF_TOKEN"]  # Armazenado com segurança em Streamlit Secrets
API_URL = "https://api-inference.huggingface.co/models/michaufsc27/pancs_modelo"
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}

# --- FUNÇÃO DE PREDIÇÃO ---
def query_huggingface_api(image_bytes):
    response = requests.post(API_URL, headers=HEADERS, files={"file": image_bytes})
    return response.json()

# --- INTERFACE STREAMLIT ---
st.set_page_config(page_title="PancsID", layout="centered")
st.title("🌿 PancsID - Identificador de Plantas PANC")

uploaded_file = st.file_uploader("Envie uma imagem da planta (jpg ou png)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagem enviada", use_column_width=True)

    with st.spinner("🔍 Analisando a imagem com o modelo Hugging Face..."):
        result = query_huggingface_api(uploaded_file.read())

    if isinstance(result, list) and len(result) > 0:
        label = result[0].get("label", "Desconhecido")
        score = result[0].get("score", 0)
        st.success(f"🌱 Previsão: **{label}**")
        st.write(f"Confiança: {score * 100:.2f}%")
    else:
        st.error("❌ Não foi possível obter a previsão. Verifique se o modelo está ativo.")
