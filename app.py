import streamlit as st
import requests
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

# --- CONFIG ---
API_TOKEN = st.secrets["HF_TOKEN"]  # Armazenado com segurança em Streamlit Secrets
API_URL = "https://api-inference.huggingface.co/models/michaufsc27/pancs_modelo"
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}
# Hugging Face repo e modelo
REPO_ID = "michaufsc27/pancs_modelo"
MODEL_FILENAME = "modelo_pancs.h5"
CLASSES_FILENAME = "classes.txt"

# --- FUNÇÃO DE PREDIÇÃO ---
def query_huggingface_api(image_bytes):
    response = requests.post(API_URL, headers=HEADERS, files={"file": image_bytes})
    return response.json()
@st.cache_resource(show_spinner=False)
def carregar_modelo():
    hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME, cache_dir=".")
    model = tf.keras.models.load_model(MODEL_FILENAME)
    return model

# --- INTERFACE STREAMLIT ---
st.set_page_config(page_title="PancsID", layout="centered")
@st.cache_resource
def carregar_classes():
    hf_hub_download(repo_id=REPO_ID, filename=CLASSES_FILENAME, cache_dir=".")
    with open(CLASSES_FILENAME, "r") as f:
        classes = [linha.strip() for linha in f]
    return classes

# Carregar modelo e classes
model = carregar_modelo()
class_names = carregar_classes()

# Título do app
st.title("🌿 PancsID - Identificador de Plantas PANC")

uploaded_file = st.file_uploader("Envie uma imagem da planta (jpg ou png)", type=["jpg", "jpeg", "png"])
# Escolha do método de envio
metodo = st.radio("Escolha como enviar a imagem:", ["📁 Enviar do dispositivo", "📷 Tirar com a câmera"])

# Upload ou câmera
if metodo == "📁 Enviar do dispositivo":
    imagem_input = st.file_uploader("Envie uma imagem da planta", type=["jpg", "jpeg", "png"])
else:
    imagem_input = st.camera_input("Tire uma foto da planta")

# Processamento e previsão
if imagem_input and model:
    image = Image.open(imagem_input).convert("RGB")
    st.image(image, caption="Imagem selecionada", use_column_width=True)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagem enviada", use_column_width=True)
    img = image.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("🔍 Analisando a imagem com o modelo Hugging Face..."):
        result = query_huggingface_api(uploaded_file.read())
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_label = class_names[predicted_index]
    confidence = 100 * np.max(prediction)

    if isinstance(result, list) and len(result) > 0:
        label = result[0].get("label", "Desconhecido")
        score = result[0].get("score", 0)
        st.success(f"🌱 Previsão: **{label}**")
        st.write(f"Confiança: {score * 100:.2f}%")
    else:
        st.error("❌ Não foi possível obter a previsão. Verifique se o modelo está ativo.")
    st.success(f"🌱 Previsão: **{predicted_label}**")
    st.write(f"Confiança: {confidence:.2f}%")
