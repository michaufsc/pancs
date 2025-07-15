import streamlit as st
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

# Hugging Face repo e modelo
REPO_ID = "michaufsc27/pancs_modelo"
MODEL_FILENAME = "modelo_pancs.h5"
CLASSES_FILENAME = "classes.txt"

@st.cache_resource(show_spinner=False)
def carregar_modelo():
    hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME, cache_dir=".")
    model = tf.keras.models.load_model(MODEL_FILENAME)
    return model

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

    img = image.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_label = class_names[predicted_index]
    confidence = 100 * np.max(prediction)

    st.success(f"🌱 Previsão: **{predicted_label}**")
    st.write(f"Confiança: {confidence:.2f}%")
