import streamlit as st
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download, HfFolder

# Token seguro via secrets
HF_TOKEN = st.secrets["HF_TOKEN"]

# Hugging Face repo e arquivos
REPO_ID = "michaufsc27/pancs_modelo"
MODEL_FILENAME = "modelo_pancs.h5"
CLASSES_FILENAME = "classes.txt"

# Autentica o huggingface_hub
HfFolder.save_token(HF_TOKEN)

@st.cache_resource(show_spinner=False)
def carregar_modelo():
    model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME, token=HF_TOKEN)
    model = tf.keras.models.load_model(model_path)
    return model

@st.cache_resource
def carregar_classes():
    classes_path = hf_hub_download(repo_id=REPO_ID, filename=CLASSES_FILENAME, token=HF_TOKEN)
    with open(classes_path, "r") as f:
        classes = [linha.strip() for linha in f]
    return classes

# Interface do app
st.title("🌿 PancsID - Identificador de Plantas PANC")

# Upload ou câmera
uploaded_file = st.file_uploader("Envie uma imagem da planta", type=["jpg", "jpeg", "png"])
if not uploaded_file:
    uploaded_file = st.camera_input("📸 Ou tire uma foto da planta agora")

# Carrega modelo e classes
model = carregar_modelo()
class_names = carregar_classes()

if uploaded_file and model:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagem enviada", use_column_width=True)

    img = image.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_index = int(np.argmax(prediction))
    predicted_label = class_names[predicted_index]
    confidence = 100 * np.max(prediction)

    st.success(f"🌱 Previsão: **{predicted_label}**")
    st.write(f"Confiança: {confidence:.2f}%")
