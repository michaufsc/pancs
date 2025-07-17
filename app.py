import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Configurações
MODEL_PATH = "modelo/modelo.h5"
LABELS_PATH = "labels.txt"

@st.cache_resource
def carregar_modelo():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_resource
def carregar_classes():
    with open(LABELS_PATH, 'r') as f:
        return [linha.strip() for linha in f]

def preprocessar_imagem(img, tamanho=(224, 224)):
    img = img.resize(tamanho)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Interface
st.title("🌿 Identificador de PANCs")
upload = st.file_uploader("Envie uma foto da planta:", type=["jpg", "png"])

if upload:
    imagem = Image.open(upload)
    st.image(imagem, width=300)
    
    if st.button("Identificar"):
        with st.spinner("Analisando..."):
            try:
                modelo = carregar_modelo()
                classes = carregar_classes()
                img_processada = preprocessar_imagem(imagem)
                
                predicao = modelo.predict(img_processada)[0]
                classe_id = np.argmax(predicao)
                
                st.success(f"**Resultado:** {classes[classe_id]}")
                st.info(f"**Confiança:** {predicao[classe_id]:.2%}")
                
            except Exception as e:
                st.error(f"Erro: {str(e)}")
                st.code("Dica: Verifique se o modelo está na pasta /modelo")
