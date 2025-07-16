import streamlit as st
import requests
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

# --- CONFIGURAÇÃO ---
API_TOKEN = st.secrets["HF_TOKEN"]  # Armazenado com segurança em Streamlit Secrets
API_URL = "https://api-inference.huggingface.co/models/michaufsc27/pancs_modelo"
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}
REPO_ID = "michaufsc27/pancs_modelo"
MODEL_FILENAME = "modelo_pancs.h5"
CLASSES_FILENAME = "classes.txt"

# --- FUNÇÕES ---
@st.cache_resource(show_spinner="Carregando modelo...")
def carregar_modelo():
    model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME, cache_dir=".")
    model = tf.keras.models.load_model(model_path)
    return model

@st.cache_resource
def carregar_classes():
    classes_path = hf_hub_download(repo_id=REPO_ID, filename=CLASSES_FILENAME, cache_dir=".")
    with open(classes_path, "r", encoding='utf-8') as f:
        classes = [linha.strip() for linha in f]
    return classes

def query_huggingface_api(image_bytes):
    response = requests.post(API_URL, headers=HEADERS, data=image_bytes)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Erro na API: {response.status_code} - {response.text}")
        return None

def preprocess_image(image, target_size=(224, 224)):
    img = image.resize(target_size)
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- INTERFACE ---
st.set_page_config(page_title="PancsID", layout="centered")
st.title("🌿 PancsID - Identificador de Plantas PANC")

# Carregar modelo e classes
try:
    model = carregar_modelo()
    class_names = carregar_classes()
except Exception as e:
    st.error(f"Erro ao carregar modelo ou classes: {str(e)}")
    st.stop()

# Seleção do método de entrada
metodo = st.radio("Escolha como enviar a imagem:", 
                 ["📁 Enviar do dispositivo", "📷 Tirar com a câmera"])

imagem_input = None
if metodo == "📁 Enviar do dispositivo":
    imagem_input = st.file_uploader("Envie uma imagem da planta", type=["jpg", "jpeg", "png"])
else:
    imagem_input = st.camera_input("Tire uma foto da planta")

# Processamento quando uma imagem é fornecida
if imagem_input is not None:
    try:
        image = Image.open(imagem_input).convert("RGB")
        st.image(image, caption="Imagem selecionada", use_column_width=True)
        
        # Opção de escolha do método de predição
        usar_api = st.checkbox("Usar API do Hugging Face (recomendado)", value=True)
        
        if usar_api:
            with st.spinner("🔍 Analisando com API Hugging Face..."):
                image_bytes = imagem_input.getvalue()
                result = query_huggingface_api(image_bytes)
                
            if result:
                if isinstance(result, list):
                    # Ordenar resultados por confiança
                    result_sorted = sorted(result, key=lambda x: x['score'], reverse=True)
                    
                    st.success(f"🌱 Previsão principal: **{result_sorted[0]['label']}**")
                    st.write(f"Confiança: {result_sorted[0]['score'] * 100:.2f}%")
                    
                    # Mostrar alternativas
                    if len(result_sorted) > 1:
                        st.subheader("Outras possibilidades:")
                        for i, pred in enumerate(result_sorted[1:3], start=1):
                            st.write(f"{i}. {pred['label']} ({pred['score'] * 100:.2f}%)")
                else:
                    st.error("Formato de resposta inesperado da API")
        
        else:
            with st.spinner("🔍 Analisando com modelo local..."):
                img_array = preprocess_image(image)
                prediction = model.predict(img_array)
                predicted_index = np.argmax(prediction)
                predicted_label = class_names[predicted_index]
                confidence = 100 * np.max(prediction)
                
                st.success(f"🌱 Previsão: **{predicted_label}**")
                st.write(f"Confiança: {confidence:.2f}%")
                
                # Mostrar top 3 previsões
                top_indices = np.argsort(prediction[0])[-3:][::-1]
                st.subheader("Outras possibilidades:")
                for i, idx in enumerate(top_indices[1:], start=1):
                    st.write(f"{i}. {class_names[idx]} ({prediction[0][idx] * 100:.2f}%)")
    
    except Exception as e:
        st.error(f"Erro ao processar imagem: {str(e)}")

# Adicionar informações de ajuda
st.sidebar.markdown("""
### Como usar:
1. Escolha como enviar a imagem (upload ou câmera)
2. Aguarde a análise (pode demorar alguns segundos)
3. Veja os resultados e possíveis alternativas

### Dicas:
- Fotograve a planta com fundo limpo
- Foque nas folhas e características distintivas
- Para melhor precisão, use a opção da API
""")
