import streamlit as st
import requests
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from huggingface_hub import hf_hub_download
import tempfile
import os

# --- CONFIGURAÇÃO ---
st.set_page_config(
    page_title="PancsID", 
    page_icon="🌿", 
    layout="centered",
    initial_sidebar_state="expanded"
)

# Carregar token de forma segura
API_TOKEN = st.secrets.get("HF_TOKEN", "")
API_URL = "https://api-inference.huggingface.co/models/michaufsc27/pancs_modelo"
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"} if API_TOKEN else {}
REPO_ID = "michaufsc27/pancs_modelo"
MODEL_FILENAME = "modelo_pancs (2).h5"
CLASSES_FILENAME = "classes (2).txt"

# --- FUNÇÕES ---
@st.cache_resource(show_spinner="🔄 Carregando modelo local...")
def carregar_modelo():
    """Carrega o modelo TensorFlow com tratamento robusto de erros"""
    try:
        # Criar diretório temporário para evitar conflitos
        cache_dir = tempfile.mkdtemp()
        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=MODEL_FILENAME,
            cache_dir=cache_dir,
            token=API_TOKEN if API_TOKEN else None
        )
        
        # Configuração de compatibilidade
        custom_objects = {
            'InputLayer': tf.keras.layers.InputLayer,
            'KerasLayer': tf.keras.layers.Layer
        }
        
        model = tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False
        )
        
        # Verificação de integridade do modelo
        if not hasattr(model, "predict"):
            raise ValueError("Modelo carregado não possui método predict")
            
        model.compile()
        return model
    except Exception as e:
        st.error(f"⛔ Falha ao carregar modelo: {str(e)}")
        st.error("Por favor, verifique os arquivos do modelo ou tente mais tarde.")
        st.stop()

@st.cache_resource(show_spinner="📚 Carregando categorias...")
def carregar_classes():
    """Carrega as classes de plantas com tratamento de encoding"""
    try:
        cache_dir = tempfile.mkdtemp()
        classes_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=CLASSES_FILENAME,
            cache_dir=cache_dir,
            token=API_TOKEN if API_TOKEN else None
        )
        
        with open(classes_path, "r", encoding='utf-8') as f:
            classes = [linha.strip() for linha in f if linha.strip()]
            
        if not classes:
            raise ValueError("Arquivo de classes vazio")
            
        return classes
    except Exception as e:
        st.error(f"⛔ Erro ao carregar categorias: {str(e)}")
        st.stop()

@st.cache_data
def preprocess_image(_image, target_size=(224, 224)):
    """Pré-processa a imagem para o formato esperado pelo modelo"""
    try:
        img = _image.resize(target_size)
        img_array = tf.keras.utils.img_to_array(img)
        img_array = img_array / 127.5 - 1.0  # Normalização para [-1, 1]
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        st.error(f"⛔ Erro no pré-processamento da imagem: {str(e)}")
        return None

def query_huggingface_api(image_bytes):
    """Consulta a API do Hugging Face com tratamento de erros"""
    if not API_TOKEN:
        st.warning("🔒 API desativada (token não configurado)")
        return None
        
    try:
        response = requests.post(API_URL, headers=HEADERS, data=image_bytes, timeout=10)
        response.raise_for_status()
        
        result = response.json()
        if not isinstance(result, list):
            raise ValueError("Resposta da API em formato inesperado")
            
        return result
    except requests.exceptions.RequestException as e:
        st.error(f"⛔ Erro na conexão com a API: {str(e)}")
        return None
    except Exception as e:
        st.error(f"⛔ Erro ao processar resposta da API: {str(e)}")
        return None

def display_results(predictions, source="API"):
    """Exibe os resultados de forma padronizada"""
    if not predictions:
        return
        
    # Cria DataFrame com resultados
    if source == "API":
        df = pd.DataFrame(predictions)
        df = df.sort_values("score", ascending=False)
        df["score"] = df["score"].apply(lambda x: f"{x*100:.2f}%")
        df.columns = ["Planta", "Confiança"]
    else:
        top_indices = np.argsort(predictions[0])[-5:][::-1]
        df = pd.DataFrame({
            "Planta": [class_names[i] for i in top_indices],
            "Confiança": [f"{predictions[0][i]*100:.2f}%" for i in top_indices]
        })
    
    # Exibe resultados
    st.success(f"🌱 **Identificação principal**: {df.iloc[0]['Planta']} (Confiança: {df.iloc[0]['Confiança']})")
    
    with st.expander("🔍 Ver todas as possibilidades"):
        st.dataframe(df, hide_index=True, use_container_width=True)
        
    # Sugestão visual
    if df.iloc[0]['Confiança'] < "70.00%":
        st.warning("⚠️ Confiança moderada - verifique com outras fontes")

# --- INTERFACE ---
st.title("🌿 PancsID - Identificador de Plantas PANC")
st.markdown("""
Identifique plantas alimentícias não convencionais (PANC) através de imagens.
""")

# Barra lateral informativa
with st.sidebar:
    st.header("ℹ️ Como usar")
    st.markdown("""
    1. Escolha o método de entrada
    2. Envie/tire uma foto da planta
    3. Aguarde a análise (10-30 segundos)
    4. Verifique os resultados
    
    **Dicas para melhor precisão:**
    - Fotograve com boa iluminação
    - Foque nas folhas e características únicas
    - Evite fundos poluídos
    """)
    
    if not API_TOKEN:
        st.warning("🔒 Modo local (sem API Hugging Face)")

# Carregar recursos
model = carregar_modelo() if not API_TOKEN else None
class_names = carregar_classes()

# Seleção do método de entrada
metodo = st.radio(
    "📷 Método de captura:",
    ["Upload de imagem", "Tirar foto"],
    horizontal=True,
    index=0
)

imagem_input = None
if metodo == "Upload de imagem":
    imagem_input = st.file_uploader(
        "Selecione uma imagem da planta",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
else:
    imagem_input = st.camera_input(
        "Tire uma foto da planta",
        label_visibility="collapsed"
    )

# Processamento da imagem
if imagem_input is not None:
    try:
        image = Image.open(imagem_input).convert("RGB")
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Imagem processada", use_column_width=True)
        
        with col2:
            st.subheader("🔎 Análise")
            
            # Opção de usar API se disponível
            usar_api = False
            if API_TOKEN:
                usar_api = st.toggle("Usar API Hugging Face (recomendado)", value=True)
            
            if usar_api:
                with st.spinner("🔍 Consultando API Hugging Face..."):
                    image_bytes = imagem_input.getvalue()
                    result = query_huggingface_api(image_bytes)
                    if result:
                        display_results(result, source="API")
            else:
                with st.spinner("🔍 Processando com modelo local..."):
                    img_array = preprocess_image(image)
                    if img_array is not None:
                        prediction = model.predict(img_array)
                        display_results(prediction, source="local")

    except Exception as e:
        st.error(f"⛔ Erro inesperado: {str(e)}")
        st.error("Por favor, tente com outra imagem.")

# Rodapé
st.markdown("---")
st.caption("""
ℹ️ PancsID v1.0 | Modelo desenvolvido por michaufsc27 | Aplicação por [seu nome]
""")
