import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image
from huggingface_hub import hf_hub_download
import os
import traceback

# --- CONFIGURAÇÃO ---
API_TOKEN = st.secrets.get("HF_TOKEN", "")  # Pegando token do secrets
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"} if API_TOKEN else {}

REPO_ID = "michaufsc27/pancs_modelo"
MODEL_FILENAME = "modelo_pancs.keras"
CLASSES_FILENAME = "labels.txt"

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
            
        # Tentativa de carregamento com tratamento para a InputLayer
        try:
            model = tf.keras.models.load_model(
                model_path, 
                compile=False,
                custom_objects={'InputLayer': tf.keras.layers.InputLayer}
            )
        except Exception as e:
            st.warning("Primeira tentativa de carregamento falhou, tentando alternativa...")
            model = tf.keras.models.load_model(
                model_path,
                compile=False,
                safe_mode=False
            )
            
        model.compile()
        return model
    except Exception as e:
        st.error(f"Erro detalhado ao carregar modelo: {str(e)}")
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
        img = image.convert("RGB").resize(target_size)
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        raise RuntimeError(f"Erro ao processar a imagem: {e}")

def mostrar_resultados(predictions, classes):
    top_indices = np.argsort(predictions[0])[::-1][:3]  # Top 3 previsões
    st.subheader("Resultados:")
    
    for i, idx in enumerate(top_indices):
        confidence = float(predictions[0][idx])
        predicted_label = classes[idx]
        
        # Barra de progresso para visualização da confiança
        st.progress(confidence)
        st.write(f"{i+1}º: **{predicted_label}** - {confidence:.2%}")

# --- INTERFACE PRINCIPAL ---
def main():
    st.set_page_config(
        page_title="Identificador de PANCs", 
        layout="centered",
        page_icon="🌿"
    )
    
    st.title("🌿 Identificador de Plantas Alimentícias Não Convencionais (PANCs)")
    st.markdown("""
    Envie uma imagem de uma planta para identificar a espécie.
    *Plantas Alimentícias Não Convencionais (PANCs)* são espécies vegetais com potencial alimentício,
    mas que não são amplamente consumidas ou conhecidas.
    """)

    with st.expander("ℹ️ Como usar"):
        st.write("""
        1. Clique em "Escolha uma imagem" ou arraste uma foto
        2. Selecione uma imagem clara da planta
        3. Clique no botão "Identificar"
        4. Veja os resultados da classificação
        """)

    uploaded_file = st.file_uploader(
        "📷 Escolha uma imagem...", 
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Imagem enviada", use_container_width=True)  # Atualizado aqui

            if st.button("🔍 Identificar", type="primary"):
                with st.spinner("Processando imagem..."):
                    try:
                        model = carregar_modelo()
                        classes = carregar_classes()
                        input_data = preprocess_image(image)
                        
                        with st.spinner("Classificando..."):
                            predictions = model.predict(input_data, verbose=0)
                        
                        mostrar_resultados(predictions, classes)
                        
                    except Exception as e:
                        st.error("⚠️ Erro durante a classificação:")
                        st.code(traceback.format_exc())

        except Exception as e:
            st.error("⚠️ Erro ao processar a imagem:")
            st.code(traceback.format_exc())

# --- EXECUÇÃO ---
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("⚠️ Erro inesperado no aplicativo:")
        st.code(traceback.format_exc())
