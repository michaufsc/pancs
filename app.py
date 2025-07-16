import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image
from huggingface_hub import hf_hub_download
import os
import traceback

# --- CONFIGURAÇÃO ---
API_TOKEN = st.secrets.get("HF_TOKEN", "")  # Token do Hugging Face
REPO_ID = "michaufsc27/pancs_modelo"  # Repositório do modelo
CLASSES_FILENAME = "labels.txt"  # Arquivo com as classes

# --- FUNÇÕES ---

@st.cache_resource(show_spinner="🔄 Carregando modelo...")
def carregar_modelo():
    try:
        # Verifica se temos um modelo local
        model_path = "modelo_pancs.h5"
        
        # Se não existir local, baixa do Hugging Face Hub
        if not os.path.exists(model_path):
            model_path = hf_hub_download(
                repo_id=REPO_ID,
                filename="modelo_pancs.h5",
                token=API_TOKEN if API_TOKEN else None
            )
        
        # Carrega o modelo H5
        model = tf.keras.models.load_model(
            model_path,
            compile=False
        )
        
        # Compila o modelo (ajuste conforme sua configuração original)
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {str(e)}")
        st.error("Verifique se o arquivo model.h5 está correto")
        raise RuntimeError(f"Falha ao carregar modelo: {e}")

@st.cache_resource
def carregar_classes():
    try:
        classes_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=CLASSES_FILENAME,
            token=API_TOKEN if API_TOKEN else None
        )
        with open(classes_path, "r", encoding="utf-8") as f:
            return [linha.strip() for linha in f if linha.strip()]
    except Exception as e:
        st.error(f"Erro ao carregar classes: {str(e)}")
        return ["Classe 1", "Classe 2"]  # Fallback básico

def preprocess_image(image, target_size=(224, 224)):
    """Pré-processamento padrão para modelos CNN"""
    img = image.convert("RGB").resize(target_size)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Adiciona dimensão do batch
    return tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

def mostrar_resultados(predictions, classes):
    """Exibe os resultados de forma organizada"""
    st.subheader("🔍 Resultados da Identificação")
    
    # Ordena as predições
    top_indices = np.argsort(predictions[0])[::-1][:3]  # Top 3
    
    for i, idx in enumerate(top_indices):
        confianca = float(predictions[0][idx])
        label = classes[idx]
        
        # Barra de progresso colorida
        if confianca > 0.7:
            st.success(f"{i+1}º: {label} - {confianca:.2%}")
        elif confianca > 0.3:
            st.warning(f"{i+1}º: {label} - {confianca:.2%}")
        else:
            st.error(f"{i+1}º: {label} - {confianca:.2%}")
        
        st.progress(confianca)

# --- INTERFACE ---
def main():
    st.set_page_config(
        page_title="Identificador de PANCs",
        page_icon="🌿",
        layout="centered"
    )
    
    st.title("🌿 Identificador de PANCs")
    st.markdown("""
    Identifique Plantas Alimentícias Não Convencionais (PANCs) através de imagens.
    Envie uma foto da planta para análise.
    """)
    
    with st.expander("ℹ️ Instruções"):
        st.write("""
        1. Clique em "Escolher arquivo" para enviar uma imagem
        2. Aguarde o processamento (pode demorar alguns segundos)
        3. Veja os resultados da identificação
        """)
    
    uploaded_file = st.file_uploader(
        "Selecione uma imagem da planta...",
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file is not None:
        try:
            # Carrega e exibe a imagem
            image = Image.open(uploaded_file)
            st.image(image, caption="Imagem enviada", use_container_width=True)
            
            if st.button("Identificar Planta", type="primary"):
                with st.spinner("Analisando a planta..."):
                    try:
                        # Carrega modelo e classes
                        model = carregar_modelo()
                        classes = carregar_classes()
                        
                        # Pré-processa e faz a predição
                        input_data = preprocess_image(image)
                        predictions = model.predict(input_data, verbose=0)
                        
                        # Mostra resultados
                        mostrar_resultados(predictions, classes)
                        
                    except Exception as e:
                        st.error("Erro durante a identificação")
                        st.code(traceback.format_exc())
        
        except Exception as e:
            st.error("Erro ao processar a imagem")
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
