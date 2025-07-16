import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image
from huggingface_hub import hf_hub_download
import os
import traceback
import h5py

# --- CONFIGURAÇÃO ---
API_TOKEN = st.secrets.get("HF_TOKEN", "")
REPO_ID = "michaufsc27/pancs_modelo"
MODEL_FILENAME = "modelo_pancs.h5"
CLASSES_FILENAME = "labels.txt"

# --- FUNÇÕES PRINCIPAIS ---

def criar_arquitetura_compativel(num_classes):
    """Cria uma arquitetura compatível com o modelo original"""
    inputs = tf.keras.Input(shape=(224, 224, 3), name='input_layer')
    
    # Camadas convolucionais (ajustar conforme arquitetura real)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    
    # Camada densa final com número correto de classes
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)

@st.cache_resource(show_spinner="🔄 Carregando modelo...")
def carregar_modelo():
    try:
        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=MODEL_FILENAME,
            token=API_TOKEN if API_TOKEN else None
        )
        
        # Carrega as classes primeiro para saber o número de saídas necessárias
        classes = carregar_classes()
        num_classes = len(classes)
        
        # 1. Tentativa de carregamento direto
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            return model
        except Exception as e:
            st.warning(f"Carregamento direto falhou: {str(e)}. Tentando abordagem alternativa...")
            
            # 2. Abordagem alternativa - reconstruir arquitetura
            new_model = criar_arquitetura_compativel(num_classes)
            
            # Carregar pesos manualmente, ignorando incompatibilidades
            new_model.load_weights(model_path, by_name=True, skip_mismatch=True)
            
            st.warning("Algumas camadas podem não ter carregado corretamente. Verifique o desempenho.")
            new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            return new_model
            
    except Exception as e:
        st.error(f"Erro crítico ao carregar modelo: {str(e)}")
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
    except:
        return [f"Classe {i}" for i in range(213)]  # Fallback baseado no shape esperado

def preprocess_image(image, target_size=(224, 224)):
    """Pré-processamento padrão para modelos CNN"""
    img = image.convert("RGB").resize(target_size)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Adiciona dimensão do batch
    return img_array / 255.0  # Normalização

def mostrar_resultados(predictions, classes):
    """Exibe os resultados de forma organizada"""
    st.subheader("🔍 Resultados da Identificação")
    top_indices = np.argsort(predictions[0])[::-1][:5]  # Top 5 resultados
    
    for i, idx in enumerate(top_indices):
        confianca = float(predictions[0][idx])
        label = classes[idx]
        
        # Barra de progresso colorida
        st.metric(label=f"{i+1}º: {label}", value=f"{confianca:.2%}")
        st.progress(confianca)

# --- INTERFACE STREAMLIT ---
def main():
    st.set_page_config(
        page_title="Identificador de PANCs",
        page_icon="🌿",
        layout="centered"
    )
    
    st.title("🌿 Identificador de PANCs")
    st.write("Envie uma imagem de planta para identificação")
    
    uploaded_file = st.file_uploader("Selecione uma imagem...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Imagem enviada", use_container_width=True)
            
            if st.button("Identificar Planta"):
                with st.spinner("Processando..."):
                    model = carregar_modelo()
                    classes = carregar_classes()
                    input_data = preprocess_image(image)
                    predictions = model.predict(input_data, verbose=0)
                    mostrar_resultados(predictions, classes)
                    
        except Exception as e:
            st.error("Erro ao processar a imagem")
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
