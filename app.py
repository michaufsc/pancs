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

def criar_arquitetura_padrao(num_classes):
    """Cria uma arquitetura CNN genérica compatível"""
    inputs = tf.keras.Input(shape=(224, 224, 3), name='input_layer')
    
    # Bloco convolucional 1
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    
    # Bloco convolucional 2
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    
    # Bloco convolucional 3
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    
    # Camadas densas
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)

@st.cache_resource(show_spinner="🔄 Carregando modelo...")
def carregar_modelo():
    try:
        # Verifica se o modelo está localmente
        model_path = "modelo_pancs.h5"
        
        if not os.path.exists(model_path):
            model_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=MODEL_FILENAME,
                token=API_TOKEN if API_TOKEN else None
            )
        
        # Primeiro carrega as classes para determinar o número de saídas
        classes = carregar_classes()
        num_classes = len(classes)
        
        # Estratégia 1: Tentar carregar com safe_mode=False
        try:
            model = tf.keras.models.load_model(
                model_path,
                compile=False,
                safe_mode=False
            )
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            st.success("Modelo carregado com sucesso!")
            return model
        except Exception as e:
            st.warning(f"Falha no carregamento direto: {str(e)}")
        
        # Estratégia 2: Carregar apenas os pesos em arquitetura nova
        st.info("Criando nova arquitetura e carregando pesos...")
        new_model = criar_arquitetura_padrao(num_classes)
        
        try:
            # Tenta carregar os pesos diretamente
            new_model.load_weights(model_path)
            st.success("Pesos carregados com sucesso na nova arquitetura!")
        except Exception as e:
            st.warning(f"Não foi possível carregar todos os pesos: {str(e)}")
            st.warning("O modelo será inicializado com pesos aleatórios")
        
        new_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return new_model
        
    except Exception as e:
        st.error(f"Erro crítico: {str(e)}")
        st.error("""
        Soluções sugeridas:
        1. Verifique se o arquivo do modelo está intacto
        2. Atualize o TensorFlow para a versão mais recente
        3. Recrie o modelo usando Input(shape=...) em vez de batch_shape
        """)
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
            classes = [linha.strip() for linha in f if linha.strip()]
            if not classes:
                raise ValueError("Arquivo de classes vazio")
            return classes
    except Exception as e:
        st.warning(f"Usando classes fallback. Erro: {str(e)}")
        return [f"Classe {i}" for i in range(213)]  # Fallback baseado no erro anterior

def preprocess_image(image, target_size=(224, 224)):
    """Pré-processamento robusto da imagem"""
    try:
        img = image.convert("RGB").resize(target_size)
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, axis=0)
        return img_array / 255.0  # Normalização
    except Exception as e:
        raise ValueError(f"Erro ao processar imagem: {str(e)}")

def mostrar_resultados(predictions, classes):
    """Exibe resultados de forma clara"""
    st.subheader("🌱 Resultados da Identificação")
    
    # Ordena as predições
    top_indices = np.argsort(predictions[0])[::-1][:5]  # Top 5
    
    for i, idx in enumerate(top_indices):
        confianca = float(predictions[0][idx])
        label = classes[idx]
        
        # Cria colunas para organizar os resultados
        col1, col2 = st.columns([1, 4])
        with col1:
            st.metric(label=f"{i+1}º", value=f"{confianca:.1%}")
        with col2:
            st.progress(confianca)
            st.caption(label)

# --- INTERFACE STREAMLIT ---
def main():
    st.set_page_config(
        page_title="Identificador de PANCs",
        page_icon="🌿",
        layout="centered"
    )
    
    st.title("🌿 Identificador de PANCs")
    st.markdown("""
    Identifique Plantas Alimentícias Não Convencionais através de imagens.
    *Funciona melhor com fotos claras das folhas ou da planta inteira.*
    """)
    
    uploaded_file = st.file_uploader(
        "Selecione uma imagem da planta...",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False
    )
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Imagem enviada", use_container_width=True)
            
            if st.button("🔍 Identificar Planta", type="primary"):
                with st.spinner("Analisando a planta..."):
                    try:
                        model = carregar_modelo()
                        classes = carregar_classes()
                        input_data = preprocess_image(image)
                        predictions = model.predict(input_data, verbose=0)
                        mostrar_resultados(predictions, classes)
                    except Exception as e:
                        st.error(f"Erro durante a identificação: {str(e)}")
                        st.code(traceback.format_exc())
                        
        except Exception as e:
            st.error(f"Erro ao processar a imagem: {str(e)}")

if __name__ == "__main__":
    main()
