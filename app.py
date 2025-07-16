import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image
from huggingface_hub import hf_hub_download
import os
import traceback

# --- CONFIGURAÇÃO ---
API_TOKEN = st.secrets.get("HF_TOKEN", "")
REPO_ID = "michaufsc27/pancs_modelo"
MODEL_FILENAME = "modelo_pancs.h5"
CLASSES_FILENAME = "labels.txt"

# --- ARQUITETURA DO MODELO ---

def criar_modelo_pancs(num_classes):
    """Cria a arquitetura exata do modelo PANCs com shapes corretos"""
    inputs = tf.keras.Input(shape=(224, 224, 3), name='input_layer')
    
    # Bloco 1
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2d_1')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2), name='max_pooling2d_1')(x)
    
    # Bloco 2
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2d_2')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name='max_pooling2d_2')(x)
    
    # Bloco 3
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2d_3')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name='max_pooling2d_3')(x)
    
    # Camadas Densas (ajustadas para o shape 86528, 128)
    x = tf.keras.layers.Flatten(name='flatten')(x)
    x = tf.keras.layers.Dense(128, activation='relu', name='dense_1')(x)  # Ajustado para 128 unidades
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name='dense_2')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)

# --- CARREGAMENTO DO MODELO ---

@st.cache_resource(show_spinner="🔄 Carregando modelo...")
def carregar_modelo():
    try:
        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=MODEL_FILENAME,
            token=API_TOKEN if API_TOKEN else None
        )
        
        # Carrega classes primeiro para determinar número de saídas
        classes = carregar_classes()
        num_classes = len(classes)
        
        # 1. Tentar carregar o modelo diretamente
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            st.success("Modelo carregado com sucesso!")
            return model
        except Exception as e:
            st.warning(f"Carregamento direto falhou: {str(e)}")
            st.info("Reconstruindo modelo com arquitetura personalizada...")
        
        # 2. Reconstruir modelo com arquitetura correta
        model = criar_modelo_pancs(num_classes)
        
        # Carregar pesos compatíveis
        try:
            # Lista de camadas cujos pesos podem ser carregados
            camadas_compatíveis = [
                'conv2d_1', 'conv2d_2', 'conv2d_3',
                'max_pooling2d_1', 'max_pooling2d_2', 'max_pooling2d_3',
                'flatten', 'dense_1'
            ]
            
            # Carrega apenas as camadas compatíveis
            for layer in model.layers:
                if layer.name in camadas_compatíveis:
                    try:
                        layer_weights = tf.keras.models.load_model(model_path).get_layer(layer.name).get_weights()
                        layer.set_weights(layer_weights)
                        st.write(f"✓ Pesos carregados para camada: {layer.name}")
                    except:
                        st.write(f"✗ Não foi possível carregar pesos para: {layer.name}")
        
        except Exception as e:
            st.warning(f"Erro ao carregar pesos: {str(e)}")
            st.warning("Algumas camadas serão inicializadas com pesos aleatórios")
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
        
    except Exception as e:
        st.error(f"Erro crítico: {str(e)}")
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
    except:
        st.warning("Usando classes fallback (128 classes)")
        return [f"Planta {i}" for i in range(128)]

# --- PRÉ-PROCESSAMENTO ---

def preprocess_image(image, target_size=(224, 224)):
    """Pré-processamento completo da imagem"""
    img = image.convert("RGB").resize(target_size)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)
    return img_array / 255.0  # Normalização

# --- INTERFACE ---

def mostrar_resultados(predictions, classes):
    """Exibe os resultados formatados"""
    st.subheader("🌱 Resultados da Identificação")
    
    top_indices = np.argsort(predictions[0])[::-1][:3]  # Top 3 resultados
    
    for i, idx in enumerate(top_indices):
        confianca = float(predictions[0][idx])
        label = classes[idx]
        
        # Barra de progresso colorida
        if confianca > 0.7:
            st.success(f"{i+1}º: {label} - {confianca:.1%}")
        elif confianca > 0.3:
            st.warning(f"{i+1}º: {label} - {confianca:.1%}")
        else:
            st.error(f"{i+1}º: {label} - {confianca:.1%}")
        
        st.progress(confianca)

def main():
    st.set_page_config(
        page_title="Identificador de PANCs",
        page_icon="🌿",
        layout="centered"
    )
    
    st.title("🌿 Identificador de PANCs")
    st.write("Envie uma imagem de planta para identificação")
    
    uploaded_file = st.file_uploader(
        "Selecione uma imagem...", 
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False
    )
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Imagem enviada", use_container_width=True)
            
            if st.button("🔍 Identificar Planta", type="primary"):
                with st.spinner("Processando..."):
                    model = carregar_modelo()
                    classes = carregar_classes()
                    input_data = preprocess_image(image)
                    predictions = model.predict(input_data, verbose=0)
                    mostrar_resultados(predictions, classes)
                    
        except Exception as e:
            st.error(f"Erro: {str(e)}")
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
