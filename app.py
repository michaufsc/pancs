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

def criar_arquitetura_customizada(input_shape=(224, 224, 3), num_classes=213):
    """Cria uma arquitetura CNN customizada compatível"""
    inputs = tf.keras.Input(shape=input_shape, name='input_layer')
    
    # Bloco 1
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    
    # Bloco 2
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    
    # Bloco 3
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    
    # Camadas fully connected
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
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
        
        classes = carregar_classes()
        num_classes = len(classes)
        
        # Estratégia 1: Tentar carregar com safe_mode=False
        try:
            model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            st.success("Modelo carregado com sucesso (método direto)!")
            return model
        except Exception as e:
            st.warning(f"Falha no carregamento direto: {str(e)}")
        
        # Estratégia 2: Reconstruir arquitetura e carregar pesos
        st.info("Tentando reconstruir arquitetura manualmente...")
        new_model = criar_arquitetura_customizada(num_classes=num_classes)
        
        # Verifica quais pesos podem ser carregados
        with h5py.File(model_path, 'r') as f:
            layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]
            st.write(f"Camadas disponíveis no arquivo: {layer_names}")
        
        # Carrega pesos compatíveis
        load_status = new_model.load_weights(model_path, by_name=True)
        
        # Mostra status do carregamento
        if load_status:
            st.warning("Aviso: Algumas camadas não carregaram corretamente:")
            for layer in new_model.layers:
                if not layer.weights:
                    st.write(f"- {layer.name}: Não carregado")
                else:
                    st.write(f"+ {layer.name}: Carregado")
        
        new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return new_model
        
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
        st.warning("Usando classes fallback (213 classes)")
        return [f"Classe {i}" for i in range(213)]

def preprocess_image(image, target_size=(224, 224)):
    """Pré-processamento completo da imagem"""
    img = image.convert("RGB").resize(target_size)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)
    return img_array / 255.0  # Normalização

def mostrar_resultados(predictions, classes):
    """Exibe resultados formatados"""
    st.subheader("📊 Resultados da Classificação")
    
    top_k = 5
    top_indices = np.argsort(predictions[0])[::-1][:top_k]
    
    cols = st.columns(top_k)
    for i, idx in enumerate(top_indices):
        with cols[i]:
            conf = float(predictions[0][idx])
            st.metric(
                label=classes[idx],
                value=f"{conf:.1%}",
                help=f"Confiança: {conf:.2%}"
            )
            st.progress(conf)

# --- INTERFACE ---
def main():
    st.set_page_config(
        page_title="Identificador de PANCs",
        page_icon="🌿",
        layout="wide"
    )
    
    st.title("🌿 Identificador de Plantas Alimentícias Não Convencionais")
    
    with st.expander("ℹ️ Como usar", expanded=True):
        st.write("""
        1. Faça upload de uma imagem de planta
        2. Clique em 'Identificar'
        3. Veja os resultados classificados por confiança
        """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Selecione uma imagem...",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
            
            if st.button("🔍 Identificar Planta", type="primary"):
                with st.spinner("Processando..."):
                    try:
                        model = carregar_modelo()
                        classes = carregar_classes()
                        input_data = preprocess_image(image)
                        predictions = model.predict(input_data, verbose=0)
                        mostrar_resultados(predictions, classes)
                    except Exception as e:
                        st.error(f"Erro durante a identificação: {str(e)}")
                        st.code(traceback.format_exc())

    with col2:
        st.subheader("📝 Informações Técnicas")
        if st.checkbox("Mostrar detalhes do modelo"):
            try:
                model = carregar_modelo()
                st.text("Arquitetura do modelo:")
                st.text(model.summary())
            except:
                st.warning("Modelo ainda não carregado")

if __name__ == "__main__":
    main()
