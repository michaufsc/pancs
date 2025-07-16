import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image
from huggingface_hub import hf_hub_download
import os
import traceback

# --- CONFIGURAÇÕES ---
API_TOKEN = st.secrets.get("HF_TOKEN", "")
REPO_ID = "michaufsc27/pancs_modelo"
MODEL_FILENAME = "modelo_pancs.h5"
CLASSES_FILENAME = "labels.txt"

# --- ARQUITETURA DO MODELO OTIMIZADA ---
def construir_modelo_otimizado(num_classes):
    """Cria uma arquitetura mais eficiente para classificação de plantas"""
    inputs = tf.keras.Input(shape=(224, 224, 3))
    
    # Camada de pré-processamento
    x = tf.keras.layers.Rescaling(1./255)(inputs)
    
    # Blocos convolucionais com BatchNorm e Dropout
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    # Camadas densas
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    # Saída
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Compilação com otimização
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# --- CARREGAMENTO INTELIGENTE ---
@st.cache_resource(show_spinner="🔄 Carregando e otimizando modelo...")
def carregar_modelo_otimizado():
    try:
        # Carrega classes primeiro
        classes = carregar_classes()
        num_classes = len(classes)
        
        # Cria novo modelo otimizado
        novo_modelo = construir_modelo_otimizado(num_classes)
        
        # Tenta carregar pesos compatíveis do modelo antigo
        try:
            model_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=MODEL_FILENAME,
                token=API_TOKEN if API_TOKEN else None
            )
            
            # Carrega apenas as camadas compatíveis
            modelo_antigo = tf.keras.models.load_model(model_path, compile=False)
            
            for layer in novo_modelo.layers:
                try:
                    if layer.name in [l.name for l in modelo_antigo.layers]:
                        layer.set_weights(modelo_antigo.get_layer(layer.name).get_weights())
                        st.success(f"✓ Transferidos pesos para: {layer.name}")
                except:
                    st.warning(f"✗ Pesos incompatíveis para: {layer.name}")
        
        except Exception as e:
            st.warning(f"Não foi possível carregar pesos existentes: {str(e)}")
            st.info("Modelo será treinado do zero quando necessário")
        
        return novo_modelo
        
    except Exception as e:
        st.error(f"Erro crítico: {str(e)}")
        raise RuntimeError(f"Falha ao inicializar modelo: {e}")

# --- PRÉ-PROCESSAMENTO AVANÇADO ---
def preprocessamento_avancado(image, target_size=(224, 224)):
    """Pré-processamento com aumento de dados em tempo real"""
    img = image.convert("RGB").resize(target_size)
    
    # Conversão para array e normalização
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)
    
    # Aplicar transformações de aumento de dados
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    return datagen.flow(img_array, batch_size=1).next() / 255.0

# --- INTERFACE APRIMORADA ---
def main():
    st.set_page_config(
        page_title="Identificador de PANCs - Versão Avançada",
        page_icon="🌿",
        layout="wide"
    )
    
    st.title("🌿 Identificador Inteligente de PANCs")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Envio de Imagem")
        uploaded_file = st.file_uploader(
            "Selecione uma imagem de planta...",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Imagem enviada", use_container_width=True)
            
            if st.button("🔍 Identificar Planta", type="primary"):
                with st.spinner("Analisando com modelo avançado..."):
                    try:
                        model = carregar_modelo_otimizado()
                        classes = carregar_classes()
                        input_data = preprocessamento_avancado(image)
                        predictions = model.predict(input_data, verbose=0)
                        
                        with col2:
                            mostrar_resultados_detalhados(predictions, classes)
                            
                    except Exception as e:
                        st.error(f"Erro durante análise: {str(e)}")
    
    with col2:
        st.subheader("Detalhes Técnicos")
        if st.checkbox("Mostrar informações do modelo"):
            try:
                model = carregar_modelo_otimizado()
                st.text("Arquitetura do modelo:")
                st.text(model.summary())
                
                # Mostrar exemplos de pré-processamento
                if uploaded_file:
                    st.subheader("Visualização do Pré-processamento")
                    img_array = preprocessamento_avancado(Image.open(uploaded_file))
                    st.image(img_array[0], caption="Imagem após pré-processamento", clamp=True)
            except:
                st.warning("Modelo ainda não carregado")

def mostrar_resultados_detalhados(predictions, classes):
    """Visualização avançada dos resultados"""
    st.subheader("📊 Análise Detalhada")
    
    # Top 5 predições
    top_k = 5
    top_indices = np.argsort(predictions[0])[::-1][:top_k]
    
    # Gráfico de barras
    chart_data = {
        "Planta": [classes[i] for i in top_indices],
        "Confiança": [float(predictions[0][i]) for i in top_indices]
    }
    
    st.bar_chart(chart_data, x="Planta", y="Confiança", height=300)
    
    # Tabela detalhada
    st.write("Detalhes das predições:")
    for i, idx in enumerate(top_indices):
        conf = float(predictions[0][idx])
        st.progress(conf, text=f"{i+1}º: {classes[idx]} - {conf:.2%}")

if __name__ == "__main__":
    main()
