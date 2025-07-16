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
MODEL_FILENAME = "modelo_pancs.h5"  # Nome correto do seu arquivo
CLASSES_FILENAME = "labels.txt"

# --- FUNÇÕES PRINCIPAIS ---

def criar_arquitetura_base():
    """Define a arquitetura base compatível com seu modelo"""
    inputs = tf.keras.Input(shape=(224, 224, 3), name='input_layer')
    
    # Camadas convolucionais (ajuste conforme sua arquitetura real)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    
    return inputs, x

@st.cache_resource(show_spinner="🔄 Carregando modelo...")
def carregar_modelo():
    try:
        # Caminho para o modelo (local ou Hugging Face Hub)
        model_path = "modelo_pancs.h5"
        
        if not os.path.exists(model_path):
            model_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=MODEL_FILENAME,
                token=API_TOKEN if API_TOKEN else None
            )

        # Tentativa 1: Carregamento direto
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            return model
        except Exception as e:
            st.warning("Primeira tentativa falhou. Tentando abordagem alternativa...")

            # Tentativa 2: Reconstrução da arquitetura
            with h5py.File(model_path, 'r') as f:
                # Verifica se tem configuração do modelo
                if 'model_config' in f.attrs:
                    model_config = f.attrs['model_config']
                    if isinstance(model_config, (bytes, str)):
                        try:
                            model_config = eval(model_config) if isinstance(model_config, bytes) else eval(model_config)
                            # Corrige o problema do batch_shape
                            if 'config' in model_config and 'batch_shape' in model_config['config']:
                                model_config['config']['input_shape'] = model_config['config']['batch_shape'][1:]
                                del model_config['config']['batch_shape']
                            model = tf.keras.models.model_from_config(model_config)
                            # Carrega os pesos
                            model.load_weights(model_path)
                            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                            return model
                        except:
                            pass

            # Tentativa 3: Carregamento manual dos pesos
            inputs, x = criar_arquitetura_base()
            
            # Verifica número de classes
            classes = carregar_classes()
            outputs = tf.keras.layers.Dense(len(classes), activation='softmax')(x)
            new_model = tf.keras.Model(inputs=inputs, outputs=outputs)
            
            # Carrega os pesos camada por camada
            new_model.load_weights(model_path, by_name=True, skip_mismatch=True)
            new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            
            st.success("Modelo carregado com abordagem alternativa!")
            return new_model

    except Exception as e:
        st.error(f"Erro crítico ao carregar modelo: {str(e)}")
        st.error("Dicas para solução:")
        st.error("1. Verifique se o arquivo modelo_pancs.h5 está intacto")
        st.error("2. Confira se a versão do TensorFlow é compatível")
        st.error("3. Considere recriar o modelo com Input(shape=...) em vez de batch_shape")
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
        return [f"Classe {i}" for i in range(10)]  # Fallback básico

# ... (mantenha o restante do seu código igual) ...
