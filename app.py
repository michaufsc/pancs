import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import sys
from huggingface_hub import hf_hub_download

# ======================
# CONFIGURAÇÕES
# ======================
REPO_ID = "michaufsc27/pancs_modelo"
MODEL_FILENAME = "modelo_pancs.h5"
CLASSES_FILE = "classes.txt"

# ======================
# FUNÇÕES PRINCIPAIS
# ======================
@st.cache_resource(show_spinner="🔍 Carregando modelo...")
def load_model():
    try:
        # Baixa o modelo garantindo a versão mais recente
        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=MODEL_FILENAME,
            force_download=True,
            cache_dir=".",
            local_dir_use_symlinks=False
        )
        
        # Carrega o modelo ignorando pequenas incompatibilidades
        model = tf.keras.models.load_model(model_path, compile=False)
        st.success("✅ Modelo carregado com sucesso!")
        return model
        
    except Exception as e:
        st.error(f"❌ Falha ao carregar modelo: {str(e)}")
        st.stop()

def load_classes():
    try:
        with open(CLASSES_FILE, "r", encoding="utf-8") as f:
            classes = [line.strip() for line in f.readlines()]
        return classes
    except Exception as e:
        st.error(f"❌ Erro ao carregar classes: {str(e)}")
        st.stop()

# ======================
# INTERFACE
# ======================
st.set_page_config(page_title="PancsID", page_icon="🌿")
st.title("🌿 PancsID - Identificador de PANC")

# Barra lateral com informações técnicas
st.sidebar.header("ℹ️ Informações Técnicas")
st.sidebar.code(f"""
Python: {sys.version.split()[0]}
TensorFlow: {tf.__version__}
Numpy: {np.__version__}
""")

# ======================
# CARREGAMENTO INICIAL
# ======================
model = load_model()
class_names = load_classes()

# ======================
# UPLOAD DE IMAGEM
# ======================
uploaded_file = st.file_uploader(
    "Envie uma imagem da planta (JPG, PNG)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False
)

if uploaded_file:
    try:
        # Pré-processamento
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Imagem enviada", use_column_width=True)
        
        # Redimensiona e normaliza (igual ao treino no Colab)
        img = img.resize((224, 224))  # Altere conforme seu modelo
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) / 255.0
        
        # Predição
        with st.spinner("🔮 Analisando a planta..."):
            predictions = model.predict(img_array)
            scores = tf.nn.softmax(predictions[0])
            predicted_class = class_names[np.argmax(scores)]
            confidence = 100 * np.max(scores)
        
        # Resultados
        st.success(f"**Identificação:** {predicted_class}")
        st.metric("Confiança", f"{confidence:.2f}%")
        
        # Debug (opcional)
        st.expander("Detalhes técnicos").write(f"""
        ```python
        Shape da entrada: {img_array.shape}
        Classe predita: {np.argmax(scores)}
        Todas as classes: {class_names}
        """)
        
    except Exception as e:
        st.error(f"⚠️ Erro ao processar imagem: {str(e)}")
        st.write("Dicas para correção:")
        st.markdown("""
        - Verifique se a imagem é válida
        - Confira o formato (RGB, não transparente)
        - Tente outra imagem
        """)

# ======================
# RODAPÉ
# ======================
st.markdown("---")
st.caption("Desenvolvido com ❤️ por [seu nome] | Modelo treinado no Google Colab")
