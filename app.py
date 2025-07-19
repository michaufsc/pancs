import streamlit as st
import pandas as pd
import requests
import os
from io import BytesIO, StringIO
from PIL import Image

# ====== CONFIGURAÇÃO ======
API_KEY = "2b10StWKYdMZlXgbScMsBcRO"
API_URL = "https://my-api.plantnet.org/v2/identify/all"

# ====== CARREGAR DATASET - VERSÃO SUPER ROBUSTA ======
@st.cache_data
def load_data(uploaded_file=None):
    # Tentar diferentes métodos de leitura
    read_attempts = [
        {'encoding': 'utf-8-sig', 'engine': 'python', 'on_bad_lines': 'warn'},
        {'encoding': 'latin1', 'engine': 'python', 'sep': ';'},
        {'encoding': 'utf-8', 'engine': 'python', 'sep': None}
    ]
    
    for attempt in read_attempts:
        try:
            if uploaded_file is not None:
                # Se foi feito upload de um arquivo
                df = pd.read_csv(uploaded_file, **attempt)
            else:
                # Tentar arquivos locais
                for filename in ["plantas_panc_limpo.csv", "plantas_panc.csv"]:
                    if os.path.exists(filename):
                        df = pd.read_csv(filename, **attempt)
                        break
                else:
                    continue
            
            # Padronizar colunas (caso o arquivo tenha formato diferente)
            required_cols = ['nome_cientifico', 'nomes_populares', 'familia', 
                           'habito', 'parte_comestivel', 'uso_culinario', 'url']
            
            # Manter apenas as colunas necessárias
            df = df[[col for col in required_cols if col in df.columns]].copy()
            
            # Adicionar colunas faltantes
            for col in required_cols:
                if col not in df.columns:
                    df[col] = None
            
            return df
            
        except Exception as e:
            st.warning(f"Tentativa com {attempt} falhou: {str(e)[:100]}...")
            continue
    
    st.error("Não foi possível ler nenhum arquivo de dados.")
    return pd.DataFrame()

# ====== INTERFACE PARA UPLOAD ======
st.set_page_config(page_title="Identificador de Plantas Comestíveis", layout="centered")

st.title("🌿 Identificador de Plantas Comestíveis (PANCs)")

# Opção para upload de arquivo
uploaded_file = st.file_uploader("Carregue seu arquivo de plantas (CSV)", type="csv")

# Carregar dados (do upload ou arquivo local)
df = load_data(uploaded_file)

if df.empty:
    st.error("""
    **Não foi possível carregar os dados. Por favor:**
    1. Verifique se o arquivo existe no diretório
    2. Ou faça upload de um arquivo CSV válido
    3. O arquivo deve conter pelo menos: nome_cientifico, nomes_populares, familia, habito, parte_comestivel, uso_culinario, url
    """)
    st.stop()

# ====== RESTANTE DO SEU CÓDIGO ORIGINAL ======
# [Seu código existente para a interface e funcionalidades]
idioma = st.selectbox("Escolha o idioma / Choose language", ["Português", "Español"])
lang = 'pt' if idioma == "Português" else 'es'

# ... continue com o resto do seu aplicativo ...
