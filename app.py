import streamlit as st
import pandas as pd
import requests
import os
from io import BytesIO
from PIL import Image

# ====== CONFIGURAÇÃO ======
API_KEY = "2b10StWKYdMZlXgbScMsBcRO"
API_URL = "https://my-api.plantnet.org/v2/identify/all"

# ====== CARREGAR DATASET - VERSÃO ROBUSTA ======
@st.cache_data
def load_data():
    try:
        # Tenta carregar o arquivo limpo primeiro
        if os.path.exists("plantas_panc_limpo.csv"):
            df = pd.read_csv("plantas_panc_limpo.csv", encoding='utf-8-sig')
            st.success("Dados carregados do arquivo limpo")
            return df
        
        # Fallback para o arquivo original se necessário
        elif os.path.exists("plantas_panc.csv"):
            st.warning("Usando arquivo original - recomendo gerar a versão limpa")
            
            # Leitura robusta do arquivo original
            try:
                df = pd.read_csv("plantas_panc.csv", 
                                encoding='utf-8-sig',
                                engine='python',
                                on_bad_lines='warn')
                
                # Limpeza básica se usar o arquivo original
                df = df.iloc[:, :7]  # Pega apenas as primeiras 7 colunas
                df.columns = ['nome_cientifico', 'nomes_populares', 'familia', 
                            'habito', 'parte_comestivel', 'uso_culinario', 'url']
                return df
                
            except Exception as e:
                st.error(f"Erro ao ler arquivo original: {e}")
                return pd.DataFrame()
                
        else:
            st.error("Nenhum arquivo de dados encontrado!")
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Erro inesperado: {e}")
        return pd.DataFrame()

df = load_data()

# Verifica se temos dados para continuar
if df.empty:
    st.error("Não foi possível carregar os dados. Verifique os arquivos.")
    st.stop()

# ====== INTERFACE DO APP ======
st.set_page_config(page_title="Identificador de Plantas Comestíveis", layout="centered")

# Restante do seu código continua igual a partir daqui...
# [Seu código existente para a interface e funções]
