import streamlit as st
import pandas as pd
import requests
from io import BytesIO
from PIL import Image
import os

# ============================================
# CONFIGURAÇÕES GLOBAIS
# ============================================
st.set_page_config(
    page_title="Identificador de PANCs",
    layout="centered",
    page_icon="🌿",
    menu_items={
        'Get Help': 'https://github.com/seu-usuario/seu-repositorio',
        'Report a bug': "https://github.com/seu-usuario/seu-repositorio/issues",
        'About': "### Aplicativo para identificação de Plantas Alimentícias Não Convencionais"
    }
)

# ============================================
# CONSTANTES E CONFIGURAÇÕES
# ============================================
CSV_PATH = "data/panc_formatado_limpo.csv"  # Caminho relativo considerando estrutura no GitHub
API_KEY = st.secrets.get("API_KEY", "2b10StWKYdMZlXgbScMsBcRO")  # Usando secrets para API key
API_URL = "https://my-api.plantnet.org/v2/identify/all"

# ============================================
# FUNÇÕES PRINCIPAIS
# ============================================
@st.cache_data(ttl=3600)  # Cache por 1 hora
def load_data():
    """Carrega e prepara os dados das PANCs"""
    try:
        # Verifica se arquivo existe
        if not os.path.exists(CSV_PATH):
            raise FileNotFoundError(f"Arquivo não encontrado em: {CSV_PATH}")
        
        # Carrega CSV com tratamento de encoding
        df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
        
        if df.empty:
            st.warning("O arquivo CSV está vazio!")
            return pd.DataFrame()

        # Pré-processamento
        df = preprocess_data(df)
        return df

    except Exception as e:
        st.error(f"Erro ao carregar dados: {str(e)}")
        return pd.DataFrame()

def preprocess_data(df):
    """Realiza transformações nos dados"""
    # Corrige URLs
    url_cols = ['url', 'imagem']
    for col in url_cols:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: f"https:{x}" if x and isinstance(x, str) and not x.startswith('http') else x
            )
    return df

def display_plant_info(row):
    """Exibe informações detalhadas sobre uma planta"""
    st.subheader(f"🌱 {row['nome_cientifico']}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**📋 Informações básicas**")
        st.write(f"**Nomes populares:** {row.get('nomes_populares', 'N/A')}")
        st.write(f"**Família:** {row.get('familia', 'N/A')}")
        st.write(f"**Hábito:** {row.get('habito', 'N/A')}")
    
    with col2:
        st.markdown("**🍽️ Uso culinário**")
        st.write(f"**Parte comestível:** {row.get('parte_comestivel', 'N/A')}")
        if pd.notna(row.get('uso_culinario')):
            st.markdown("**Receitas:**")
            st.write(row['uso_culinario'])
    
    display_plant_image(row.get('imagem'))
    
    if pd.notna(row.get('url')):
        st.markdown(f"🔗 [Mais informações]({row['url']}")

def display_plant_image(image_url):
    """Exibe imagem da planta com tratamento de erros"""
    if pd.notna(image_url):
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            st.image(img, caption="Imagem da planta", use_column_width=True)
        except Exception as e:
            st.warning(f"Não foi possível carregar a imagem: {str(e)}")

# ============================================
# INTERFACE DO USUÁRIO
# ============================================
def main():
    # Cabeçalho
    st.title("🌿 Identificador de PANCs")
    
    # Seleção de idioma
    language = st.radio(
        "Idioma/Language",
        ["Português", "Español"],
        horizontal=True
    )
    lang = 'pt' if language == "Português" else 'es'
    
    # Carrega dados
    df = load_data()
    if df.empty:
        st.error("Dados não puderam ser carregados. Verifique o arquivo CSV.")
        st.stop()
    
    # Modo de operação
    mode = st.radio(
        "Modo de uso",
        ["📷 Identificar por imagem", "🔎 Buscar por nome"],
        horizontal=True
    )
    
    if mode == "📷 Identificar por imagem":
        identify_by_image(df)
    else:
        search_by_name(df)

def identify_by_image(df):
    """Interface para identificação por imagem"""
    st.markdown("### 📷 Identifique plantas por foto")
    
    uploaded_file = st.file_uploader(
        "Selecione uma imagem (JPG, PNG)",
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file:
        st.image(uploaded_file, caption="Sua imagem", use_column_width=True)
        
        if st.button("🔍 Identificar Planta", type="primary"):
            with st.spinner("Analisando a planta..."):
                try:
                    result = identify_plant_api(uploaded_file)
                    process_api_result(result, df)
                except Exception as e:
                    st.error(f"Falha na identificação: {str(e)}")

def identify_plant_api(image_file):
    """Faz requisição à API PlantNet"""
    files = {"images": (image_file.name, image_file.getvalue())}
    data = {"organs": ["leaf", "flower", "fruit"]}
    
    response = requests.post(
        f"{API_URL}?api-key={API_KEY}",
        files=files,
        data=data,
        timeout=15
    )
    response.raise_for_status()
    return response.json()

def process_api_result(result, df):
    """Processa o resultado da API e exibe na interface"""
    if "results" in result and result["results"]:
        best_match = result["results"][0]
        sci_name = best_match["species"]["scientificNameWithoutAuthor"]
        confidence = best_match["score"]
        
        st.success(f"**Planta identificada:** {sci_name} (confiança: {confidence:.0%})")
        
        # Busca no banco de PANCs
        panc_match = df[df['nome_cientifico'].str.contains(sci_name, case=False, na=False)]
        
        if not panc_match.empty:
            st.success("✅ Esta planta está no nosso banco de PANCs!")
            display_plant_info(panc_match.iloc[0])
        else:
            st.warning("ℹ️ Planta identificada não consta em nosso banco.")
            st.info("""Nem todas as plantas são comestíveis. 
                   Consulte um especialista antes de consumir.""")
    else:
        st.warning("Não foi possível identificar a planta com segurança.")

def search_by_name(df):
    """Interface para busca por nome"""
    st.markdown("### 🔎 Busca por nome")
    
    search_term = st.text_input(
        "Digite o nome da planta:",
        placeholder="Ex: ora-pro-nobis, taioba, etc."
    )
    
    if search_term:
        results = df[
            df["nome_cientifico"].str.contains(search_term, case=False, na=False) |
            df["nomes_populares"].str.contains(search_term, case=False, na=False)
        ]
        
        if results.empty:
            st.warning("Nenhuma PANC encontrada com esse nome.")
            st.info("Dica: Tente nomes científicos ou populares alternativos.")
        else:
            st.success(f"Encontradas {len(results)} PANCs:")
            for _, row in results.iterrows():
                display_plant_info(row)
                st.divider()

# ============================================
# EXECUÇÃO PRINCIPAL
# ============================================
if __name__ == "__main__":
    main()
