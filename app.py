import streamlit as st
import pandas as pd
import requests
from io import BytesIO
from PIL import Image
import os
from pathlib import Path

# ======================
# CONFIGURAÇÕES INICIAIS
# ======================
st.set_page_config(
    page_title="Identificador de PANCs",
    layout="centered",
    page_icon="🌿",
    menu_items={
        'Get Help': 'https://github.com/michaufsc/pancs',
        'Report a bug': "https://github.com/michaufsc/pancs/issues",
        'About': "### App para identificação de Plantas Alimentícias Não Convencionais"
    }
)

# ======================
# CONSTANTES
# ======================
DATA_DIR = Path(__file__).parent / "data"
CSV_PATH = DATA_DIR / "plantas_panc.csv"  # Caminho corrigido
API_URL = "https://my-api.plantnet.org/v2/identify/all"

# ======================
# FUNÇÕES PRINCIPAIS
# ======================
@st.cache_data(ttl=3600)
def load_data():
    """Carrega os dados das PANCs com tratamento robusto"""
    try:
        # Verifica se a pasta data existe
        DATA_DIR.mkdir(exist_ok=True)
        
        # Verifica se o arquivo existe
        if not CSV_PATH.exists():
            st.error(f"Arquivo CSV não encontrado em: {CSV_PATH}")
            st.info("Tentando baixar automaticamente...")
            
            try:
                # Tenta baixar do GitHub
                download_csv()
                if not CSV_PATH.exists():
                    st.error("Falha ao baixar o arquivo CSV")
                    return pd.DataFrame()
            except Exception as e:
                st.error(f"Erro ao baixar: {str(e)}")
                return pd.DataFrame()

        # Tenta ler o arquivo
        try:
            df = pd.read_csv(CSV_PATH, encoding='utf-8')
            if df.empty:
                st.error("O arquivo CSV está vazio!")
                return pd.DataFrame()
            
            # Pré-processamento
            df = preprocess_data(df)
            return df
            
        except Exception as e:
            st.error(f"Erro ao ler CSV: {str(e)}")
            return pd.DataFrame()

    except Exception as e:
        st.error(f"Erro crítico: {str(e)}")
        return pd.DataFrame()

def download_csv():
    """Baixa o arquivo CSV do repositório GitHub"""
    csv_url = "https://raw.githubusercontent.com/michaufsc/pancs/main/plantas_panc.csv"
    response = requests.get(csv_url, timeout=10)
    response.raise_for_status()
    
    # Garante que a pasta existe
    DATA_DIR.mkdir(exist_ok=True)
    
    with open(CSV_PATH, 'wb') as f:
        f.write(response.content)

def preprocess_data(df):
    """Prepara os dados para uso"""
    # Corrige URLs
    for col in ['url', 'imagem']:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: f"https:{x}" if x and isinstance(x, str) and not x.startswith('http') else x
            )
    return df

def display_plant_info(row):
    """Exibe informações detalhadas da planta"""
    st.subheader(f"🌱 {row['nome_cientifico']}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**📋 Informações básicas**")
        st.write(f"**Nomes populares:** {row.get('nomes_populares', 'N/A')}")
        st.write(f"**Família:** {row.get('familia', 'N/A')}")
    
    with col2:
        st.markdown("**🍽️ Uso culinário**")
        st.write(f"**Parte comestível:** {row.get('parte_comestivel', 'N/A')}")
        if pd.notna(row.get('uso_culinario')):
            st.markdown("**Receitas:**")
            st.write(row['uso_culinario'])
    
    display_plant_image(row.get('imagem'))
    
    if pd.notna(row.get('url')):
        st.markdown(f"🔗 [Mais informações]({row['url']})")

def display_plant_image(image_url):
    """Exibe imagem da planta com tratamento de erros"""
    if pd.notna(image_url):
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            st.image(img, caption="Imagem da planta", use_column_width=True)
        except Exception as e:
            st.warning(f"Erro ao carregar imagem: {str(e)}")

# ======================
# INTERFACE PRINCIPAL
# ======================
def main():
    st.title("🌿 Identificador de PANCs")
    
    # Carrega dados
    df = load_data()
    if df.empty:
        st.error("""
        **Dados não carregados!**  
        Verifique:  
        1. Arquivo `plantas_panc.csv` existe em `data/`  
        2. O arquivo tem conteúdo válido  
        3. O formato está correto
        """)
        
        if st.button("🔄 Tentar baixar novamente"):
            try:
                download_csv()
                st.rerun()
            except Exception as e:
                st.error(f"Falha ao baixar: {str(e)}")
        return

    # Seleção de modo
    mode = st.radio(
        "Selecione o modo:",
        ["📷 Identificar por imagem", "🔍 Buscar por nome"],
        horizontal=True
    )

    if mode == "📷 Identificar por imagem":
        identify_by_image(df)
    else:
        search_by_name(df)

def identify_by_image(df):
    """Modo de identificação por imagem"""
    st.markdown("### 📷 Envie uma foto da planta")
    
    uploaded_file = st.file_uploader(
        "Selecione uma imagem (JPG/PNG)",
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file and st.button("🔍 Identificar", type="primary"):
        with st.spinner("Analisando a planta..."):
            try:
                # Configuração da API
                files = {"images": (uploaded_file.name, uploaded_file.getvalue())}
                data = {"organs": ["leaf", "flower", "fruit"]}
                
                # Requisição
                response = requests.post(
                    f"{API_URL}?api-key={st.secrets.get('API_KEY', '')}",
                    files=files,
                    data=data,
                    timeout=15
                )
                response.raise_for_status()
                
                process_api_result(response.json(), df)
                
            except requests.exceptions.RequestException as e:
                st.error(f"Erro na API: {str(e)}")
            except Exception as e:
                st.error(f"Erro inesperado: {str(e)}")

def process_api_result(result, df):
    """Processa o resultado da API"""
    if not result.get("results"):
        st.warning("Nenhum resultado encontrado")
        return
    
    best_match = result["results"][0]
    sci_name = best_match["species"]["scientificNameWithoutAuthor"]
    confidence = best_match["score"]
    
    st.success(f"**Identificado:** {sci_name} (Confiança: {confidence:.0%})")
    
    # Busca correspondência no banco de PANCs
    panc_match = df[
        df['nome_cientifico'].str.contains(sci_name, case=False, na=False)
    ]
    
    if not panc_match.empty:
        st.success("✅ Esta planta é uma PANC!")
        display_plant_info(panc_match.iloc[0])
    else:
        st.warning("⚠️ Planta identificada não consta como PANC")
        st.info("Nem todas as plantas são comestíveis. Consulte um especialista.")

def search_by_name(df):
    """Modo de busca por nome"""
    st.markdown("### 🔍 Busque por nome científico ou popular")
    
    search_term = st.text_input(
        "Digite o nome da planta:",
        placeholder="Ex: Ora-pro-nóbis, Talinum paniculatum..."
    )
    
    if search_term:
        results = df[
            df["nome_cientifico"].str.contains(search_term, case=False, na=False) |
            df["nomes_populares"].str.contains(search_term, case=False, na=False)
        ]
        
        if results.empty:
            st.warning("Nenhuma PANC encontrada")
            st.info("Tente nomes alternativos ou científicos")
        else:
            st.success(f"🔎 {len(results)} resultado(s) encontrado(s)")
            for _, row in results.iterrows():
                display_plant_info(row)
                st.divider()

if __name__ == "__main__":
    main()
