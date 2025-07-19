import streamlit as st
import pandas as pd
import requests
from io import BytesIO
from PIL import Image

# ====== CONFIGURAÇÃO ======
API_KEY = "2b10StWKYdMZlXgbScMsBcRO"
API_URL = "https://my-api.plantnet.org/v2/identify/all"

# ====== CARREGAR DATASET LIMPO ======
@st.cache_data  # Cache para melhor performance
def load_data():
    try:
        df = pd.read_csv("plantas_panc_limpo.csv", encoding='utf-8-sig')
        
        # Verifica e padroniza colunas essenciais
        required_cols = ['nome_cientifico', 'nomes_populares', 'familia', 
                        'habito', 'parte_comestivel', 'uso_culinario', 'url', 'imagem']
        
        for col in required_cols:
            if col not in df.columns:
                df[col] = None  # Cria coluna faltante com valores nulos
                
        return df
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return pd.DataFrame(columns=required_cols)  # Retorna DataFrame vazio se falhar

df = load_data()

# ====== INTERFACE DO APP ======
st.set_page_config(page_title="Identificador de Plantas Comestíveis", layout="centered")

# Seletor de idioma
idioma = st.selectbox("Escolha o idioma / Choose language", ["Português", "Español"])
lang = 'pt' if idioma == "Português" else 'es'

st.title("🌿 Identificador de Plantas Comestíveis (PANCs)")

# ====== FUNÇÃO PARA EXIBIR INFORMAÇÕES DE UMA PLANTA ======
def mostrar_info_planta(linha):
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if pd.notna(linha["imagem"]):
            try:
                # Tenta carregar imagem da URL
                response = requests.get(linha["imagem"])
                img = Image.open(BytesIO(response.content))
                st.image(img, caption="🖼️ Imagem educativa", use_container_width=True)
            except:
                st.warning("Imagem não disponível")
    
    with col2:
        st.write(f"**Nome científico:** {linha['nome_cientifico']}")
        st.write(f"**Nomes populares:** {linha['nomes_populares']}")
        st.write(f"**Família:** {linha['familia']}")
        st.write(f"**Hábito de crescimento:** {linha['habito']}")
        st.write(f"**Parte comestível:** {linha['parte_comestivel']}")
        
        if pd.notna(linha['uso_culinario']):
            with st.expander("🍽️ Uso culinário"):
                st.markdown(linha['uso_culinario'])

    if pd.notna(linha['url']):
        st.markdown(f"🔗 [Ver na fonte]({linha['url']})")

# ====== MODOS DE OPERAÇÃO ======
modo = st.radio("Modo de uso", ["📷 Identificar por imagem", "🔎 Buscar por nome"], horizontal=True)

if modo == "📷 Identificar por imagem":
    st.subheader("Identificação por imagem")
    imagem = st.file_uploader("Envie uma imagem da planta", type=["jpg", "jpeg", "png"])
    
    if imagem:
        st.image(imagem, caption="📸 Imagem enviada", use_container_width=True)

        if st.button("🔍 Identificar", type="primary"):
            with st.spinner("Analisando a imagem..."):
                files = {"images": imagem}
                data = {"organs": ["leaf", "flower", "fruit"]}  # Melhor precisão
                
                try:
                    response = requests.post(
                        f"{API_URL}?api-key={API_KEY}",
                        files=files,
                        data=data
                    )
                    response.raise_for_status()
                    resultado = response.json()

                    if not resultado.get("results"):
                        st.warning("Não foi possível identificar a planta. Tente com uma imagem mais clara.")
                        return

                    melhor_resultado = resultado["results"][0]
                    nome = melhor_resultado["species"]["scientificNameWithoutAuthor"]
                    score = melhor_resultado["score"]
                    
                    st.success(f"🌱 Identificado como: *{nome}* (precisão: {score:.1%})")

                    # Mostrar imagem de referência se disponível
                    if 'images' in melhor_resultado and melhor_resultado['images']:
                        try:
                            imagem_url = melhor_resultado['images'][0]['url']['o']
                            st.image(imagem_url, caption="Imagem de referência", use_container_width=True)
                        except (KeyError, IndexError):
                            pass

                    # Verificar se está no banco de dados
                    planta = df[df["nome_cientifico"].str.lower() == nome.lower().strip()]
                    
                    if not planta.empty:
                        st.success("✅ Esta planta é comestível (PANC)!")
                        mostrar_info_planta(planta.iloc[0])
                    else:
                        st.warning("⚠️ Planta identificada, mas não consta no banco de plantas comestíveis.")
                        st.info("Consulte um especialista antes de consumir qualquer planta desconhecida.")

                except requests.exceptions.RequestException as e:
                    st.error(f"Erro na conexão com a API: {e}")
                except Exception as e:
                    st.error(f"Erro inesperado: {str(e)}")

else:  # Modo de busca por nome
    st.subheader("Busca por nome")
    termo = st.text_input("Digite o nome científico ou popular da planta:")
    
    if termo:
        resultados = df[
            df["nome_cientifico"].str.lower().str.contains(termo.lower()) |
            df["nomes_populares"].str.lower().str.contains(termo.lower())
        ]

        if resultados.empty:
            st.warning("Nenhuma planta encontrada com esse termo.")
        else:
            st.success(f"Encontradas {len(resultados)} plantas:")
            for _, linha in resultados.iterrows():
                with st.expander(f"🌿 {linha['nome_cientifico']}"):
                    mostrar_info_planta(linha)

# Adicionar disclaimer
st.markdown("---")
st.warning("""
**Aviso importante:** Este aplicativo é apenas para fins educativos. 
Sempre consulte um especialista antes de consumir qualquer planta.
""")
