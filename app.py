import streamlit as st
import pandas as pd
import requests
from io import BytesIO
from PIL import Image

# ====== CONFIGURAÇÃO ======
API_KEY = "2b10StWKYdMZlXgbScMsBcRO"  # Sua chave API do PlantNet
API_URL = "https://my-api.plantnet.org/v2/identify/all"

# ====== CARREGAR DATASET LOCAL DE PLANTAS COMESTÍVEIS ======
@st.cache_data
def carregar_dados():
    try:
        df = pd.read_csv("panc_corrigido.csv")
        # Garantir que as URLs das imagens estejam completas (se necessário)
        if 'imagem' in df.columns:
            df['imagem'] = df['imagem'].apply(lambda x: f"https://hortodidatico.ufsc.br/{x}" if pd.notna(x) and not x.startswith('http') else x)
        return df
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo: {e}")
        return pd.DataFrame()

df = carregar_dados()

# ====== INTERFACE DO APP ======
st.set_page_config(page_title="Identificador de PANCs", layout="centered", page_icon="🌿")

idioma = st.selectbox("Escolha o idioma / Choose language", ["Português", "Español"])
lang = 'pt' if idioma == "Português" else 'es'

st.title("🌿 Identificador de Plantas Alimentícias Não Convencionais (PANCs)")

modo = st.radio("Modo de uso", ["📷 Identificar por imagem", "🔎 Buscar por nome"], horizontal=True)

# ====== FUNÇÃO PARA EXIBIR INFORMAÇÕES DE UMA PLANTA ======
def mostrar_info_planta(linha):
    st.subheader(f"🌱 {linha['nome_cientifico']}")
    
    # Exibir imagem se disponível
    if 'imagem' in linha and pd.notna(linha["imagem"]):
        try:
            response = requests.get(linha["imagem"])
            img = Image.open(BytesIO(response.content))
            st.image(img, caption="Imagem da planta", use_column_width=True)
        except:
            st.warning("Não foi possível carregar a imagem")
    
    # Criar colunas para melhor organização
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📋 Informações básicas**")
        st.write(f"**Nomes populares:** {linha.get('nomes_populares', 'Não disponível')}")
        st.write(f"**Família:** {linha.get('familia', 'Não disponível')}")
        st.write(f"**Hábito:** {linha.get('habito', 'Não disponível')}")
    
    with col2:
        st.markdown("**🍽️ Uso culinário**")
        st.write(f"**Parte comestível:** {linha.get('parte_comestivel', 'Não disponível')}")
        if 'uso_culinario' in linha and pd.notna(linha['uso_culinario']):
            st.markdown("**Receitas:**")
            st.write(linha['uso_culinario'])
    
    # Link para mais informações
    if 'url' in linha and pd.notna(linha['url']):
        st.markdown(f"🔗 [Mais informações no Horto Didático UFSC]({linha['url']})")

# ====== IDENTIFICAÇÃO POR IMAGEM ======
if modo == "📷 Identificar por imagem":
    st.markdown("""
    ### Identifique plantas comestíveis por foto
    Tire uma foto ou faça upload de uma imagem da planta que deseja identificar.
    """)
    
    imagem = st.file_uploader("Selecione uma imagem (JPG, PNG)", type=["jpg", "jpeg", "png"])
    
    if imagem:
        st.image(imagem, caption="Sua imagem", use_column_width=True)
        
        if st.button("🔍 Identificar Planta", type="primary"):
            with st.spinner("Analisando a planta..."):
                try:
                    # Preparar dados para a API
                    files = {"images": (imagem.name, imagem.getvalue())}
                    data = {"organs": ["leaf", "flower", "fruit"]}  # Tentar identificar por folha, flor ou fruto
                    
                    # Fazer requisição para a API PlantNet
                    response = requests.post(
                        f"{API_URL}?api-key={API_KEY}",
                        files=files,
                        data=data
                    )
                    response.raise_for_status()
                    resultado = response.json()
                    
                    # Mostrar resultados
                    if "results" in resultado and len(resultado["results"]) > 0:
                        melhor_resultado = resultado["results"][0]
                        nome_cientifico = melhor_resultado["species"]["scientificNameWithoutAuthor"]
                        confianca = melhor_resultado["score"]
                        
                        st.success(f"**Planta identificada:** {nome_cientifico} (confiança: {confianca:.0%})")
                        
                        # Verificar se a planta está no banco de PANCs
                        planta_panc = df[df['nome_cientifico'].str.contains(nome_cientifico, case=False, na=False)]
                        
                        if not planta_panc.empty:
                            st.success("✅ Esta planta está no nosso banco de PANCs!")
                            mostrar_info_planta(planta_panc.iloc[0])
                        else:
                            st.warning("ℹ️ Planta identificada, mas não consta em nosso banco de PANCs.")
                            st.info("""
                            Nem todas as plantas identificadas são comestíveis. 
                            Consulte um especialista antes de consumir qualquer planta desconhecida.
                            """)
                    else:
                        st.warning("Não foi possível identificar a planta com certeza suficiente.")
                
                except Exception as e:
                    st.error(f"Erro na identificação: {str(e)}")

# ====== BUSCA MANUAL ======
else:
    st.markdown("""
    ### Busque plantas comestíveis por nome
    Digite o nome científico ou popular da planta que deseja encontrar.
    """)
    
    termo = st.text_input("Digite o nome da planta:", placeholder="Ex: ora-pro-nobis, taioba, etc.")
    
    if termo:
        # Buscar no dataframe
        resultados = df[
            df["nome_cientifico"].str.contains(termo, case=False, na=False) |
            df["nomes_populares"].str.contains(termo, case=False, na=False)
        ]
        
        if resultados.empty:
            st.warning("Nenhuma PANC encontrada com esse nome.")
            st.info("""
            Dicas para busca:
            - Tente nomes científicos (ex: Pereskia aculeata)
            - Ou nomes populares (ex: ora-pro-nobis)
            - Verifique a ortografia
            """)
        else:
            st.success(f"Encontradas {len(resultados)} PANCs correspondentes:")
            
            for _, planta in resultados.iterrows():
                mostrar_info_planta(planta)
                st.divider()
