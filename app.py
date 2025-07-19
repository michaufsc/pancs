import streamlit as st
import pandas as pd
import requests

# ====== CONFIGURAÇÃO ======
API_KEY = "SUA_API_KEY_PLANTNET_AQUI"
API_URL = "https://my-api.plantnet.org/v2/identify/all"

# ====== CARREGAR DATASET LOCAL DE PLANTAS COMESTÍVEIS ======
df = pd.read_csv("plantas_panc.csv")

# ====== INTERFACE DO APP ======
st.set_page_config(page_title="Identificador de Plantas Comestíveis", layout="centered")

idioma = st.selectbox("Escolha o idioma / Choose language", ["Português", "Español"])
lang = 'pt' if idioma == "Português" else 'es'

st.title("🌿 Identificador de Plantas Comestíveis (PANCs)")

modo = st.radio("Modo de uso", ["📷 Identificar por imagem", "🔎 Buscar por nome"])

# ====== FUNÇÃO PARA EXIBIR INFORMAÇÕES DE UMA PLANTA ======
def mostrar_info_planta(linha):
    if pd.notna(linha["imagem"]):
        st.image(linha["imagem"], caption="🖼️ Imagem educativa (Horto Didático)", use_container_width=True)

    st.write(f"**Nome científico:** {linha['nome_cientifico']}")
    st.write(f"**Nomes populares:** {linha['nomes_populares']}")
    st.write(f"**Família:** {linha['familia']}")
    st.write(f"**Origem:** {linha['origem']}")
    st.write(f"**Hábito de crescimento:** {linha['habito']}")
    st.write(f"**Parte comestível – preparo:** {linha['parte_comestivel']}")

    if pd.notna(linha['receita']) and linha['receita'].strip():
        st.markdown(f"🍽️ **Receita:** {linha['receita']}")

    st.markdown(f"🔗 [Ver na fonte]({linha['fonte']})")

# ====== IDENTIFICAÇÃO POR IMAGEM ======
if modo == "📷 Identificar por imagem":
    imagem = st.file_uploader("Envie uma imagem da planta", type=["jpg", "jpeg", "png"])
    
    if imagem:
        st.image(imagem, caption="📸 Imagem enviada pelo usuário", use_container_width=True)

        if st.button("🔍 Identificar"):
            files = {"images": imagem}
            data = {"organs": ["leaf"]}
            url = f"{API_URL}?api-key={API_KEY}"

            with st.spinner("🔎 Identificando planta..."):
                try:
                    r = requests.post(url, files=files, data=data)
                    r.raise_for_status()
                    resultado = r.json()

                    nome = resultado["results"][0]["species"]["scientificNameWithoutAuthor"]
                    score = resultado["results"][0]["score"]
                    st.success(f"🌱 Nome científico identificado: *{nome}* (confiança: {score:.1%})")

                    # Imagem da API PlantNet (se houver)
                    try:
                        imagem_url = resultado['results'][0]['images'][0]['url']['o']
                        st.image(imagem_url, caption="🖼️ Imagem de referência (PlantNet)", use_container_width=True)
                    except (KeyError, IndexError, TypeError):
                        st.warning("⚠️ Nenhuma imagem de referência disponível.")

                    planta = df[df["nome_cientifico"].str.lower() == nome.lower()]
                    if not planta.empty:
                        st.success("✅ Esta planta está no banco de plantas comestíveis!")
                        mostrar_info_planta(planta.iloc[0])
                    else:
                        st.warning("⚠️ Planta identificada, mas não está no banco de plantas comestíveis.")

                except Exception as e:
                    st.error(f"❌ Erro ao identificar a planta: {e}")

# ====== BUSCA MANUAL ======
else:
    termo = st.text_input("Digite o nome científico ou popular da planta:")

    if termo:
        resultados = df[
            df["nome_cientifico"].str.lower().str.contains(termo.lower()) |
            df["nomes_populares"].str.lower().str.contains(termo.lower())
        ]

        if resultados.empty:
            st.warning("❌ Nenhuma planta encontrada com esse nome.")
        else:
            for _, linha in resultados.iterrows():
                st.subheader(f"🌿 {linha['nome_cientifico']}")
                mostrar_info_planta(linha)
                st.markdown("---")
