import pandas as pd
import re
from google.colab import files
from io import StringIO

def limpar_texto(texto):
    """Limpa e padroniza textos removendo caracteres especiais e espaços extras"""
    if pd.isna(texto):
        return ""
    texto = str(texto)
    # Remove múltiplos espaços, caracteres especiais, mas mantém pontuação básica
    texto = re.sub(r'[^\w\s,;.:-]', '', texto)
    texto = texto.replace('\t', ' ').replace('\n', ' ')
    texto = re.sub(' +', ' ', texto)
    return texto.strip(' .,;')

def corrigir_formatacao_csv(conteudo):
    """Corrige problemas específicos de formatação no arquivo CSV"""
    # Padroniza quebras de linha
    conteudo = conteudo.replace('\r\n', '\n').replace('\r', '\n')
    
    # Corrige padrões problemáticos com aspas
    conteudo = re.sub(r'",\s*",', '","', conteudo)  # Remove espaços entre aspas
    conteudo = re.sub(r'""",', '","', conteudo)     # Corrige aspas triplas
    conteudo = re.sub(r',\s*"",', ',', conteudo)    # Remove campos vazios
    
    # Remove linhas completamente vazias
    linhas = [linha for linha in conteudo.split('\n') if linha.strip()]
    return '\n'.join(linhas)

def ler_csv_corrigido(conteudo):
    """Tenta ler o CSV com diferentes abordagens até ter sucesso"""
    tentativas = [
        {'sep': ',', 'quotechar': '"', 'engine': 'python'},
        {'sep': ';', 'quotechar': '"', 'engine': 'python'},
        {'sep': ',', 'quotechar': None, 'engine': 'python'},
        {'sep': '\t', 'quotechar': '"', 'engine': 'python'}
    ]
    
    for config in tentativas:
        try:
            return pd.read_csv(StringIO(conteudo), **config), None
        except pd.errors.ParserError as e:
            ultimo_erro = e
    
    # Se todas as tentativas falharem, retorna o erro
    return None, ultimo_erro

def processar_dados(df):
    """Processa e limpa o dataframe"""
    # Normaliza nomes de colunas
    df.columns = [limpar_texto(col).lower().replace(' ', '_') for col in df.columns]
    
    # Identifica colunas importantes (algumas podem ter nomes diferentes)
    colunas_esperadas = {
        'nome_cientifico': ['nome_cientifico', 'cientifico', 'especie'],
        'nomes_populares': ['nomes_populares', 'populares', 'vulgares'],
        'familia': ['familia', 'familia_botanica'],
        'habito': ['habito', 'crescimento', 'porte'],
        'parte_comestivel': ['parte_comestivel', 'comestivel', 'partes_utilizadas'],
        'uso_culinario': ['uso_culinario', 'culinario', 'receitas'],
        'url': ['url', 'link', 'fonte']
    }
    
    # Mapeia colunas existentes para os nomes padronizados
    mapeamento_colunas = {}
    for padrao, alternativas in colunas_esperadas.items():
        for col in df.columns:
            if any(alt in col for alt in alternativas):
                mapeamento_colunas[col] = padrao
                break
    
    # Renomeia colunas
    df = df.rename(columns=mapeamento_colunas)
    
    # Mantém apenas as colunas padronizadas
    colunas_finais = [c for c in colunas_esperadas.keys() if c in df.columns]
    df = df[colunas_finais]
    
    # Limpeza dos dados
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(limpar_texto)
    
    # Remove linhas completamente vazias
    df = df.dropna(how='all')
    
    return df

def contar_plantas_unicas(df):
    """Conta quantas plantas únicas existem na base"""
    if 'nome_cientifico' not in df.columns:
        return 0
    
    # Remove duplicatas baseadas no nome científico
    plantas_unicas = df['nome_cientifico'].drop_duplicates()
    
    return len(plantas_unicas)

def main():
    print("1. Faça o upload do arquivo CSV com os dados das PANC")
    uploaded = files.upload()

    if not uploaded:
        print("Nenhum arquivo enviado!")
        return

    file_name = list(uploaded.keys())[0]
    
    print("\n2. Processando arquivo...")
    try:
        # Lê o conteúdo do arquivo
        with open(file_name, 'r', encoding='utf-8') as f:
            conteudo = f.read()
        
        # Corrige problemas de formatação
        conteudo_corrigido = corrigir_formatacao_csv(conteudo)
        
        # Tenta ler o CSV corrigido
        df, erro = ler_csv_corrigido(conteudo_corrigido)
        if erro:
            raise erro
        
        # Processa os dados
        df_limpo = processar_dados(df)
        
        # Conta plantas únicas
        num_plantas = contar_plantas_unicas(df_limpo)
        
        print(f"\n3. Dados limpos - Total de {num_plantas} plantas únicas encontradas:")
        display(df_limpo.head())
        
        # Salva o resultado
        output_file = 'panc_corrigido.csv'
        df_limpo.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"\n4. Download do arquivo processado: {output_file}")
        files.download(output_file)
        
    except Exception as e:
        print(f"\nErro ao processar o arquivo: {str(e)}")
        print("\nDicas para resolver:")
        print("- Verifique se o arquivo é um CSV válido")
        print("- Abra o arquivo em um editor de texto e confira se não há linhas quebradas")
        print("- Tente remover manualmente caracteres especiais problemáticos")

if __name__ == "__main__":
    main()
