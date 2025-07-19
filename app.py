import pandas as pd
from google.colab import files

def analisar_panc(df):
    """Realiza análises específicas no dataframe das PANC"""
    # Análise básica
    num_plantas = df['nome_cientifico'].nunique()
    familias_unicas = df['familia'].nunique()
    habitos = df['habito'].value_counts()
    
    print(f"Total de plantas únicas: {num_plantas}")
    print(f"Famílias botânicas distintas: {familias_unicas}")
    print("\nDistribuição por hábito de crescimento:")
    print(habitos)
    
    # Plantas com receitas
    if 'uso_culinario' in df.columns:
        com_receitas = df[df['uso_culinario'].str.contains('👨‍🍳|Receitas|receitas', na=False)]
        print(f"\nPlantas com receitas disponíveis: {len(com_receitas)}")
    
    return df

def main():
    print("1. Faça o upload do arquivo panc_corrigido.csv")
    uploaded = files.upload()

    if not uploaded:
        print("Nenhum arquivo enviado!")
        return

    file_name = list(uploaded.keys())[0]
    
    if file_name != 'panc_corrigido.csv':
        print("Por favor, faça upload do arquivo panc_corrigido.csv")
        return
    
    print("\n2. Carregando e analisando os dados...")
    try:
        # Carrega o arquivo já corrigido
        df = pd.read_csv(file_name)
        
        # Realiza análises
        df_analisado = analisar_panc(df)
        
        # Mostra exemplos
        print("\n3. Exemplo de dados:")
        display(df_analisado.sample(5))
        
        # Opção para salvar análises
        output_file = 'analise_panc.csv'
        df_analisado.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"\n4. Download do arquivo com análises: {output_file}")
        files.download(output_file)
        
    except Exception as e:
        print(f"\nErro ao processar o arquivo: {str(e)}")
        print("\nDicas para resolver:")
        print("- Verifique se o arquivo está no formato CSV válido")
        print("- Confira se o arquivo tem as colunas esperadas")

if __name__ == "__main__":
    main()
