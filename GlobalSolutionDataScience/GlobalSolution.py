import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Assumindo que o dataset está em um arquivo CSV chamado 'dataset.csv'
# Se for outro formato (como Excel), ajuste a função de leitura (ex: pd.read_excel)
df = pd.read_csv('wildfires.csv')

# Exibir as primeiras linhas do DataFrame para verificar
print(df.head())

# Exibir informações sobre o DataFrame
print(df.info())

# Remoção de duplicatas
print(f"Número de linhas antes da remoção de duplicatas: {len(df)}")
df = df.drop_duplicates()
print(f"Número de linhas após a remoção de duplicatas: {len(df)}")

try:
    df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
except KeyError:
    print("Coluna 'Data' não encontrada. Ajuste o nome da coluna se necessário.")

# Padronização de campos de categorias (Exemplo: convertendo para minúsculas e removendo espaços extras)
# Substitua 'Categoria' pelo nome real da sua coluna categórica
try:
    df['Estado'] = df['Estado'].str.lower().str.strip()
except KeyError:
    print("Coluna 'Estado' não encontrada. Ajuste o nome da coluna se necessário.")
except AttributeError:
     print("Coluna 'Estado' não é do tipo string ou não existe.")

# Identificação e tratamento de valores nulos
print("\nValores nulos por coluna antes do tratamento:")
print(df.isnull().sum())

# Exemplo de tratamento: preencher valores nulos em colunas numéricas com a média
for col in df.select_dtypes(include=['number']).columns:
    if df[col].isnull().any():
        mean_val = df[col].mean()
        df[col].fillna(mean_val, inplace=True)
        print(f"Valores nulos na coluna '{col}' preenchidos com a média ({mean_val:.2f}).")

# Exemplo de tratamento: preencher valores nulos em colunas categóricas com a moda (valor mais frequente)
for col in df.select_dtypes(include=['object']).columns:
     if df[col].isnull().any():
         mode_val = df[col][~df[col].isnull()].mode()
         if not mode_val.empty:
             df[col].fillna(mode_val[0], inplace=True)
             print(f"Valores nulos na coluna '{col}' preenchidos com a moda ('{mode_val[0]}').")


print("\nValores nulos por coluna após o tratamento:")
print(df.isnull().sum())


print("\nIdentificação de outliers usando Z-score (limite Z > 3 ou Z < -3):")
for col in df.select_dtypes(include=np.number).columns:
    # Calcula o Z-score, ignorando NaNs
    df[f'{col}_zscore'] = np.abs(stats.zscore(df[col].dropna()))
    outliers_count = df[df[f'{col}_zscore'] > 3].shape[0]
    print(f"Coluna '{col}': {outliers_count} outliers.")
    #calculando Z_Score
    if f'{col}_zscore' in df.columns:
        df = df.drop(columns=[f'{col}_zscore'])


# Exibir as primeiras linhas do DataFrame após a limpeza
print("\nDataFrame após limpeza:")
print(df.head())

# Exibir informações finais sobre o DataFrame
print("\nInformações finais do DataFrame:")
print(df.info())

# Exibir estatísticas descritivas após a limpeza
print("\nEstatísticas descritivas após limpeza:")
print(df.describe(include='all'))

plt.style.use('ggplot')
sns.set_palette("viridis")

# 1. Tendências temporais para a ocorrência de incêndios
if 'FIRE_YEAR' in df.columns:
    plt.figure(figsize=(12, 6))
    df['FIRE_YEAR'].value_counts().sort_index().plot(kind='bar')
    plt.title('Ocorrência de Incêndios por Ano')
    plt.xlabel('Ano')
    plt.ylabel('Número de Incêndios')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 2. Causas principais de incêndios
if 'NWCG_CAUSE_CLASSIFICATION' in df.columns:
    causas_principais = df['NWCG_CAUSE_CLASSIFICATION'].value_counts().head(5)

    plt.figure(figsize=(12, 6))
    causas_principais.plot(kind='pie', autopct='%1.1f%%', startangle=90)
    plt.title('Top 5 Causas de Incêndios (Porcentagem)')
    plt.ylabel('')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.barplot(x=causas_principais.values, y=causas_principais.index)
    plt.title('Top 5 Causas de Incêndios (Contagem)')
    plt.xlabel('Número de Incêndios')
    plt.ylabel('Causa')
    plt.tight_layout()
    plt.show()

# 3. Estados onde ocorreram com maior e menor frequência
if 'STATE' in df.columns:
    incendios_por_estado = df['STATE'].value_counts().head(10)

    plt.figure(figsize=(12, 6))
    sns.barplot(x=incendios_por_estado.values, y=incendios_por_estado.index)
    plt.title('Top 10 Estados com Maior Ocorrência de Incêndios')
    plt.xlabel('Número de Incêndios')
    plt.ylabel('Estado')
    plt.tight_layout()
    plt.show()

# 4. Categoria com maior quantidade de incêndios
if 'FIRE_SIZE_CLASS' in df.columns:
    categorias_incendios = df['FIRE_SIZE_CLASS'].value_counts()

    plt.figure(figsize=(12, 6))
    categorias_incendios.plot(kind='bar')
    plt.title('Distribuição de Incêndios por Categoria de Tamanho')
    plt.xlabel('Categoria de Tamanho')
    plt.ylabel('Número de Incêndios')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 5. Maiores quantidades de acres por incêndio
if 'FIRE_SIZE' in df.columns:
    acres_por_incendio = df.sort_values(by='FIRE_SIZE', ascending=False).head(10)

    plt.figure(figsize=(12, 6))
    sns.barplot(x='FIRE_SIZE', y='FIRE_NAME', data=acres_por_incendio)
    plt.title('Top 10 Maiores Incêndios por Tamanho (Acres)')
    plt.xlabel('Tamanho do Incêndio (Acres)')
    plt.ylabel('Nome do Incêndio')
    plt.tight_layout()
    plt.show()

# Filtrar o DataFrame para incluir apenas incêndios com a causa 'HUMAN'
df_humanos = df[df['NWCG_CAUSE_CLASSIFICATION'] == 'Human']

# Encontrar os maiores incêndios causados por humanos (baseado no tamanho, FIRE_SIZE)
maiores_incendios_humanos = df_humanos.sort_values(by='FIRE_SIZE', ascending=False).head(10)
maiores_incendios_humanos[['FIRE_SIZE', 'FIRE_NAME', 'NWCG_CAUSE_CLASSIFICATION']]

# Criar um gráfico dos maiores incêndios causados por humanos
plt.figure(figsize=(12, 6))
sns.barplot(x='FIRE_SIZE', y='FIRE_NAME', data=maiores_incendios_humanos, palette='viridis')
plt.title('Top 10 Maiores Incêndios Causados por Humanos por Tamanho (Acres)')
plt.xlabel('Tamanho do Incêndio (Acres)')
plt.ylabel('Nome do Incêndio')
plt.tight_layout()
plt.show()

# Defina a semente aleatória (somatório do último número da matrícula de todos os integrantes do grupo)
# Substitua 'sua_semente' pelo valor calculado
semente_aleatoria = 42

# Definir o tamanho da amostra
tamanho_amostra = 500000

# Verificar se o tamanho do DataFrame é maior ou igual ao tamanho da amostra desejada
if len(df) >= tamanho_amostra:
    # Selecionar a amostra aleatória sem reposição
    df_amostra = df.sample(n=tamanho_amostra, random_state=semente_aleatoria, replace=False)

    print(f"\nDataFrame original possui {len(df)} registros.")
    print(f"Amostra de {tamanho_amostra} observações selecionada com sucesso.")
    print("\nPrimeiras linhas da amostra:")
    print(df_amostra.head())

    print("\nInformações da amostra:")
    print(df_amostra.info())

# Para verificar o número total de incêndios no dataset
numero_total_incendios = len(df)
print(f"\nO número total de incêndios no dataset é: {numero_total_incendios}")

# Contar a ocorrência de incêndios por ano
incendios_por_ano = df['FIRE_YEAR'].value_counts().sort_index()

# Calcular a diferença de incêndios entre anos consecutivos
diferenca_anual = incendios_por_ano.diff()

# Calcular a porcentagem de crescimento anual
# Evitar divisão por zero para o primeiro ano (que terá NaN na diferença)
# Para o primeiro ano com dados, não há crescimento em relação a um ano anterior no dataset
porcentagem_crescimento_anual = (diferenca_anual / incendios_por_ano.shift(1)) * 100

print("\nPorcentagem de crescimento de incêndios por FIRE_YEAR:")
print(porcentagem_crescimento_anual.dropna()) # Remover o NaN do primeiro ano

# Opcionalmente, você pode plotar o crescimento anual
plt.figure(figsize=(12, 6))
porcentagem_crescimento_anual.plot(kind='bar')
plt.title('Porcentagem de Crescimento Anual de Incêndios')
plt.xlabel('Ano')
plt.ylabel('Crescimento (%)')
plt.xticks(rotation=45)
plt.axhline(0, color='grey', linestyle='--', linewidth=0.8) # Adiciona linha no 0 para visualização
plt.tight_layout()
plt.show()

# 6. Distribuição geográfica dos incêndios
# Isso requer dados de latitude e longitude (LATITUDE, LONGITUDE)
if 'LATITUDE' in df.columns and 'LONGITUDE' in df.columns:
    # Plotar um scatter plot simples no mapa (pode ser lento com muitos pontos)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='LONGITUDE', y='LATITUDE', data=df_amostra, alpha=0.5, s=5, palette='viridis') # Usando a amostra para performance
    plt.title('Distribuição Geográfica dos Incêndios (Amostra)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 7. Distribuição geográfica dos incêndios com heatmap
    # Para uma visualização mais densa, um mapa de calor 2D
    # Ajuste 'bins' e 'cmap' conforme necessário para melhor visualização
    plt.figure(figsize=(10, 8))
    plt.hist2d(df_amostra['LONGITUDE'], df_amostra['LATITUDE'], bins=100, cmap='inferno')
    plt.colorbar(label='Contagem de Incêndios')
    plt.title('Mapa de Calor da Distribuição Geográfica dos Incêndios (Amostra)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    plt.tight_layout()
    plt.show()




# 8. Incêndios por Dia da Semana
# Se a coluna 'Data' (ou 'DISCOVERY_DATE') já foi convertida para datetime
if 'Data' in df.columns:
    df['Dia_Semana'] = df['Data'].dt.day_name() # Cria coluna com nome do dia da semana
    dias_ordem = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    plt.figure(figsize=(12, 6))
    sns.countplot(x='Dia_Semana', data=df, order=dias_ordem)
    plt.title('Ocorrência de Incêndios por Dia da Semana')
    plt.xlabel('Dia da Semana')
    plt.ylabel('Número de Incêndios')
    plt.tight_layout()
    plt.show()
elif 'DISCOVERY_DATE' in df.columns:
    df['Dia_Semana_Descobrimento'] = pd.to_datetime(df['DISCOVERY_DATE'], errors='coerce').dt.day_name()
    dias_ordem = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    plt.figure(figsize=(12, 6))
    sns.countplot(x='Dia_Semana_Descobrimento', data=df, order=dias_ordem)
    plt.title('Ocorrência de Incêndios por Dia da Semana de Descobrimento')
    plt.xlabel('Dia da Semana')
    plt.ylabel('Número de Incêndios')
    plt.tight_layout()
    plt.show()
