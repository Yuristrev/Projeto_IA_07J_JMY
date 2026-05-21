"""
Projeto: Classificação de Currículos com IA

Integrantes:
- Yuri Trevisan - RA: 10417375
- João Victor Mota - RA: 10418226
- Matheus Leonardo José - RA: 10341130

Descrição:
Este arquivo realiza o pré-processamento dos dados presentes no dataset referente
à análise de currículos, transformação de texto, treinamento do modelo de
classificação e avaliação de desempenho.

Histórico:
O código foi feito e testado no aplicativo local do VS Code e depois copiado para o GitHub.
Neste arquivo, foram adicionadas análises exploratórias numéricas e gráficos para melhorar
a apresentação dos dados utilizados no projeto.

Versão 01: 24/03/2026 - Código incluído no GitHub
Versão 02: 24/03/2026 - Ajustes no caminho do dataset e saída exibida
Versão 03: 20/05/2026 - Inclusão de análise exploratória, gráficos e matriz de confusão
"""

import pandas as pd
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Leitura do dataset em formato CSV
df = pd.read_csv("/workspaces/Projeto_IA_07J_JMY/projeto-ia-curriculos/dataset/curriculos.csv")

# Exibe os primeiros registros para validar a estrutura dos dados
print("Prévia do dataset:")
print(df.head())

# Informações gerais utilizadas na análise exploratória
print("\nQuantidade de registros:")
print(len(df))

print("\nQuantidade de colunas:")
print(len(df.columns))

print("\nColunas do dataset:")
print(df.columns.tolist())

print("\nInformações gerais do dataset:")
print(df.info())

# Verificação de possíveis valores ausentes
print("\nValores nulos por coluna:")
print(df.isnull().sum())

# Quantidade de candidatos aptos e não aptos
print("\nDistribuição das classes:")
print(df["classificacao"].value_counts())

# Contagem de palavras dos currículos para análise textual
df["qtd_palavras"] = df["texto_curriculo"].apply(
    lambda x: len(str(x).split())
)

print("\nMédia de palavras por currículo:")
print(round(df["qtd_palavras"].mean(), 2))

print("\nMenor currículo em quantidade de palavras:")
print(df["qtd_palavras"].min())

print("\nMaior currículo em quantidade de palavras:")
print(df["qtd_palavras"].max())

# Geração de gráfico para visualizar a distribuição das classes
classes = df["classificacao"].value_counts()

plt.bar(classes.index, classes.values)
plt.title("Distribuição das Classes")
plt.xlabel("Classificação")
plt.ylabel("Quantidade")
plt.show()

# Gráfico para visualizar a quantidade de palavras dos currículos
plt.hist(df["qtd_palavras"], bins=10)
plt.title("Quantidade de Palavras nos Currículos")
plt.xlabel("Quantidade de Palavras")
plt.ylabel("Frequência")
plt.show()

# Remove possíveis registros nulos
df = df.dropna()

# Função de limpeza textual utilizada antes da vetorização
def limpar_texto(texto):
    texto = str(texto).lower()
    texto = re.sub(r"\d+", "", texto)
    texto = texto.translate(str.maketrans("", "", string.punctuation))
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto

# Combinação das informações textuais em uma única variável
df["texto_completo"] = (
    df["texto_curriculo"] + " " +
    df["formacao"] + " " +
    df["habilidades"] + " " +
    df["vaga"]
)

# Aplicação da limpeza nos textos
df["texto_completo"] = df["texto_completo"].apply(limpar_texto)

# Separação entre entrada e saída
X = df["texto_completo"]
y = df["classificacao"]

# Transformação dos textos em dados numéricos utilizando TF-IDF
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

# Divisão entre dados de treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

# Criação e treinamento do modelo de Regressão Logística
modelo = LogisticRegression()
modelo.fit(X_train, y_train)

# Realiza as previsões utilizando os dados de teste
y_pred = modelo.predict(X_test)

# Exibição da acurácia do modelo
print("\nAcurácia:")
print(accuracy_score(y_test, y_pred))

# Relatório de desempenho por classe
report = classification_report(y_test, y_pred, output_dict=True)

print("\nRelatório simplificado:\n")

for classe in ["apto", "não apto"]:
    print(f"Classe: {classe}")
    print(f"Precisão: {report[classe]['precision']:.2f}")
    print()

# Matriz de confusão para visualizar os acertos e erros do modelo
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d")
plt.title("Matriz de Confusão")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.show()
