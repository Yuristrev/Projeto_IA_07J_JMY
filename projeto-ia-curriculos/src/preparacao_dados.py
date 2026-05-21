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

df = pd.read_csv("/workspaces/Projeto_IA_07J_JMY/projeto-ia-curriculos/dataset/curriculos.csv")

print("Prévia do dataset:")
print(df.head())

print("\nQuantidade de registros:")
print(len(df))

print("\nQuantidade de colunas:")
print(len(df.columns))

print("\nColunas do dataset:")
print(df.columns.tolist())

print("\nInformações gerais do dataset:")
print(df.info())

print("\nValores nulos por coluna:")
print(df.isnull().sum())

print("\nDistribuição das classes:")
print(df["classificacao"].value_counts())

df["qtd_palavras"] = df["texto_curriculo"].apply(
    lambda x: len(str(x).split())
)

print("\nMédia de palavras por currículo:")
print(round(df["qtd_palavras"].mean(), 2))

print("\nMenor currículo em quantidade de palavras:")
print(df["qtd_palavras"].min())

print("\nMaior currículo em quantidade de palavras:")
print(df["qtd_palavras"].max())

classes = df["classificacao"].value_counts()

plt.bar(classes.index, classes.values)
plt.title("Distribuição das Classes")
plt.xlabel("Classificação")
plt.ylabel("Quantidade")
plt.show()

# Gráfico de quantidade de palavras
plt.hist(df["qtd_palavras"], bins=10)
plt.title("Quantidade de Palavras nos Currículos")
plt.xlabel("Quantidade de Palavras")
plt.ylabel("Frequência")
plt.show()

df = df.dropna()

def limpar_texto(texto):
    texto = str(texto).lower()
    texto = re.sub(r"\d+", "", texto)
    texto = texto.translate(str.maketrans("", "", string.punctuation))
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto

df["texto_completo"] = (
    df["texto_curriculo"] + " " +
    df["formacao"] + " " +
    df["habilidades"] + " " +
    df["vaga"]
)

df["texto_completo"] = df["texto_completo"].apply(limpar_texto)

X = df["texto_completo"]
y = df["classificacao"]

vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

modelo = LogisticRegression()
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

print("\nAcurácia:")
print(accuracy_score(y_test, y_pred))

report = classification_report(y_test, y_pred, output_dict=True)

print("\nRelatório simplificado:\n")

for classe in ["apto", "não apto"]:
    print(f"Classe: {classe}")
    print(f"Precisão: {report[classe]['precision']:.2f}")
    print()

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d")
plt.title("Matriz de Confusão")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.show()
