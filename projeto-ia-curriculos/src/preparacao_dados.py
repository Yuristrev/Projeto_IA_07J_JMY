"""
Projeto: Classificação de Currículos com IA

Integrantes:
- Yuri Trevisan - RA: 10417375
- João Victor Mota - RA: 10418226
- Matheus Leonardo José - RA: 10341130

Descrição:
Este arquivo realiza o pré-processamento dos dados, transformação de texto, treinamento de modelo de classificação e avaliaçãode desempenho.

Histórico:
Neste arquivo apenas alteramos o caminho para ler o Dataset e o que está sendo exibido da linha 68 ao 71 desse código. O restante do codigo permaneceu igual de quando o código foi implementado

"""

import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv("/workspaces/Projeto_IA_07J_JMY/projeto-ia-curriculos/dataset/curriculos.csv")

print("Prévia do dataset:")
print(df.head())

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

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.3, stratify=y, random_state=42)

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