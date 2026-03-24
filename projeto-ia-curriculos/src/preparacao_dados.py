"""
Projeto: Classificação de Currículos com IA

Integrantes:
- Yuri Trevisan - RA: 10417375
- João Victor Mota - RA: 10418226
- Matheus Leonardo José - RA: 10341130

Descrição:
Este arquivo realiza o pré-processamento dos dados, transformação de texto, treinamento de modelo de classificação e avaliaçãode desempenho.

"""

import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# 1. Ler o CSV
df = pd.read_csv("/workspaces/Projeto_IA_07J_JMY/projeto-ia-curriculos/dataset/curriculos.csv")

print("Prévia do dataset:")
print(df.head())

# 2. Remover valores nulos
df = df.dropna()

# 3. Função de limpeza
def limpar_texto(texto):
    texto = str(texto).lower()
    texto = re.sub(r"\d+", "", texto)
    texto = texto.translate(str.maketrans("", "", string.punctuation))
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto

# 4. Criar texto completo
df["texto_completo"] = (
    df["texto_curriculo"] + " " +
    df["formacao"] + " " +
    df["habilidades"] + " " +
    df["vaga"]
)

# 5. Limpar texto
df["texto_completo"] = df["texto_completo"].apply(limpar_texto)

# 6. Separar X e y
X = df["texto_completo"]
y = df["classificacao"]

# 7. TF-IDF
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

# 8. Dividir treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.3, stratify=y, random_state=42)

# 9. Modelo
modelo = LogisticRegression()
modelo.fit(X_train, y_train)

# 10. Previsão
y_pred = modelo.predict(X_test)

# 11. Avaliação
print("\nAcurácia:")
print(accuracy_score(y_test, y_pred))

report = classification_report(y_test, y_pred, output_dict=True)

print("\nRelatório simplificado:\n")

for classe in report.keys():
    if classe not in ["accuracy", "macro avg", "weighted avg"]:
        print(f"Classe: {classe}")
        print(f"Precisão: {report[classe]['precision']:.2f}")
        print()