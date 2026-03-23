"""
Projeto: Classificação de Currículos com IA
Autor: Yuri Trevisan
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

df = pd.read_csv("dataset/curriculos.csv")
df = df.dropna(subset=["texto_curriculo", "classificacao"])

df["classificacao"] = df["classificacao"].map({"apto": 1, "não apto": 0})

vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df["texto_curriculo"])

y = df["classificacao"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Dados preparados!")
