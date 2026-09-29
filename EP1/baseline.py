import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report


# ============================================================
# 1. LEITURA DO DATASET
# ============================================================

arquivo = "train.xlsx"

df = pd.read_excel(arquivo)

print("Dimensões do dataset:", df.shape)
print("Colunas:", df.columns.tolist())


# ============================================================
# 2. SEPARAÇÃO ENTRE TEXTO E RÓTULO
# ============================================================

textos = df["resp_text"].astype(str)
rotulos = df["clarity"]


# ============================================================
# 3. DIVISÃO TREINO / TESTE
# ============================================================

texto_train, texto_test, y_train, y_test = train_test_split(
    textos,
    rotulos,
    test_size=0.20,
    random_state=123,
    stratify=rotulos
)

print("\nTamanho do conjunto de treinamento:", len(texto_train))
print("Tamanho do conjunto de teste:", len(texto_test))


# ============================================================
# 4. REPRESENTAÇÃO TF-IDF
# ============================================================

vect = TfidfVectorizer()

x_train = vect.fit_transform(texto_train)
x_test = vect.transform(texto_test)

print("\nCaracterísticas TF-IDF:")
print("Treinamento:", x_train.shape)
print("Teste:", x_test.shape)


# ============================================================
# 5. REGRESSÃO LOGÍSTICA
# ============================================================

clf = LogisticRegression(
    class_weight="balanced"
)

clf.fit(x_train, y_train)


# ============================================================
# 6. PREDIÇÃO
# ============================================================

predicted = clf.predict(x_test)


# ============================================================
# 7. AVALIAÇÃO
# ============================================================

score = f1_score(
    y_test,
    predicted,
    average="macro"
)

print("\n======================================")
print("BASELINE: TF-IDF + REGRESSÃO LOGÍSTICA")
print("======================================")
print(f"F1 macro: {score:.4f}")

print("\nRelatório de classificação:")
print(classification_report(y_test, predicted))