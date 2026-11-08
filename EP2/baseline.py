import eli5
import pandas as pd
from eli5.formatters import format_as_text

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


# ============================================================
# 8. INTERPRETABILIDADE COM ELI5
# ============================================================

print("\n======================================")
print("INTERPRETABILIDADE GLOBAL: PESOS DAS PALAVRAS")
print("======================================")
explicacao_pesos = eli5.explain_weights(
    clf,
    vec=vect,
    top=20
)
print(format_as_text(explicacao_pesos))

print("\n======================================")
print("INTERPRETABILIDADE LOCAL: PREVISÕES DO TESTE")
print("======================================")

num_exemplos = min(3, len(texto_test))
for indice in range(num_exemplos):
    texto = texto_test.iloc[indice]
    rotulo_real = y_test.iloc[indice]
    rotulo_previsto = predicted[indice]

    print(f"\nExemplo {indice + 1}")
    trecho_texto = texto[:500]
    if len(texto) > 500:
        trecho_texto += "..."
    print(f"Texto: {trecho_texto}")
    print(f"Rótulo real: {rotulo_real}")
    print(f"Rótulo previsto: {rotulo_previsto}")
    print("Contribuição dos termos para a previsão:")
    explicacao_previsao = eli5.explain_prediction(
        clf,
        texto,
        vec=vect,
        top=10
    )
    print(format_as_text(explicacao_previsao))