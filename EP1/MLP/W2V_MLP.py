import os
import random
import gc
import numpy as np
import pandas as pd

from gensim.models import Word2Vec

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# CONFIGURAÇÕES
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Dispositivo:", DEVICE)


# ============================================================
# CAMINHOS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_PATH = os.path.join(BASE_DIR, "train.xlsx")
TEST_PATH = os.path.join(BASE_DIR, "test1.xlsx")


# ============================================================
# CARREGAMENTO
# ============================================================

df = pd.read_excel(TRAIN_PATH)

texts = df["resp_text"].fillna("").astype(str).tolist()

label_map = {
    "c1": 0,
    "c234": 1,
    "c5": 2
}

labels = df["clarity"].map(label_map).values


# ============================================================
# TOKENIZAÇÃO
# ============================================================

def tokenize(text):
    return text.lower().split()


tokenized_texts = [
    tokenize(text)
    for text in texts
]


# ============================================================
# WORD2VEC
# ============================================================

def train_word2vec(
    sentences,
    vector_size=200,
    window=5,
    min_count=2,
    sg=1,
    epochs=10
):

    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        epochs=epochs,
        workers=os.cpu_count(),
        seed=SEED
    )

    return model


# ============================================================
# EMBEDDING DOS DOCUMENTOS
# ============================================================

def document_embedding(tokens, w2v_model):

    vectors = [
        w2v_model.wv[word]
        for word in tokens
        if word in w2v_model.wv
    ]

    if len(vectors) == 0:

        return np.zeros(
            w2v_model.vector_size,
            dtype=np.float32
        )

    return np.mean(
        vectors,
        axis=0
    ).astype(np.float32)


def create_embeddings(texts_tokens, w2v_model):

    return np.array([
        document_embedding(tokens, w2v_model)
        for tokens in texts_tokens
    ])


# ============================================================
# MODELO MLP
# ============================================================

class MLPClassifier(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden1=128,
        hidden2=64,
        dropout=0.3,
        num_classes=3
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_dim, hidden1),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(hidden1, hidden2),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(hidden2, num_classes)
        )

    def forward(self, x):

        return self.network(x)


# ============================================================
# TREINAMENTO DO MLP
# ============================================================

def train_mlp(
    X_train,
    y_train,
    X_val,
    y_val,
    hidden1,
    hidden2,
    dropout,
    learning_rate,
    batch_size,
    epochs
):

    X_train = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    y_train = torch.tensor(
        y_train,
        dtype=torch.long
    )

    X_val = torch.tensor(
        X_val,
        dtype=torch.float32
    )

    y_val = torch.tensor(
        y_val,
        dtype=torch.long
    )

    train_dataset = TensorDataset(
        X_train,
        y_train
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    model = MLPClassifier(
        input_dim=X_train.shape[1],
        hidden1=hidden1,
        hidden2=hidden2,
        dropout=dropout
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    # --------------------------------------------------------
    # TREINAMENTO
    # --------------------------------------------------------

    model.train()

    for epoch in range(epochs):

        for batch_X, batch_y in train_loader:

            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(batch_X)

            loss = criterion(
                outputs,
                batch_y
            )

            loss.backward()

            optimizer.step()

    # --------------------------------------------------------
    # VALIDAÇÃO
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        X_val = X_val.to(DEVICE)

        outputs = model(X_val)

        predictions = torch.argmax(
            outputs,
            dim=1
        ).cpu().numpy()

    accuracy = accuracy_score(
        y_val.cpu().numpy(),
        predictions
    )

    precision = precision_score(
        y_val.cpu().numpy(),
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_val.cpu().numpy(),
        predictions,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_val.cpu().numpy(),
        predictions,
        average="macro",
        zero_division=0
    )

    del model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return accuracy, precision, recall, f1


# ============================================================
# AVALIAÇÃO DE UM INDIVÍDUO
# ============================================================

def evaluate_individual(params):

    print("\n==============================================")
    print("AVALIANDO INDIVÍDUO")
    print(params)
    print("==============================================")

    skf = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=SEED
    )

    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(tokenized_texts, labels),
        start=1
    ):

        print(f"\n--- Fold {fold}/3 ---")

        # ----------------------------------------------------
        # Dados do fold
        # ----------------------------------------------------

        train_tokens = [
            tokenized_texts[i]
            for i in train_idx
        ]

        val_tokens = [
            tokenized_texts[i]
            for i in val_idx
        ]

        y_train = labels[train_idx]
        y_val = labels[val_idx]

        # ----------------------------------------------------
        # WORD2VEC
        #
        # IMPORTANTE:
        # treinado SOMENTE no conjunto de treino do fold
        # ----------------------------------------------------

        w2v = train_word2vec(
            train_tokens,

            vector_size=params["vector_size"],

            window=params["window"],

            min_count=params["min_count"],

            sg=params["sg"],

            epochs=params["w2v_epochs"]
        )

        # ----------------------------------------------------
        # EMBEDDINGS
        # ----------------------------------------------------

        X_train = create_embeddings(
            train_tokens,
            w2v
        )

        X_val = create_embeddings(
            val_tokens,
            w2v
        )

        # ----------------------------------------------------
        # MLP
        # ----------------------------------------------------

        metrics = train_mlp(

            X_train,
            y_train,

            X_val,
            y_val,

            hidden1=params["hidden1"],

            hidden2=params["hidden2"],

            dropout=params["dropout"],

            learning_rate=params["learning_rate"],

            batch_size=params["batch_size"],

            epochs=params["mlp_epochs"]
        )

        accuracy, precision, recall, f1 = metrics

        print(
            f"Accuracy:  {accuracy:.4f}"
        )

        print(
            f"Precision: {precision:.4f}"
        )

        print(
            f"Recall:    {recall:.4f}"
        )

        print(
            f"F1-macro:  {f1:.4f}"
        )

        fold_metrics.append(metrics)

        del w2v
        del X_train
        del X_val

        gc.collect()

    # ========================================================
    # MÉDIA DOS FOLDS
    # ========================================================

    fold_metrics = np.array(
        fold_metrics
    )

    mean_accuracy = fold_metrics[:, 0].mean()
    mean_precision = fold_metrics[:, 1].mean()
    mean_recall = fold_metrics[:, 2].mean()
    mean_f1 = fold_metrics[:, 3].mean()

    std_accuracy = fold_metrics[:, 0].std()
    std_f1 = fold_metrics[:, 3].std()

    print("\n==============================================")
    print("RESULTADO DO INDIVÍDUO")
    print("==============================================")

    print(
        f"Accuracy:  {mean_accuracy:.4f} ± {std_accuracy:.4f}"
    )

    print(
        f"Precision: {mean_precision:.4f}"
    )

    print(
        f"Recall:    {mean_recall:.4f}"
    )

    print(
        f"F1-macro:  {mean_f1:.4f} ± {std_f1:.4f}"
    )

    return {
        "accuracy": mean_accuracy,
        "precision": mean_precision,
        "recall": mean_recall,
        "f1": mean_f1
    }


# ============================================================
# ESPAÇO DE BUSCA
# ============================================================

SEARCH_SPACE = {

    "vector_size": [100, 200],

    "window": [5, 7],

    "min_count": [1, 2],

    "sg": [0, 1],

    "w2v_epochs": [10, 20],

    "hidden1": [128, 256],

    "hidden2": [64, 128],

    "dropout": [0.2, 0.3],

    "learning_rate": [1e-3, 1e-2],

    "batch_size": [32, 64],

    "mlp_epochs": [10, 20]
}


# ============================================================
# GERAR INDIVÍDUO ALEATÓRIO
# ============================================================

def random_individual():

    return {
        key: random.choice(values)
        for key, values in SEARCH_SPACE.items()
    }


# ============================================================
# ALGORITMO GENÉTICO
# ============================================================

POPULATION_SIZE = 8
N_GENERATIONS = 5
MUTATION_RATE = 0.20


def fitness(metrics):

    # Accuracy é o critério principal
    # F1-macro é o critério secundário

    return (
        metrics["accuracy"],
        metrics["f1"]
    )


def crossover(parent1, parent2):

    child = {}

    for key in SEARCH_SPACE:

        if random.random() < 0.5:

            child[key] = parent1[key]

        else:

            child[key] = parent2[key]

    return child


def mutate(individual):

    child = individual.copy()

    for key in SEARCH_SPACE:

        if random.random() < MUTATION_RATE:

            child[key] = random.choice(
                SEARCH_SPACE[key]
            )

    return child


# ============================================================
# EXECUÇÃO DO AG
# ============================================================

def genetic_algorithm():

    population = [
        random_individual()
        for _ in range(POPULATION_SIZE)
    ]

    best_individual = None
    best_metrics = None

    history = []

    for generation in range(N_GENERATIONS):

        print("\n\n")
        print("##############################################")
        print(
            f"GERAÇÃO {generation + 1}/{N_GENERATIONS}"
        )
        print("##############################################")

        evaluated_population = []

        for i, individual in enumerate(population):

            print(
                f"\nIndivíduo "
                f"{i + 1}/{POPULATION_SIZE}"
            )

            metrics = evaluate_individual(
                individual
            )

            evaluated_population.append({
                "params": individual,
                "metrics": metrics
            })

        # ----------------------------------------------------
        # RANKING
        # ----------------------------------------------------

        evaluated_population.sort(
            key=lambda x: fitness(x["metrics"]),
            reverse=True
        )

        generation_best = evaluated_population[0]

        print("\n==============================================")
        print("MELHOR DA GERAÇÃO")
        print("==============================================")

        print(
            generation_best["params"]
        )

        print(
            generation_best["metrics"]
        )

        # ----------------------------------------------------
        # MELHOR GLOBAL
        # ----------------------------------------------------

        if (
            best_metrics is None
            or fitness(generation_best["metrics"])
            > fitness(best_metrics)
        ):

            best_individual = (
                generation_best["params"]
            )

            best_metrics = (
                generation_best["metrics"]
            )

        history.append({
            "generation": generation + 1,
            "accuracy": generation_best["metrics"]["accuracy"],
            "f1": generation_best["metrics"]["f1"]
        })

        # ----------------------------------------------------
        # SELEÇÃO
        #
        # Mantemos os 2 melhores
        # ----------------------------------------------------

        parents = [
            evaluated_population[0]["params"],
            evaluated_population[1]["params"]
        ]

        # ----------------------------------------------------
        # NOVA POPULAÇÃO
        # ----------------------------------------------------

        new_population = parents.copy()

        while len(new_population) < POPULATION_SIZE:

            parent1 = random.choice(parents)
            parent2 = random.choice(parents)

            child = crossover(
                parent1,
                parent2
            )

            child = mutate(child)

            new_population.append(child)

        population = new_population

    return (
        best_individual,
        best_metrics,
        history
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    best_params, best_metrics, history = genetic_algorithm()

    print("\n\n")
    print("##############################################")
    print("RESULTADO FINAL")
    print("##############################################")

    print("\nMelhores parâmetros:")

    for key, value in best_params.items():

        print(
            f"{key}: {value}"
        )

    print("\nMelhores métricas:")

    print(
        f"Accuracy:  {best_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: {best_metrics['precision']:.4f}"
    )

    print(
        f"Recall:    {best_metrics['recall']:.4f}"
    )

    print(
        f"F1-macro:  {best_metrics['f1']:.4f}"
    )

    # --------------------------------------------------------
    # HISTÓRICO
    # --------------------------------------------------------

    history_df = pd.DataFrame(history)

    history_path = os.path.join(
        BASE_DIR,
        "historico_ag_word2vec_mlp.xlsx"
    )

    history_df.to_excel(
        history_path,
        index=False
    )

    print(
        f"\nHistórico salvo em: {history_path}"
    )