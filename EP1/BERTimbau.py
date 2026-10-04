# ============================================================
# 1. IMPORTS
# ============================================================

import os
import gc
import random
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup
)

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 2. CONFIGURAÇÃO
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("CONFIGURAÇÃO")
print("=" * 60)

print("PyTorch:", torch.__version__)
print("Device:", DEVICE)--

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# 3. CAMINHOS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_PATH = os.path.join(BASE_DIR, "train.xlsx")
TEST_PATH = os.path.join(BASE_DIR, "test1.xlsx")

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"

OUTPUT_PATH = "test1_predictions_transformer.xlsx"


# ============================================================
# 4. CARREGAR DADOS
# ============================================================

train_df = pd.read_excel(TRAIN_PATH)
test_df = pd.read_excel(TEST_PATH)

print("\nTRAIN:", train_df.shape)
print("TEST :", test_df.shape)

print("\nColunas:")
print(train_df.columns.tolist())


# ============================================================
# 5. PREPARAR DADOS
# ============================================================

X = (
    train_df["resp_text"]
    .fillna("")
    .astype(str)
    .values
)

X_test = (
    test_df["resp_text"]
    .fillna("")
    .astype(str)
    .values
)

label_map = {
    "c1": 0,
    "c234": 1,
    "c5": 2
}

inverse_label_map = {
    0: "c1",
    1: "c234",
    2: "c5"
}

y = np.array([
    label_map[label]
    for label in train_df["clarity"].astype(str)
])

print("\nDistribuição das classes:")
print(train_df["clarity"].value_counts())

print("\nTreino:", len(X))
print("Teste :", len(X_test))


# ============================================================
# 6. TOKENIZER
# ============================================================

print("\nCarregando tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("Tokenizer carregado.")


# ============================================================
# 7. DATASET
# ============================================================

class TextDataset(Dataset):

    def __init__(
        self,
        texts,
        labels=None,
        max_length=256
    ):

        self.texts = texts
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        encoding = tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length
        )

        item = {
            key: torch.tensor(
                value,
                dtype=torch.long
            )
            for key, value in encoding.items()
        }

        if self.labels is not None:

            item["labels"] = torch.tensor(
                self.labels[idx],
                dtype=torch.long
            )

        return item


# ============================================================
# 8. COLLATOR
# ============================================================

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding=True
)


# ============================================================
# 9. CRIAR MODELO
# ============================================================

def create_model():

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3
    )

    return model.to(DEVICE)


# ============================================================
# 10. TREINAR E AVALIAR UM FOLD
# ============================================================

def train_one_fold(
    X_train,
    y_train,
    X_val,
    y_val,
    learning_rate,
    batch_size,
    epochs,
    weight_decay,
    max_length
):

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_dataset = TextDataset(
        X_train,
        y_train,
        max_length=max_length
    )

    val_dataset = TextDataset(
        X_val,
        y_val,
        max_length=max_length
    )

    # --------------------------------------------------------
    # DATALOADERS
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator
    )

    # --------------------------------------------------------
    # MODELO
    # --------------------------------------------------------

    model = create_model()

    # --------------------------------------------------------
    # OTIMIZADOR
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    total_steps = (
        len(train_loader) * epochs
    )

    warmup_steps = int(
        total_steps * 0.1
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # --------------------------------------------------------
    # TREINAMENTO
    # --------------------------------------------------------

    model.train()

    for epoch in range(epochs):

        total_loss = 0.0

        for batch in train_loader:

            batch = {
                key: value.to(DEVICE)
                for key, value in batch.items()
            }

            optimizer.zero_grad()

            outputs = model(**batch)

            loss = outputs.loss

            loss.backward()

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = (
            total_loss /
            len(train_loader)
        )

        print(
            f"      Epoch "
            f"{epoch + 1}/{epochs} "
            f"- Loss: {avg_loss:.4f}"
        )

    # --------------------------------------------------------
    # VALIDAÇÃO
    # --------------------------------------------------------

    model.eval()

    predictions = []
    true_labels = []

    with torch.no_grad():

        for batch in val_loader:

            labels = batch["labels"]

            batch = {
                key: value.to(DEVICE)
                for key, value in batch.items()
            }

            outputs = model(**batch)

            preds = torch.argmax(
                outputs.logits,
                dim=1
            )

            predictions.extend(
                preds.cpu().numpy()
            )

            true_labels.extend(
                labels.cpu().numpy()
            )

    # --------------------------------------------------------
    # MÉTRICAS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        true_labels,
        predictions
    )

    precision = precision_score(
        true_labels,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        true_labels,
        predictions,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        true_labels,
        predictions,
        average="macro",
        zero_division=0
    )

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    # --------------------------------------------------------
    # LIMPEZA
    # --------------------------------------------------------

    del model
    del optimizer
    del scheduler
    del train_loader
    del val_loader

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


# ============================================================
# 11. ESPAÇO DE HIPERPARÂMETROS
# ============================================================

SEARCH_SPACE = {

    "learning_rate": [
        1e-5,
        2e-5,
        3e-5,
        5e-5
    ],

    "batch_size": [
        8,
        16
    ],

    "epochs": [
        2,
        3,
        4,
        5
    ],

    "weight_decay": [
        0.0,
        0.01,
        0.05,
        0.1
    ],

    "max_length": [
        128,
        256,
        384,
        512
    ]
}


# ============================================================
# 12. PARÂMETROS DO ALGORITMO GENÉTICO
# ============================================================

POPULATION_SIZE = 2
N_GENERATIONS = 1
MUTATION_RATE = 0.20

N_FOLDS = 5


# ============================================================
# 13. CRIAR INDIVÍDUO
# ============================================================

def random_individual():

    return {
        key: random.choice(values)
        for key, values in SEARCH_SPACE.items()
    }


# ============================================================
# 14. AVALIAR INDIVÍDUO
# ============================================================

def evaluate_individual(individual):

    print("\n")
    print("=" * 70)
    print("AVALIANDO INDIVÍDUO")
    print("=" * 70)

    print(individual)

    skf = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=SEED
    )

    fold_results = []

    # --------------------------------------------------------
    # CROSS VALIDATION
    # --------------------------------------------------------

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(X, y),
        start=1
    ):

        print(
            f"\n"
            f"-----------------------------\n"
            f"FOLD {fold}/{N_FOLDS}\n"
            f"-----------------------------"
        )

        X_train = X[train_idx]
        X_val = X[val_idx]

        y_train = y[train_idx]
        y_val = y[val_idx]

        metrics = train_one_fold(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,

            learning_rate=(
                individual["learning_rate"]
            ),

            batch_size=(
                individual["batch_size"]
            ),

            epochs=(
                individual["epochs"]
            ),

            weight_decay=(
                individual["weight_decay"]
            ),

            max_length=(
                individual["max_length"]
            )
        )

        fold_results.append(metrics)

        print(
            f"\n"
            f"      Accuracy : "
            f"{metrics['accuracy']:.4f}\n"
            f"      Precision: "
            f"{metrics['precision']:.4f}\n"
            f"      Recall   : "
            f"{metrics['recall']:.4f}\n"
            f"      F1-macro : "
            f"{metrics['f1']:.4f}"
        )

    # --------------------------------------------------------
    # MÉDIA DOS FOLDS
    # --------------------------------------------------------

    mean_accuracy = np.mean([
        result["accuracy"]
        for result in fold_results
    ])

    mean_precision = np.mean([
        result["precision"]
        for result in fold_results
    ])

    mean_recall = np.mean([
        result["recall"]
        for result in fold_results
    ])

    mean_f1 = np.mean([
        result["f1"]
        for result in fold_results
    ])

    std_accuracy = np.std([
        result["accuracy"]
        for result in fold_results
    ])

    std_f1 = np.std([
        result["f1"]
        for result in fold_results
    ])

    results = {

        "accuracy": mean_accuracy,
        "precision": mean_precision,
        "recall": mean_recall,
        "f1": mean_f1,

        "accuracy_std": std_accuracy,
        "f1_std": std_f1
    }

    print("\n")
    print("MÉDIA DOS 5 FOLDS")
    print("-" * 40)

    print(
        f"Accuracy : "
        f"{mean_accuracy:.4f} "
        f"(± {std_accuracy:.4f})"
    )

    print(
        f"Precision: "
        f"{mean_precision:.4f}"
    )

    print(
        f"Recall   : "
        f"{mean_recall:.4f}"
    )

    print(
        f"F1-macro : "
        f"{mean_f1:.4f} "
        f"(± {std_f1:.4f})"
    )

    return results


# ============================================================
# 15. SELEÇÃO POR TORNEIO
# ============================================================

def tournament_selection(
    population,
    evaluated_population,
    tournament_size=2
):

    selected = random.sample(
        evaluated_population,
        tournament_size
    )

    # Accuracy é o primeiro critério.
    # F1 é o segundo.
    selected.sort(
        key=lambda item: (
            item["metrics"]["accuracy"],
            item["metrics"]["f1"]
        ),
        reverse=True
    )

    return selected[0]["individual"].copy()


# ============================================================
# 16. CROSSOVER
# ============================================================

def crossover(
    parent1,
    parent2
):

    child = {}

    for key in SEARCH_SPACE.keys():

        if random.random() < 0.5:

            child[key] = parent1[key]

        else:

            child[key] = parent2[key]

    return child


# ============================================================
# 17. MUTAÇÃO
# ============================================================

def mutate(
    individual,
    mutation_rate=MUTATION_RATE
):

    child = individual.copy()

    for key, values in SEARCH_SPACE.items():

        if random.random() < mutation_rate:

            child[key] = random.choice(
                values
            )

    return child


# ============================================================
# 18. ALGORITMO GENÉTICO
# ============================================================

def genetic_algorithm():

    # --------------------------------------------------------
    # POPULAÇÃO INICIAL
    # --------------------------------------------------------

    population = [
        random_individual()
        for _ in range(POPULATION_SIZE)
    ]

    best_individual = None
    best_metrics = None

    history = []

    # --------------------------------------------------------
    # GERAÇÕES
    # --------------------------------------------------------

    for generation in range(
        N_GENERATIONS
    ):

        print("\n\n")
        print("#" * 80)
        print(
            f"GERAÇÃO "
            f"{generation + 1}/"
            f"{N_GENERATIONS}"
        )
        print("#" * 80)

        evaluated_population = []

        # ----------------------------------------------------
        # AVALIAR POPULAÇÃO
        # ----------------------------------------------------

        for i, individual in enumerate(
            population
        ):

            print("\n")
            print(
                f"INDIVÍDUO "
                f"{i + 1}/"
                f"{POPULATION_SIZE}"
            )

            metrics = evaluate_individual(
                individual
            )

            evaluated_population.append({
                "individual": individual,
                "metrics": metrics
            })

            # Guardar histórico
            history.append({

                "generation": (
                    generation + 1
                ),

                "individual": (
                    i + 1
                ),

                **individual,

                "accuracy": (
                    metrics["accuracy"]
                ),

                "precision": (
                    metrics["precision"]
                ),

                "recall": (
                    metrics["recall"]
                ),

                "f1": (
                    metrics["f1"]
                ),

                "accuracy_std": (
                    metrics["accuracy_std"]
                ),

                "f1_std": (
                    metrics["f1_std"]
                )
            })

        # ----------------------------------------------------
        # RANKING
        # ----------------------------------------------------

        evaluated_population.sort(
            key=lambda item: (
                item["metrics"]["accuracy"],
                item["metrics"]["f1"]
            ),
            reverse=True
        )

        generation_best = (
            evaluated_population[0]
        )

        generation_metrics = (
            generation_best["metrics"]
        )

        print("\n")
        print(
            "=" * 70
        )
        print(
            "MELHOR DA GERAÇÃO"
        )
        print(
            "=" * 70
        )

        print(
            generation_best["individual"]
        )

        print(
            f"\nAccuracy : "
            f"{generation_metrics['accuracy']:.4f}"
        )

        print(
            f"Precision: "
            f"{generation_metrics['precision']:.4f}"
        )

        print(
            f"Recall   : "
            f"{generation_metrics['recall']:.4f}"
        )

        print(
            f"F1-macro : "
            f"{generation_metrics['f1']:.4f}"
        )

        # ----------------------------------------------------
        # MELHOR GLOBAL
        # ----------------------------------------------------

        if best_metrics is None:

            is_better = True

        elif (
            generation_metrics["accuracy"]
            > best_metrics["accuracy"]
        ):

            is_better = True

        elif (
            generation_metrics["accuracy"]
            == best_metrics["accuracy"]
            and
            generation_metrics["f1"]
            > best_metrics["f1"]
        ):

            is_better = True

        else:

            is_better = False

        if is_better:

            best_individual = (
                generation_best[
                    "individual"
                ].copy()
            )

            best_metrics = (
                generation_metrics.copy()
            )

            print(
                "\n*** NOVO MELHOR GLOBAL ***"
            )

        # ----------------------------------------------------
        # ÚLTIMA GERAÇÃO
        # ----------------------------------------------------

        if (
            generation
            ==
            N_GENERATIONS - 1
        ):
            break

        # ----------------------------------------------------
        # ELITISMO
        # ----------------------------------------------------

        elite = (
            evaluated_population[0]
            ["individual"]
            .copy()
        )

        new_population = [
            elite
        ]

        # ----------------------------------------------------
        # NOVA POPULAÇÃO
        # ----------------------------------------------------

        while len(new_population) < POPULATION_SIZE:

            parent1 = tournament_selection(
                population,
                evaluated_population
            )

            parent2 = tournament_selection(
                population,
                evaluated_population
            )

            child = crossover(
                parent1,
                parent2
            )

            child = mutate(
                child
            )

            new_population.append(
                child
            )

        population = new_population

    history_df = pd.DataFrame(
        history
    )

    return (
        best_individual,
        best_metrics,
        history_df
    )


# ============================================================
# 19. EXECUTAR ALGORITMO GENÉTICO
# ============================================================

best_params, best_metrics, ga_history = (
    genetic_algorithm()
)


# ============================================================
# 20. RESULTADO DO AG
# ============================================================

print("\n\n")
print("=" * 80)
print("RESULTADO FINAL DO ALGORITMO GENÉTICO")
print("=" * 80)

print("\nMelhores hiperparâmetros:")

for key, value in best_params.items():

    print(
        f"{key}: {value}"
    )

print("\nMétricas:")

print(
    f"Accuracy : "
    f"{best_metrics['accuracy']:.4f}"
)

print(
    f"Precision: "
    f"{best_metrics['precision']:.4f}"
)

print(
    f"Recall   : "
    f"{best_metrics['recall']:.4f}"
)

print(
    f"F1-macro : "
    f"{best_metrics['f1']:.4f}"
)


# ============================================================
# 21. HISTÓRICO DO ALGORITMO GENÉTICO
# ============================================================

ga_history = ga_history.sort_values(
    [
        "accuracy",
        "f1"
    ],
    ascending=False
)

print("\nTOP CONFIGURAÇÕES:")

print(
    ga_history.head(10)
)


# ============================================================
# 22. TREINAR MODELO FINAL COM TODO O TRAIN
# ============================================================

print("\n")
print("=" * 80)
print("TREINAMENTO DO MODELO FINAL")
print("=" * 80)

final_model = create_model()

final_train_dataset = TextDataset(
    X,
    y,
    max_length=best_params["max_length"]
)

final_test_dataset = TextDataset(
    X_test,
    labels=None,
    max_length=best_params["max_length"]
)

final_train_loader = DataLoader(
    final_train_dataset,
    batch_size=best_params["batch_size"],
    shuffle=True,
    collate_fn=data_collator
)

final_test_loader = DataLoader(
    final_test_dataset,
    batch_size=best_params["batch_size"],
    shuffle=False,
    collate_fn=data_collator
)

final_optimizer = torch.optim.AdamW(
    final_model.parameters(),
    lr=best_params["learning_rate"],
    weight_decay=best_params["weight_decay"]
)

total_steps = (
    len(final_train_loader)
    *
    best_params["epochs"]
)

warmup_steps = int(
    total_steps * 0.1
)

final_scheduler = (
    get_linear_schedule_with_warmup(
        final_optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
)


# ============================================================
# 23. FINE-TUNING FINAL
# ============================================================

final_model.train()

for epoch in range(
    best_params["epochs"]
):

    total_loss = 0.0

    for batch in final_train_loader:

        batch = {
            key: value.to(DEVICE)
            for key, value in batch.items()
        }

        final_optimizer.zero_grad()

        outputs = final_model(
            **batch
        )

        loss = outputs.loss

        loss.backward()

        final_optimizer.step()
        final_scheduler.step()

        total_loss += loss.item()

    avg_loss = (
        total_loss /
        len(final_train_loader)
    )

    print(
        f"Epoch "
        f"{epoch + 1}/"
        f"{best_params['epochs']} "
        f"- Loss: {avg_loss:.4f}"
    )


# ============================================================
# 24. PREDIÇÃO NO TESTE
# ============================================================

final_model.eval()

test_predictions = []

with torch.no_grad():

    for batch in final_test_loader:

        batch = {
            key: value.to(DEVICE)
            for key, value in batch.items()
        }

        outputs = final_model(
            **batch
        )

        predictions = torch.argmax(
            outputs.logits,
            dim=1
        )

        test_predictions.extend(
            predictions.cpu().numpy()
        )


# ============================================================
# 25. CONVERTER CÓDIGOS PARA CLASSES
# ============================================================

test_predictions_labels = [
    inverse_label_map[pred]
    for pred in test_predictions
]


# ============================================================
# 26. SALVAR RESULTADO
# ============================================================

output_df = test_df.copy()

output_df["clarity"] = (
    test_predictions_labels
)

output_df.to_excel(
    OUTPUT_PATH,
    index=False
)

print("\n")
print("=" * 80)
print("FINALIZADO")
print("=" * 80)

print(
    "Arquivo salvo em:",
    OUTPUT_PATH
)

print("\nDistribuição das previsões:")

print(
    output_df["clarity"].value_counts()
)