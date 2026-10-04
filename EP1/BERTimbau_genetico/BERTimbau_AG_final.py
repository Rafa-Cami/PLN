import os
import gc
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

SEED = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "train.xlsx")
TEST_PATH = os.path.join(BASE_DIR, "test1.xlsx")
OUTPUT_PATH = os.path.join(BASE_DIR, "test1_predictions_transformer.xlsx")

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"

N_FOLDS = 5
POPULATION_SIZE = 8
N_GENERATIONS = 5
MUTATION_RATE = 0.20

# 128 foi retirado: a análise dos seus dados mostrou 63,12% de truncamento.
SEARCH_SPACE = {
    "learning_rate": [1e-5, 2e-5, 3e-5, 5e-5],
    "batch_size": [8, 16],
    "epochs": [2, 3, 4, 5],
    "weight_decay": [0.0, 0.001, 0.01, 0.05, 0.1],
    "max_length": [256, 384, 512],
}

label_map = {"c1": 0, "c234": 1, "c5": 2}
inverse_label_map = {0: "c1", 1: "c234", 2: "c5"}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("BERTIMBAU - CLASSIFICAÇÃO DE TEXTOS")
print("=" * 70)
print(f"PyTorch: {torch.__version__}")
print(f"Dispositivo: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("=" * 70)

print("\nCarregando dados...")
train_df = pd.read_excel(TRAIN_PATH)
test_df = pd.read_excel(TEST_PATH)

X = train_df["resp_text"].fillna("").astype(str).tolist()
y = train_df["clarity"].map(label_map).values
X_test = test_df["resp_text"].fillna("").astype(str).tolist()

print(f"Train: {train_df.shape}")
print(f"Test:  {test_df.shape}")
print("\nDistribuição das classes:")
print(train_df["clarity"].value_counts())

print("\nCarregando tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

class TextDataset(Dataset):
    def __init__(self, texts, labels=None, max_length=256):
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
        item = {k: torch.tensor(v, dtype=torch.long) for k, v in encoding.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def create_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3
    )
    return model.to(DEVICE)

def train_and_evaluate_fold(X_train, y_train, X_val, y_val, params, fold_number):
    set_seed(SEED + fold_number)

    batch_size = params["batch_size"]
    epochs = params["epochs"]
    learning_rate = params["learning_rate"]
    weight_decay = params["weight_decay"]
    max_length = params["max_length"]

    train_dataset = TextDataset(X_train, y_train, max_length)
    val_dataset = TextDataset(X_val, y_val, max_length)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=data_collator, num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=data_collator, num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    model = create_model()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * 0.10)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0

        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(
            f"      Fold {fold_number} | "
            f"Época {epoch + 1}/{epochs} | Loss: {avg_loss:.4f}"
        )

    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in val_loader:
            labels = batch.pop("labels")
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy().tolist())
            true_labels.extend(labels.numpy().tolist())

    metrics = {
        "accuracy": accuracy_score(true_labels, predictions),
        "precision": precision_score(true_labels, predictions, average="macro", zero_division=0),
        "recall": recall_score(true_labels, predictions, average="macro", zero_division=0),
        "f1": f1_score(true_labels, predictions, average="macro", zero_division=0),
    }

    del model, optimizer, scheduler, train_loader, val_loader
    del train_dataset, val_dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics

def evaluate_configuration(params):
    print("\n" + "=" * 70)
    print("CONFIGURAÇÃO")
    print("=" * 70)
    for key, value in params.items():
        print(f"{key}: {value}")
    print("=" * 70)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_results = []

    for fold_number, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\n--- Fold {fold_number}/{N_FOLDS} ---")

        X_train = [X[i] for i in train_idx]
        X_val = [X[i] for i in val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]

        metrics = train_and_evaluate_fold(
            X_train, y_train, X_val, y_val, params, fold_number
        )
        fold_results.append(metrics)

        print(
            f"      Accuracy:  {metrics['accuracy']:.4f}\n"
            f"      Precision: {metrics['precision']:.4f}\n"
            f"      Recall:    {metrics['recall']:.4f}\n"
            f"      F1-macro:  {metrics['f1']:.4f}"
        )

    def mean_std(name):
        values = [r[name] for r in fold_results]
        return float(np.mean(values)), float(np.std(values))

    accuracy, accuracy_std = mean_std("accuracy")
    precision, precision_std = mean_std("precision")
    recall, recall_std = mean_std("recall")
    f1, f1_std = mean_std("f1")

    result = {
        "params": params.copy(),
        "metrics": {
            "accuracy": accuracy, "accuracy_std": accuracy_std,
            "precision": precision, "precision_std": precision_std,
            "recall": recall, "recall_std": recall_std,
            "f1": f1, "f1_std": f1_std,
        },
        "fold_results": fold_results
    }

    print("\n" + "-" * 70)
    print("RESULTADO MÉDIO")
    print("-" * 70)
    print(f"Accuracy:  {accuracy:.4f} ± {accuracy_std:.4f}")
    print(f"Precision: {precision:.4f} ± {precision_std:.4f}")
    print(f"Recall:    {recall:.4f} ± {recall_std:.4f}")
    print(f"F1-macro:  {f1:.4f} ± {f1_std:.4f}")

    return result

PARAMETER_NAMES = list(SEARCH_SPACE.keys())

def random_individual():
    return {k: random.choice(SEARCH_SPACE[k]) for k in PARAMETER_NAMES}

def individual_key(params):
    return tuple(params[k] for k in PARAMETER_NAMES)

def crossover(parent1, parent2):
    return {
        k: (parent1[k] if random.random() < 0.5 else parent2[k])
        for k in PARAMETER_NAMES
    }

def mutate(individual):
    child = individual.copy()
    for key in PARAMETER_NAMES:
        if random.random() < MUTATION_RATE:
            alternatives = [v for v in SEARCH_SPACE[key] if v != child[key]]
            if alternatives:
                child[key] = random.choice(alternatives)
    return child

def run_genetic_algorithm():
    evaluated_cache = {}
    history = []

    population = []
    seen = set()
    while len(population) < POPULATION_SIZE:
        individual = random_individual()
        key = individual_key(individual)
        if key not in seen:
            population.append(individual)
            seen.add(key)

    for generation in range(1, N_GENERATIONS + 1):
        print("\n\n" + "#" * 70)
        print(f"GERAÇÃO {generation}/{N_GENERATIONS}")
        print("#" * 70)

        evaluated_population = []

        for number, individual in enumerate(population, start=1):
            print(f"\nINDIVÍDUO {number}/{len(population)}")
            key = individual_key(individual)

            if key in evaluated_cache:
                print("Configuração já avaliada. Usando resultado armazenado.")
                result = evaluated_cache[key]
            else:
                result = evaluate_configuration(individual)
                evaluated_cache[key] = result

            evaluated_population.append(result)
            m = result["metrics"]

            history.append({
                "generation": generation,
                "accuracy": m["accuracy"],
                "accuracy_std": m["accuracy_std"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "f1_std": m["f1_std"],
                **individual
            })

        # Accuracy é o critério principal; F1-macro é desempate.
        evaluated_population.sort(
            key=lambda item: (item["metrics"]["accuracy"], item["metrics"]["f1"]),
            reverse=True
        )

        best = evaluated_population[0]

        print("\n" + "=" * 70)
        print(f"MELHOR DA GERAÇÃO {generation}")
        print("=" * 70)
        print(f"Accuracy: {best['metrics']['accuracy']:.4f}")
        print(f"F1-macro: {best['metrics']['f1']:.4f}")
        print("\nParâmetros:")
        for key, value in best["params"].items():
            print(f"  {key}: {value}")

        if generation == N_GENERATIONS:
            break

        n_parents = max(2, POPULATION_SIZE // 2)
        parents = [r["params"] for r in evaluated_population[:n_parents]]

        new_population = parents.copy()
        new_keys = {individual_key(x) for x in new_population}

        # Garante que a nova população tenha exatamente POPULATION_SIZE indivíduos.
        while len(new_population) < POPULATION_SIZE:
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            child = mutate(crossover(parent1, parent2))
            key = individual_key(child)

            if key not in new_keys:
                new_population.append(child)
                new_keys.add(key)

        population = new_population

    all_results = list(evaluated_cache.values())
    all_results.sort(
        key=lambda item: (item["metrics"]["accuracy"], item["metrics"]["f1"]),
        reverse=True
    )

    history_df = pd.DataFrame(history)
    history_path = os.path.join(BASE_DIR, "ga_history_transformer.xlsx")
    history_df.to_excel(history_path, index=False)

    return all_results[0], all_results, history_df

def train_final_model(params):
    print("\n\n" + "#" * 70)
    print("TREINAMENTO FINAL COM TODOS OS DADOS")
    print("#" * 70)

    for key, value in params.items():
        print(f"  {key}: {value}")

    set_seed(SEED)

    max_length = params["max_length"]
    batch_size = params["batch_size"]
    epochs = params["epochs"]
    learning_rate = params["learning_rate"]
    weight_decay = params["weight_decay"]

    train_dataset = TextDataset(X, y, max_length)
    test_dataset = TextDataset(X_test, labels=None, max_length=max_length)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=data_collator, num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=data_collator, num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    model = create_model()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * 0.10)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0

        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        print(
            f"Época {epoch + 1}/{epochs} | "
            f"Loss: {total_loss / len(train_loader):.4f}"
        )

    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy().tolist())

    predicted_labels = [inverse_label_map[p] for p in predictions]

    output_df = test_df.copy()
    output_df["clarity"] = predicted_labels
    output_df.to_excel(OUTPUT_PATH, index=False)

    print("\n" + "=" * 70)
    print("PREDIÇÕES GERADAS")
    print("=" * 70)
    print(output_df["clarity"].value_counts())
    print(f"\nArquivo salvo em:\n{OUTPUT_PATH}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    best_global, all_results, history_df = run_genetic_algorithm()

    print("\n\n" + "#" * 70)
    print("MELHOR CONFIGURAÇÃO GLOBAL")
    print("#" * 70)

    m = best_global["metrics"]
    print(f"Accuracy:  {m['accuracy']:.4f} ± {m['accuracy_std']:.4f}")
    print(f"Precision: {m['precision']:.4f} ± {m['precision_std']:.4f}")
    print(f"Recall:    {m['recall']:.4f} ± {m['recall_std']:.4f}")
    print(f"F1-macro:  {m['f1']:.4f} ± {m['f1_std']:.4f}")

    print("\nParâmetros vencedores:")
    for key, value in best_global["params"].items():
        print(f"  {key}: {value}")

    top_rows = []
    for rank, result in enumerate(all_results[:10], start=1):
        m = result["metrics"]
        top_rows.append({
            "rank": rank,
            "accuracy": m["accuracy"],
            "accuracy_std": m["accuracy_std"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "f1_std": m["f1_std"],
            **result["params"]
        })

    top_df = pd.DataFrame(top_rows)
    top_path = os.path.join(BASE_DIR, "top_configurations_transformer.xlsx")
    top_df.to_excel(top_path, index=False)

    print("\nTOP 10 CONFIGURAÇÕES:")
    print(top_df.to_string(index=False))
    print(f"\nRanking salvo em: {top_path}")
    print(f"Histórico salvo em: {os.path.join(BASE_DIR, 'ga_history_transformer.xlsx')}")

    # O test1.xlsx só entra aqui, depois da escolha dos hiperparâmetros.
    train_final_model(best_global["params"])

    print("\n" + "=" * 70)
    print("PROCESSO CONCLUÍDO")
    print("=" * 70)

if __name__ == "__main__":
    main()
