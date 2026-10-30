from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lime.lime_text import LimeTextExplainer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

# ==========================================
# 0. AMBIENTE E SEED OFICIAL (123)
# ==========================================

SEED = 123
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "train.xlsx"
CHECKPOINT_PATH = BASE_DIR / "bertimbau_large_best.pt"
REPORT_PATH = BASE_DIR / "bertimbau_large_lime.txt"


def write_report(text):
    with REPORT_PATH.open("a", encoding="utf-8") as report_file:
        report_file.write(f"{text}\n")


REPORT_PATH.write_text(
    "BERTIMBAU LARGE - MÉTRICAS E INTERPRETABILIDADE LIME\n"
    "====================================================\n",
    encoding="utf-8",
)
print(f"Utilizando dispositivo: {device}")
print(f"Dataset: {DATA_PATH}")
print(f"Relatório TXT: {REPORT_PATH}")
write_report(f"Dispositivo: {device}")
write_report(f"Dataset: {DATA_PATH}")

df = pd.read_excel(DATA_PATH)
print("Dimensões do dataset:", df.shape)
write_report(f"Dimensões do dataset: {df.shape}")


def normalizar_texto(texto):
    if not isinstance(texto, str):
        return ""
    texto = re.sub(r"https?://\S+|www\.\S+", "LINK_URL", texto)
    texto = re.sub(
        r"(?i)\b(lei|decreto)\s*(?:nº|n°|nº\.|º|°)?\s*\d+[\d\./]*",
        "LEI_DECRETO",
        texto,
    )
    texto = re.sub(r"\b\d{5,}\.\d{6}/\d{4}-\d{2}\b", "NUMERO_PROCESSO", texto)
    texto = re.sub(
        r"\(\?\d{2,4}\)?\s*\d{4,5}[-\s]?\d{4}",
        "NUMERO_TELEFONE",
        texto,
    )
    return texto.lower()


df["resp_text_clean"] = df["resp_text"].apply(normalizar_texto)

label2id = {"c1": 0, "c234": 1, "c5": 2}
id2label = {value: key for key, value in label2id.items()}
df["label"] = df["clarity"].map(label2id)

df_train, df_val = train_test_split(
    df,
    test_size=0.20,
    random_state=SEED,
    stratify=df["label"],
)

# ==========================================
# 1. MODELO LARGE (335M PARÂMETROS)
# ==========================================
MODEL_NAME = "neuralmind/bert-large-portuguese-cased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
MAX_LEN = 256


def encode_texts(texts):
    return tokenizer(
        [str(text) for text in texts],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )


class TextDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None

    def __getitem__(self, idx):
        item = {key: value[idx] for key, value in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


train_dataset = TextDataset(
    encode_texts(df_train["resp_text_clean"].to_numpy()),
    df_train["label"].to_numpy(),
)
val_dataset = TextDataset(
    encode_texts(df_val["resp_text_clean"].to_numpy()),
    df_val["label"].to_numpy(),
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
).to(device)

# ==========================================
# 2. HIPERPARÂMETROS PARA MODELOS LARGE
# ==========================================
epochs = 4
lr = 1e-5
accum_steps = 4

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
total_steps = (len(train_loader) // accum_steps) * epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(total_steps * 0.1),
    num_training_steps=total_steps,
)
criterion = nn.CrossEntropyLoss()
write_report(
    f"Configuração: epochs={epochs}, lr={lr}, "
    f"batch_size=8, accum_steps={accum_steps}, max_len={MAX_LEN}"
)

# ==========================================
# 3. LOOP DE TREINO E VALIDAÇÃO
# ==========================================
print("\nIniciando Treinamento com BERTIMBAU LARGE...")
best_acc = 0.0
write_report("\nMÉTRICAS POR ÉPOCA\n------------------")

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    for step, batch in enumerate(
        tqdm(train_loader, desc=f"Época {epoch + 1}/{epochs} [Treino]")
    ):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels) / accum_steps
        loss.backward()

        if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(batch["labels"].numpy())

    acc = accuracy_score(labels_all, preds_all)
    f1_macro = f1_score(labels_all, preds_all, average="macro")
    epoch_metrics = (
        f"Época {epoch + 1} | Val Acc: {acc:.4f} | "
        f"Val F1-Macro: {f1_macro:.4f}"
    )
    print(f"\n--> {epoch_metrics}")
    write_report(epoch_metrics)

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        checkpoint_message = (
            f"Melhor checkpoint salvo em {CHECKPOINT_PATH} "
            f"(Acc: {acc:.4f})"
        )
        print(f"    [NOVO RECORD] {checkpoint_message}")
        write_report(checkpoint_message)

# ==========================================
# 4. AVALIAÇÃO FINAL DO CAMPEÃO LARGE
# ==========================================
print("\n==================================================")
print(f"RESULTADO FINAL DO BERTIMBAU LARGE (Pico: {best_acc:.4f})")
print("==================================================")
write_report("\nAVALIAÇÃO FINAL\n---------------")
write_report(f"Melhor Val Acc por época: {best_acc:.4f}")

model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()

preds_all, labels_all = [], []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        preds_all.extend(preds.cpu().numpy())
        labels_all.extend(batch["labels"].numpy())

final_accuracy = accuracy_score(labels_all, preds_all)
final_f1_macro = f1_score(labels_all, preds_all, average="macro")
final_report = classification_report(
    labels_all,
    preds_all,
    target_names=["c1", "c234", "c5"],
    digits=4,
)
print(f"Val Accuracy: {final_accuracy:.4f}")
print(f"Val F1-Macro: {final_f1_macro:.4f}")
print(final_report)
write_report(f"Val Accuracy: {final_accuracy:.4f}")
write_report(f"Val F1-Macro: {final_f1_macro:.4f}")
write_report(final_report)

# ==========================================
# 5. INTERPRETABILIDADE LOCAL COM LIME
# ==========================================
# LIME consulta o modelo em textos perturbados; inferir em lotes limita
# a memória usada durante as explicações.
LIME_NUM_EXAMPLES = 3
LIME_NUM_SAMPLES = 500
LIME_BATCH_SIZE = 16
LIME_NUM_FEATURES = 10

lime_explainer = LimeTextExplainer(
    class_names=[id2label[index] for index in range(len(id2label))],
    random_state=SEED,
)


def predict_proba(texts):
    model.eval()
    probabilities = []

    for start in range(0, len(texts), LIME_BATCH_SIZE):
        batch_texts = [
            normalizar_texto(text)
            for text in texts[start : start + LIME_BATCH_SIZE]
        ]
        encodings = tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        encodings = {key: value.to(device) for key, value in encodings.items()}

        with torch.no_grad():
            logits = model(**encodings).logits
            batch_probabilities = torch.softmax(logits, dim=1)
        probabilities.extend(batch_probabilities.cpu().numpy())

    return np.asarray(probabilities)


print(
    f"\nGerando explicações LIME para "
    f"{min(LIME_NUM_EXAMPLES, len(df_val))} exemplos..."
)
write_report(
    f"\nEXPLICAÇÕES LOCAIS LIME "
    f"(num_samples={LIME_NUM_SAMPLES}, num_features={LIME_NUM_FEATURES})\n"
    "------------------------"
)

for example_number, (_, row) in enumerate(
    df_val.head(LIME_NUM_EXAMPLES).iterrows(),
    start=1,
):
    text = row["resp_text_clean"]
    true_label_id = int(row["label"])
    probabilities = predict_proba([text])
    predicted_label_id = int(np.argmax(probabilities[0]))

    explanation = lime_explainer.explain_instance(
        text,
        predict_proba,
        labels=tuple(range(len(id2label))),
        num_features=LIME_NUM_FEATURES,
        num_samples=LIME_NUM_SAMPLES,
    )

    probability_text = ", ".join(
        f"{id2label[index]}: {probability:.4f}"
        for index, probability in enumerate(probabilities[0])
    )
    sample_report = (
        f"\nExemplo {example_number}\n"
        f"Rótulo real: {id2label[true_label_id]}\n"
        f"Rótulo previsto: {id2label[predicted_label_id]}\n"
        f"Probabilidades: {probability_text}\n"
        f"Texto analisado: {text}\n"
    )

    for class_id, class_name in id2label.items():
        sample_report += (
            f"Contribuições LIME para {class_name}: "
            f"{explanation.as_list(label=class_id)}\n"
        )

    print(sample_report)
    write_report(sample_report)
