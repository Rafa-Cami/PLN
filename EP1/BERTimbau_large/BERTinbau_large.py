import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from google.colab import drive
from tqdm import tqdm

# ==========================================
# 0. AMBIENTE E SEED OFICIAL (123)
# ==========================================

SEED = 123
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilizando dispositivo: {device}")

DATA_PATH = "/content/drive/MyDrive/epPLN/train.xlsx"
df = pd.read_excel(DATA_PATH)

def normalizar_texto(texto):
    if not isinstance(texto, str):
        return ""
    texto = re.sub(r'https?://\S+|www\.\S+', 'LINK_URL', texto)
    texto = re.sub(r'(?i)\b(lei|decreto)\s*(?:nº|n°|nº\.|º|°)?\s*\d+[\d\./]*', 'LEI_DECRETO', texto)
    texto = re.sub(r'\b\d{5,}\.\d{6}/\d{4}-\d{2}\b', 'NUMERO_PROCESSO', texto)
    texto = re.sub(r'\(\?\d{2,4}\)?\s*\d{4,5}[-\s]?\d{4}', 'NUMERO_TELEFONE', texto)
    return texto.lower()

df["resp_text_clean"] = df["resp_text"].apply(normalizar_texto)

label2id = {"c1": 0, "c234": 1, "c5": 2}
id2label = {v: k for k, v in label2id.items()}
df["label"] = df["clarity"].map(label2id)

df_train, df_val = train_test_split(df, test_size=0.20, random_state=SEED, stratify=df["label"])

# ==========================================
# 1. MODELO LARGE (335M Parâmetros)
# ==========================================
MODEL_NAME = 'neuralmind/bert-large-portuguese-cased'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
MAX_LEN = 256

def encode_texts(texts):
    return tokenizer([str(t) for t in texts], padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors="pt")

class TextDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

train_dataset = TextDataset(encode_texts(df_train["resp_text_clean"].to_numpy()), df_train["label"].to_numpy())
val_dataset = TextDataset(encode_texts(df_val["resp_text_clean"].to_numpy()), df_val["label"].to_numpy())

# Batch size 8 para caber na VRAM do Colab
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3).to(device)

# ==========================================
# 2. HIPERPARÂMETROS PARA MODELOS LARGE
# ==========================================
epochs = 4
lr = 1e-5 # Learning rate menor para não desestabilizar as 24 camadas
accum_steps = 4 # Batch efetivo = 8 * 4 = 32

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
total_steps = (len(train_loader) // accum_steps) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)
criterion = nn.CrossEntropyLoss()

# ==========================================
# 3. LOOP DE TREINO E VALIDAÇÃO
# ==========================================
print("\nIniciando Treinamento com BERTIMBAU LARGE...")
best_acc = 0.0

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    for step, batch in enumerate(tqdm(train_loader, desc=f"Época {epoch+1}/{epochs} [Treino]")):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels) / accum_steps
        loss.backward()
        
        if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
    # Validação
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(batch['labels'].numpy())
            
    acc = accuracy_score(labels_all, preds_all)
    f1_macro = f1_score(labels_all, preds_all, average='macro')
    print(f"\n--> Época {epoch+1} | Val Acc: {acc:.4f} | Val F1-Macro: {f1_macro:.4f}")
    
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), 'bertimbau_large_best.pt')
        print(f"    [NOVO RECORD] Modelo LARGE salvo! (Acc: {acc:.4f})")

# ==========================================
# 4. AVALIAÇÃO FINAL DO CAMPEÃO LARGE
# ==========================================
print(f"\n==================================================")
print(f"RESULTADO FINAL DO BERTIMBAU LARGE (Pico: {best_acc:.4f})")
print("==================================================")
model.load_state_dict(torch.load('bertimbau_large_best.pt'))
model.eval()

preds_all, labels_all = [], []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        preds_all.extend(preds.cpu().numpy())
        labels_all.extend(batch['labels'].numpy())

print(classification_report(labels_all, preds_all, target_names=["c1", "c234", "c5"], digits=4))