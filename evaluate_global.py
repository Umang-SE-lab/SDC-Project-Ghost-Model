import os
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

from torch.utils.data import DataLoader, TensorDataset

from model import Ghost1D_GRU
from dataset import CICIoTDataset

# --------------------------------------------------
# Configuration
# --------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "saved_models/global_model_round_10.pth"
BATCH_SIZE = 256

os.makedirs("results_global", exist_ok=True)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

print("Loading dataset...")

dataset = CICIoTDataset()

(
    _,
    _,
    _,
    _,
    x_test,
    y_test,
    _,
    _,
    features,
) = dataset.load_data()

print("Dataset loaded successfully.")

testloader = DataLoader(
    TensorDataset(
        torch.tensor(x_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    ),
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# --------------------------------------------------
# Load Global Model
# --------------------------------------------------

model = Ghost1D_GRU(
    input_dim=len(features),
    num_classes=2,
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# --------------------------------------------------
# Prediction
# --------------------------------------------------

all_preds = []
all_labels = []

with torch.no_grad():
    for x, y in testloader:

        x = x.to(DEVICE)

        outputs = model(x)

        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.numpy())

# --------------------------------------------------
# Metrics
# --------------------------------------------------

accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, zero_division=0)
recall = recall_score(all_labels, all_preds, zero_division=0)
f1 = f1_score(all_labels, all_preds, zero_division=0)

print("\n========== GLOBAL MODEL RESULTS ==========\n")
print(f"Accuracy : {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall   : {recall*100:.2f}%")
print(f"F1 Score : {f1*100:.2f}%")

print("\nClassification Report\n")
print(classification_report(
    all_labels,
    all_preds,
    target_names=["Attack", "Benign"],
))

# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(all_labels, all_preds)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Attack", "Benign"],
)

disp.plot(cmap="Blues")

plt.tight_layout()

plt.savefig("results_global/confusion_matrix.png")
plt.close()

# --------------------------------------------------
# Save Metrics
# --------------------------------------------------

metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1,
}

pd.DataFrame([metrics]).to_csv(
    "results_global/metrics.csv",
    index=False,
)

with open("results_global/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\nSaved:")
print("results_global/confusion_matrix.png")
print("results_global/metrics.csv")
print("results_global/metrics.json")