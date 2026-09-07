import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from dataset import CICIoTDataset
from model import Ghost1D_GRU

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model(model_path):
    dataset = CICIoTDataset()

    (
        _,
        _,
        _,
        _,
        x_test,
        y_test,
        _,
        encoder,
        features,
    ) = dataset.load_data()

    model = Ghost1D_GRU(
        input_dim=len(features),
        num_classes=2,
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(model_path, map_location=DEVICE)
    )

    model.eval()

    x_test = torch.tensor(
        x_test,
        dtype=torch.float32,
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(x_test)
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print("\n========== RESULTS ==========")
    print(f"Accuracy : {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall   : {recall*100:.2f}%")
    print(f"F1 Score : {f1*100:.2f}%")

    print("\nClassification Report\n")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=encoder.classes_,
        )
    )

    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=encoder.classes_,
        yticklabels=encoder.classes_,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    os.makedirs("results", exist_ok=True)

    plt.savefig("results/confusion_matrix.png")

    metrics = {
        "Accuracy": float(accuracy),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1 Score": float(f1),
    }

    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    pd.DataFrame([metrics]).to_csv(
        "results/metrics.csv",
        index=False,
    )

    print("\nSaved:")
    print("results/confusion_matrix.png")
    print("results/metrics.csv")
    print("results/metrics.json")


if __name__ == "__main__":
    evaluate_model(
        "saved_models/global_model_round_10.pth"
    )