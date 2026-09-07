import os
import torch
import flwr as fl
import csv

from flwr.server.strategy import FedAvg
from flwr.server import ServerConfig

from model import Ghost1D_GRU
from utils import set_weights

CSV_FILE = "results/federated_metrics.csv"

os.makedirs("results", exist_ok=True)

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Round",
            "Loss",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
        ])

# --------------------------------------------------
# Create folder for global models
# --------------------------------------------------

os.makedirs("saved_models", exist_ok=True)

INPUT_DIM = 39          # change if your feature count is different
NUM_CLASSES = 2


class SaveModelStrategy(FedAvg):
    def aggregate_fit(
        self,
        server_round,
        results,
        failures,
    ):
        aggregated = super().aggregate_fit(
            server_round,
            results,
            failures,
        )

        if aggregated is not None:
            parameters, _ = aggregated

            model = Ghost1D_GRU(
                input_dim=INPUT_DIM,
                num_classes=NUM_CLASSES,
            )

            set_weights(model, parameters)

            torch.save(
                model.state_dict(),
                f"saved_models/global_model_round_{server_round}.pth",
            )

            print(
                f"\nGlobal model saved: saved_models/global_model_round_{server_round}.pth"
            )

        return aggregated

    def aggregate_evaluate(
        self,
        server_round,
        results,
        failures,
    ):
        aggregated = super().aggregate_evaluate(
            server_round,
            results,
            failures,
        )

        if aggregated is not None:
            loss, metrics = aggregated

            accuracy = metrics.get("accuracy", 0)

            with open(CSV_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    server_round,
                    loss,
                    metrics["accuracy"],
                    metrics["precision"],
                    metrics["recall"],
                    metrics["f1"],
                ])

            print(
                f"Round {server_round} | Loss={loss:.4f} | Accuracy={accuracy:.4f}"
            )

        return aggregated

    def weighted_average(metrics):
        total_examples = sum(num_examples for num_examples, _ in metrics)

        accuracy = sum(
            num_examples * m["accuracy"]
            for num_examples, m in metrics
        ) / total_examples

        return {
            "accuracy": sum(n*m["accuracy"] for n,m in metrics)/total,
            "precision": sum(n*m["precision"] for n,m in metrics)/total,
            "recall": sum(n*m["recall"] for n,m in metrics)/total,
            "f1": sum(n*m["f1"] for n,m in metrics)/total,
        }


strategy = SaveModelStrategy(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=3,
    min_evaluate_clients=3,
    min_available_clients=3,
    evaluate_metrics_aggregation_fn=weighted_average
)


if __name__ == "__main__":
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=ServerConfig(num_rounds=10),
        strategy=strategy,
    )