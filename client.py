import sys
import torch
from torch.utils.data import DataLoader, TensorDataset
import flwr as fl
import torch.nn as nn
from model import Ghost1D_GRU
from dataset import CICIoTDataset
from trainer import train_local
from utils import get_weights, set_weights
from sklearn.metrics import precision_score, recall_score, f1_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
LOCAL_EPOCHS = 2
LEARNING_RATE = 0.001


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = int(client_id)
        self.criterion = nn.CrossEntropyLoss()
        dataset = CICIoTDataset()

        (
            x_train,
            y_train,
            x_val,
            y_val,
            _,
            _,
            _,
            _,
            features,
        ) = dataset.load_client_data(self.client_id, 3)

        self.trainloader = DataLoader(
            TensorDataset(
                torch.tensor(x_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.long),
            ),
            batch_size=BATCH_SIZE,
            shuffle=True,
            pin_memory=True,
            num_workers=2,
            persistent_workers=True,
        )

        self.valloader = DataLoader(
            TensorDataset(
                torch.tensor(x_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.long),
            ),
            batch_size=BATCH_SIZE,
            shuffle=False,
            pin_memory=True,
            num_workers=2,
            persistent_workers=True,
        )

        self.model = Ghost1D_GRU(
            input_dim=len(features),
            num_classes=2,
        ).to(DEVICE)

    def get_parameters(self, config):
        return get_weights(self.model)

    def fit(self, parameters, config):
        set_weights(self.model, parameters)

        self.model = train_local(
            self.model,
            self.trainloader,
            epochs=LOCAL_EPOCHS,
            lr=LEARNING_RATE,
        )

        return (
            get_weights(self.model),
            len(self.trainloader.dataset),
            {},
        )

    def evaluate(self, parameters, config):
        set_weights(self.model, parameters)
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in self.trainloader:   # We'll change this to validation later
                x = x.to(DEVICE)
                y = y.to(DEVICE)

                outputs = self.model(x)
                loss = self.criterion(outputs, y)

                total_loss += loss.item() * x.size(0)
                _, predicted = torch.max(outputs, 1)

                total += y.size(0)
                correct += (predicted == y).sum().item()

        avg_loss = total_loss / total
        accuracy = correct / total
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x, y in self.valloader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)

                outputs = self.model(x)
                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        precision = precision_score(
            all_labels,
            all_preds,
            zero_division=0,
        )

        recall = recall_score(
            all_labels,
            all_preds,
            zero_division=0,
        )

        f1 = f1_score(
            all_labels,
            all_preds,
            zero_division=0,
        )

        print(f"Client {self.client_id} | Loss: {avg_loss:.4f} | Accuracy: {accuracy*100:.2f}%")

        return (
            avg_loss,
            total,
            {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            },
        )


if __name__ == "__main__":
    client_id = int(sys.argv[1])

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(client_id),
    )