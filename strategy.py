import os
import torch
import flwr as fl

from model import Ghost1D_GRU
from utils import set_weights

SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, input_dim):
        super().__init__(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=3,
            min_evaluate_clients=3,
            min_available_clients=3,
        )

        self.input_dim = input_dim

    def aggregate_fit(
        self,
        server_round,
        results,
        failures,
    ):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round,
            results,
            failures,
        )

        if aggregated_parameters is not None:

            model = Ghost1D_GRU(
                input_dim=self.input_dim,
                num_classes=2,
            )

            weights = fl.common.parameters_to_ndarrays(
                aggregated_parameters
            )

            set_weights(model, weights)

            save_path = os.path.join(
                SAVE_DIR,
                f"global_model_round_{server_round}.pth",
            )

            torch.save(
                model.state_dict(),
                save_path,
            )

            print(f"\nGlobal model saved: {save_path}")

        return aggregated_parameters, aggregated_metrics