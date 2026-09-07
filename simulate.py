import flwr as fl

from dataset import CICIoTDataset
from client import FlowerClient
from strategy import SaveModelStrategy


def client_fn(context):
    client_id = context.node_config["partition-id"]
    return FlowerClient(client_id).to_client()


if __name__ == "__main__":

    dataset = CICIoTDataset()

    (
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        features,
    ) = dataset.load_data()

    strategy = SaveModelStrategy(
        input_dim=len(features),
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=3,
        config=fl.server.ServerConfig(
            num_rounds=10,
        ),
        strategy=strategy,
    )