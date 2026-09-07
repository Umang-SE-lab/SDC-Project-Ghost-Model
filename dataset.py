import json
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
_DATA_CACHE = None

class CICIoTDataset:
    def __init__(
        self,
        dataset_path="data/CICIOT23",
        sample_fraction=0.10,   # 10% while developing
        random_state=42,
    ):
        self.dataset_path = dataset_path
        self.sample_fraction = sample_fraction
        self.random_state = random_state

        self._loaded = False

    def _binary_labels(self, frame):
        return np.where(
            frame["label"].astype(str).str.contains(
                "benign",
                case=False,
                na=False,
            ),
            "Benign",
            "Attack",
        )

    def _load_once(self):
        global _DATA_CACHE

        if _DATA_CACHE is not None:
            (
                self.features,
                self.x_train,
                self.y_train,
                self.x_validation,
                self.y_validation,
                self.x_test,
                self.y_test,
                self.scaler,
                self.encoder,
            ) = _DATA_CACHE

            self._loaded = True
            return

        print("Loading dataset...")

        train = pd.read_csv(
            os.path.join(self.dataset_path, "train", "train.csv")
        )

        validation = pd.read_csv(
            os.path.join(self.dataset_path, "validation", "validation.csv")
        )

        test = pd.read_csv(
            os.path.join(self.dataset_path, "test", "test.csv")
        )

        if self.sample_fraction < 1.0:
            train = train.sample(
                frac=self.sample_fraction,
                random_state=self.random_state,
            )

            validation = validation.sample(
                frac=self.sample_fraction,
                random_state=self.random_state,
            )

            test = test.sample(
                frac=self.sample_fraction,
                random_state=self.random_state,
            )

        with open("selected_features.json") as f:
            features = json.load(f)

        features = [f for f in features if f in train.columns]

        x_train = train[features].apply(pd.to_numeric, errors="coerce")
        x_validation = validation[features].apply(pd.to_numeric, errors="coerce")
        x_test = test[features].apply(pd.to_numeric, errors="coerce")

        medians = x_train.median()

        x_train = (
            x_train.replace([np.inf, -np.inf], np.nan)
            .fillna(medians)
        )

        x_validation = (
            x_validation.replace([np.inf, -np.inf], np.nan)
            .fillna(medians)
        )

        x_test = (
            x_test.replace([np.inf, -np.inf], np.nan)
            .fillna(medians)
        )

        scaler = StandardScaler()

        x_train = scaler.fit_transform(x_train)
        x_validation = scaler.transform(x_validation)
        x_test = scaler.transform(x_test)

        encoder = LabelEncoder()

        y_train = encoder.fit_transform(
            self._binary_labels(train)
        )

        y_validation = encoder.transform(
            self._binary_labels(validation)
        )

        y_test = encoder.transform(
            self._binary_labels(test)
        )

        self.features = features
        self.x_train = x_train
        self.y_train = y_train
        self.x_validation = x_validation
        self.y_validation = y_validation
        self.x_test = x_test
        self.y_test = y_test
        self.scaler = scaler
        self.encoder = encoder

        _DATA_CACHE = (
            self.features,
            self.x_train,
            self.y_train,
            self.x_validation,
            self.y_validation,
            self.x_test,
            self.y_test,
            self.scaler,
            self.encoder,
        )

        self._loaded = True

        print("Dataset loaded successfully.")

    def load_client_data(
        self,
        client_id,
        num_clients=3,
    ):

        self._load_once()

        indices = np.array_split(
            np.arange(len(self.x_train)),
            num_clients,
        )

        idx = indices[client_id]

        return (
            self.x_train[idx],
            self.y_train[idx],
            self.x_validation,
            self.y_validation,
            self.x_test,
            self.y_test,
            self.scaler,
            self.encoder,
            self.features,
        )

    def load_data(self):

        self._load_once()

        return (
            self.x_train,
            self.y_train,
            self.x_validation,
            self.y_validation,
            self.x_test,
            self.y_test,
            self.scaler,
            self.encoder,
            self.features,
        )