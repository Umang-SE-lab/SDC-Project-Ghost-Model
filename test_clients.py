from dataset import CICIoTDataset

dataset = CICIoTDataset()

for i in range(3):
    x_train, y_train, _, _, _, _, _, _, _ = dataset.load_client_data(i, 3)

    print(f"\nClient {i+1}")
    print("Samples:", len(x_train))