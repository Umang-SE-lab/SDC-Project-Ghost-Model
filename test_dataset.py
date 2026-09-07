from dataset import CICIoTDataset

print("Starting...")

dataset = CICIoTDataset()

print("Loading data...")

x_train, y_train, x_val, y_val, x_test, y_test, scaler, encoder, features = dataset.load_data()

print("Training Shape:", x_train.shape)
print("Validation Shape:", x_val.shape)
print("Testing Shape:", x_test.shape)

print("Features:", len(features))
print("Classes:", encoder.classes_)

print("Done!")