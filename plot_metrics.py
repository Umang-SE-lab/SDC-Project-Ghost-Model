import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)

df = pd.read_csv("results/training_metrics.csv")

# ---------------- Loss ----------------

plt.figure(figsize=(7,5))
plt.plot(df["Round"], df["Loss"], marker="o")
plt.title("Global Loss vs Federated Rounds")
plt.xlabel("Round")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/loss_curve.png")
plt.close()

# ---------------- Accuracy ----------------

plt.figure(figsize=(7,5))
plt.plot(df["Round"], df["Accuracy"], marker="o")
plt.title("Global Accuracy vs Federated Rounds")
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/accuracy_curve.png")
plt.close()

print("Graphs saved in results/")