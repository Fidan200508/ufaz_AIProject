import pickle
import numpy as np
from pathlib import Path
from starter_pack.train_nn import train_nn

# -----------------------
# Load data
# -----------------------
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

data = np.load(BASE_DIR / "data/digits_data.npz")
split = np.load(BASE_DIR / "data/digits_split_indices.npz")

X = data["X"]
y = data["y"]

train_idx = split["train_idx"]
val_idx = split["val_idx"]

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]

# -----------------------
# Train model
# -----------------------
model, _, _ = train_nn(
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    input_dim=64,
    hidden_dim=32,
    output_dim=10,
    optimizer="adam",
    lr=0.001,
    reg_lambda=1e-4,
    batch_size=64,
    epochs=200,
    seed=42,
    verbose=False,
)

# -----------------------
# Save weights
# -----------------------
weights = {
    "W1": model.W1,
    "b1": model.b1,
    "W2": model.W2,
    "b2": model.b2,
}

with open("digits_model.pkl", "wb") as f:
    pickle.dump(weights, f)

print("✅ Model saved as digits_model.pkl")