import sys
import pickle
from pathlib import Path
import numpy as np

# Fix import paths
root = Path(__file__).resolve().parents[3]  # ufaz_AIProject
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "starter_pack" / "src"))  # so nn_model is found

from starter_pack.src.train_nn import train_nn

# -----------------------
# Load data
# -----------------------
BASE_DIR = Path(__file__).resolve().parents[1]  # points to digit-recognition-system-app

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

save_path = BASE_DIR / "models" / "digits_model.pkl"
save_path.parent.mkdir(parents=True, exist_ok=True)

with open(save_path, "wb") as f:
    pickle.dump(weights, f)

print(" Model saved as digits_model.pkl")