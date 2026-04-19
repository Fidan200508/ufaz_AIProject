import numpy as np
import matplotlib.pyplot as plt
import copy
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]  # points to starter_pack

data = np.load(BASE_DIR / 'data/digits_data.npz')
indices = np.load(BASE_DIR / 'data/digits_split_indices.npz')

X_train, y_train = data['X'][indices['train_idx']], data['y'][indices['train_idx']]
X_val, y_val = data['X'][indices['val_idx']], data['y'][indices['val_idx']]
X_test, y_test = data['X'][indices['test_idx']], data['y'][indices['test_idx']]

D_IN = 64
D_HIDDEN = 32
D_OUT = 10
REG_LAMBDA = 1e-4
BATCH_SIZE = 64
MAX_EPOCHS = 200


# 2. MODEL FUNKSİYALARI
def softmax(S):
    S_max = np.max(S, axis=1, keepdims=True)
    exps = np.exp(S - S_max)
    return exps / np.sum(exps, axis=1, keepdims=True)


def forward_mlp(X, W1, b1, W2, b2):
    Z1 = X @ W1.T + b1.T
    H = np.tanh(Z1)
    S = H @ W2.T + b2.T
    P = softmax(S)
    return Z1, H, S, P


def compute_loss(probs, y_true, W1, W2, reg_lambda):
    n = probs.shape[0]
    log_p = -np.log(probs[np.arange(n), y_true] + 1e-12)
    data_loss = np.mean(log_p)
    reg_loss = 0.5 * reg_lambda * (np.sum(W1 ** 2) + np.sum(W2 ** 2))
    return data_loss + reg_loss


def get_gradients(X, H, probs, y_true, W1, W2, reg_lambda):
    n = X.shape[0]
    Y = np.zeros_like(probs)
    Y[np.arange(n), y_true] = 1
    dL_dS = (probs - Y) / n

    dW2 = dL_dS.T @ H + reg_lambda * W2
    db2 = np.sum(dL_dS, axis=0, keepdims=True).T

    dL_dZ1 = (dL_dS @ W2) * (1 - H ** 2)
    dW1 = dL_dZ1.T @ X + reg_lambda * W1
    db1 = np.sum(dL_dZ1, axis=0, keepdims=True).T

    return dW1, db1, dW2, db2

def train_model(optimizer_type='sgd', seed=42):
    np.random.seed(seed)

    W1 = np.random.randn(D_HIDDEN, D_IN) * np.sqrt(2 / D_IN)
    b1 = np.zeros((D_HIDDEN, 1))
    W2 = np.random.randn(D_OUT, D_HIDDEN) * np.sqrt(2 / D_HIDDEN)
    b2 = np.zeros((D_OUT, 1))

    # Optimizer state
    vW1, vb1, vW2, vb2 = [np.zeros_like(p) for p in [W1, b1, W2, b2]]  # Momentum/Adam
    sW1, sb1, sW2, sb2 = [np.zeros_like(p) for p in [W1, b1, W2, b2]]  # Adam second moment

    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')
    best_params = None
    t = 0  # Adam step counter

    cfg = {
        'sgd': {'lr': 0.05},
        'momentum': {'lr': 0.05, 'beta': 0.9},
        'adam': {'lr': 0.001, 'b1': 0.9, 'b2': 0.999, 'eps': 1e-8}
    }[optimizer_type]

    for epoch in range(MAX_EPOCHS):
        indices = np.random.permutation(X_train.shape[0])
        X_sh, y_sh = X_train[indices], y_train[indices]

        epoch_losses = []
        for i in range(0, len(X_train), BATCH_SIZE):
            t += 1
            xb, yb = X_sh[i:i + BATCH_SIZE], y_sh[i:i + BATCH_SIZE]

            Z1, H, S, P = forward_mlp(xb, W1, b1, W2, b2)
            loss = compute_loss(P, yb, W1, W2, REG_LAMBDA)
            epoch_losses.append(loss)

            gW1, gb1, gW2, gb2 = get_gradients(xb, H, P, yb, W1, W2, REG_LAMBDA)

            # Update Rules
            if optimizer_type == 'sgd':
                W1 -= cfg['lr'] * gW1;
                b1 -= cfg['lr'] * gb1
                W2 -= cfg['lr'] * gW2;
                b2 -= cfg['lr'] * gb2

            elif optimizer_type == 'momentum':
                vW1 = cfg['beta'] * vW1 + cfg['lr'] * gW1;
                W1 -= vW1
                vb1 = cfg['beta'] * vb1 + cfg['lr'] * gb1;
                b1 -= vb1
                vW2 = cfg['beta'] * vW2 + cfg['lr'] * gW2;
                W2 -= vW2
                vb2 = cfg['beta'] * vb2 + cfg['lr'] * gb2;
                b2 -= vb2

            elif optimizer_type == 'adam':
                for p, g, m, s in zip(['W1', 'b1', 'W2', 'b2'], [gW1, gb1, gW2, gb2],
                                      [vW1, vb1, vW2, vb2], [sW1, sb1, sW2, sb2]):
                    m[:] = cfg['b1'] * m + (1 - cfg['b1']) * g
                    s[:] = cfg['b2'] * s + (1 - cfg['b2']) * (g ** 2)
                    m_hat = m / (1 - cfg['b1'] ** t)
                    s_hat = s / (1 - cfg['b2'] ** t)
                    update = cfg['lr'] * m_hat / (np.sqrt(s_hat) + cfg['eps'])
                    if p == 'W1':
                        W1 -= update
                    elif p == 'b1':
                        b1 -= update
                    elif p == 'W2':
                        W2 -= update
                    elif p == 'b2':
                        b2 -= update

        # Validation
        _, _, _, P_v = forward_mlp(X_val, W1, b1, W2, b2)
        v_loss = compute_loss(P_v, y_val, W1, W2, REG_LAMBDA)
        v_acc = np.mean(np.argmax(P_v, axis=1) == y_val)

        history['train_loss'].append(np.mean(epoch_losses))
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)

        # Checkpoint: Save best validation loss model
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_params = (copy.deepcopy(W1), copy.deepcopy(b1), copy.deepcopy(W2), copy.deepcopy(b2))

    return history, best_params

results = {}
optimizers = ['sgd', 'momentum', 'adam']

for opt in optimizers:
    print(f"Training with {opt}...")
    hist, best_p = train_model(opt)

    _, _, _, P_test = forward_mlp(X_test, *best_p)
    test_acc = np.mean(np.argmax(P_test, axis=1) == y_test)
    print(f"-> {opt.upper()} Test Acc: {test_acc:.4f}")
    results[opt] = hist

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for opt in optimizers:
    plt.plot(results[opt]['val_loss'], label=opt)
plt.title('Validation Cross-Entropy')
plt.xlabel('Epoch');
plt.ylabel('Loss');
plt.legend()

plt.subplot(1, 2, 2)
for opt in optimizers:
    plt.plot(results[opt]['val_acc'], label=opt)
plt.title('Validation Accuracy')
plt.xlabel('Epoch');
plt.ylabel('Accuracy');
plt.legend()
plt.show()


seeds = [1, 10, 42, 100, 2026]
adam_test_accs = []

for s in seeds:
    print(f"Running Adam with Seed {s}...")
    _, best_p = train_model('adam', seed=s)
    _, _, _, P_test = forward_mlp(X_test, *best_p)
    acc = np.mean(np.argmax(P_test, axis=1) == y_test)
    adam_test_accs.append(acc)

mean_acc = np.mean(adam_test_accs)
std_acc = np.std(adam_test_accs, ddof=1)
ci_95 = 2.776 * (std_acc / np.sqrt(5)) # t-distribution for N=5

print(f"\nFinal Adam Statistics (5 seeds):")
print(f"Mean Test Accuracy: {mean_acc:.4f} ± {ci_95:.4f}")

import matplotlib.pyplot as plt


def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix (Adam)'):
    # 10x10 matris yaradırıq (0-9 rəqəmləri üçün)
    cm = np.zeros((10, 10), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[true, pred] += 1

    # Vizualizasiya
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(10)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)

    thresh = cm.max() / 2.
    for i in range(10):
        for j in range(10):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Həqiqi Rəqəm (True Label)')
    plt.xlabel('Proqnoz Edilən (Predicted Label)')
    plt.tight_layout()
    plt.show()


# Ən yaxşı Adam modeli ilə test setində proqnoz veririk
_, _, _, P_test = forward_mlp(X_test, *best_p)
y_pred = np.argmax(P_test, axis=1)

# Matrisi çəkirik
plot_confusion_matrix(y_test, y_pred)

results = {}
optimizers = ['sgd', 'momentum', 'adam']

best_overall_params = None
best_overall_val_loss = float('inf')

for opt in optimizers:
    print(f"Training with {opt}...")

    hist, best_p = train_model(opt)

    # Əgər bu optimizatorun verdiyi nəticə indiyə qədərkinin ən yaxşısıdırsa:
    current_min_val_loss = min(hist['val_loss'])
    if current_min_val_loss < best_overall_val_loss:
        best_overall_val_loss = current_min_val_loss
        best_overall_params = best_p  # Burada best_p-ni yadda saxlayırıq

    results[opt] = hist

import pickle

if best_overall_params is not None:
    model_data = {
        'W1': best_overall_params[0],
        'b1': best_overall_params[1],
        'W2': best_overall_params[2],
        'b2': best_overall_params[3]
    }

    with open('digits_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    print("\nƏn yaxşı model 'digits_model.pkl' olaraq yadda saxlanıldı!")
else:
    print("\nXəta: Heç bir parametr yadda saxlanılmadı.")