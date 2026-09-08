"""Shared code: the data, the partitioning, and the model.

Everything here is plain NumPy on purpose. The point of the demo is the
federation, not the model, so the model is deliberately small enough to read
in one sitting and fast enough to train live on a laptop.

The task: classify a short clinical note into one of three departments --
cardiology, paediatrics, emergency.
"""

import numpy as np

DEPARTMENTS = ["cardiology", "paediatrics", "emergency"]
NUM_CLASSES = len(DEPARTMENTS)
VOCAB_SIZE = 256  # hashed bag-of-words dimension

# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
# Word pools per department. Real deployments use real notes; a Software
# Freedom Day stage does not, which is itself a small illustration of the
# problem this talk is about.

WORD_POOLS = {
    0: [  # cardiology
        "chest", "pain", "pressure", "blood", "palpitations", "arrhythmia",
        "ecg", "coronary", "artery", "stenosis", "breath", "cholesterol",
        "infarction", "cardiac", "murmur",
    ],
    1: [  # paediatrics
        "child", "infant", "fever", "vaccine", "growth", "weight", "newborn",
        "ear", "cough", "diarrhoea", "seizure", "mother", "feeding",
        "teething", "anaemia",
    ],
    2: [  # emergency
        "trauma", "bleeding", "fracture", "burn", "unconscious", "urgent",
        "wound", "poisoning", "fall", "ambulance", "resuscitation", "blunt",
        "head", "severe", "triage",
    ],
}


def _hash_token(token: str) -> int:
    """Stable hash of a token into the feature space (no external deps)."""
    h = 2166136261
    for ch in token:
        h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
    return h % VOCAB_SIZE


def _make_note(rng: np.random.Generator, label: int, n_words: int = 8) -> str:
    """Generate one synthetic clinical note for a department.

    Most words come from that department's pool; a few are drawn from the
    other pools, so the classes overlap and the task is not trivial.
    """
    words = []
    for _ in range(n_words):
        pool = label if rng.random() < 0.75 else int(rng.integers(NUM_CLASSES))
        words.append(str(rng.choice(WORD_POOLS[pool])))
    return " ".join(words)


def _vectorize(notes: list[str]) -> np.ndarray:
    """Hashed bag-of-words, L2-normalised."""
    x = np.zeros((len(notes), VOCAB_SIZE), dtype=np.float32)
    for i, note in enumerate(notes):
        for token in note.split():
            x[i, _hash_token(token)] += 1.0
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-8)


def _sample(rng: np.random.Generator, label_probs, n: int):
    labels = rng.choice(NUM_CLASSES, size=n, p=label_probs)
    notes = [_make_note(rng, int(y)) for y in labels]
    return _vectorize(notes), labels.astype(np.int64)


# Each hospital sees a different case mix. THIS IS THE POINT: client data in
# federated learning is never identically distributed, and you can watch what
# that does to training by changing these rows.
#
#   Hospital 0 -- a cardiac centre
#   Hospital 1 -- a children's hospital
#   Hospital 2 -- a general hospital with a busy emergency department
#
# Set every row to [1/3, 1/3, 1/3] to simulate the IID case and compare.
HOSPITAL_MIX = [
    [0.94, 0.03, 0.03],
    [0.03, 0.94, 0.03],
    [0.10, 0.10, 0.80],
]

# Deliberately unequal dataset sizes, so FedAvg's weighting does visible work.
HOSPITAL_SIZES = [100, 300, 600]


def load_hospital_data(partition_id: int, seed: int = 42, iid: bool = False):
    """Return (x_train, y_train, x_test, y_test) for one hospital.

    Set iid=True to give every hospital the same balanced case mix. Running
    the demo both ways is the single most informative thing you can do with
    it -- see the README.
    """
    rng = np.random.default_rng(seed + partition_id)
    n = HOSPITAL_SIZES[partition_id % len(HOSPITAL_SIZES)]
    mix = ([1 / 3] * NUM_CLASSES if iid
           else HOSPITAL_MIX[partition_id % len(HOSPITAL_MIX)])
    x_train, y_train = _sample(rng, mix, n)
    x_test, y_test = _sample(rng, mix, max(n // 4, 25))
    return x_train, y_train, x_test, y_test


def load_global_testset(seed: int = 7, n: int = 900):
    """A balanced test set held by the server, for centralized evaluation.

    Note the asymmetry: no hospital's local test set looks like this one.
    A model can score well on every local test set and still be mediocre here.
    """
    rng = np.random.default_rng(seed)
    return _sample(rng, [1 / 3, 1 / 3, 1 / 3], n)


# --------------------------------------------------------------------------
# Model: multinomial logistic regression, trained with plain gradient descent
# --------------------------------------------------------------------------


def init_params(seed: int = 0):
    """Initial weights and bias. The server calls this once, at round 0."""
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 0.01, size=(VOCAB_SIZE, NUM_CLASSES)).astype(np.float32)
    b = np.zeros(NUM_CLASSES, dtype=np.float32)
    return [w, b]


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def train_local(params, x, y, epochs: int = 5, lr: float = 0.5):
    """Local training: a few epochs of full-batch gradient descent.

    This is Step 2 of a federated round. Note that we do NOT train to
    convergence -- just a little, then send the result back.
    """
    w, b = [p.copy() for p in params]
    n = len(y)
    onehot = np.zeros((n, NUM_CLASSES), dtype=np.float32)
    onehot[np.arange(n), y] = 1.0

    for _ in range(epochs):
        probs = _softmax(x @ w + b)
        grad = (probs - onehot) / n
        w -= lr * (x.T @ grad)
        b -= lr * grad.sum(axis=0)

    loss = evaluate_local([w, b], x, y)[0]
    return [w, b], loss


def evaluate_local(params, x, y):
    """Return (cross-entropy loss, accuracy)."""
    w, b = params
    probs = _softmax(x @ w + b)
    loss = float(-np.log(np.maximum(probs[np.arange(len(y)), y], 1e-9)).mean())
    acc = float((probs.argmax(axis=1) == y).mean())
    return loss, acc
