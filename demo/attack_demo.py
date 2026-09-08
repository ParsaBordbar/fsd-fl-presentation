"""Slide 13, live: reconstructing training data from a model update.

Run:  python attack_demo.py

This is the honest, minimal version of Deep Leakage from Gradients (Zhu, Liu
& Han, NeurIPS 2019, arXiv:1906.08935). Our model is a linear classifier, so
the inversion is exact arithmetic rather than the optimisation loop the paper
needs for deep networks. Say that on stage. What carries over unchanged is the
thing the talk is about:

    the update is a deterministic function of the training data,
    and functions can be run backwards.

The demo then turns on the two defences from slide 15 and shows the attack
degrade -- which is the point of that slide, and also its cost.
"""

import numpy as np

from hospitals_fl.task import (
    DEPARTMENTS,
    NUM_CLASSES,
    VOCAB_SIZE,
    WORD_POOLS,
    _hash_token,
    _make_note,
    _softmax,
    _vectorize,
    init_params,
)

LR = 0.5


def client_update(params, x, y, lr: float = LR):
    """One honest local step. This is exactly what a client sends back."""
    w, b = params
    n = len(y)
    onehot = np.zeros((n, NUM_CLASSES), dtype=np.float32)
    onehot[np.arange(n), y] = 1.0
    probs = _softmax(x @ w + b)
    grad = (probs - onehot) / n
    return -lr * (x.T @ grad), -lr * grad.sum(axis=0)  # (dW, db)


def invert(dw, db):
    """The attack. Input: only what the server received.

    For a linear classifier the update is an outer product of the input
    vector with the per-class error, so:

      * db is positive in exactly one coordinate -- the true label.
        (This is the iDLG observation: the label falls out analytically.)
      * dW is that same input vector, scaled column by column. Divide any
        column by its db entry and the input comes straight back out.
    """
    label = int(np.argmax(db))
    c = int(np.argmax(np.abs(db)))
    x_hat = dw[:, c] / db[c]
    return label, x_hat


def words_from_vector(x_vec, threshold: float = 1e-3):
    """Map a recovered feature vector back to words via the hash table."""
    buckets = {i for i in np.nonzero(np.abs(x_vec) > threshold)[0]}
    found = []
    for pool in WORD_POOLS.values():
        for word in pool:
            if _hash_token(word) in buckets and word not in found:
                found.append(word)
    return found


def banner(text):
    print("\n" + "=" * 64)
    print(f"  {text}")
    print("=" * 64)


def main():
    rng = np.random.default_rng(3)
    params = init_params()

    # ---------------------------------------------------------------- 1
    banner("1. One patient note, one honest update")

    true_label = 2  # emergency
    note = _make_note(rng, true_label, n_words=8)
    x = _vectorize([note])
    y = np.array([true_label])

    print(f"\n  The note that never leaves the hospital:")
    print(f"    \"{note}\"")
    print(f"    department: {DEPARTMENTS[true_label]}")

    dw, db = client_update(params, x, y)
    print(f"\n  What the hospital actually sends:")
    print(f"    a weight update of shape {dw.shape} and a bias update of shape {db.shape}")
    print(f"    {dw.size + db.size} floats. No text. No records. Nothing readable.")

    # ---------------------------------------------------------------- 2
    banner("2. The server runs it backwards")

    label_hat, x_hat = invert(dw, db)
    cos = float(x_hat @ x[0] / (np.linalg.norm(x_hat) * np.linalg.norm(x[0])))

    print(f"\n  recovered department : {DEPARTMENTS[label_hat]}"
          f"   {'CORRECT' if label_hat == true_label else 'wrong'}")
    print(f"  cosine similarity    : {cos:.6f}   (1.000000 = exact)")
    print(f"\n  recovered words:")
    print(f"    {' '.join(words_from_vector(x_hat))}")
    print(f"\n  original words:")
    print(f"    {' '.join(sorted(set(note.split())))}")
    print("\n  Any extra words in the recovered set are hash collisions in a")
    print("  256-dimensional feature space, not attack failures. A larger")
    print("  vocabulary removes them.")

    # ---------------------------------------------------------------- 3
    banner("3. Defence one: don't send an update computed from one example")

    print("\n  batch size   cosine similarity of the recovered vector")
    print("  " + "-" * 52)
    for batch in (1, 2, 4, 16, 64, 256):
        labels = rng.integers(0, NUM_CLASSES, size=batch)
        notes = [_make_note(rng, int(l)) for l in labels]
        xb = _vectorize(notes)
        dwb, dbb = client_update(params, xb, labels)
        _, xb_hat = invert(dwb, dbb)
        target = xb[0]
        c = float(xb_hat @ target /
                  max(np.linalg.norm(xb_hat) * np.linalg.norm(target), 1e-12))
        print(f"  {batch:>10}   {c:>6.3f}")
    print("\n  Larger batches mix examples together, and one example stops")
    print("  being separable. This is a mitigation, not a guarantee -- the")
    print("  server chooses the model and the batch policy it hands you.")

    # ---------------------------------------------------------------- 4
    banner("4. Defence two: clip and add noise (differential privacy)")

    print("\n  noise multiplier   cosine similarity   (single example, worst case)")
    print("  " + "-" * 62)
    clip = 0.5
    for noise in (0.0, 0.01, 0.1, 1.0):
        dwc = dw * min(1.0, clip / np.linalg.norm(dw))
        dbc = db * min(1.0, clip / np.linalg.norm(db))
        sigma = noise * clip
        dwn = dwc + rng.normal(0, sigma, dwc.shape)
        dbn = dbc + rng.normal(0, sigma, dbc.shape)
        _, xn_hat = invert(dwn, dbn)
        c = float(xn_hat @ x[0] /
                  max(np.linalg.norm(xn_hat) * np.linalg.norm(x[0]), 1e-12))
        print(f"  {noise:>16}   {c:>17.3f}")
    print("\n  Flower's docs suggest a noise multiplier of 1.0 or above for")
    print("  strong privacy. Run the federation at that setting and watch")
    print("  the accuracy table move. There is no free privacy.")

    print()


if __name__ == "__main__":
    main()
