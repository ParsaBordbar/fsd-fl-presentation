"""The same federation as `flwr run`, in plain NumPy, in under a second.

Run:  python fedavg_numpy.py [--iid] [--rounds 10] [--noise 0.0 --clip 0.5]

Same data, same model, same local training and the same weighted average as
the ServerApp/ClientApp pair -- just without Ray. It exists for one reason:
the tables and the accuracy chart in the deck come from here, so they can be
regenerated offline the night before the talk, and so the audience can check
that the slide numbers are not made up.

With --noise > 0 every hospital clips its update to --clip and adds Gaussian
noise (noise multiplier x clip), which is the client-side differential
privacy recipe from slide "Differential privacy: bounded influence". Sweep it
to see the accuracy-vs-noise curve.
"""

import argparse

import numpy as np

from hospitals_fl.task import (
    HOSPITAL_SIZES,
    evaluate_local,
    init_params,
    load_global_testset,
    load_hospital_data,
    train_local,
)


def run(iid=False, rounds=10, epochs=5, noise=0.0, clip=0.5, seed=0):
    rng = np.random.default_rng(seed)
    hospitals = [load_hospital_data(i, iid=iid) for i in range(len(HOSPITAL_SIZES))]
    x_glob, y_glob = load_global_testset()
    params = init_params()
    history = []

    for rnd in range(1, rounds + 1):
        updates, sizes = [], []
        for x_train, y_train, _, _ in hospitals:
            local, _ = train_local(params, x_train, y_train, epochs=epochs)
            delta = [l - p for l, p in zip(local, params)]
            if noise > 0:
                norm = np.sqrt(sum(float((d ** 2).sum()) for d in delta))
                delta = [d * min(1.0, clip / max(norm, 1e-12)) for d in delta]
                delta = [d + rng.normal(0.0, noise * clip, d.shape) for d in delta]
            updates.append(delta)
            sizes.append(len(y_train))

        total = sum(sizes)
        params = [
            p + sum(n / total * u[i] for u, n in zip(updates, sizes))
            for i, p in enumerate(params)
        ]

        global_acc = evaluate_local(params, x_glob, y_glob)[1]
        local_acc = [evaluate_local(params, xt, yt)[1] for _, _, xt, yt in hospitals]
        history.append((rnd, global_acc, local_acc))
    return history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iid", action="store_true")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--noise", type=float, default=0.0, help="DP noise multiplier")
    ap.add_argument("--clip", type=float, default=0.5, help="DP clipping norm")
    args = ap.parse_args()

    hist = run(args.iid, args.rounds, args.epochs, args.noise, args.clip)
    names = "ABC"
    print(f"  data: {'IID' if args.iid else 'non-IID'}   "
          f"sizes: {HOSPITAL_SIZES}   noise: {args.noise}")
    print("  round   global   " + "   ".join(f"   {n}" for n in names))
    for rnd, g, loc in hist:
        print(f"  {rnd:>5}   {g:>6.3f}   " + "   ".join(f"{a:.3f}" for a in loc))


if __name__ == "__main__":
    main()
