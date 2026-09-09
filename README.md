# Federated Learning and the Future of Private, Open AI

Software Freedom Day talk. Slides live at <https://fsd.bbrand.ir/> (the QR code on slide 1).

## Open the slides

```bash
open index.html
```

Works offline, everything is in `vendor/`.

Keys: `←` `→` slides · `Esc` overview · `S` speaker notes · `F` fullscreen.

## Build the single-file deck

```bash
python3 build_single.py
```

Writes `dist/index.html` (about 1.2 MB) with everything inlined. Serve that one file; `deploy/nginx.conf` has a server block for it. Rebuild after every edit to `index.html`.

## Run the demos

One-time setup:

```bash
cd demo
uv venv --python 3.13 .venv
source .venv/bin/activate
uv pip install "flwr[simulation]>=1.36" numpy
```

Add these two federations to `~/.flwr/config.toml`:

```toml
[superlink.local-simulation]
address = ":local:"
options.num-supernodes = 3
options.backend.client-resources.num-cpus = 1
options.backend.init-args.num-cpus = 4

[superlink.local-sim-5]
address = ":local:"
options.num-supernodes = 5
options.backend.client-resources.num-cpus = 1
options.backend.init-args.num-cpus = 4
```

Every time, activate the environment first. `flwr` needs `flower-superlink` on `PATH`, so this step is required:

```bash
cd demo
source .venv/bin/activate
```

Then run any demo:

```bash
# Demo 0: the deck's numbers, no Ray, under a second
python fedavg_numpy.py                     # non-IID: 0.399 -> 0.859
python fedavg_numpy.py --iid               # IID:     0.991 -> 0.994
python fedavg_numpy.py --iid --noise 1.0   # DP:      -> 0.652

# Demo 1: baseline, three hospitals, prints the FedAvg weights
RAY_DEDUP_LOGS=0 flwr run . local-simulation --stream

# Demo 2: non-IID vs IID
flwr run . local-simulation --stream --run-config "num-server-rounds=10"
flwr run . local-simulation --stream --run-config "num-server-rounds=10 iid=true"

# Demo 3: reconstruct the training note from one update
python attack_demo.py

# Demo 4: poisoning and robust aggregation, five hospitals
flwr run . local-sim-5 --stream --run-config "num-server-rounds=6 iid=true poisoned-hospitals=1 attack-boost=20.0"                   # FedAvg: 0.150
flwr run . local-sim-5 --stream --run-config "num-server-rounds=6 iid=true strategy='krum' poisoned-hospitals=1 attack-boost=20.0"   # Krum:   0.993
flwr run . local-sim-5 --stream --run-config "num-server-rounds=6 strategy='krum' poisoned-hospitals=1 attack-boost=20.0"            # non-IID Krum: 0.326
```

Each Flower run takes about 20 seconds. The results table prints at the bottom. `deactivate` when done.

Full walkthrough of what each demo shows, and what to say on stage, is in `demo/README.md`.
