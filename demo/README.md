# hospitals-fl — demo code for the talk

Three hospitals train a shared clinical-note classifier without a shared
database. Everything is NumPy and Flower; each demo runs on a laptop in about
twenty seconds, offline.

| File | Which slide it serves |
|---|---|
| `hospitals_fl/task.py` | Data, non-IID partitioning, the model |
| `hospitals_fl/client_app.py` | Slides 7–9 — what runs at the hospital, what leaves it |
| `hospitals_fl/server_app.py` | Slides 9–10, 19 — the aggregator and the weighting |
| `attack_demo.py` | **Slide 13** — reconstructing training data from an update |
| — via run-config flags | Slides 16, 17–19 — non-IID, poisoning, robust aggregation |
| `fedavg_numpy.py` | The same federation without Ray, < 1 s — regenerates every table and chart in the deck |

The task: sort a short clinical note into cardiology, paediatrics, or emergency.
The notes are synthetic bags of clinical words — which is itself a small
illustration of the problem the talk is about, since real notes are exactly what
you cannot put in a public repository.

Layout: `hospitals_fl/` is the Flower app package (`pyproject.toml` points
`serverapp`/`clientapp` at it); `attack_demo.py` and `fedavg_numpy.py` sit next
to it and import from it, so run everything from this directory.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install "flwr[simulation]>=1.36" numpy
```

## Federation config — read this before the talk

Flower moved federation settings in 1.36. Check with `flwr --version`.

**Flower ≥ 1.36** — settings live in `~/.flwr/config.toml`:

```toml
[superlink]
default = "local-simulation"

[superlink.local-simulation]
address = ":local:"
options.num-supernodes = 3
options.backend.client-resources.num-cpus = 1
options.backend.init-args.num-cpus = 4
```

Then run `flwr run . local-simulation --stream`.

**Flower < 1.36** — put this at the end of `pyproject.toml` instead, and run
`flwr run .`:

```toml
[tool.flwr.federations]
default = "local-simulation"

[tool.flwr.federations.local-simulation]
options.num-supernodes = 3
```

Two practical notes. `options.backend.init-args.num-cpus` is what stops Ray
refusing to start with "ActorPool is empty" on a low-core machine. And prefix any
command with `RAY_DEDUP_LOGS=0` to see every hospital's log line instead of Ray
collapsing the repeats.

**Use 3 supernodes for demos 1–3** (it matches the three hospitals in your slides
and gives clean weights). **Switch to 5 for demo 4** — Krum needs enough honest
participants to have something to compare against.

---

## Demo 0 — the numbers in the deck, without Ray

```bash
python fedavg_numpy.py            # non-IID, 10 rounds: 0.399 → 0.859
python fedavg_numpy.py --iid      # 0.991 → 0.994
python fedavg_numpy.py --iid --noise 1.0   # DP: clip 0.5, noise ×1.0 → 0.652
```

Plain NumPy, same data, model, local epochs and weighted average as the
Flower run, so it prints the same accuracies to three decimals. The accuracy
chart and the per-hospital table in the deck come from here; the
accuracy-vs-noise chart is the `--iid --noise` sweep over
0, 0.25, 0.5, 1, 1.5, 2, 3, 4. Use it when Ray refuses to start.

## Demo 1 — the baseline (slides 9–10)

```bash
RAY_DEDUP_LOGS=0 flwr run . local-simulation --stream
```

Two things to point at.

**The weighting**, printed every round — slide 10 made real:

```
  FedAvg weights (round 1):
    hospital 0:  n= 100  w=0.100  ███
    hospital 1:  n= 300  w=0.300  █████████
    hospital 2:  n= 600  w=0.600  ██████████████████
```

**The accuracy table** at the end. Say out loud that the hospital with 600
records has six times the influence of the one with 100, and that this is a
policy decision wearing the costume of a default. It sets up the fairness point
on slide 16 and the governance argument in section 7.

## Demo 2 — non-IID (slide 16) ★

The single most informative flag in the package. Each hospital has a different
case mix by default. Run it both ways:

```bash
flwr run . local-simulation --stream --run-config "num-server-rounds=10"
flwr run . local-simulation --stream --run-config "num-server-rounds=10 iid=true"
```

| round | non-IID | IID |
|---|---|---|
| 1 | 0.399 | **0.991** |
| 4 | 0.639 | 0.996 |
| 6 | 0.650 | 0.994 |
| 10 | 0.859 | 0.994 |

Same model, same data volume, same number of rounds. The IID federation is
essentially finished after one round; the non-IID one is still at 0.86 after ten,
with a visible plateau around rounds 4–6 as the three local models pull in
different directions. That plateau is client drift, on a projector.

One more thing worth showing in the non-IID run: the *federated* accuracy column
(0.912) is higher than the *global* column (0.859). Each hospital evaluates on
its own skewed test set, so local evaluation flatters the model. Who holds the
evaluation set is a governance question, and here it's worth about five points.

## Demo 3 — reconstructing the data (slide 13) ★★

```bash
python attack_demo.py
```

No Flower, no server, one second. It takes a single honest client update and
recovers the training example from it:

```
  The note that never leaves the hospital:
    "fracture cardiac fall ambulance poisoning trauma bleeding wound"
    department: emergency

  What the hospital actually sends:
    a weight update of shape (256, 3) and a bias update of shape (3,)
    771 floats. No text. No records. Nothing readable.

  recovered department : emergency   CORRECT
  cosine similarity    : 1.000000   (1.000000 = exact)

  recovered words:
    cardiac trauma bleeding fracture wound poisoning fall ambulance
```

Then it turns on the two defences and shows the attack degrading — larger batches
first, then clipping and noise.

**Be honest about what this is.** Our model is linear, so the inversion is exact
arithmetic; Deep Leakage from Gradients needs an optimisation loop for deep
networks. Say that on stage. What transfers unchanged is the principle: the
update is a deterministic function of the data, and functions can be run
backwards. Point at the 771-floats line, then at the recovered note.

Occasionally an extra word appears in the recovered set. Those are hash
collisions in a 256-dimensional feature space, not attack failures, and the
script says so. Mentioning it unprompted buys more credibility than hiding it.

## Demo 4 — poisoning and robust aggregation (slides 17–19) ★★★

Set `options.num-supernodes = 5` first. One hospital flips its labels and scales
its update up — the model-replacement idea at its simplest.

**(a) FedAvg has no defence.** IID data, one attacker in five:

```bash
flwr run . local-simulation --stream --run-config \
  "num-server-rounds=6 iid=true poisoned-hospitals=1 attack-boost=20.0"
```
→ global accuracy collapses to **0.150**.

**(b) Robust aggregation fixes it.** Same attack, different rule:

```bash
flwr run . local-simulation --stream --run-config \
  "num-server-rounds=6 iid=true strategy='krum' poisoned-hospitals=1 attack-boost=20.0"
```
→ **0.993**. `strategy='trimmed'` gives 0.997. Not one line of client code
changed; only the aggregation rule did.

**(c) Now turn the non-IID data back on.** Drop `iid=true`:

```bash
flwr run . local-simulation --stream --run-config \
  "num-server-rounds=6 strategy='krum' poisoned-hospitals=1 attack-boost=20.0"
```
→ back to **0.326**. Trimmed mean fails identically. The defence stops working.

| | FedAvg | Krum | Trimmed |
|---|---|---|---|
| IID + 1 attacker | 0.150 | **0.993** | **0.997** |
| non-IID + 1 attacker | 0.326 | 0.326 | 0.326 |

That bottom row is the most valuable thing in the package, and it's the live
version of slide 19: when clients genuinely differ, an honest minority client is
hard to tell apart from an attacker, so outlier rejection discards the
participants you federated in order to include. It's the Karimireddy, He & Jaggi
bucketing result, reproduced on stage in twenty seconds.

And the framing line for section 7: every one of these defences works by
**inspecting individual updates** — exactly what secure aggregation exists to
prevent. The room has now watched the conflict rather than heard it described.

`strategy` accepts `fedavg`, `krum`, `multikrum`, `median`, `trimmed`.

---

## Optional: Flower's own secure aggregation example

To show SecAgg+ actually executing rather than described, Flower ships a runnable
example that logs each protocol stage:
<https://flower.ai/docs/examples/flower-secure-aggregation.html>

It's marked experimental and the API has moved across 1.x, so test it on the
presentation laptop before relying on it. Screenshot the four stages as a
fallback either way.

For differential privacy, wrap any strategy in
`DifferentialPrivacyClientSideFixedClipping` (or the adaptive / server-side
variants) and add `fixedclipping_mod` to the ClientApp:
<https://flower.ai/docs/framework/how-to-use-differential-privacy.html>

## Stage checklist

- [ ] Pin the Flower version and test **every** command on the presentation laptop, **offline**
- [ ] Terminal font ≥ 18pt
- [ ] Remember to switch `num-supernodes` from 3 to 5 before demo 4
- [ ] Screenshot each expected output as a backup layer in the deck
- [ ] 60-second screen recording as a second backup
- [ ] Accuracy tables already pasted into the slides, so you can skip the live runs entirely and lose nothing
- [ ] Decide in advance what gets cut if you run long — demos 3 and 4c carry arguments nothing else in the talk can carry
