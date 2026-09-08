"""The ClientApp -- this is the code that runs at each hospital.

In a real deployment this process runs inside the hospital's network, on a
machine the hospital controls, and the patient records never leave it. What
leaves is whatever this file puts into the reply Message.

That is worth saying out loud during the demo: the privacy promise of
federated learning is, quite literally, a claim about the last ten lines of
this file. If you cannot read this file, you cannot check the promise.
"""

import numpy as np
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from hospitals_fl.task import (
    DEPARTMENTS,
    evaluate_local,
    load_hospital_data,
    train_local,
)

app = ClientApp()


@app.train()
def train(msg: Message, context: Context) -> Message:
    """Steps 2 and 3 of a federated round: train locally, send back an update."""
    partition_id = int(context.node_config["partition-id"])
    iid = bool(msg.content["config"]["iid"])
    x_train, y_train, _, _ = load_hospital_data(partition_id, iid=iid)

    # Step 2: unpack the global model and train it on local data only.
    params = [np.asarray(a) for a in msg.content["arrays"].to_numpy_ndarrays()]
    epochs = int(msg.content["config"]["local-epochs"])
    global_params = [p.copy() for p in params]

    # ---- section 6 of the talk, made runnable -------------------------
    # A "poisoned" hospital trains on deliberately wrong labels and then
    # scales its update up, so that after averaging it dominates the result.
    # This is the model-replacement idea from Bagdasaryan et al. (2020),
    # reduced to its simplest form.
    poisoned = int(msg.content["config"]["poisoned-hospitals"])
    boost = float(msg.content["config"]["attack-boost"])
    is_attacker = partition_id < poisoned

    if is_attacker:
        y_train = (y_train + 1) % len(DEPARTMENTS)  # flip every label

    params, loss = train_local(params, x_train, y_train, epochs=epochs)

    if is_attacker and boost > 1.0:
        params = [g + boost * (p - g) for p, g in zip(params, global_params)]
        print(f"  [hospital {partition_id}] *** POISONED *** "
              f"flipped labels, update scaled x{boost:g}")

    # A quick look at what this hospital actually holds, so the non-IID
    # partitioning is visible in the logs rather than just asserted.
    mix = np.bincount(y_train, minlength=len(DEPARTMENTS)) / len(y_train)
    mix_str = " ".join(f"{d}={p:.0%}" for d, p in zip(DEPARTMENTS, mix))
    print(f"  [hospital {partition_id}] n={len(y_train):>4}  {mix_str}")

    # Step 3: send back the trained parameters -- NOT the data.
    #
    # `num-examples` is what FedAvg weights by. It is also, note, a small
    # piece of information about this hospital that does leave the building.
    return Message(
        RecordDict(
            {
                "arrays": ArrayRecord(params),
                "metrics": MetricRecord(
                    {
                        "train_loss": loss,
                        "num-examples": len(y_train),
                        "hospital-id": partition_id,
                    }
                ),
            }
        ),
        reply_to=msg,
    )


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """Federated evaluation: score the global model on local test data."""
    partition_id = int(context.node_config["partition-id"])
    iid = bool(msg.content["config"]["iid"])
    _, _, x_test, y_test = load_hospital_data(partition_id, iid=iid)

    params = [np.asarray(a) for a in msg.content["arrays"].to_numpy_ndarrays()]
    loss, acc = evaluate_local(params, x_test, y_test)

    return Message(
        RecordDict(
            {
                "metrics": MetricRecord(
                    {
                        "eval_loss": loss,
                        "eval_acc": acc,
                        "num-examples": len(y_test),
                    }
                )
            }
        ),
        reply_to=msg,
    )
