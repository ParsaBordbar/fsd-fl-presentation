"""The ServerApp -- the aggregator.

This is the component the whole last third of the talk is about. Whoever runs
this process decides who participates, which model is trained, how updates are
combined, and who receives the result. Distributing the data does not
distribute that.
"""

import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import (
    FedAvg,
    FedMedian,
    FedTrimmedAvg,
    Krum,
    MultiKrum,
)

from hospitals_fl.task import (
    evaluate_local,
    init_params,
    load_global_testset,
)

app = ServerApp()


class LoggingFedAvg(FedAvg):
    """FedAvg, plus a printout of the weight each hospital gets.

    This subclass changes nothing about the algorithm. It exists so the
    audience can see the weighted average happening: the hospital with 600
    examples gets six times the influence of the one with 100, and that is a
    policy decision that looks like a default.
    """

    def aggregate_train(self, server_round, replies):
        replies = list(replies)

        rows = []
        for reply in replies:
            if reply.has_content():
                m = reply.content["metrics"]
                rows.append((int(m["hospital-id"]), float(m["num-examples"])))
        rows.sort()
        total = sum(n for _, n in rows) or 1.0

        print(f"  FedAvg weights (round {server_round}):")
        for hid, n in rows:
            bar = "\u2588" * int(round(30 * n / total))
            print(f"    hospital {hid}:  n={int(n):>4}  w={n / total:.3f}  {bar}")

        return super().aggregate_train(server_round, replies)


@app.main()
def main(grid: Grid, context: Context) -> None:
    num_rounds = int(context.run_config["num-server-rounds"])
    local_epochs = int(context.run_config["local-epochs"])
    fraction_train = float(context.run_config["fraction-train"])
    poisoned_cfg = int(context.run_config["poisoned-hospitals"])
    boost = float(context.run_config["attack-boost"])
    iid = bool(context.run_config["iid"])

    # Step 0: initialise the global model.
    initial_arrays = ArrayRecord(init_params())

    # A balanced test set the server holds, so we can score the global model
    # on a distribution that matches no single hospital.
    x_test, y_test = load_global_testset()

    def global_evaluate(server_round: int, arrays: ArrayRecord):
        params = [np.asarray(a) for a in arrays.to_numpy_ndarrays()]
        loss, acc = evaluate_local(params, x_test, y_test)
        print(f"  >> global model: acc={acc:.3f}  loss={loss:.3f}")
        return MetricRecord({"global_loss": loss, "global_acc": acc})

    # Robust aggregation, section 6. Swap the rule without touching a single
    # line of client code -- and notice that every one of these works by
    # INSPECTING individual updates, which is exactly what secure aggregation
    # is designed to prevent. That is slide 19.
    name = str(context.run_config["strategy"]).lower()
    poisoned = int(context.run_config["poisoned-hospitals"])
    common = dict(
        fraction_train=fraction_train,
        fraction_evaluate=1.0,
        min_train_nodes=2,
        min_evaluate_nodes=2,
        min_available_nodes=2,
    )
    if name == "krum":
        strategy = Krum(**common, num_malicious_nodes=poisoned)
    elif name == "multikrum":
        strategy = MultiKrum(**common, num_malicious_nodes=poisoned,
                             num_nodes_to_select=2)
    elif name == "median":
        strategy = FedMedian(**common)
    elif name == "trimmed":
        strategy = FedTrimmedAvg(**common, beta=0.2)
    else:
        strategy = LoggingFedAvg(**common)
    print(f"  aggregation rule: {strategy.__class__.__name__}"
          f"   poisoned hospitals: {poisoned}"
          f"   data: {'IID' if iid else 'non-IID'}")

    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        train_config=ConfigRecord(
            {
                "local-epochs": local_epochs,
                "poisoned-hospitals": poisoned_cfg,
                "attack-boost": boost,
                "iid": iid,
            }
        ),
        evaluate_config=ConfigRecord({"iid": iid}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    print("\n" + "=" * 62)
    print("  round   global acc   federated acc")
    print("=" * 62)
    for rnd in sorted(result.evaluate_metrics_serverapp):
        g = result.evaluate_metrics_serverapp[rnd]["global_acc"]
        fed = result.evaluate_metrics_clientapp.get(rnd, {}).get("eval_acc")
        fed_str = f"{fed:.3f}" if fed is not None else "   -"
        print(f"  {rnd:>5}   {g:>10.3f}   {fed_str:>13}")
    print("=" * 62)
