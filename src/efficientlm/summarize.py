"""Collect results/<method>/seed<k>/metrics.json -> mean +- std per method.
Pure-python aggregation (no torch/pandas) so the agent can reuse it; plotting is optional."""
import argparse
import csv
import glob
import json
import os
import statistics

METRICS = ["test_acc", "test_f1", "dev_acc", "trainable_params", "trainable_pct",
           "model_size_mb", "adapter_size_kb", "latency_ms", "train_time_s"]


def load_runs(results_dir):
    runs = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*", "seed*", "metrics.json"))):
        with open(path) as f:
            runs.append(json.load(f))
    return runs


def aggregate(runs):
    by_method = {}
    for r in runs:
        by_method.setdefault(r["method"], []).append(r)
    agg = {}
    for method, rs in by_method.items():
        entry = {"n_seeds": len(rs)}
        for m in METRICS:
            vals = [r[m] for r in rs if r.get(m) is not None]
            if vals:
                entry[m] = (statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0)
        agg[method] = entry
    return agg


def to_markdown(agg):
    head = "| method | seeds | test acc (%) | test F1 | trainable params | model MB | adapter KB | latency ms/batch |\n"
    head += "|---|---|---|---|---|---|---|---|\n"
    rows = []
    for method, e in sorted(agg.items(), key=lambda kv: -kv[1].get("test_acc", (0, 0))[0]):
        g = lambda k, f: (f % e[k][0]) if k in e else "-"
        acc = f"{100 * e['test_acc'][0]:.2f} ± {100 * e['test_acc'][1]:.2f}" if "test_acc" in e else "-"
        rows.append(f"| {method} | {e['n_seeds']} | {acc} | {g('test_f1', '%.4f')} | "
                    f"{int(e['trainable_params'][0]):,} | {g('model_size_mb', '%.1f')} | "
                    f"{g('adapter_size_kb', '%.0f')} | {g('latency_ms', '%.1f')} |")
    return head + "\n".join(rows) + "\n"


def write_csv(agg, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "n_seeds"] + [f"{m}_{s}" for m in METRICS for s in ("mean", "std")])
        for method, e in agg.items():
            row = [method, e["n_seeds"]]
            for m in METRICS:
                row += list(e[m]) if m in e else ["", ""]
            w.writerow(row)


def plot(agg, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping plots")
        return
    methods = sorted(agg, key=lambda m: agg[m]["test_acc"][0])
    acc = [100 * agg[m]["test_acc"][0] for m in methods]
    err = [100 * agg[m]["test_acc"][1] for m in methods]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(methods, acc, xerr=err, color="#4c72b0")
    ax.set_xlabel("SST-2 test accuracy (%)")
    ax.set_xlim(max(0, min(acc) - 5), min(100, max(acc) + 2))
    ax.set_title("Accuracy by method (mean ± std over seeds)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "accuracy_by_method.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for m in methods:
        ax.errorbar(agg[m]["trainable_params"][0], 100 * agg[m]["test_acc"][0],
                    yerr=100 * agg[m]["test_acc"][1], fmt="o", capsize=3)
        ax.annotate(m, (agg[m]["trainable_params"][0], 100 * agg[m]["test_acc"][0]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("trainable parameters (log scale)")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("Accuracy vs. trainable parameters")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "accuracy_vs_params.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    args = ap.parse_args()
    runs = load_runs(args.results_dir)
    if not runs:
        raise SystemExit(f"No metrics.json files found under {args.results_dir}/")
    agg = aggregate(runs)
    md = to_markdown(agg)
    print(md)
    write_csv(agg, os.path.join(args.results_dir, "summary.csv"))
    with open(os.path.join(args.results_dir, "summary.md"), "w") as f:
        f.write(md)
    plot(agg, args.results_dir)
    print(f"Saved summary.csv, summary.md and plots in {args.results_dir}/")


if __name__ == "__main__":
    main()
