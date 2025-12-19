import argparse
import csv
from pathlib import Path

import numpy as np
import torch

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:
    raise SystemExit(
        "Missing scikit-learn. Install with: pip install scikit-learn"
    ) from exc


def load_activations(path: Path) -> tuple[np.ndarray, list[str]]:
    payload = torch.load(path, map_location="cpu")
    acts = payload["activations"]
    if isinstance(acts, torch.Tensor):
        acts = acts.detach().cpu().numpy()
    ids = payload["ids"]
    return acts, list(ids)


def load_data(path: Path) -> dict[str, dict[str, str]]:
    out = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[row["id"]] = row
    return out


def build_task_indices(ids: list[str], data_by_id: dict[str, dict[str, str]], task: str):
    labels = []
    indices = []
    for i, row_id in enumerate(ids):
        row = data_by_id.get(row_id)
        if not row:
            continue
        stance = row["stance_label"]
        prop_id = row["proposition_id"]
        is_true = prop_id.endswith("__true")
        is_false = prop_id.endswith("__false")

        if task == "A":
            if not is_true or stance not in {"declared_true", "declared_false"}:
                continue
            label = 1 if stance == "declared_true" else 0
        elif task == "B":
            if not is_false or stance not in {"declared_true", "declared_false"}:
                continue
            label = 1 if stance == "declared_true" else 0
        elif task == "C":
            if not is_false or stance not in {"bare", "declared_false"}:
                continue
            label = 1 if stance == "declared_false" else 0
        else:
            raise ValueError(f"Unknown task: {task}")

        indices.append(i)
        labels.append(label)

    return np.array(indices, dtype=np.int64), np.array(labels, dtype=np.int64)


def fit_and_score(X_train, y_train, X_test, y_test, seed: int) -> float:
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = LogisticRegression(
        penalty="l2",
        solver="saga",
        max_iter=2000,
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, probs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train linear probes on activations.")
    parser.add_argument("--train-acts", default="dataset/activations_train.pt")
    parser.add_argument(
        "--eval-acts",
        default="dataset/activations_test.pt",
        help="Evaluation activations. Set to the train activations for in-sample eval.",
    )
    parser.add_argument("--data", default="dataset/data.csv")
    parser.add_argument("--tasks", default="A,B,C")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument(
        "--out",
        default="dataset/probe_aurocs.csv",
        help="Output CSV with AUROC by layer/position/task.",
    )
    args = parser.parse_args()

    train_acts, train_ids = load_activations(Path(args.train_acts))
    eval_acts, eval_ids = load_activations(Path(args.eval_acts))
    data_by_id = load_data(Path(args.data))

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    if train_acts.ndim != 4 or eval_acts.ndim != 4:
        raise ValueError("Expected activations with shape [n, layers, positions, d_model].")

    n_layers = train_acts.shape[1]
    n_positions = train_acts.shape[2]

    rows_out = []
    for task in tasks:
        train_idx, y_train = build_task_indices(train_ids, data_by_id, task)
        eval_idx, y_eval = build_task_indices(eval_ids, data_by_id, task)
        if len(train_idx) == 0 or len(eval_idx) == 0:
            raise ValueError(f"No data for task {task} in train or test split.")
        if len(np.unique(y_train)) < 2 or len(np.unique(y_eval)) < 2:
            raise ValueError(f"Need both classes present for task {task}.")

        for layer in range(n_layers):
            for pos in range(n_positions):
                X_train = train_acts[train_idx, layer, pos, :].astype(np.float32, copy=False)
                X_test = eval_acts[eval_idx, layer, pos, :].astype(np.float32, copy=False)
                scores = []
                for seed in seeds:
                    auc = fit_and_score(X_train, y_train, X_test, y_eval, seed)
                    scores.append(auc)
                scores = np.array(scores, dtype=np.float64)
                rows_out.append(
                    {
                        "task": task,
                        "layer": layer,
                        "position": "entity" if pos == 0 else "final",
                        "mean_auroc": float(scores.mean()),
                        "std_auroc": float(scores.std(ddof=0)),
                        "n_seeds": len(scores),
                        "n_train": len(y_train),
                        "n_test": len(y_eval),
                    }
                )

    out_path = Path(args.out)
    fieldnames = [
        "task",
        "layer",
        "position",
        "mean_auroc",
        "std_auroc",
        "n_seeds",
        "n_train",
        "n_test",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Wrote {len(rows_out)} rows to {out_path}")


if __name__ == "__main__":
    main()
