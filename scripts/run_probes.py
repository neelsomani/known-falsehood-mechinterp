import argparse
import csv
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from joblib import Parallel, delayed
except ImportError as exc:
    raise SystemExit(
        "Missing dependencies. Install with: pip install scikit-learn joblib"
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


def save_cache(path: Path, scaler: StandardScaler, clf: LogisticRegression) -> None:
    np.savez(
        path,
        mean=scaler.mean_,
        scale=scaler.scale_,
        coef=clf.coef_,
        intercept=clf.intercept_,
        classes=clf.classes_,
    )


def score_with_cache(path: Path, X_test_scaled: np.ndarray, y_test: np.ndarray) -> float:
    data = np.load(path)
    coef = data["coef"]
    intercept = data["intercept"]
    classes = data["classes"]
    if classes.shape[0] != 2 or classes[0] != 0 or classes[1] != 1:
        raise ValueError(f"Unexpected classes in cache: {classes}")

    logits = X_test_scaled @ coef.T + intercept
    probs = 1.0 / (1.0 + np.exp(-logits))
    return roc_auc_score(y_test, probs.ravel())


def run_one(
    task: str,
    layer: int,
    pos: int,
    train_acts: np.ndarray,
    eval_acts: np.ndarray,
    train_idx: np.ndarray,
    y_train: np.ndarray,
    eval_idx: np.ndarray,
    y_eval: np.ndarray,
    seeds: list[int],
    cache_root: Path | None,
) -> dict[str, object]:
    X_train = train_acts[train_idx, layer, pos, :].astype(np.float32, copy=False)
    X_test = eval_acts[eval_idx, layer, pos, :].astype(np.float32, copy=False)

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    scores = []
    for seed in seeds:
        cache_path = None
        if cache_root is not None:
            cache_path = cache_root / f"{task}_L{layer}_P{pos}_S{seed}.npz"

        if cache_path is not None and cache_path.exists():
            auc = score_with_cache(cache_path, X_test_scaled, y_eval)
        else:
            clf = LogisticRegression(
                penalty="l2",
                solver="lbfgs",
                max_iter=500,
                random_state=seed,
                n_jobs=1,
            )
            clf.fit(X_train_scaled, y_train)
            probs = clf.predict_proba(X_test_scaled)[:, 1]
            auc = roc_auc_score(y_eval, probs)
            if cache_path is not None:
                save_cache(cache_path, scaler, clf)
        scores.append(auc)

    scores = np.array(scores, dtype=np.float64)
    return {
        "task": task,
        "layer": layer,
        "position": "entity" if pos == 0 else "final",
        "mean_auroc": float(scores.mean()),
        "std_auroc": float(scores.std(ddof=0)),
        "n_seeds": len(scores),
        "n_train": len(y_train),
        "n_test": len(y_eval),
    }


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
        "--cache-dir",
        default="dataset/probe_cache",
        help="Directory to save/load cached probe weights.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable loading/saving cached probe weights.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip tasks that don't have both classes in the eval split.",
    )
    parser.add_argument(
        "--out",
        default="dataset/probe_aurocs.csv",
        help="Output CSV with AUROC by layer/position/task.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=96,
        help="Number of parallel workers for probe fits.",
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

    train_acts = np.ascontiguousarray(train_acts, dtype=np.float32)
    eval_acts = np.ascontiguousarray(eval_acts, dtype=np.float32)
    memmap_root = Path(args.cache_dir) if args.cache_dir else Path("dataset/probe_cache")
    memmap_dir = memmap_root / "_memmap"
    memmap_dir.mkdir(parents=True, exist_ok=True)
    train_mm_path = memmap_dir / "train_acts.npy"
    eval_mm_path = memmap_dir / "eval_acts.npy"

    def ensure_memmap(path: Path, array: np.ndarray) -> None:
        if path.exists():
            existing = np.load(path, mmap_mode="r")
            if existing.shape == array.shape and existing.dtype == array.dtype:
                return
        np.save(path, array)

    ensure_memmap(train_mm_path, train_acts)
    ensure_memmap(eval_mm_path, eval_acts)
    train_acts = np.load(train_mm_path, mmap_mode="r")
    eval_acts = np.load(eval_mm_path, mmap_mode="r")

    cache_root = Path(args.cache_dir) if args.cache_dir and not args.no_cache else None
    if cache_root is not None:
        cache_root.mkdir(parents=True, exist_ok=True)

    rows_out = []
    for task in tasks:
        train_idx, y_train = build_task_indices(train_ids, data_by_id, task)
        eval_idx, y_eval = build_task_indices(eval_ids, data_by_id, task)
        if len(train_idx) == 0 or len(eval_idx) == 0:
            raise ValueError(f"No data for task {task} in train or test split.")
        if len(np.unique(y_train)) < 2 or len(np.unique(y_eval)) < 2:
            if args.skip_missing:
                print(f"Skipping task {task}: missing class in train or eval split.")
                continue
            raise ValueError(f"Need both classes present for task {task}.")

        jobs = [(layer, pos) for layer in range(n_layers) for pos in range(n_positions)]
        n_workers = min(args.workers, os.cpu_count() or 1)
        print(f"Running {len(jobs)} probes for task {task} on {n_workers} workers.")
        task_start = time.time()
        rows_task = Parallel(n_jobs=n_workers, backend="loky", verbose=10)(
            delayed(run_one)(
                task,
                layer,
                pos,
                train_acts,
                eval_acts,
                train_idx,
                y_train,
                eval_idx,
                y_eval,
                seeds,
                cache_root,
            )
            for (layer, pos) in jobs
        )
        rows_out.extend(rows_task)
        elapsed = (time.time() - task_start) / 60.0
        print(f"Finished task {task} in {elapsed:.1f}m.")

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
