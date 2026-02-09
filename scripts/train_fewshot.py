"""Few-shot MLP regression: train on embeddings, evaluate Spearman correlation.

Usage:
    python scripts/train_fewshot.py \
        --emb-path embs/antiberty/thermostability/jain2017biophysical_Tm/embeddings.pt \
        --csv-path data/thermostability/jain2017biophysical_Tm.csv

    # Run all eligible datasets for a model:
    python scripts/train_fewshot.py --run-all --model antiberty
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

FLAB_DIR = Path(__file__).resolve().parent.parent
MIN_SIZE = 30


def get_args():
    parser = argparse.ArgumentParser(description="Few-shot MLP regression on embeddings")
    parser.add_argument("--emb-path", type=str, default=None,
                        help="Path to embeddings.pt")
    parser.add_argument("--csv-path", type=str, default=None,
                        help="Path to dataset CSV")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: fewshot/{model}/{category}/{dataset}/)")
    parser.add_argument("--hidden-dim", type=int, default=256,
                        help="MLP hidden dimension")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--run-all", action="store_true",
                        help="Run all eligible datasets for the specified model")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (required with --run-all)")
    return parser.parse_args()


class FewShotMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_single_trial(embeddings, fitness, hidden_dim, lr, epochs, patience, seed, device):
    """Train one trial and return test Spearman rho and p-value."""
    device_obj = torch.device(device)
    rng = np.random.RandomState(seed)

    n = len(fitness)
    indices = np.arange(n)

    # 80/10/10 split
    train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=rng)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=rng)

    # Z-score normalize embeddings (fit on train)
    emb_train = embeddings[train_idx]
    emb_mean = emb_train.mean(dim=0)
    emb_std = emb_train.std(dim=0)
    emb_std[emb_std == 0] = 1.0  # avoid division by zero

    X_train = ((embeddings[train_idx] - emb_mean) / emb_std).to(device_obj)
    X_val = ((embeddings[val_idx] - emb_mean) / emb_std).to(device_obj)
    X_test = ((embeddings[test_idx] - emb_mean) / emb_std).to(device_obj)

    # Z-score normalize fitness (fit on train)
    fit_train = fitness[train_idx]
    fit_mean = fit_train.mean()
    fit_std = fit_train.std()
    if fit_std == 0:
        fit_std = torch.tensor(1.0)

    y_train = ((fitness[train_idx] - fit_mean) / fit_std).to(device_obj)
    y_val = ((fitness[val_idx] - fit_mean) / fit_std).to(device_obj)
    # For test evaluation, use raw fitness (Spearman is rank-based)
    y_test_raw = fitness[test_idx].numpy()

    # Model
    input_dim = embeddings.shape[1]
    model = FewShotMLP(input_dim, hidden_dim).to(device_obj)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    # Load best model and evaluate on test
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test).cpu().numpy()

    rho, p_value = spearmanr(y_test_raw, test_pred)
    return rho, p_value, len(train_idx), len(val_idx), len(test_idx)


def run_fewshot(emb_path, csv_path, output_dir, hidden_dim, lr, epochs,
                patience, base_seed, n_trials, device):
    """Run few-shot learning on one dataset."""
    # Load embeddings
    emb_data = torch.load(emb_path, map_location="cpu")
    embeddings = emb_data["embeddings"].float()
    model_name = emb_data["model"]

    # Load fitness
    df = pd.read_csv(csv_path)
    if len(df) < MIN_SIZE:
        print(f"SKIP: {csv_path} has {len(df)} samples (< {MIN_SIZE})")
        return None
    fitness = torch.tensor(df["fitness"].values, dtype=torch.float32)

    assert len(embeddings) == len(fitness), \
        f"Embedding count {len(embeddings)} != CSV rows {len(fitness)}"

    # Infer output dir
    if output_dir is None:
        parts = Path(csv_path).parts
        category = parts[-2]
        dataset_name = Path(csv_path).stem
        output_dir = str(FLAB_DIR / "fewshot" / model_name / category / dataset_name)

    output_path = os.path.join(output_dir, "results.json")
    if os.path.exists(output_path):
        print(f"SKIP: {output_path} already exists")
        return None

    # Run trials with different seeds
    trials = []
    for trial_idx in range(n_trials):
        seed = base_seed + trial_idx
        rho, p_val, n_train, n_val, n_test = train_single_trial(
            embeddings, fitness, hidden_dim, lr, epochs, patience, seed, device
        )
        trials.append({
            "seed": seed,
            "spearman_rho": float(rho),
            "spearman_p": float(p_val),
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
        })

    rhos = [t["spearman_rho"] for t in trials]
    p_vals = [t["spearman_p"] for t in trials]

    parts = Path(csv_path).parts
    category = parts[-2]
    dataset_name = Path(csv_path).stem

    results = {
        "model": model_name,
        "dataset": dataset_name,
        "category": category,
        "n_samples": len(df),
        "embedding_dim": embeddings.shape[1],
        "hidden_dim_mlp": hidden_dim,
        "n_trials": n_trials,
        "trials": trials,
        "mean_spearman_rho": float(np.mean(rhos)),
        "std_spearman_rho": float(np.std(rhos)),
        "mean_spearman_p": float(np.mean(p_vals)),
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"{model_name} | {category}/{dataset_name} | "
          f"N={len(df)} | rho={results['mean_spearman_rho']:.4f} ± {results['std_spearman_rho']:.4f}")

    return results


def main():
    args = get_args()

    if args.run_all:
        if args.model is None:
            raise ValueError("--model is required with --run-all")

        eligible_csv = FLAB_DIR / "data" / "fewshot_eligible_datasets.csv"
        if not eligible_csv.exists():
            raise FileNotFoundError(f"{eligible_csv} not found. Run generate_fewshot_datasets.py first.")

        eligible = pd.read_csv(eligible_csv)
        all_results = []

        for _, row in eligible.iterrows():
            csv_path = str(FLAB_DIR / row["csv_path"])
            category = row["category"]
            dataset_name = row["dataset_name"]

            emb_path = str(FLAB_DIR / "embs" / args.model / category / dataset_name / "embeddings.pt")
            if not os.path.exists(emb_path):
                print(f"SKIP: {emb_path} not found")
                continue

            output_dir = str(FLAB_DIR / "fewshot" / args.model / category / dataset_name)

            result = run_fewshot(
                emb_path=emb_path,
                csv_path=csv_path,
                output_dir=output_dir,
                hidden_dim=args.hidden_dim,
                lr=args.lr,
                epochs=args.epochs,
                patience=args.patience,
                base_seed=args.seed,
                n_trials=args.n_trials,
                device=args.device,
            )
            if result is not None:
                all_results.append(result)

        # Summary
        if all_results:
            summary_dir = FLAB_DIR / "fewshot" / args.model
            summary_dir.mkdir(parents=True, exist_ok=True)
            with open(summary_dir / "summary.json", "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"\nSummary: {len(all_results)} datasets processed")
            rhos = [r["mean_spearman_rho"] for r in all_results]
            print(f"Overall mean rho: {np.mean(rhos):.4f} ± {np.std(rhos):.4f}")

    else:
        if args.emb_path is None or args.csv_path is None:
            raise ValueError("--emb-path and --csv-path are required (or use --run-all)")

        run_fewshot(
            emb_path=args.emb_path,
            csv_path=args.csv_path,
            output_dir=args.output_dir,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            epochs=args.epochs,
            patience=args.patience,
            base_seed=args.seed,
            n_trials=args.n_trials,
            device=args.device,
        )


if __name__ == "__main__":
    main()
