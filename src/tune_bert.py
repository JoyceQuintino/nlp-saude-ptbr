"""Busca em grade de hiperparâmetros do BERTimbau, escolhendo pelo F1 macro de VALIDAÇÃO.

O teste não é tocado aqui: depois da busca, treine o modelo final com a melhor
configuração (reports/best_hparams.json) usando src.train_bert.

Uso:
  python -m src.tune_bert --epochs 2
  python -m src.train_bert --lr <melhor_lr> --batch_size <melhor_bs> --epochs 3
"""
import argparse
import itertools
import json
from pathlib import Path

import mlflow

from src import train_bert
from src.data import DEFAULT_PATH

GRID = {"lr": [1e-5, 2e-5, 5e-5], "batch_size": [16]}


def main(args):
    mlflow.set_experiment("nlp-saude-ptbr")
    best_f1, best_cfg = -1.0, None
    with mlflow.start_run(run_name="tuning_bertimbau"):
        for lr, bs in itertools.product(GRID["lr"], GRID["batch_size"]):
            trial = argparse.Namespace(
                data=args.data, out="models/_tune", epochs=args.epochs, lr=lr,
                batch_size=bs, max_len=args.max_len, seed=args.seed,
                run_name=f"lr{lr}_bs{bs}",
            )
            val_f1 = train_bert.main(trial, eval_test=False)
            print(f"lr={lr} batch_size={bs} -> val_f1_macro={val_f1:.4f}")
            if val_f1 > best_f1:
                best_f1, best_cfg = val_f1, {"lr": lr, "batch_size": bs}

        mlflow.log_params({f"best_{k}": v for k, v in best_cfg.items()})
        mlflow.log_metric("best_val_f1_macro", best_f1)

    Path("reports").mkdir(exist_ok=True)
    with open("reports/best_hparams.json", "w") as f:
        json.dump({"val_f1_macro": best_f1, **best_cfg}, f, indent=2)
    print(f"Melhor configuração: {best_cfg} (val F1 macro = {best_f1:.4f})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=DEFAULT_PATH)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
