"""Fine-tuning do BERTimbau com loop de treino em PyTorch puro."""
import argparse
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from src.data import DEFAULT_PATH, load_splits
from src.metrics import evaluate, save_confusion_matrix

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.enc = tokenizer(list(texts), truncation=True, max_length=max_len)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = int(self.labels[i])
        return item


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        preds.extend(model(**batch).logits.argmax(-1).cpu().tolist())
    return np.array(preds)


def main(args, eval_test=True):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train, val, test, label2id = load_splits(args.data)
    names = list(label2id)
    id2label = {i: l for l, i in label2id.items()}

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    collate = DataCollatorWithPadding(tok)
    def mk(df, shuffle):
        return DataLoader(
            TextDataset(df["text"], df["y"], tok, args.max_len),
            batch_size=args.batch_size,
            shuffle=shuffle,
            collate_fn=collate,
        )

    train_dl, val_dl, test_dl = mk(train, True), mk(val, False), mk(test, False)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(names), id2label=id2label, label2id=label2id
    ).to(device)

    # pesos por classe para lidar com desbalanceamento
    counts = np.bincount(train["y"], minlength=len(names))
    weights = torch.tensor(len(train) / (len(names) * counts), dtype=torch.float, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = len(train_dl) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)

    out_dir = Path(args.out)
    mlflow.set_experiment("nlp-saude-ptbr")
    run_name = getattr(args, "run_name", "bertimbau")
    with mlflow.start_run(run_name=run_name, nested=mlflow.active_run() is not None):
        mlflow.log_params(
            {"model": MODEL_NAME, "epochs": args.epochs, "lr": args.lr,
             "batch_size": args.batch_size, "max_len": args.max_len}
        )
        best_f1 = -1.0
        for epoch in range(args.epochs):
            model.train()
            running = 0.0
            for batch in train_dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch.pop("labels")
                loss = loss_fn(model(**batch).logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                running += loss.item()

            val_pred = predict(model, val_dl, device)
            val_f1 = evaluate(val["y"], val_pred, names)["f1_macro"]
            mlflow.log_metrics({"train_loss": running / len(train_dl), "val_f1_macro": val_f1}, step=epoch)
            print(f"epoch {epoch + 1}: loss={running / len(train_dl):.4f} val_f1={val_f1:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                model.save_pretrained(out_dir)
                tok.save_pretrained(out_dir)

        if not eval_test:  # usado na busca de hiperparâmetros: escolhe só pela validação
            return best_f1

        # avaliação final com o melhor checkpoint (selecionado na validação)
        model = AutoModelForSequenceClassification.from_pretrained(out_dir).to(device)
        test_pred = predict(model, test_dl, device)
        res = evaluate(test["y"], test_pred, names)
        mlflow.log_metrics({"test_f1_macro": res["f1_macro"], "test_accuracy": res["accuracy"]})
        save_confusion_matrix(test["y"], test_pred, names, "reports/cm_bert.png")
        mlflow.log_artifact("reports/cm_bert.png")
        print(res["report"])
        print(f"F1 macro (teste): {res['f1_macro']:.4f}")
        return best_f1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=DEFAULT_PATH)
    p.add_argument("--out", default="models/bert")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
