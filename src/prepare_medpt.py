"""Gera data/raw/dataset.csv (colunas text,label) a partir do MedPT (AKCIT/MedPT).

Tarefas:
  specialty      especialidade médica de quem respondeu (top-N mais frequentes)
  question_type  tipo da pergunta (rótulo gerado por LLM no próprio dataset)

Uso:
  python -m src.prepare_medpt --task specialty --top 10 --max_per_class 2000
  python -m src.prepare_medpt --csv medpt_local.csv --task question_type
"""
import argparse
from pathlib import Path

import pandas as pd

HF_NAME = "AKCIT/MedPT"


def _find(df, name):
    """Acha a coluna ignorando maiúsculas/espaços (ex.: 'Medical Specialty')."""
    norm = {c.lower().replace(" ", "_"): c for c in df.columns}
    if name not in norm:
        raise KeyError(f"Coluna '{name}' não encontrada. Colunas: {list(df.columns)}")
    return norm[name]


def build_dataset(df, task="specialty", top=10, max_per_class=None, seed=42, min_chars=20):
    label_col = _find(df, "medical_specialty" if task == "specialty" else "question_type")
    out = df[[_find(df, "question"), label_col]].copy()
    out.columns = ["text", "label"]
    out = out.dropna()
    out["text"] = out["text"].astype(str).str.strip()
    out["label"] = out["label"].astype(str).str.strip()
    out = out[out["text"].str.len() >= min_chars]

    if task == "specialty":
        # Profissionais podem listar várias especialidades ("Psicólogo, Psicanalista").
        # Esses rótulos combinados se sobrepõem às classes simples e confundem o modelo.
        out = out[~out["label"].str.contains(",")]
        keep = out["label"].value_counts().head(top).index
        out = out[out["label"].isin(keep)]

    # O dataset tem ~2 respostas por pergunta: uma linha por pergunta, e
    # descarta perguntas com rótulos conflitantes (ex.: respondidas por 2 especialidades).
    n_labels = out.groupby("text")["label"].transform("nunique")
    out = out[n_labels == 1].drop_duplicates(subset="text")

    if max_per_class:
        out = out.sample(frac=1, random_state=seed).groupby("label").head(max_per_class)
    return out.reset_index(drop=True)


def main(args):
    if args.csv:
        df = pd.read_csv(args.csv)
    else:
        from datasets import load_dataset

        ds = load_dataset(HF_NAME)
        df = next(iter(ds.values())).to_pandas()
    print("Colunas:", list(df.columns), "| linhas brutas:", len(df))

    out = build_dataset(df, args.task, args.top, args.max_per_class, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Salvo {len(out)} perguntas únicas em {args.out}")
    print(out["label"].value_counts())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["specialty", "question_type"], default="specialty")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--max_per_class", type=int, default=2000)
    p.add_argument("--csv", default=None)
    p.add_argument("--out", default="data/raw/dataset.csv")
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
