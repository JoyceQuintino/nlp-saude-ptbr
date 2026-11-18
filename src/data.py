"""Carregamento e divisão estratificada do dataset (CSV com colunas: text, label)."""
import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_PATH = "data/raw/dataset.csv"


def load_splits(path: str = DEFAULT_PATH, seed: int = 42):
    """Retorna train, val, test (70/15/15), label2id."""
    df = pd.read_csv(path).dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]

    # Mesmo texto em mais de uma linha vira vazamento treino/teste: mantém só
    # textos únicos e descarta os que aparecem com rótulos conflitantes.
    n_labels = df.groupby("text")["label"].transform("nunique")
    df = df[n_labels == 1].drop_duplicates(subset="text")

    labels = sorted(df["label"].unique())
    label2id = {label: i for i, label in enumerate(labels)}
    df["y"] = df["label"].map(label2id)

    train, temp = train_test_split(df, test_size=0.30, stratify=df["y"], random_state=seed)
    val, test = train_test_split(temp, test_size=0.50, stratify=temp["y"], random_state=seed)
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
        label2id,
    )
