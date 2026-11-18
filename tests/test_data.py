import pandas as pd

from src.data import load_splits
from src.metrics import evaluate


def _fake_csv(tmp_path):
    rows = [(f"texto {i} da classe {c}", c) for c in ("a", "b", "c") for i in range(20)]
    path = tmp_path / "d.csv"
    pd.DataFrame(rows, columns=["text", "label"]).to_csv(path, index=False)
    return str(path)


def test_splits_sem_vazamento(tmp_path):
    train, val, test, label2id = load_splits(_fake_csv(tmp_path))
    assert set(label2id) == {"a", "b", "c"}
    assert not set(train["text"]) & set(test["text"])
    assert not set(val["text"]) & set(test["text"])
    assert len(train) + len(val) + len(test) == 60


def test_f1_perfeito():
    res = evaluate([0, 1, 2], [0, 1, 2], ["a", "b", "c"])
    assert res["f1_macro"] == 1.0


def test_duplicados_nao_vazam_entre_splits(tmp_path):
    rows = []
    for c in ("a", "b"):
        for i in range(30):
            rows += [(f"pergunta repetida {c} {i}", c)] * 2  # cada texto 2x
    path = tmp_path / "dup.csv"
    pd.DataFrame(rows, columns=["text", "label"]).to_csv(path, index=False)
    train, val, test, _ = load_splits(str(path))
    assert len(train) + len(val) + len(test) == 60
    assert not set(train["text"]) & set(test["text"])
