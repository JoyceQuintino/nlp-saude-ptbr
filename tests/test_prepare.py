import pandas as pd

from src.prepare_medpt import build_dataset


def _df():
    rows = [
        ("Estou com dor de cabeça forte há dias", "Neurologia"),
        ("Estou com dor de cabeça forte há dias", "Neurologia"),  # duplicada
        ("Tenho ansiedade e não consigo dormir bem", "Psicologia"),
        ("Tenho ansiedade e não consigo dormir bem", "Psiquiatria"),  # conflito
        ("curta", "Neurologia"),  # curta demais
        ("Quais exames fazer para checar a tireoide?", "Endocrinologia"),
    ]
    return pd.DataFrame(rows, columns=["Question", "Medical Specialty"])


def test_build_dataset_limpa_duplicados_conflitos_e_curtas():
    out = build_dataset(_df(), task="specialty", top=10)
    assert out["text"].is_unique
    assert not out["text"].str.contains("ansiedade").any()  # rótulos conflitantes
    assert not (out["text"] == "curta").any()
    assert len(out) == 2


def test_descarta_rotulos_com_varias_especialidades():
    rows = [
        ("Tenho ansiedade e não consigo dormir bem à noite", "Psicólogo"),
        ("Sinto tristeza profunda há várias semanas seguidas", "Psicólogo, Psicanalista"),
    ]
    df = pd.DataFrame(rows, columns=["Question", "Medical Specialty"])
    out = build_dataset(df, task="specialty", top=10)
    assert list(out["label"]) == ["Psicólogo"]
