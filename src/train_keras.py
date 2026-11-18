"""Modelo em TensorFlow/Keras: TextVectorization + Embedding + BiLSTM."""
import argparse

import mlflow
import numpy as np
import tensorflow as tf

from src.data import DEFAULT_PATH, load_splits
from src.metrics import evaluate, save_confusion_matrix


def main(args):
    tf.keras.utils.set_random_seed(args.seed)
    train, val, test, label2id = load_splits(args.data)
    names = list(label2id)

    def x(df):
        return tf.constant(df["text"].astype(str).tolist())

    vec = tf.keras.layers.TextVectorization(
        max_tokens=args.vocab, output_sequence_length=args.max_len
    )
    vec.adapt(x(train))

    inp = tf.keras.Input(shape=(), dtype=tf.string)
    h = vec(inp)
    h = tf.keras.layers.Embedding(args.vocab, 128, mask_zero=True)(h)
    h = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(h)
    h = tf.keras.layers.Dropout(0.3)(h)
    out = tf.keras.layers.Dense(len(names), activation="softmax")(h)
    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    counts = np.bincount(train["y"], minlength=len(names))
    cw = {i: len(train) / (len(names) * c) for i, c in enumerate(counts)}

    mlflow.set_experiment("nlp-saude-ptbr")
    with mlflow.start_run(run_name="keras_bilstm"):
        mlflow.log_params({"model": "bilstm", "epochs": args.epochs, "lr": args.lr,
                           "vocab": args.vocab, "max_len": args.max_len})
        model.fit(
            x(train), train["y"].to_numpy(),
            validation_data=(x(val), val["y"].to_numpy()),
            epochs=args.epochs, batch_size=32, class_weight=cw,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)],
        )
        preds = model.predict(x(test)).argmax(-1)
        res = evaluate(test["y"], preds, names)
        mlflow.log_metrics({"test_f1_macro": res["f1_macro"], "test_accuracy": res["accuracy"]})
        save_confusion_matrix(test["y"], preds, names, "reports/cm_keras.png")
        mlflow.log_artifact("reports/cm_keras.png")
        print(res["report"])
        print(f"F1 macro (teste): {res['f1_macro']:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=DEFAULT_PATH)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--vocab", type=int, default=20000)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
