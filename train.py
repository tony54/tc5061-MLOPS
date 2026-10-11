"""
train.py - Entrenamiento parametrizado de un clasificador de ingreso (UCI Adult)
con registro de parametros, metricas, artefactos y modelo en MLflow Tracking.

Uso basico:
    python train.py
    python train.py --C 0.1 --penalty l1 --class-weight balanced --seed 42
    python train.py --help
"""
import argparse
import hashlib
import json
import os
import random
import sys
import warnings

warnings.filterwarnings("ignore", message=".*Inferred schema contains integer column.*")
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             classification_report, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

COLUMNAS = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
            "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
            "hours_per_week", "native_country", "income"]
NUM_COLS = ["age", "education_num", "hours_per_week", "capital_gain_log", "capital_loss_log"]
CAT_COLS = ["workclass", "marital_status", "occupation", "relationship", "race", "sex",
            "native_country"]


# ----------------------------------------------------------------------------- argumentos
def parse_args():
    p = argparse.ArgumentParser(
        description="Entrena una regresion logistica sobre UCI Adult y registra la corrida en MLflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data", default="data/adult.data", help="Ruta al archivo adult.data")
    p.add_argument("--C", type=float, default=1.0,
                   help="Inverso de la fuerza de regularizacion (menor = mas regularizacion)")
    p.add_argument("--penalty", choices=["l1", "l2"], default="l2", help="Tipo de regularizacion")
    p.add_argument("--class-weight", choices=["none", "balanced"], default="none",
                   help="Ponderacion de clases para compensar el desbalance")
    p.add_argument("--max-iter", type=int, default=1000, help="Iteraciones maximas del optimizador")
    p.add_argument("--test-size", type=float, default=0.2, help="Fraccion del conjunto de prueba")
    p.add_argument("--seed", type=int, default=42, help="Semilla aleatoria (particion y modelo)")
    p.add_argument("--experiment", default="adult_income_logreg", help="Nombre del experimento MLflow")
    p.add_argument("--run-name", default=None, help="Nombre de la corrida (opcional)")
    p.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"),
                   help="Servidor de MLflow; si no responde se usa ./mlruns")
    p.add_argument("--output-dir", default="outputs", help="Carpeta local para los artefactos")
    return p.parse_args()


# ----------------------------------------------------------------------------- datos
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def md5_de_archivo(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def load_data(path):
    return pd.read_csv(path, header=None, names=COLUMNAS, skipinitialspace=True)


def clean_data(df):
    """Mismas decisiones documentadas en el notebook de exploracion."""
    df = df.replace("?", np.nan).dropna().drop_duplicates()
    y = (df["income"] == ">50K").astype(int)
    X = df.drop(columns=["income", "fnlwgt", "education", "capital_gain", "capital_loss"])
    X["capital_gain_log"] = np.log1p(df["capital_gain"])
    X["capital_loss_log"] = np.log1p(df["capital_loss"])
    return X, y


def split_data(X, y, test_size, seed):
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


# ----------------------------------------------------------------------------- modelo
def build_pipeline(args):
    preproc = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ])
    clf = LogisticRegression(
        C=args.C,
        penalty=args.penalty,
        solver="liblinear" if args.penalty == "l1" else "lbfgs",
        class_weight=None if args.class_weight == "none" else "balanced",
        max_iter=args.max_iter,
        random_state=args.seed,
    )
    return Pipeline([("preproc", preproc), ("clf", clf)])


def evaluate(model, X, y):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_proba),
    }
    return metrics, y_pred, y_proba


# ----------------------------------------------------------------------------- artefactos
def save_artifacts(output_dir, y_test, y_pred, y_proba, metrics):
    os.makedirs(output_dir, exist_ok=True)
    rutas = []

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["<=50K", ">50K"],
                                            cmap="Blues", ax=axes[0], colorbar=False)
    axes[0].set_title("Matriz de confusion (prueba)")
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=axes[1], name="Reg. logistica")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].set_title("Curva ROC")
    plt.tight_layout()
    rutas.append(os.path.join(output_dir, "evaluation.png"))
    fig.savefig(rutas[-1], dpi=110, bbox_inches="tight")
    plt.close(fig)

    rutas.append(os.path.join(output_dir, "metrics.json"))
    with open(rutas[-1], "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    rutas.append(os.path.join(output_dir, "classification_report.txt"))
    with open(rutas[-1], "w", encoding="utf-8") as f:
        f.write(classification_report(y_test, y_pred, target_names=["<=50K", ">50K"], digits=4))
    return rutas


# ----------------------------------------------------------------------------- mlflow
def configurar_tracking(uri):
    try:
        requests.get(uri, timeout=3)
    except Exception:
        print(f"[aviso] {uri} no responde; se registra en ./mlruns")
        uri = "mlruns"
    mlflow.set_tracking_uri(uri)


def log_run(args, model, metrics, train_metrics, artifacts, X_train, X_test, data_md5):
    mlflow.log_params({
        "model": "LogisticRegression",
        "C": args.C,
        "penalty": args.penalty,
        "class_weight": args.class_weight,
        "max_iter": args.max_iter,
        "solver": model["clf"].solver,
        "test_size": args.test_size,
        "seed": args.seed,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "data_md5": data_md5,
    })
    mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
    mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
    mlflow.log_metric("n_iter", int(np.ravel(model["clf"].n_iter_)[0]))
    for ruta in artifacts:
        mlflow.log_artifact(ruta)
    ejemplo = X_train.head(5).astype({c: "float64" for c in ["age", "education_num", "hours_per_week"]})
    mlflow.sklearn.log_model(model, name="model", input_example=ejemplo)


# ----------------------------------------------------------------------------- main
def main():
    args = parse_args()
    set_seed(args.seed)

    X, y = clean_data(load_data(args.data))
    X_train, X_test, y_train, y_test = split_data(X, y, args.test_size, args.seed)

    model = build_pipeline(args)
    model.fit(X_train, y_train)

    metrics, y_pred, y_proba = evaluate(model, X_test, y_test)
    train_metrics = {k: v for k, v in evaluate(model, X_train, y_train)[0].items() if k in ("accuracy", "f1")}
    artifacts = save_artifacts(args.output_dir, y_test, y_pred, y_proba, metrics)

    configurar_tracking(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=args.run_name) as run:
        log_run(args, model, metrics, train_metrics, artifacts, X_train, X_test,
                md5_de_archivo(args.data))
        run_id = run.info.run_id

    print(f"Corrida      : {run_id}")
    print(f"Configuracion: C={args.C} penalty={args.penalty} class_weight={args.class_weight} "
          f"max_iter={args.max_iter} seed={args.seed}")
    for k, v in metrics.items():
        print(f"  test_{k:<10}: {v:.4f}")


if __name__ == "__main__":
    main()
