# TC5061 · Experimentación con MLflow Tracking

Script de entrenamiento parametrizado para el problema **UCI Adult (Census Income)**: predecir si el ingreso anual de una persona supera los 50 000 USD. Convierte el notebook base de la semana anterior en código reutilizable, instrumentado con MLflow para registrar parámetros, métricas, artefactos y el modelo de cada corrida.

Autor: Antonio Díaz · Maestría en Inteligencia Artificial Aplicada · Tecnológico de Monterrey

## Estructura

```
train.py              script de entrenamiento y evaluación (argparse + MLflow)
run_experiments.ps1   lanza las 8 corridas del experimento (Windows)
run_experiments.sh    lanza las 8 corridas del experimento (Linux / WSL / macOS)
requirements.txt      dependencias con versiones fijadas
data/adult.data       dataset UCI Adult (32 561 registros)
notebooks/eda_adult.ipynb   notebook de exploración de datos (solo EDA y diagnóstico de calidad)
```

## Instalación

Probado con Python 3.13.9.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / WSL / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## Ejecución

1. Levanta el servidor de MLflow desde la raíz del repositorio (en otra terminal):

   ```bash
   mlflow ui --host 127.0.0.1 --port 5000
   ```

   Si el servidor no está disponible, el script registra la corrida en la carpeta local `./mlruns`.

2. Corre el script con los valores por defecto:

   ```bash
   python train.py
   ```

3. O cambia los hiperparámetros desde la línea de comandos, sin editar el código:

   ```bash
   python train.py --C 0.1 --penalty l1 --class-weight balanced --seed 42 --run-name mi_corrida
   ```

4. Abre http://localhost:5000, entra al experimento `adult_income_logreg` y compara las corridas.

### Argumentos

| Argumento | Por defecto | Descripción |
|---|---|---|
| `--C` | `1.0` | Inverso de la fuerza de regularización (menor = más regularización) |
| `--penalty` | `l2` | Tipo de regularización: `l1` o `l2` |
| `--class-weight` | `none` | `balanced` pondera las clases para compensar el desbalance 76/24 |
| `--max-iter` | `1000` | Iteraciones máximas del optimizador |
| `--test-size` | `0.2` | Fracción del conjunto de prueba |
| `--seed` | `42` | Semilla aleatoria para la partición y el modelo |
| `--experiment` | `adult_income_logreg` | Nombre del experimento en MLflow |
| `--run-name` | *(vacío)* | Nombre de la corrida |
| `--tracking-uri` | `http://127.0.0.1:5000` | Servidor de MLflow (o variable `MLFLOW_TRACKING_URI`) |
| `--data` | `data/adult.data` | Ruta al dataset |
| `--output-dir` | `outputs` | Carpeta local donde se escriben los artefactos |

`python train.py --help` muestra la ayuda completa.

### Reproducir las corridas del reporte

```bash
# Windows
.\run_experiments.ps1
# Linux / WSL / macOS
bash run_experiments.sh
```

Lanza ocho corridas: la configuración base, tres valores de `C`, regularización `l1`, dos variantes con `class_weight=balanced` y una repetición exacta de la base para comprobar la reproducibilidad.

## Qué se registra en MLflow

- **Parámetros:** `C`, `penalty`, `class_weight`, `max_iter`, `solver`, `test_size`, `seed`, tamaños de train/test y el hash MD5 del archivo de datos.
- **Métricas:** `test_accuracy`, `test_precision`, `test_recall`, `test_f1`, `test_roc_auc`, `train_accuracy`, `train_f1` y `n_iter`.
- **Artefactos:** `evaluation.png` (matriz de confusión y curva ROC), `metrics.json` y `classification_report.txt`.
- **Modelo:** el `Pipeline` completo (preprocesamiento + regresión logística) con `mlflow.sklearn.log_model`, con ejemplo de entrada para inferir la firma.

## Reproducibilidad

- La semilla (`--seed`) se aplica a `random`, `numpy`, a la partición estratificada `train_test_split` y al `random_state` de `LogisticRegression`. Dos ejecuciones con los mismos argumentos producen exactamente las mismas métricas (corridas `baseline_C1_l2` y `baseline_repro` en el experimento).
- `requirements.txt` fija las versiones de todas las librerías usadas.
- El hash MD5 del dataset se registra como parámetro para detectar cambios en los datos entre corridas.

## Preprocesamiento

Replica las decisiones del notebook de exploración: se eliminan las filas con `?` (2 399) y los duplicados (23); se descartan `fnlwgt` y `education`; se aplica `log1p` a `capital_gain` y `capital_loss`; las numéricas se estandarizan y las categóricas se codifican con *one-hot* dentro del `Pipeline`, ajustado solo con los datos de entrenamiento.
