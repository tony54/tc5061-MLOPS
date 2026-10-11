#!/usr/bin/env bash
# Ejecuta las corridas del experimento (Linux / WSL / macOS).
# Requiere `mlflow ui` corriendo en http://127.0.0.1:5000 (o se usa ./mlruns).
set -e

python train.py --run-name baseline_C1_l2
python train.py --run-name C0.01_l2         --C 0.01
python train.py --run-name C0.1_l2          --C 0.1
python train.py --run-name C10_l2           --C 10
python train.py --run-name C1_l1            --penalty l1
python train.py --run-name C1_l2_balanced   --class-weight balanced
python train.py --run-name C0.1_l1_balanced --C 0.1 --penalty l1 --class-weight balanced
# Misma configuracion que la primera: comprueba que la semilla hace reproducible el resultado
python train.py --run-name baseline_repro
