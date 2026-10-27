"""
Utilidades compartidas por todos los modelos (XGBoost, red neuronal): carga de datos y métricas.
Así los modelos se evalúan EXACTAMENTE igual y la comparación es justa.

MÉTRICAS (sobre el conjunto de prueba)
  En escala logarítmica (lo que el modelo optimiza):
    RMSE_log  raíz del error cuadrático medio de log(precio)
    R2_log    proporción de la variación de log(precio) que explica el modelo (1 = perfecto)
  En pesos (lo que entiende el cliente):
    MAE       error absoluto medio en pesos
    MAPE      error porcentual medio
    MdAPE     error porcentual MEDIANO (menos sensible a casos extremos)
    dentro_10 / dentro_20: % de viviendas cuyo precio se predijo con error menor al 10 % / 20 %
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
DATOS = RAIZ / "data" / "processed" / "modelo"
MODELOS = RAIZ / "models"
REPORTES = RAIZ / "reports"
FIG = REPORTES / "figures"


def cargar():
    """Devuelve (train, val, test, metadata)."""
    meta = json.loads((DATOS / "metadata.json").read_text(encoding="utf-8"))
    conjuntos = [pd.read_csv(DATOS / f"{n}.csv", dtype={"id": str}, encoding="utf-8-sig") for n in ["train", "val", "test"]]
    return (*conjuntos, meta)


def columnas_entrada(meta: dict) -> tuple[list[str], list[str]]:
    """(columnas numéricas/binarias, columnas categóricas como índice)."""
    num = meta["numericas"] + meta["indicadores_falta"] + list(meta["amenidades"])
    cat = [f"{c}_idx" for c in meta["categoricas"]]
    return num, cat


def nombre_legible(col: str, meta: dict) -> str:
    """'am_03' -> 'amenidad: Zona de lavandería';  'barrio_modelo_idx' -> 'barrio_modelo'."""
    if col in meta["amenidades"]:
        return f"amenidad: {meta['amenidades'][col]}"
    return col.replace("_idx", "")


def metricas(y_log_real, y_log_pred) -> dict:
    y_log_real, y_log_pred = np.asarray(y_log_real, float), np.asarray(y_log_pred, float)
    real, pred = np.exp(y_log_real), np.exp(y_log_pred)
    ape = np.abs(pred - real) / real
    ss_res = ((y_log_real - y_log_pred) ** 2).sum()
    ss_tot = ((y_log_real - y_log_real.mean()) ** 2).sum()
    return {
        "RMSE_log": float(np.sqrt(np.mean((y_log_real - y_log_pred) ** 2))),
        "R2_log": float(1 - ss_res / ss_tot),
        "MAE_pesos": float(np.mean(np.abs(pred - real))),
        "MAPE_%": float(ape.mean() * 100),
        "MdAPE_%": float(np.median(ape) * 100),
        "dentro_10_%": float((ape <= 0.10).mean() * 100),
        "dentro_20_%": float((ape <= 0.20).mean() * 100),
    }


def linea_base(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """
    Modelo "ingenuo" de referencia: predice la mediana de log(precio) de viviendas del mismo
    municipio, tipo y estrato en el train (o la mediana general si el grupo no existe).
    Un modelo de verdad tiene que ganarle claramente; si no, no aporta nada.
    """
    claves = ["municipio", "tipo_inmueble", "estrato"]
    medianas = train.groupby(claves)["log_precio"].median()
    pred = test[claves].apply(tuple, axis=1).map(medianas)
    return pred.fillna(train["log_precio"].median()).to_numpy()


def imprimir_metricas(nombre: str, m: dict) -> None:
    print(f"  {nombre:<22} RMSE_log {m['RMSE_log']:.4f} | R2_log {m['R2_log']:.3f} | "
          f"MAE ${m['MAE_pesos']:,.0f} | MAPE {m['MAPE_%']:.1f}% | MdAPE {m['MdAPE_%']:.1f}% | "
          f"±10%: {m['dentro_10_%']:.1f}% | ±20%: {m['dentro_20_%']:.1f}%")
