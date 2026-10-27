"""
Paso 2 del modelo: XGBoost, el modelo de REFERENCIA para el precio justo.

    models/xgboost_precio.json                 modelo entrenado
    reports/metricas_xgboost.json              métricas en el conjunto de prueba
    reports/predicciones_test_xgboost.csv      predicción de cada vivienda de prueba
    reports/figures/xgb_curva_aprendizaje.png  error en train y validación por iteración
    reports/figures/xgb_importancia.png        variables que más usa el modelo

¿QUÉ ES XGBOOST?
    "Gradient boosting" de árboles de decisión: entrena cientos de árboles pequeños, uno tras otro,
    y cada árbol nuevo intenta corregir los errores que dejaron los anteriores. Es el modelo que
    suele ganar en datos de tabla, por eso es la referencia con la que comparamos la red neuronal.

EARLY STOPPING (parada temprana)
    Se le permite crecer hasta 5.000 árboles, pero después de cada uno se mide el error en VALIDACIÓN.
    Si en 200 árboles seguidos no mejora, se detiene y se queda con el mejor. Así evitamos el
    sobreajuste (que el modelo memorice el train y falle con datos nuevos).

Ejecutar:  python -u -m src.models.xgboost_precio
"""

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402

from src.models.comun import (FIG, MODELOS, REPORTES, cargar, columnas_entrada, imprimir_metricas,  # noqa: E402
                              linea_base, metricas, nombre_legible)

PARAMETROS = dict(
    n_estimators=5000, learning_rate=0.03, max_depth=8, min_child_weight=3,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    tree_method="hist", device="cuda",            # entrenar en la GPU (RTX 5070)
    enable_categorical=True, max_cat_to_onehot=1, # el barrio, la comuna... como categorías nativas
    early_stopping_rounds=200, eval_metric="rmse", random_state=42,
)


def preparar_X(df: pd.DataFrame, num: list[str], cat: list[str], meta: dict) -> pd.DataFrame:
    X = df[num + cat].copy()
    for c in cat:   # mismas categorías posibles en train/val/test (las del vocabulario)
        n = len(meta["vocabulario"][c.replace("_idx", "")])
        X[c] = pd.Categorical(X[c], categories=range(n))
    return X


def main() -> None:
    train, val, test, meta = cargar()
    num, cat = columnas_entrada(meta)
    Xtr, Xva, Xte = (preparar_X(d, num, cat, meta) for d in (train, val, test))
    ytr, yva, yte = train["log_precio"], val["log_precio"], test["log_precio"]
    print(f"Train {len(Xtr):,} | Val {len(Xva):,} | Test {len(Xte):,} | variables: {Xtr.shape[1]}")

    modelo = xgb.XGBRegressor(**PARAMETROS)
    t0 = time.time()
    modelo.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xva, yva)], verbose=250)
    print(f"Entrenado en {time.time() - t0:.0f} s | mejor iteración: {modelo.best_iteration}")

    # --- Evaluación en PRUEBA (datos nunca vistos) -------------------------------------
    pred = modelo.predict(Xte)
    m_xgb = metricas(yte, pred)
    m_base = metricas(yte, linea_base(train, test))
    print("\nResultados en el conjunto de PRUEBA:")
    imprimir_metricas("Línea base (mediana)", m_base)
    imprimir_metricas("XGBoost", m_xgb)

    # Error por tipo de inmueble y por municipio (¿dónde se equivoca más?)
    test = test.assign(pred_log=pred)
    print("\nMdAPE (%) por tipo de inmueble:")
    for t, g in test.groupby("tipo_inmueble"):
        print(f"  {t:<14} n={len(g):>5}  MdAPE {metricas(g['log_precio'], g['pred_log'])['MdAPE_%']:.1f}%")
    print("MdAPE (%) por municipio (con al menos 30 viviendas):")
    for mu, g in test.groupby("municipio"):
        if len(g) >= 30:
            print(f"  {mu:<14} n={len(g):>5}  MdAPE {metricas(g['log_precio'], g['pred_log'])['MdAPE_%']:.1f}%")

    # --- Guardar ---------------------------------------------------------------------------
    MODELOS.mkdir(exist_ok=True)
    modelo.save_model(MODELOS / "xgboost_precio.json")
    (REPORTES / "metricas_xgboost.json").write_text(json.dumps(
        {"xgboost": m_xgb, "linea_base": m_base, "mejor_iteracion": int(modelo.best_iteration),
         "parametros": {k: v for k, v in PARAMETROS.items()}}, indent=1), encoding="utf-8")
    test[["id", "precio", "log_precio", "pred_log", "municipio", "tipo_inmueble"]].to_csv(
        REPORTES / "predicciones_test_xgboost.csv", index=False, encoding="utf-8-sig")

    # --- Gráficas ----------------------------------------------------------------------------
    FIG.mkdir(parents=True, exist_ok=True)
    hist = modelo.evals_result()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hist["validation_0"]["rmse"], label="Entrenamiento")
    ax.plot(hist["validation_1"]["rmse"], label="Validación")
    ax.axvline(modelo.best_iteration, color="gray", ls="--", label=f"Mejor iteración ({modelo.best_iteration})")
    ax.set_xlabel("Número de árboles"); ax.set_ylabel("RMSE de log(precio)")
    ax.set_title("XGBoost: curva de aprendizaje"); ax.legend()
    fig.savefig(FIG / "xgb_curva_aprendizaje.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    gain = modelo.get_booster().get_score(importance_type="gain")
    imp = (pd.Series(gain).rename(index=lambda c: nombre_legible(c, meta)).sort_values().tail(20))
    imp = imp / imp.sum() * 100
    fig, ax = plt.subplots(figsize=(8, 7))
    imp.plot.barh(ax=ax)
    ax.set_xlabel("Importancia relativa (% de ganancia, top 20)")
    ax.set_title("XGBoost: variables que más explican el precio")
    fig.savefig(FIG / "xgb_importancia.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print("\nTop 10 variables (ganancia):")
    print(imp.sort_values(ascending=False).head(10).round(1).to_string())
    print(f"\nGuardado: models/xgboost_precio.json, reports/metricas_xgboost.json, gráficas en {FIG}")


if __name__ == "__main__":
    main()
