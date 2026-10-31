"""
Paso 4 del modelo: "aumento de datos" = SEGMENTACIÓN y CLASIFICACIÓN de cada vivienda.

    data/processed/valoracion_<fecha>.csv   por vivienda: precio justo, diferencia %, clase, segmento
    reports/valoracion_resumen.json         umbrales, métricas, número de segmentos
    reports/segmentos_perfil.csv            cómo es cada segmento
    reports/valoracion_peores_errores.csv   las viviendas donde el modelo más se equivoca (diagnóstico)
    reports/figures/valor_diferencias.png   distribución de (precio publicado / precio justo)
    reports/figures/segmentos.png           elección del número de segmentos y mapa precio-área

1) PRECIO JUSTO SIN TRAMPA: VALIDACIÓN CRUZADA (out-of-fold)
    Si le pedimos al modelo el precio de una vivienda que vio al entrenar, casi acierta porque
    la memorizó, y ninguna saldría "ganga" ni "cara". Por eso:
      - Se reparten TODAS las viviendas en 5 partes (respetando los grupos de duplicados).
      - 5 veces: se entrena con 4 partes y se predice la parte que quedó por fuera.
      - Así cada vivienda recibe un precio justo de un modelo que NUNCA la vio.
    Precio justo = promedio de XGBoost y la red neuronal (el mejor modelo en prueba).
    Dentro de cada entrenamiento, un 15 % se aparta para el early stopping de ambos modelos.

2) CLASIFICACIÓN (el umbral lo fijan los datos)
    diferencia = log(precio publicado) - log(precio justo)
    sigma = error típico del modelo (desviación robusta con la MAD: no la arrastran los extremos)
        tipo con menos de MIN_POR_TIPO anuncios -> "sin_valoracion" (el modelo casi no lo vio)
        |diferencia| >= 3 sigma                 -> "atipico" (probable error o algo que el anuncio no dice)
        diferencia <= -sigma                    -> "ganga"
        diferencia >= +sigma                    -> "caro"
        en medio                                -> "justo"   (dentro del margen de error del modelo)

3) SEGMENTACIÓN: K-means
    Agrupa viviendas parecidas en precio, tamaño, estrato, distancia a universidad/Metro y seguridad.
    La SILUETA (1 = grupos muy separados, 0 = se solapan) dio 0.18-0.22 para TODOS los k: el mercado
    es un continuo sin grupos naturales marcados (hallazgo). Como la separación casi no cambia con k,
    se fija k por utilidad para el estudiante (K_FIJO) y se reportan las siluetas.

Ejecutar:  python -u -m src.models.valoracion
"""

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402
from sklearn.model_selection import GroupKFold, GroupShuffleSplit  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.models.comun import FIG, RAIZ, REPORTES, cargar, columnas_entrada, imprimir_metricas, metricas  # noqa: E402

PROCESSED = RAIZ / "data" / "processed"
K_FOLDS = 5
SEMILLA = 42
K_CANDIDATOS = range(3, 10)
K_FIJO = 5             # silueta plana (0.18-0.22) -> se elige por utilidad; None = el de mayor silueta
MIN_POR_TIPO = 100     # con menos anuncios de un tipo, el modelo no puede valorarlo (p. ej. Habitación: 32)
K_ATIPICO = 3          # diferencias de más de 3 errores típicos: probable error o dato oculto -> verificar

# Variables de la segmentación (transformadas para que las distancias tengan sentido)
VARS_SEGMENTO = {
    "log_precio":        lambda d: d["log_precio"],
    "log_area":          lambda d: np.log(d["area_m2"]),
    "habitaciones":      lambda d: d["habitaciones"],
    "estrato":           lambda d: d["estrato"],
    "log_dist_univ":     lambda d: np.log1p(d["dist_universidad_min_km"]),
    "log_dist_metro":    lambda d: np.log1p(d["dist_metro_km"]),
    "indice_seguridad":  lambda d: d["indice_seguridad"],
}


# --------------------------------------------------------------------------------------------
# 1) Precio justo out-of-fold
# --------------------------------------------------------------------------------------------
def precio_justo_oof(todo: pd.DataFrame, meta: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch
    import xgboost as xgb

    from src.models.red_experimentos import entrenar
    from src.models.red_precio import DROPOUT, INIT_EMBEDDING, TOPE_EMBEDDING, WEIGHT_DECAY, a_tensores, predecir
    from src.models.xgboost_precio import PARAMETROS, preparar_X

    cfg_red = dict(tope=TOPE_EMBEDDING, init=INIT_EMBEDDING, dropout=DROPOUT, wd=WEIGHT_DECAY)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num, cat = columnas_entrada(meta)
    card = [len(meta["vocabulario"][c.replace("_idx", "")]) for c in cat]

    oof_xgb, oof_red = np.full(len(todo), np.nan), np.full(len(todo), np.nan)
    particion = np.zeros(len(todo), dtype=int)
    gkf = GroupKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEMILLA)
    for f, (i_ent, i_fuera) in enumerate(gkf.split(todo, groups=todo["grupo"]), start=1):
        t0 = time.time()
        particion[i_fuera] = f
        ent, fuera = todo.iloc[i_ent], todo.iloc[i_fuera]
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEMILLA)
                    .split(ent, groups=ent["grupo"]))
        ajuste, parada = ent.iloc[a], ent.iloc[b]

        # XGBoost
        modelo = xgb.XGBRegressor(**PARAMETROS)
        modelo.fit(preparar_X(ajuste, num, cat, meta), ajuste["log_precio"],
                   eval_set=[(preparar_X(parada, num, cat, meta), parada["log_precio"])], verbose=False)
        oof_xgb[i_fuera] = modelo.predict(preparar_X(fuera, num, cat, meta))

        # Red neuronal (misma configuración final, objetivo estandarizado con la media de ESTE ajuste)
        y_media, y_desv = float(ajuste["log_precio"].mean()), float(ajuste["log_precio"].std())
        y_aj = torch.tensor(((ajuste["log_precio"] - y_media) / y_desv).to_numpy(), dtype=torch.float32,
                            device=dispositivo)
        datos = {"train": (*a_tensores(ajuste, num, cat, meta, dispositivo), y_aj),
                 "val": (*a_tensores(parada, num, cat, meta, dispositivo), parada["log_precio"].to_numpy())}
        red, _ = entrenar(cfg_red, SEMILLA, datos, len(num), card, y_media, y_desv)
        oof_red[i_fuera] = predecir(red, *a_tensores(fuera, num, cat, meta, dispositivo), y_media, y_desv)

        m = metricas(fuera["log_precio"], (oof_xgb[i_fuera] + oof_red[i_fuera]) / 2)
        print(f"  Partición {f}/{K_FOLDS}: {len(fuera):>5} viviendas | grupo más grande {fuera['grupo'].value_counts().iat[0]:>4} | "
              f"XGBoost {modelo.best_iteration} árboles | ensemble MdAPE {m['MdAPE_%']:.1f}% "
              f"R2 {m['R2_log']:.3f} RMSE {m['RMSE_log']:.3f} | {time.time() - t0:.0f} s")
    return oof_xgb, oof_red, particion


# --------------------------------------------------------------------------------------------
# 2) Clasificación
# --------------------------------------------------------------------------------------------
def clasificar(log_precio, log_justo, tipo) -> tuple[np.ndarray, float]:
    d = np.asarray(log_precio) - np.asarray(log_justo)
    tipo = pd.Series(np.asarray(tipo))
    valorable = tipo.map(tipo.value_counts()).to_numpy() >= MIN_POR_TIPO
    dv = d[valorable]                                               # sigma solo con los valorables
    sigma = float(1.4826 * np.median(np.abs(dv - np.median(dv))))  # 1.4826: MAD -> desv. estándar
    clase = np.select(
        [~valorable, np.abs(d) >= K_ATIPICO * sigma, d <= -sigma, d >= sigma],
        ["sin_valoracion", "atipico", "ganga", "caro"], "justo")
    return clase, sigma


# --------------------------------------------------------------------------------------------
# 3) Segmentación
# --------------------------------------------------------------------------------------------
def segmentar(df: pd.DataFrame) -> tuple[np.ndarray, int, dict]:
    X = pd.DataFrame({k: f(df) for k, f in VARS_SEGMENTO.items()})
    X = X.fillna(X.median())
    Xs = StandardScaler().fit_transform(X)        # misma escala: ninguna variable domina por sus unidades
    siluetas = {}
    for k in K_CANDIDATOS:
        etiquetas = KMeans(n_clusters=k, n_init=10, random_state=SEMILLA).fit_predict(Xs)
        siluetas[k] = float(silhouette_score(Xs, etiquetas, sample_size=min(8000, len(Xs)), random_state=SEMILLA))
        print(f"  k={k}: silueta {siluetas[k]:.3f}")
    k = K_FIJO or max(siluetas, key=siluetas.get)
    etiquetas = KMeans(n_clusters=k, n_init=10, random_state=SEMILLA).fit_predict(Xs)
    return etiquetas, k, siluetas


def perfil_segmentos(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("segmento")
    perfil = pd.DataFrame({
        "viviendas": g.size(),
        "pct": g.size() / len(df) * 100,
        "precio_mediano": g["precio"].median(),
        "area_mediana": g["area_m2"].median(),
        "habitaciones": g["habitaciones"].median(),
        "estrato": g["estrato"].median(),
        "km_universidad": g["dist_universidad_min_km"].median(),
        "km_metro": g["dist_metro_km"].median(),
        "indice_seguridad": g["indice_seguridad"].median(),
        "tipo_principal": g["tipo_inmueble"].agg(lambda s: s.mode().iat[0]),
        "municipio_principal": g["municipio"].agg(lambda s: s.mode().iat[0]),
        "pct_ganga": g["clase_precio"].agg(lambda s: (s == "ganga").mean() * 100),
    })
    # Nombre automático y descriptivo (se puede ajustar a mano después de revisarlo)
    p33, p66 = df["precio"].quantile([1 / 3, 2 / 3])
    def nombre(fila):
        precio = "económico" if fila.precio_mediano <= p33 else "medio" if fila.precio_mediano <= p66 else "alto"
        tam = "pequeño" if fila.area_mediana < 50 else "mediano" if fila.area_mediana < 90 else "amplio"
        extras = []
        if fila.km_universidad < 1:
            extras.append("cerca de universidad")
        if fila.indice_seguridad < 30:
            extras.append("seguridad baja")
        elif fila.indice_seguridad >= 65:
            extras.append("seguridad alta")
        return f"{fila.tipo_principal} {tam}, precio {precio}" + (", " + ", ".join(extras) if extras else "")
    perfil.insert(0, "nombre", [f"S{i}: {nombre(f)}" for i, f in perfil.iterrows()])
    return perfil.sort_values("precio_mediano")


# --------------------------------------------------------------------------------------------
def main() -> None:
    from src.models.preparar_datos import grupos_duplicados

    train, val, test, meta = cargar()
    todo = pd.concat([train, val, test], ignore_index=True)
    viv = pd.read_csv(PROCESSED / meta["fuente"], dtype={"id": str, "comuna_codigo": str},
                      encoding="utf-8-sig", low_memory=False)
    fecha = meta["fuente"].replace("viviendas_", "").replace(".csv", "")
    # Grupos de duplicados con los datos ORIGINALES (sin imputar), igual que en preparar_datos
    viv_coord = viv[viv["lat"].notna()]
    todo["grupo"] = todo["id"].map(pd.Series(grupos_duplicados(viv_coord).to_numpy(), index=viv_coord["id"]))
    assert todo["grupo"].notna().all(), "hay viviendas sin grupo de duplicados"
    print(f"Viviendas: {len(todo):,} | grupos de duplicados: {todo['grupo'].nunique():,}")

    print(f"\n1) Precio justo por validación cruzada ({K_FOLDS} particiones):")
    oof_xgb, oof_red, particion = precio_justo_oof(todo, meta)
    oof = (oof_xgb + oof_red) / 2
    print("\n   Error out-of-fold sobre TODAS las viviendas (MdAPE debe parecerse al de prueba, ~10.5%):")
    m_oof = {n: metricas(todo["log_precio"], p) for n, p in [("xgboost", oof_xgb), ("red", oof_red), ("ensemble", oof)]}
    for n, m in m_oof.items():
        imprimir_metricas(n, m)

    # Diagnóstico: ¿el RMSE/R2 lo arrastran pocos anuncios extremos?
    err = np.abs(todo["log_precio"].to_numpy() - oof)
    peores = np.argsort(-err)[:50]
    sin_peores = np.setdiff1d(np.arange(len(todo)), peores)
    m_sin = metricas(todo["log_precio"].iloc[sin_peores], oof[sin_peores])
    print(f"   Sin los 50 peores errores (0.16 % de los datos): R2 {m_sin['R2_log']:.3f} | RMSE {m_sin['RMSE_log']:.3f}")
    print("   Los 10 peores errores:")
    diag = (todo.iloc[peores][["id", "precio", "area_m2", "tipo_inmueble", "municipio", "grupo"]]
            .assign(precio_justo=np.exp(oof[peores]).round(-3), veces=np.exp(err[peores]).round(1),
                    particion=particion[peores], tam_grupo=lambda d: d["grupo"].map(todo["grupo"].value_counts())))
    print(diag.head(10).to_string(index=False))
    diag.merge(viv[["id", "url", "barrio"]], on="id", how="left").to_csv(
        REPORTES / "valoracion_peores_errores.csv", index=False, encoding="utf-8-sig")

    print("\n2) Clasificación")
    clase, sigma = clasificar(todo["log_precio"], oof, todo["tipo_inmueble"])
    res = pd.DataFrame({
        "id": todo["id"], "precio": todo["precio"],
        "precio_justo": np.exp(oof).round(-3), "precio_justo_xgb": np.exp(oof_xgb).round(-3),
        "precio_justo_red": np.exp(oof_red).round(-3),
        "diferencia_pct": (np.exp(todo["log_precio"] - oof) - 1) * 100,
        "clase_precio": clase,
    })
    print(f"   Error típico del modelo (sigma robusta): {sigma:.3f} en log  ->  "
          f"ganga si el precio está {abs(np.exp(-sigma) - 1) * 100:.0f} % o más por DEBAJO del justo; "
          f"caro si está {(np.exp(sigma) - 1) * 100:.0f} % o más por ENCIMA")
    print(f"   Atípico (verificar) si se aleja más de {K_ATIPICO} sigmas: por debajo de "
          f"{(np.exp(-K_ATIPICO * sigma) - 1) * 100:.0f} % o por encima de +{(np.exp(K_ATIPICO * sigma) - 1) * 100:.0f} %")
    print("   Reparto:", (res["clase_precio"].value_counts(normalize=True) * 100).round(1).to_dict())

    # Variables descriptivas (sin imputar) para segmentar y revisar
    cols = ["id", "municipio", "barrio", "tipo_inmueble", "area_m2", "habitaciones", "estrato", "url",
            "dist_universidad_min_km", "universidad_cercana", "dist_metro_km", "indice_seguridad"]
    det = res.merge(viv[cols], on="id", how="left").assign(log_precio=todo["log_precio"].to_numpy())
    print("\n   % por clase y tipo de inmueble:")
    print((pd.crosstab(det["tipo_inmueble"], det["clase_precio"], normalize="index") * 100).round(1).to_string())
    print("\n   % por clase y municipio:")
    print((pd.crosstab(det["municipio"], det["clase_precio"], normalize="index") * 100).round(1).to_string())

    print("\n3) Segmentación (K-means):")
    seg_df = det.copy()
    for c in ["area_m2", "habitaciones", "estrato"]:     # faltantes -> valor ya imputado de la matriz del modelo
        seg_df[c] = seg_df[c].fillna(todo[c])
    etiquetas, k, siluetas = segmentar(seg_df)
    det["segmento"] = etiquetas
    perfil = perfil_segmentos(det)
    det["segmento_nombre"] = det["segmento"].map(perfil["nombre"])
    print(f"\n   k = {k} ({'fijado por utilidad' if K_FIJO else 'mayor silueta'}). Perfil de cada segmento (medianas):")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(perfil.round(1).to_string())

    print("\n4) Revisión manual: gangas para un estudiante (hasta $2.5M, sin atípicas). Ábrelas en el navegador:")
    top = det[(det["clase_precio"] == "ganga") & (det["precio"] <= 2_500_000)].nsmallest(10, "diferencia_pct")
    for _, f in top.iterrows():
        print(f"   {f.diferencia_pct:+.0f}% | publicado ${f.precio:,.0f} vs justo ${f.precio_justo:,.0f} | "
              f"{f.tipo_inmueble}, {f.area_m2} m², {f.municipio} - {f.barrio}\n      {f.url}")

    # --- Guardar ----------------------------------------------------------------------------------
    salida = PROCESSED / f"valoracion_{fecha}.csv"
    det[["id", "precio", "precio_justo", "precio_justo_xgb", "precio_justo_red", "diferencia_pct",
         "clase_precio", "segmento", "segmento_nombre"]].round({"diferencia_pct": 1}).to_csv(
        salida, index=False, encoding="utf-8-sig")
    perfil.round(2).to_csv(REPORTES / "segmentos_perfil.csv", encoding="utf-8-sig")
    (REPORTES / "valoracion_resumen.json").write_text(json.dumps({
        "fuente": meta["fuente"], "k_folds": K_FOLDS, "metricas_oof": m_oof, "metricas_oof_sin_50_peores": m_sin,
        "sigma_log": sigma, "umbral_ganga_pct": (np.exp(-sigma) - 1) * 100, "umbral_caro_pct": (np.exp(sigma) - 1) * 100,
        "umbral_atipico_pct": [(np.exp(-K_ATIPICO * sigma) - 1) * 100, (np.exp(K_ATIPICO * sigma) - 1) * 100],
        "reparto_clases_pct": (res["clase_precio"].value_counts(normalize=True) * 100).to_dict(),
        "k_segmentos": k, "k_fijo": K_FIJO, "siluetas": siluetas, "variables_segmento": list(VARS_SEGMENTO),
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    # --- Gráficas -----------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(det["diferencia_pct"].clip(-70, 150), bins=110, color="tab:gray")
    for v, t in [((np.exp(-K_ATIPICO * sigma) - 1) * 100, "atípico"), ((np.exp(-sigma) - 1) * 100, "ganga"),
                 ((np.exp(sigma) - 1) * 100, "caro"), ((np.exp(K_ATIPICO * sigma) - 1) * 100, "atípico")]:
        ax.axvline(v, color="tab:red" if t != "atípico" else "black", ls="--")
        ax.text(v, ax.get_ylim()[1] * 0.92, f" {t}\n ({v:+.0f} %)", fontsize=8,
                color="tab:red" if t != "atípico" else "black")
    ax.set_xlabel("Precio publicado frente al precio justo (%)"); ax.set_ylabel("Viviendas")
    ax.set_title("¿Cuánto se aleja cada anuncio de su precio justo? (validación cruzada)")
    fig.savefig(FIG / "valor_diferencias.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1, 2]})
    a1.plot(list(siluetas), list(siluetas.values()), marker="o")
    a1.axvline(k, color="gray", ls="--"); a1.set_xlabel("Número de segmentos (k)"); a1.set_ylabel("Silueta")
    a1.set_ylim(0, max(0.4, max(siluetas.values()) * 1.2)); a1.set_title("Elección de k")
    muestra = det.sample(min(8000, len(det)), random_state=SEMILLA)
    for s in perfil.index:
        g = muestra[muestra["segmento"] == s]
        a2.scatter(g["area_m2"], g["precio"], s=4, alpha=0.5, label=perfil.loc[s, "nombre"][:60])
    a2.set_xscale("log"); a2.set_yscale("log"); a2.set_xlabel("Área (m², escala log)")
    a2.set_ylabel("Precio (COP, escala log)"); a2.set_title("Segmentos del mercado")
    a2.legend(markerscale=4, fontsize=7, loc="upper left")
    fig.savefig(FIG / "segmentos.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"\nGuardado: {salida.name}, reports/segmentos_perfil.csv, reports/valoracion_peores_errores.csv, "
          f"reports/valoracion_resumen.json, gráficas en {FIG}")


if __name__ == "__main__":
    main()
