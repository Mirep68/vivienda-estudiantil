"""
Paso 3 del modelo: RED NEURONAL (MLP con embeddings) para el precio justo, en PyTorch.

    models/red_precio.pt                       pesos de la mejor época
    models/red_precio_config.json              arquitectura (la API la necesita para reconstruir la red)
    reports/metricas_red.json                  métricas en prueba: red, XGBoost y ensemble
    reports/historial_red.csv                  pérdida y error por época
    reports/predicciones_test_red.csv
    reports/figures/red_curva_aprendizaje.png  convergencia (pérdida y RMSE por época)
    reports/figures/red_embeddings_barrios.png los barrios según el vector que aprendió la red

ARQUITECTURA
    categóricas (barrio, comuna, municipio, tipo, estado) -> EMBEDDINGS (un vector aprendido por categoría)
    numéricas estandarizadas + binarias  ─┐
    embeddings concatenados              ─┴-> Linear(256) -> ReLU -> Dropout
                                              -> Linear(128) -> ReLU -> Dropout
                                              -> Linear(64)  -> ReLU -> Linear(1) = log(precio) estandarizado

ENTRENAMIENTO
    - Pérdida Huber (robusta a anuncios mal digitados), optimizador AdamW.
    - Mini-lotes de 512 viviendas; una ÉPOCA = una pasada por todo el train.
    - La tasa de aprendizaje se reduce a la mitad si la validación se estanca 8 épocas.
    - Early stopping: si la validación no mejora en 25 épocas, se detiene y se usa la mejor.

Ejecutar:  python -u -m src.models.red_precio
"""

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from src.models.comun import (FIG, MODELOS, REPORTES, cargar, columnas_entrada, imprimir_metricas,  # noqa: E402
                              linea_base, metricas)

SEMILLA = 42
CAPAS = [256, 128, 64]
# Hiperparámetros elegidos con red_experimentos.py (config. D: ganadora por validación, 3 semillas)
DROPOUT = 0.3
TOPE_EMBEDDING = 16      # dimensión máxima de cada embedding
INIT_EMBEDDING = 0.05    # desviación inicial de los embeddings (PyTorch usa 1.0 -> vectores al azar)
LOTE = 512
LR = 1e-3
WEIGHT_DECAY = 1e-3
MAX_EPOCAS = 400
PACIENCIA = 25


def dim_embedding(n_categorias: int, tope: int = 50) -> int:
    """Regla práctica (fast.ai): más categorías -> vector más largo, con un tope."""
    return int(min(tope, round(1.6 * n_categorias ** 0.56)))


class RedPrecio(nn.Module):
    def __init__(self, n_numericas: int, cardinalidades: list[int], capas=CAPAS, dropout=DROPOUT,
                 tope_embedding: int = TOPE_EMBEDDING, init_embedding: float | None = INIT_EMBEDDING):
        super().__init__()
        # Un embedding por cada variable categórica: tabla de (n_categorías x dim) números aprendibles
        self.embeddings = nn.ModuleList([nn.Embedding(n, dim_embedding(n, tope_embedding)) for n in cardinalidades])
        # PyTorch inicia los embeddings con N(0, 1): vectores grandes y AL AZAR. Con init_embedding
        # (p. ej. 0.05) arrancan pequeños y lo que terminen codificando lo aprendieron de los datos.
        if init_embedding is not None:
            for e in self.embeddings:
                nn.init.normal_(e.weight, mean=0.0, std=init_embedding)
        entrada = n_numericas + sum(e.embedding_dim for e in self.embeddings)
        bloques = []
        for salida in capas:
            bloques += [nn.Linear(entrada, salida), nn.ReLU(), nn.Dropout(dropout)]
            entrada = salida
        bloques.append(nn.Linear(entrada, 1))
        self.mlp = nn.Sequential(*bloques)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        vectores = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat([x_num] + vectores, dim=1)
        return self.mlp(x).squeeze(1)


def a_tensores(df: pd.DataFrame, num: list[str], cat: list[str], meta: dict, dispositivo):
    X = df[num].astype("float32").copy()
    for c, p in meta["estandarizacion"].items():          # solo las numéricas continuas
        X[c] = (X[c] - p["media"]) / p["desv"]
    x_num = torch.tensor(X.to_numpy(), dtype=torch.float32, device=dispositivo)
    x_cat = torch.tensor(df[cat].to_numpy(), dtype=torch.long, device=dispositivo)
    return x_num, x_cat


@torch.no_grad()
def predecir(modelo, x_num, x_cat, y_media, y_desv) -> np.ndarray:
    modelo.eval()                                           # apaga el dropout
    return (modelo(x_num, x_cat) * y_desv + y_media).cpu().numpy()


def main() -> None:
    torch.manual_seed(SEMILLA)
    np.random.seed(SEMILLA)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {dispositivo}" + (f" ({torch.cuda.get_device_name(0)})" if dispositivo.type == "cuda" else ""))

    train, val, test, meta = cargar()
    num, cat = columnas_entrada(meta)
    cardinalidades = [len(meta["vocabulario"][c.replace("_idx", "")]) for c in cat]

    # Objetivo estandarizado (media 0, desv 1): la red entrena más estable
    y_media, y_desv = float(train["log_precio"].mean()), float(train["log_precio"].std())
    datos = {}
    for nombre, df in [("train", train), ("val", val), ("test", test)]:
        x_num, x_cat = a_tensores(df, num, cat, meta, dispositivo)
        y = torch.tensor(((df["log_precio"] - y_media) / y_desv).to_numpy(), dtype=torch.float32, device=dispositivo)
        datos[nombre] = (x_num, x_cat, y)

    modelo = RedPrecio(len(num), cardinalidades).to(dispositivo)
    n_param = sum(p.numel() for p in modelo.parameters())
    print(f"Embeddings: {dict(zip(meta['categoricas'], [e.embedding_dim for e in modelo.embeddings]))}")
    print(f"Parámetros entrenables: {n_param:,}")

    optimizador = torch.optim.AdamW(modelo.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    planificador = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizador, mode="min", factor=0.5, patience=8)
    perdida_fn = nn.HuberLoss(delta=1.0)

    x_tr, c_tr, y_tr = datos["train"]
    n = len(y_tr)
    historial, mejor, mejor_estado, sin_mejora = [], np.inf, None, 0
    t0 = time.time()
    for epoca in range(1, MAX_EPOCAS + 1):
        modelo.train()
        orden = torch.randperm(n, device=dispositivo)        # barajar en cada época
        suma = 0.0
        for i in range(0, n, LOTE):
            lote = orden[i:i + LOTE]
            optimizador.zero_grad()
            perdida = perdida_fn(modelo(x_tr[lote], c_tr[lote]), y_tr[lote])
            perdida.backward()                               # retropropagación: calcula los gradientes
            optimizador.step()                               # actualiza los pesos
            suma += perdida.item() * len(lote)

        pred_tr = predecir(modelo, x_tr, c_tr, y_media, y_desv)
        pred_va = predecir(modelo, *datos["val"][:2], y_media, y_desv)
        rmse_tr = metricas(train["log_precio"], pred_tr)["RMSE_log"]
        rmse_va = metricas(val["log_precio"], pred_va)["RMSE_log"]
        lr = optimizador.param_groups[0]["lr"]
        historial.append({"epoca": epoca, "perdida_huber": suma / n, "rmse_train": rmse_tr, "rmse_val": rmse_va, "lr": lr})
        planificador.step(rmse_va)

        if rmse_va < mejor - 1e-4:
            mejor, sin_mejora = rmse_va, 0
            mejor_estado = {k: v.detach().clone() for k, v in modelo.state_dict().items()}
            marca = " *"
        else:
            sin_mejora += 1
            marca = ""
        if epoca == 1 or epoca % 10 == 0 or marca:
            print(f"Época {epoca:>3} | pérdida {suma / n:.4f} | RMSE train {rmse_tr:.4f} | RMSE val {rmse_va:.4f} | lr {lr:.1e}{marca}")
        if sin_mejora >= PACIENCIA:
            print(f"Early stopping en la época {epoca} (sin mejora en {PACIENCIA} épocas)")
            break
    print(f"Entrenado en {time.time() - t0:.0f} s | mejor RMSE val: {mejor:.4f}")
    modelo.load_state_dict(mejor_estado)

    # --- Evaluación en PRUEBA ------------------------------------------------------------
    pred_te = predecir(modelo, *datos["test"][:2], y_media, y_desv)
    m_red = metricas(test["log_precio"], pred_te)
    m_base = metricas(test["log_precio"], linea_base(train, test))
    resultados = {"linea_base": m_base, "red_neuronal": m_red}
    xgb_path = REPORTES / "predicciones_test_xgboost.csv"
    if xgb_path.exists():
        pxgb = pd.read_csv(xgb_path, dtype={"id": str}).set_index("id")["pred_log"].reindex(test["id"]).to_numpy()
        resultados["xgboost"] = metricas(test["log_precio"], pxgb)
        resultados["ensemble_50_50"] = metricas(test["log_precio"], (pxgb + pred_te) / 2)
    print("\nResultados en el conjunto de PRUEBA:")
    for k, m in resultados.items():
        imprimir_metricas(k, m)

    # --- Guardar ----------------------------------------------------------------------------
    MODELOS.mkdir(exist_ok=True)
    torch.save(modelo.state_dict(), MODELOS / "red_precio.pt")
    (MODELOS / "red_precio_config.json").write_text(json.dumps({
        "n_numericas": len(num), "cardinalidades": cardinalidades, "capas": CAPAS, "dropout": DROPOUT,
        "tope_embedding": TOPE_EMBEDDING, "init_embedding": INIT_EMBEDDING,
        "columnas_numericas": num, "columnas_categoricas": cat, "y_media": y_media, "y_desv": y_desv,
    }, indent=1), encoding="utf-8")
    (REPORTES / "metricas_red.json").write_text(json.dumps(
        {**resultados, "epocas": len(historial), "parametros": n_param,
         "hiperparametros": {"capas": CAPAS, "dropout": DROPOUT, "tope_embedding": TOPE_EMBEDDING,
                             "init_embedding": INIT_EMBEDDING, "lote": LOTE, "lr": LR,
                             "weight_decay": WEIGHT_DECAY, "paciencia": PACIENCIA}}, indent=1), encoding="utf-8")
    hist = pd.DataFrame(historial)
    hist.to_csv(REPORTES / "historial_red.csv", index=False)
    test.assign(pred_log=pred_te)[["id", "precio", "log_precio", "pred_log", "municipio", "tipo_inmueble"]].to_csv(
        REPORTES / "predicciones_test_red.csv", index=False, encoding="utf-8-sig")

    # --- Gráfica de convergencia -------------------------------------------------------------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.plot(hist["epoca"], hist["perdida_huber"], color="tab:purple")
    a1.set_title("Pérdida Huber (entrenamiento)"); a1.set_xlabel("Época"); a1.set_yscale("log")
    a2.plot(hist["epoca"], hist["rmse_train"], label="Entrenamiento")
    a2.plot(hist["epoca"], hist["rmse_val"], label="Validación")
    mejor_ep = int(hist.loc[hist["rmse_val"].idxmin(), "epoca"])
    a2.axvline(mejor_ep, color="gray", ls="--", label=f"Mejor época ({mejor_ep})")
    if "xgboost" in resultados:
        a2.axhline(resultados["xgboost"]["RMSE_log"], color="tab:red", ls=":", label="XGBoost (test)")
    a2.set_title("RMSE de log(precio)"); a2.set_xlabel("Época"); a2.legend()
    fig.suptitle("Red neuronal: convergencia del entrenamiento")
    fig.savefig(FIG / "red_curva_aprendizaje.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # --- Embeddings de barrios en 2D (PCA) -------------------------------------------------------
    i_barrio = meta["categoricas"].index("barrio_modelo")
    E = modelo.embeddings[i_barrio].weight.detach().cpu().numpy()
    vocab = meta["vocabulario"]["barrio_modelo"]
    Ec = E - E.mean(axis=0)
    _, _, Vt = np.linalg.svd(Ec, full_matrices=False)       # PCA: las 2 direcciones de mayor variación
    xy = Ec @ Vt[:2].T
    precio_barrio = train.groupby("barrio_modelo_idx")["precio"].median()
    color = precio_barrio.reindex(range(len(vocab))).to_numpy()
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=np.log10(color), cmap="viridis", s=18)
    for j in np.argsort(-np.nan_to_num(color))[:12]:          # etiquetar los 12 más caros
        ax.annotate(vocab[j][:22], xy[j], fontsize=7)
    for j in np.argsort(np.nan_to_num(color, nan=np.inf))[:8]:  # y los 8 más baratos
        ax.annotate(vocab[j][:22], xy[j], fontsize=7)
    plt.colorbar(sc, label="log10(precio mediano del barrio)")
    ax.set_title("Embeddings de barrios aprendidos por la red (PCA 2D)")
    ax.set_xlabel("Componente 1"); ax.set_ylabel("Componente 2")
    fig.savefig(FIG / "red_embeddings_barrios.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    corr = np.corrcoef(xy[~np.isnan(color), 0], np.log(color[~np.isnan(color)]))[0, 1]
    print(f"\nCorrelación entre el componente 1 de los embeddings y el precio del barrio: {corr:.2f}")
    from src.models.red_experimentos import coherencia_vecinos   # import aquí para evitar import circular
    print(f"Coherencia de vecinos (5 barrios más parecidos según la red): {coherencia_vecinos(modelo, meta, train):.2f}")
    print(f"Guardado: models/red_precio.pt, reports/metricas_red.json, gráficas en {FIG}")


if __name__ == "__main__":
    main()
