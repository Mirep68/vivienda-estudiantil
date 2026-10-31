"""
Paso 5a del recomendador: el VECTOR DE CONTENIDO de cada vivienda, aprendido por la red neuronal.

    models/vectores_red.npz   ids + matriz (viviendas x 64): la representación de cada vivienda

¿QUÉ ES ESTE VECTOR? (la parte "CBNN" = recomendación basada en contenido con red neuronal)
    La red de precio recibe ~60 variables + los embeddings de barrio, comuna, municipio, tipo y estado,
    y las transforma capa a capa: 256 -> 128 -> 64 -> precio. La última capa oculta (64 números) es
    un RESUMEN que la red aprendió de "lo que hace que una vivienda valga lo que vale": tamaño,
    ubicación, estrato, entorno... Dos viviendas con vectores parecidos (similitud coseno alta)
    son parecidas en lo que importa, aunque difieran en variables sueltas.
    Con eso el recomendador puede hacer "más como esta": el estudiante marca las que le gustan
    y se le muestran las más parecidas.

¿SIRVE DE VERDAD? Evaluación (sin usuarios reales todavía, se evalúa con los propios datos)
    Para 1.000 viviendas al azar se buscan sus 10 vecinas más parecidas con tres métodos:
      red     -> vector aprendido por la red (64 dimensiones)
      crudo   -> las variables de entrada estandarizadas, sin red (lo que haría un recomendador simple)
      azar    -> 10 viviendas cualquiera (piso de comparación)
    y se mide si las vecinas comparten segmento, tipo y rango de precio, y qué tan lejos están.
    Se excluyen los duplicados del mismo inmueble (serían "vecinas" triviales).

Ejecutar:  python -u -m src.recomendador.vectores
"""

import json

import numpy as np
import pandas as pd
import torch

from src.models.comun import MODELOS, RAIZ, REPORTES, cargar, columnas_entrada
from src.models.preparar_datos import grupos_duplicados
from src.models.red_precio import RedPrecio, a_tensores

PROCESSED = RAIZ / "data" / "processed"
N_MUESTRA = 1000
K_VECINAS = 10
SEMILLA = 42


@torch.no_grad()
def representacion(modelo: RedPrecio, x_num: torch.Tensor, x_cat: torch.Tensor) -> np.ndarray:
    """Igual que forward(), pero se detiene en la última capa oculta (antes de la salida de precio)."""
    modelo.eval()
    vectores = [emb(x_cat[:, i]) for i, emb in enumerate(modelo.embeddings)]
    x = torch.cat([x_num] + vectores, dim=1)
    return modelo.mlp[:-1](x).cpu().numpy()          # mlp[:-1] = todas las capas menos Linear(64 -> 1)


def normalizar(M: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(M, axis=1, keepdims=True)
    return M / np.where(n == 0, 1, n)


def km(lat1, lon1, lat2, lon2):
    """Distancia haversine en km (vectorizada)."""
    p = np.pi / 180
    a = (np.sin((lat2 - lat1) * p / 2) ** 2 +
         np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 12742 * np.arcsin(np.sqrt(a))


def evaluar(nombre: str, M: np.ndarray | None, info: pd.DataFrame, muestra: np.ndarray, rng) -> dict:
    Mn = normalizar(M) if M is not None else None
    grupo = info["grupo"].to_numpy()
    seg, tipo = info["segmento"].to_numpy(), info["tipo_inmueble"].to_numpy()
    lp, lat, lon = np.log(info["precio"].to_numpy()), info["lat"].to_numpy(), info["lon"].to_numpy()
    r = {"mismo_segmento_%": [], "mismo_tipo_%": [], "dif_precio_mediana_%": [], "distancia_mediana_km": []}
    for i in muestra:
        validos = grupo != grupo[i]                                  # ni ella misma ni sus duplicados
        if Mn is None:
            vec = rng.choice(np.flatnonzero(validos), K_VECINAS, replace=False)
        else:
            sim = Mn @ Mn[i]
            sim[~validos] = -np.inf
            vec = np.argpartition(-sim, K_VECINAS)[:K_VECINAS]
        r["mismo_segmento_%"].append((seg[vec] == seg[i]).mean() * 100)
        r["mismo_tipo_%"].append((tipo[vec] == tipo[i]).mean() * 100)
        r["dif_precio_mediana_%"].append(np.median(np.abs(np.exp(lp[vec] - lp[i]) - 1)) * 100)
        r["distancia_mediana_km"].append(np.median(km(lat[i], lon[i], lat[vec], lon[vec])))
    return {"metodo": nombre, **{k: float(np.mean(v)) for k, v in r.items()}}


def main() -> None:
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, val, test, meta = cargar()
    todo = pd.concat([train, val, test], ignore_index=True)
    num, cat = columnas_entrada(meta)
    cfg = json.loads((MODELOS / "red_precio_config.json").read_text(encoding="utf-8"))
    modelo = RedPrecio(cfg["n_numericas"], cfg["cardinalidades"], capas=cfg["capas"], dropout=cfg["dropout"],
                       tope_embedding=cfg["tope_embedding"], init_embedding=cfg["init_embedding"]).to(dispositivo)
    modelo.load_state_dict(torch.load(MODELOS / "red_precio.pt", map_location=dispositivo))

    x_num, x_cat = a_tensores(todo, num, cat, meta, dispositivo)
    V = representacion(modelo, x_num, x_cat).astype("float32")
    activas = (V > 0).any(axis=0).sum()
    print(f"Vectores: {V.shape[0]:,} viviendas x {V.shape[1]} dimensiones "
          f"({activas} dimensiones activas; las demás son neuronas ReLU que nunca se encienden)")
    np.savez_compressed(MODELOS / "vectores_red.npz", ids=todo["id"].to_numpy(), V=V)

    # --- Evaluación --------------------------------------------------------------------------
    viv = pd.read_csv(PROCESSED / meta["fuente"], dtype={"id": str}, encoding="utf-8-sig", low_memory=False)
    viv = viv[viv["lat"].notna()]
    viv["grupo"] = grupos_duplicados(viv)
    val_csv = sorted(PROCESSED.glob("valoracion_*.csv"))[-1]
    seg = pd.read_csv(val_csv, dtype={"id": str}, encoding="utf-8-sig")[["id", "segmento"]]
    info = (todo[["id", "precio", "tipo_inmueble"]]
            .merge(viv[["id", "lat", "lon", "grupo"]], on="id", how="left")
            .merge(seg, on="id", how="left"))
    X_crudo = x_num.cpu().numpy()                                    # las mismas entradas, sin la red
    rng = np.random.default_rng(SEMILLA)
    muestra = rng.choice(len(info), N_MUESTRA, replace=False)
    filas = [evaluar("red (vector aprendido)", V, info, muestra, rng),
             evaluar("crudo (variables sin red)", X_crudo, info, muestra, rng),
             evaluar("azar", None, info, muestra, rng)]
    res = pd.DataFrame(filas).set_index("metodo")
    print(f"\n¿Las {K_VECINAS} vecinas más parecidas se parecen de verdad? (promedio sobre {N_MUESTRA} viviendas)")
    print(res.round(1).to_string())
    res.to_csv(REPORTES / "evaluacion_vectores.csv", encoding="utf-8-sig")
    print(f"\nGuardado: models/vectores_red.npz, reports/evaluacion_vectores.csv")


if __name__ == "__main__":
    main()
