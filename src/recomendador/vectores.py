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


def zscore(M: np.ndarray) -> np.ndarray:
    """Centra y escala cada columna; quita las columnas constantes."""
    M = M[:, M.std(axis=0) > 1e-6]
    return (M - M.mean(axis=0)) / M.std(axis=0)


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
    activas = V.std(axis=0) > 1e-6
    print(f"Vectores: {V.shape[0]:,} viviendas x {V.shape[1]} dimensiones ({int(activas.sum())} activas)")
    # ¿Por qué estandarizar? Tras la ReLU todos los vectores son positivos y apuntan casi en la misma
    # dirección (coseno ~1.0 entre casi cualquier par): la similitud coseno "cruda" casi no distingue.
    # Restar la media y dividir por la desviación de cada dimensión (z-score) centra la nube y hace
    # que el coseno mida en qué se DIFERENCIA cada vivienda del promedio.
    X_crudo = x_num.cpu().numpy()                                    # las mismas entradas, sin la red
    variantes = {
        "red sin estandarizar": V,
        "red estandarizada": zscore(V[:, activas]),
        # Entradas tal como las recibe la red: continuas ya estandarizadas con el train y binarias 0/1.
        # (z-score también a las binarias infla el peso de amenidades raras: se probó y empeora)
        "crudo (variables sin red)": X_crudo[:, X_crudo.std(axis=0) > 1e-6],
    }
    Zr, Zc = variantes["red estandarizada"], variantes["crudo (variables sin red)"]
    # Híbrido: cada bloque pesa lo mismo en el coseno (se divide por la raíz de su número de columnas)
    variantes["híbrido red + crudo"] = np.hstack([Zr / np.sqrt(Zr.shape[1]), Zc / np.sqrt(Zc.shape[1])])
    rng0 = np.random.default_rng(SEMILLA)
    par = rng0.choice(len(V), (2000, 2))
    cos = (normalizar(V)[par[:, 0]] * normalizar(V)[par[:, 1]]).sum(axis=1)
    print(f"Coseno entre pares AL AZAR con el vector sin estandarizar: mediana {np.median(cos):.3f} "
          "(cerca de 1 = el coseno casi no distingue)")

    # --- Evaluación --------------------------------------------------------------------------
    viv = pd.read_csv(PROCESSED / meta["fuente"], dtype={"id": str}, encoding="utf-8-sig", low_memory=False)
    viv = viv[viv["lat"].notna()]
    viv["grupo"] = grupos_duplicados(viv)
    val_csv = sorted(PROCESSED.glob("valoracion_*.csv"))[-1]
    seg = pd.read_csv(val_csv, dtype={"id": str}, encoding="utf-8-sig")[["id", "segmento"]]
    info = (todo[["id", "precio", "tipo_inmueble"]]
            .merge(viv[["id", "lat", "lon", "grupo"]], on="id", how="left")
            .merge(seg, on="id", how="left"))
    rng = np.random.default_rng(SEMILLA)
    muestra = rng.choice(len(info), N_MUESTRA, replace=False)
    filas = [evaluar(n, M, info, muestra, rng) for n, M in variantes.items()]
    filas.append(evaluar("azar", None, info, muestra, rng))
    res = pd.DataFrame(filas).set_index("metodo")
    print(f"\n¿Las {K_VECINAS} vecinas más parecidas se parecen de verdad? (promedio sobre {N_MUESTRA} viviendas)")
    print(res.round(1).to_string())

    # Elección: el mejor rango promedio en las 4 métricas (sin contar el azar)
    r = res.drop(index="azar")
    rangos = pd.concat([r[["mismo_segmento_%", "mismo_tipo_%"]].rank(ascending=False),
                        r[["dif_precio_mediana_%", "distancia_mediana_km"]].rank(ascending=True)], axis=1)
    res["rango_promedio"] = rangos.mean(axis=1)
    elegido = res["rango_promedio"].idxmin()
    print(f"\nRepresentación elegida para 'más como esta' (mejor rango promedio): {elegido}")
    np.savez_compressed(MODELOS / "vectores_red.npz", ids=todo["id"].to_numpy(),
                        V=variantes[elegido].astype("float32"), metodo=elegido)
    res.to_csv(REPORTES / "evaluacion_vectores.csv", encoding="utf-8-sig")
    print("Guardado: models/vectores_red.npz, reports/evaluacion_vectores.csv")

if __name__ == "__main__":
    main()
