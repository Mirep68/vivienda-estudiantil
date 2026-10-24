"""
Diagnóstico de anuncios repetidos en el scraping de Finca Raíz.
Solo LEE data/raw/fincaraiz/<fecha>/*.json (no descarga ni modifica nada).

Pregunta: ¿por qué un mismo anuncio aparece varias veces?
  Hipótesis 1 - REORDENAMIENTO: el orden cambió durante la corrida y el anuncio
                "se corrió" de página -> aparece en páginas CERCANAS (p300 y p301).
  Hipótesis 2 - DESTACADOS: anuncios pagados que se repiten en muchas páginas
                -> aparecen MUCHAS veces y en páginas LEJANAS.

Ejecutar:  python -m src.utils.diagnostico_duplicados
"""

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
CARPETA = RAIZ / "data" / "raw" / "fincaraiz"

fecha = sorted(d.name for d in CARPETA.iterdir() if d.is_dir())[-1]
filas = []
for archivo in sorted((CARPETA / fecha).glob("*.json")):
    pag = json.loads(archivo.read_text(encoding="utf-8"))
    for posicion, a in enumerate(pag["data"], start=1):
        filas.append({
            "id": str(a.get("id")),
            "busqueda": f'{pag["municipio_busqueda"]}/{pag["tipo_busqueda"]}/{pag.get("orden", "popularidad")}',
            "pagina": pag["pagina"],
            "posicion": posicion,            # lugar dentro de la página (1 a 21)
            "highlight": a.get("highlight"),
            "turbo": bool((a.get("turbo_product") or {}).get("id")),
        })
df = pd.DataFrame(filas)
print(f"Fecha: {fecha} | apariciones: {len(df):,} | anuncios únicos: {df['id'].nunique():,}\n")

# 1) ¿Cuántas veces aparece cada anuncio?
veces = df.groupby("id").size()
print("1) Cuántas veces aparece cada anuncio (número de anuncios):")
print(veces.value_counts().sort_index().rename("anuncios").to_string(), "\n")

# 2) ¿Los repetidos están en la MISMA búsqueda o en búsquedas distintas?
rep = df[df["id"].isin(veces[veces > 1].index)]
busquedas_por_id = rep.groupby("id")["busqueda"].nunique()
print("2) Repetidos dentro de la misma búsqueda:", int((busquedas_por_id == 1).sum()),
      "| en búsquedas distintas:", int((busquedas_por_id > 1).sum()), "\n")

# 3) Dentro de una misma búsqueda: ¿qué tan lejos están las páginas donde se repite?
distancias = []
for (_, _), g in rep.groupby(["id", "busqueda"]):
    p = sorted(g["pagina"])
    distancias += [b - a for a, b in zip(p, p[1:])]
d = pd.Series(distancias)
print("3) Distancia entre páginas donde se repite (dentro de la misma búsqueda):")
print(f"   0 (misma página): {(d == 0).sum():,} | 1: {(d == 1).sum():,} | 2-5: {d.between(2, 5).sum():,} | >5: {(d > 5).sum():,}\n")

# 4) ¿Los repetidos tienden a ser destacados (turbo) o a estar en las primeras posiciones?
print("4) Comparación repetidos vs no repetidos:")
df["repetido"] = df["id"].isin(veces[veces > 1].index)
print(df.groupby("repetido").agg(
    pct_turbo=("turbo", lambda s: round(s.mean() * 100, 1)),
    posicion_media=("posicion", "mean"),
).round(1).to_string(), "\n")

# 5) Por búsqueda: cuánto se pierde
res = df.groupby("busqueda").agg(apariciones=("id", "size"), unicos=("id", "nunique"))
res["pct_repetido"] = ((1 - res["unicos"] / res["apariciones"]) * 100).round(1)
print("5) Repetición por búsqueda (las 10 con más):")
print(res.sort_values("pct_repetido", ascending=False).head(10).to_string())
