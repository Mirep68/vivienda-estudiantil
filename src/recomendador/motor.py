"""
Paso 5b: el MOTOR DE RECOMENDACIÓN (lo que usará la API y la página web).

CÓMO RECOMIENDA (la "matriz de criterios impuestos + deseados" del documento)
  1. CRITERIOS IMPUESTOS (filtros duros): lo que el estudiante NO negocia.
       presupuesto máximo, universidad + distancia máxima, habitaciones mínimas, tipos de inmueble,
       municipios. Una vivienda que no los cumple no aparece, por buena que sea.
  2. CRITERIOS DESEADOS (con pesos 0-5): lo que le gustaría.
       cercania, seguridad, precio, valor (qué tan por debajo de su precio justo está), espacio,
       metro, parques, servicios, amenidades y, si marcó viviendas que le gustan, parecido.
     Cada criterio se convierte en un PERCENTIL dentro de las candidatas (0 = la peor, 1 = la mejor),
     así km, pesos, m² e índices quedan en la misma escala sin inventar factores de conversión.
     puntaje = promedio de percentiles ponderado por los pesos del estudiante (0-100).
  3. EXPLICACIÓN: cada recomendación dice POR QUÉ ("más segura que el 85 % de tus opciones").
  4. SIN REPETIDOS: un mismo inmueble publicado por varias inmobiliarias aparece una sola vez.

  "parecido" usa el vector que aprendió la red (src/recomendador/vectores.py): similitud coseno
  con el promedio de las viviendas que el estudiante marcó. Esa es la parte de deep learning del
  recomendador basado en contenido (CBNN).

Ejecutar el caso de estudio:  python -u -m src.recomendador.motor
"""

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.models.comun import MODELOS, REPORTES

ALTURA_LISTA = 10

# criterio -> (columna, sentido): +1 = más es mejor, -1 = menos es mejor
CRITERIOS = {
    "cercania":   ("dist_univ_km", -1),
    "seguridad":  ("indice_seguridad", +1),
    "precio":     ("precio", -1),
    "valor":      ("diferencia_pct", -1),
    "espacio":    ("area_m2", +1),
    "metro":      ("dist_metro_km", -1),
    "parques":    ("dist_parque_km", -1),
    "servicios":  ("servicios_500m", +1),
    "amenidades": ("n_amenidades", +1),
    "parecido":   ("similitud", +1),
}
FRASES = {
    "cercania": "más cerca de tu universidad que el {p} % de tus opciones",
    "seguridad": "zona más segura que el {p} % de tus opciones",
    "precio": "más barata que el {p} % de tus opciones",
    "valor": "mejor precio frente a su valor justo que el {p} % de tus opciones",
    "espacio": "más amplia que el {p} % de tus opciones",
    "metro": "más cerca del Metro que el {p} % de tus opciones",
    "parques": "más cerca de un parque que el {p} % de tus opciones",
    "servicios": "más supermercados y farmacias cerca que el {p} % de tus opciones",
    "amenidades": "más amenidades que el {p} % de tus opciones",
    "parecido": "se parece a las que te gustaron más que el {p} % de tus opciones",
}


@dataclass
class Perfil:
    universidad: str                                  # sigla, p. ej. "USB"
    presupuesto_max: float
    distancia_max_km: float = 3.0
    habitaciones_min: int = 0
    tipos: list[str] = field(default_factory=lambda: ["Apartaestudio", "Apartamento", "Casa", "Habitación"])
    municipios: list[str] | None = None
    incluir_atipicos: bool = False                    # anuncios "verificar": fuera por defecto
    pesos: dict[str, float] = field(default_factory=lambda: {"cercania": 3, "seguridad": 3, "precio": 3, "valor": 2})
    me_gustan: list[str] = field(default_factory=list)  # ids de viviendas que le gustaron


def columna_universidad(sigla: str) -> str:
    from src.db.conexion import nombre_columna
    return nombre_columna(f"dist_{sigla}_km")


def cargar_datos() -> pd.DataFrame:
    """Viviendas + valoración desde PostgreSQL (la misma fuente que usará la API)."""
    from sqlalchemy import create_engine, text

    from src.db.conexion import url
    motor = create_engine(url())
    q = text("""SELECT v.*, a.precio_justo, a.diferencia_pct, a.clase_precio, a.segmento_nombre
                FROM viviendas v JOIN valoracion a USING (id)""")
    with motor.connect() as c:
        df = pd.read_sql(q, c)
    # Textos largos que la web no usa: fuera, para ahorrar memoria en el servidor
    return df.drop(columns=["geom", "descripcion", "titulo", "direccion"], errors="ignore")


def cargar_vectores() -> tuple[np.ndarray, dict]:
    z = np.load(MODELOS / "vectores_red.npz", allow_pickle=True)
    print(f"Vectores de contenido: {z['V'].shape} | representación: {z['metodo'] if 'metodo' in z else 'red'}")
    V = z["V"] / np.maximum(np.linalg.norm(z["V"], axis=1, keepdims=True), 1e-9)
    return V, {i: k for k, i in enumerate(z["ids"].astype(str))}


def recomendar(df: pd.DataFrame, perfil: Perfil, vectores=None, n: int = ALTURA_LISTA) -> tuple[pd.DataFrame, dict]:
    """Devuelve (recomendaciones, diagnóstico del embudo de filtros)."""
    col_u = columna_universidad(perfil.universidad)
    if col_u not in df.columns:
        raise ValueError(f"Universidad desconocida: {perfil.universidad}")
    d = df.assign(dist_univ_km=df[col_u])
    embudo = {"total": len(d)}

    # 1) Criterios impuestos -------------------------------------------------------------
    d = d[d["precio"] <= perfil.presupuesto_max];                embudo["presupuesto"] = len(d)
    d = d[d["dist_univ_km"] <= perfil.distancia_max_km];          embudo["distancia"] = len(d)
    d = d[d["tipo_inmueble"].isin(perfil.tipos)];                 embudo["tipo"] = len(d)
    if perfil.habitaciones_min:
        d = d[d["habitaciones"].fillna(0) >= perfil.habitaciones_min]
    embudo["habitaciones"] = len(d)
    if perfil.municipios:
        d = d[d["municipio"].isin(perfil.municipios)]
    embudo["municipio"] = len(d)
    if not perfil.incluir_atipicos:
        d = d[d["clase_precio"] != "atipico"]
    embudo["sin_atipicos"] = len(d)
    # Sin repetidos: mismo inmueble publicado varias veces -> se deja el más barato
    clave = [d["lat"].round(4), d["lon"].round(4), d["area_m2"].fillna(-1).round(0), d["habitaciones"].fillna(-1)]
    d = d.assign(_c=pd.Series(list(zip(*clave)), index=d.index)).sort_values("precio").drop_duplicates("_c")
    embudo["sin_repetidos"] = len(d)
    if d.empty:
        return d, embudo

    # 2) Criterios deseados -> percentiles ------------------------------------------------
    pesos = {k: float(v) for k, v in perfil.pesos.items() if v and k in CRITERIOS}
    if perfil.me_gustan and vectores is not None:
        V, pos = vectores
        idx = [pos[i] for i in perfil.me_gustan if i in pos]
        if idx:
            centro = V[idx].mean(axis=0)
            d["similitud"] = [float(V[pos[i]] @ centro) if i in pos else np.nan for i in d["id"]]
            pesos.setdefault("parecido", 3)
    if "similitud" not in d.columns:
        pesos.pop("parecido", None)          # "parecido" solo tiene sentido si marcó viviendas que le gustan
    if not pesos:
        pesos = {"precio": 1}
    if "valor" in pesos:            # sin valoración confiable -> neutro, no premiado ni castigado
        d.loc[d["clase_precio"].isin(["sin_valoracion", "atipico"]), "diferencia_pct"] = np.nan
    pct = pd.DataFrame(index=d.index)
    for k in pesos:
        col, sentido = CRITERIOS[k]
        pct[k] = (sentido * d[col]).rank(pct=True).fillna(0.5)     # faltante = neutro
    w = pd.Series(pesos)
    d["puntaje"] = (pct[w.index] * w).sum(axis=1) / w.sum() * 100

    # 3) Explicación: los 2 criterios que más aportan (solo si la vivienda es buena en ellos)
    def porque(i):
        aporte = (pct.loc[i, w.index] * w).sort_values(ascending=False)
        razones = [FRASES[k].format(p=round(pct.loc[i, k] * 100)) for k in aporte.index[:2] if pct.loc[i, k] >= 0.6]
        return "; ".join(razones) or "equilibrada en tus criterios"
    top = d.nlargest(n, "puntaje").copy()
    top["por_que"] = [porque(i) for i in top.index]
    return top, embudo


def similares(df: pd.DataFrame, vectores, vivienda_id: str, n: int = ALTURA_LISTA, presupuesto_max=None) -> pd.DataFrame:
    """'Más como esta': las n viviendas con el vector de la red más parecido (sin la misma ni copias)."""
    V, pos = vectores
    if vivienda_id not in pos:
        raise ValueError("Vivienda sin vector")
    ref = df.loc[df["id"] == vivienda_id].iloc[0]
    d = df[df["id"].isin(list(pos))].copy()
    d["similitud"] = V[[pos[i] for i in d["id"]]] @ V[pos[vivienda_id]]
    copia = (d["lat"].round(4) == round(ref["lat"], 4)) & (d["lon"].round(4) == round(ref["lon"], 4))
    d = d[~copia & (d["clase_precio"] != "atipico")]
    if presupuesto_max:
        d = d[d["precio"] <= presupuesto_max]
    return d.nlargest(n, "similitud")


# --------------------------------------------------------------------------------------------
# Caso de estudio + evaluación
# --------------------------------------------------------------------------------------------
COLS = ["id", "municipio", "barrio", "tipo_inmueble", "precio", "precio_justo", "clase_precio", "area_m2",
        "habitaciones", "dist_univ_km", "indice_seguridad", "puntaje", "por_que", "url"]


def main() -> None:
    df = cargar_datos()
    try:
        vectores = cargar_vectores()
    except FileNotFoundError:
        vectores = None
        print("AVISO: no hay models/vectores_red.npz (corre src.recomendador.vectores); sin 'parecido'.")
    print(f"Viviendas disponibles: {len(df):,}")

    # Caso: estudiante de la USB (sede San Benito), presupuesto $2.000.000, al menos 1 habitación,
    # a máximo 3 km; le importa sobre todo la seguridad y la cercanía, luego el buen precio.
    perfil = Perfil(universidad="USB", presupuesto_max=2_000_000, distancia_max_km=3, habitaciones_min=1,
                    pesos={"seguridad": 5, "cercania": 4, "valor": 3, "precio": 2, "espacio": 1})
    top, embudo = recomendar(df, perfil, vectores)
    print("\nCASO DE ESTUDIO: estudiante USB, $2.000.000, 1+ habitación, máx. 3 km")
    print("Embudo de criterios impuestos:", embudo)
    with pd.option_context("display.width", 250, "display.max_colwidth", 90):
        print(top[COLS].round({"dist_univ_km": 2, "puntaje": 1}).to_string(index=False))

    # Evaluación 1: los impuestos se cumplen SIEMPRE
    ok = ((top["precio"] <= perfil.presupuesto_max).all() and (top["dist_univ_km"] <= perfil.distancia_max_km).all()
          and (top["habitaciones"] >= perfil.habitaciones_min).all() and (top["clase_precio"] != "atipico").all())
    print(f"\nEvaluación 1 - criterios impuestos cumplidos por las {len(top)}: {'SÍ' if ok else 'NO'}")

    # Evaluación 2: ¿la lista es mejor que el conjunto de candidatas en lo que el estudiante pidió?
    cand, _ = recomendar(df, perfil, vectores, n=10**9)
    comp = pd.DataFrame({
        "candidatas (mediana)": cand[["indice_seguridad", "dist_univ_km", "precio", "area_m2"]].median(),
        "top 10 (mediana)": top[["indice_seguridad", "dist_univ_km", "precio", "area_m2"]].median()})
    comp["gangas_%"] = np.nan
    comp.loc["% gangas", ["candidatas (mediana)", "top 10 (mediana)"]] = [
        (cand["clase_precio"] == "ganga").mean() * 100, (top["clase_precio"] == "ganga").mean() * 100]
    print("\nEvaluación 2 - top 10 frente a todas las candidatas que cumplen los impuestos:")
    print(comp.drop(columns="gangas_%").round(2).to_string())

    # Evaluación 3: sensibilidad -> si cambia la prioridad, cambia la lista (el motor obedece al perfil)
    p2 = Perfil(**{**perfil.__dict__, "pesos": {"precio": 5, "espacio": 2}})
    top2, _ = recomendar(df, p2, vectores)
    comunes = len(set(top["id"]) & set(top2["id"]))
    print(f"\nEvaluación 3 - con prioridad 'precio' en vez de 'seguridad': {comunes}/10 viviendas en común; "
          f"seguridad mediana {top['indice_seguridad'].median():.0f} -> {top2['indice_seguridad'].median():.0f}, "
          f"precio mediano ${top['precio'].median():,.0f} -> ${top2['precio'].median():,.0f}")

    # Evaluación 4: "más como esta" con la red, partiendo de la 1.ª recomendación
    if vectores is not None and len(top):
        ref = top.iloc[0]
        sim = similares(df, vectores, ref["id"], n=5, presupuesto_max=perfil.presupuesto_max)
        print(f"\nEvaluación 4 - 'más como esta' para {ref['tipo_inmueble']} en {ref['barrio']} "
              f"(${ref['precio']:,.0f}, {ref['area_m2']} m²):")
        print(sim[["municipio", "barrio", "tipo_inmueble", "precio", "area_m2", "habitaciones", "similitud"]]
              .round(3).to_string(index=False))
        # Recomendación híbrida: perfil + "me gusta" sobre la 1.ª
        p3 = Perfil(**{**perfil.__dict__, "me_gustan": [ref["id"]]})
        top3, _ = recomendar(df, p3, vectores)
        print(f"   Con 'me gusta' en esa vivienda, la nueva lista comparte {len(set(top3['id']) & set(top['id']))}/10 "
              "con la original (el resto son parecidas a la que le gustó).")

    top[COLS].to_csv(REPORTES / "caso_estudio_usb.csv", index=False, encoding="utf-8-sig")
    print("\nGuardado: reports/caso_estudio_usb.csv")


if __name__ == "__main__":
    main()
