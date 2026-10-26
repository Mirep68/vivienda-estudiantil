"""
Limpieza de los anuncios de Finca Raíz.

    data/interim/fincaraiz_<fecha>.csv  ->  data/interim/fincaraiz_limpio_<fecha>.csv
                                        ->  reports/limpieza_fincaraiz_<fecha>.csv  (impacto de cada regla)

CRITERIO GENERAL
    - Se ELIMINA la fila solo si el precio es inválido (es lo que el modelo va a
      predecir) o si el anuncio está fuera del Valle de Aburrá.
    - Cualquier otro valor erróneo se convierte en FALTANTE (NaN) para completarlo
      después. Así no perdemos un anuncio bueno por un campo mal digitado.
    - Se conserva TODO el rango real del mercado (de una habitación barata a un
      apartamento de lujo): el filtro por presupuesto lo hace el usuario en la web.

Cada regla queda registrada con cuántos anuncios afectó (trazabilidad para el artículo).

Ejecutar:  python -m src.limpieza.fincaraiz
"""

import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
CARPETA = RAIZ / "data" / "interim"

# Nombre oficial de los 10 municipios del Valle de Aburrá.
MUNICIPIOS_VALLE = {
    "medellin": "Medellín", "bello": "Bello", "itagui": "Itagüí", "envigado": "Envigado",
    "sabaneta": "Sabaneta", "la estrella": "La Estrella", "caldas": "Caldas",
    "copacabana": "Copacabana", "girardota": "Girardota", "barbosa": "Barbosa",
}

# Límites (ver la discusión en el chat / artículo).
PRECIO_MIN, PRECIO_MAX = 300_000, 100_000_000
AREA_MIN, AREA_MAX = 15, 1_000
PRECIO_M2_MIN, PRECIO_M2_MAX = 10_000, 200_000
MAX_HABITACIONES, MAX_BANOS, MAX_PARQUEADEROS, MAX_PISO = 10, 10, 10, 45
# Caja geográfica del Valle de Aburrá (la misma del script de universidades).
LAT_MIN, LAT_MAX, LON_MIN, LON_MAX = 6.00, 6.45, -75.75, -75.40


def sin_tildes(texto: str) -> str:
    """'Itaguí' -> 'itagui'. Sirve para comparar nombres escritos de formas distintas."""
    texto = unicodedata.normalize("NFKD", str(texto))
    return "".join(c for c in texto if not unicodedata.combining(c)).lower().strip()


class Registro:
    """Lleva la cuenta de qué hizo cada regla."""

    def __init__(self):
        self.filas = []

    def anotar(self, n_regla, descripcion, accion, afectados):
        self.filas.append({"regla": n_regla, "descripcion": descripcion, "accion": accion, "afectados": int(afectados)})
        print(f"  Regla {n_regla:>2} | {accion:<24} | {int(afectados):>6,} | {descripcion}")

    def tabla(self):
        return pd.DataFrame(self.filas)


def a_faltante(df, mascara, columna, reg, n, descripcion):
    """Convierte en NaN los valores de `columna` donde `mascara` es verdadera."""
    mascara = mascara & df[columna].notna()
    df.loc[mascara, columna] = np.nan
    reg.anotar(n, descripcion, f"faltante ({columna})", mascara.sum())


def limpiar(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    reg = Registro()
    n_inicial = len(df)

    # --- 1-2: reglas que ELIMINAN filas ----------------------------------------
    clave = df["municipio"].map(sin_tildes)
    fuera = ~clave.isin(MUNICIPIOS_VALLE)
    reg.anotar(1, "Municipio fuera del Valle de Aburrá", "eliminar fila", fuera.sum())
    df = df[~fuera].copy()

    precio_malo = df["precio"].isna() | (df["precio"] < PRECIO_MIN) | (df["precio"] > PRECIO_MAX)
    reg.anotar(2, f"Precio fuera de ${PRECIO_MIN:,} - ${PRECIO_MAX:,}", "eliminar fila", precio_malo.sum())
    df = df[~precio_malo].copy()

    # --- 3: corregir nombres ---------------------------------------------------
    nombre_oficial = df["municipio"].map(sin_tildes).map(MUNICIPIOS_VALLE)
    cambiados = nombre_oficial != df["municipio"]
    df["municipio"] = nombre_oficial
    reg.anotar(3, "Nombre de municipio normalizado", "corregir", cambiados.sum())

    # --- 4-5: área y precio por m² ---------------------------------------------
    a_faltante(df, (df["area_m2"] < AREA_MIN) | (df["area_m2"] > AREA_MAX), "area_m2", reg, 4,
               f"Área fuera de {AREA_MIN}-{AREA_MAX} m²")

    # Si el precio por m² es absurdo, el precio o el área están mal digitados y no
    # podemos saber cuál: el anuncio no es confiable (ni para el modelo ni para mostrarlo).
    precio_m2 = df["precio"] / df["area_m2"]
    m2_malo = precio_m2.notna() & ~precio_m2.between(PRECIO_M2_MIN, PRECIO_M2_MAX)
    reg.anotar(5, f"Precio por m² fuera de ${PRECIO_M2_MIN:,} - ${PRECIO_M2_MAX:,}", "eliminar fila", m2_malo.sum())
    df = df[~m2_malo].copy()

    # --- 6-9: valores imposibles -> faltante -----------------------------------
    es_estudio = df["tipo_inmueble"].eq("Apartaestudio")
    hab_mala = (df["habitaciones"] < 0) | (df["habitaciones"] > MAX_HABITACIONES) | ((df["habitaciones"] == 0) & ~es_estudio)
    a_faltante(df, hab_mala, "habitaciones", reg, 6,
               f"Habitaciones <0, >{MAX_HABITACIONES}, o 0 sin ser apartaestudio")
    a_faltante(df, (df["banos"] <= 0) | (df["banos"] > MAX_BANOS), "banos", reg, 7,
               f"Baños 0 o >{MAX_BANOS}")
    a_faltante(df, (df["parqueaderos"] < 0) | (df["parqueaderos"] > MAX_PARQUEADEROS), "parqueaderos", reg, 8,
               f"Parqueaderos <0 o >{MAX_PARQUEADEROS} (el 0 es real y se conserva)")
    a_faltante(df, (df["piso"] <= 0) | (df["piso"] > MAX_PISO), "piso", reg, 9,
               f"Piso 0 (no informado), negativo o >{MAX_PISO}")

    # --- 10-12: estrato ---------------------------------------------------------
    a_faltante(df, ~df["estrato"].between(1, 6), "estrato", reg, 10, "Estrato fuera de 1-6")
    # El estrato 1 aparece en barrios de estrato 4-6 (Laureles, El Poblado...): es el
    # valor que queda cuando el anunciante no lo llena, no un estrato real.
    a_faltante(df, df["estrato"] == 1, "estrato", reg, 11, "Estrato 1 (valor de relleno del portal)")

    # Estimar el estrato faltante con el más común de su barrio, si el barrio tiene
    # >= 5 anuncios con estrato. Se excluyen barrios "genéricos" (cuando el portal
    # repite el nombre del municipio, p. ej. barrio = "Medellín"): no son un barrio real.
    df["estrato_imputado"] = False
    barrio_generico = df["barrio"].map(sin_tildes) == df["municipio"].map(sin_tildes)
    validos = df[df["estrato"].notna() & ~barrio_generico]
    grupos = validos.groupby(["municipio", "barrio"])["estrato"]
    moda = grupos.agg(lambda s: s.mode().iloc[0])[grupos.size() >= 5]
    estimado = pd.Series(list(zip(df["municipio"], df["barrio"])), index=df.index).map(moda)
    usar = df["estrato"].isna() & ~barrio_generico & estimado.notna()
    df.loc[usar, "estrato"] = estimado[usar]
    df.loc[usar, "estrato_imputado"] = True
    reg.anotar(12, "Estrato faltante estimado con la moda del barrio (>=5 anuncios, sin barrios genéricos)",
               "imputar (estrato)", usar.sum())

    # --- 13: columnas que se descartan ------------------------------------------
    df = df.drop(columns=["administracion", "precio_con_admin"])
    reg.anotar(13, "Se eliminan 'administracion' y 'precio_con_admin' (98 % sin informar)", "eliminar columna", 2)

    # --- 14: coordenadas ----------------------------------------------------------
    coord_mala = ~(df["lat"].between(LAT_MIN, LAT_MAX) & df["lon"].between(LON_MIN, LON_MAX))
    coord_mala &= df["lat"].notna()
    df.loc[coord_mala, ["lat", "lon"]] = np.nan
    reg.anotar(14, "Coordenadas fuera del Valle de Aburrá", "faltante (lat, lon)", coord_mala.sum())

    # Marca útil para la siguiente etapa: el barrio del texto no es confiable aquí.
    df["barrio_generico"] = barrio_generico

    print(f"\n  Anuncios: {n_inicial:,} -> {len(df):,}  (eliminados: {n_inicial - len(df):,})")
    print(f"  Anuncios con barrio genérico (= nombre del municipio): {int(barrio_generico.sum()):,}")
    return df, reg.tabla()


def main() -> None:
    archivos = sorted(a for a in CARPETA.glob("fincaraiz_*.csv") if "_limpio" not in a.name)
    if not archivos:
        raise SystemExit("No hay datos en data/interim. Corre primero el parser.")
    entrada = archivos[-1]
    fecha = entrada.stem.replace("fincaraiz_", "")
    print(f"Limpiando {entrada.name}\n")

    df = pd.read_csv(entrada, dtype={"id": str})
    limpio, registro = limpiar(df)

    salida = CARPETA / f"fincaraiz_limpio_{fecha}.csv"
    limpio.to_csv(salida, index=False, encoding="utf-8-sig")
    reporte = RAIZ / "reports" / f"limpieza_fincaraiz_{fecha}.csv"
    reporte.parent.mkdir(parents=True, exist_ok=True)
    registro.to_csv(reporte, index=False, encoding="utf-8-sig")
    print(f"\nGuardado:\n  {salida}\n  {reporte}")

    print("\n% de faltantes después de limpiar:")
    cols = ["precio", "area_m2", "habitaciones", "banos", "parqueaderos", "piso", "estrato", "lat", "antiguedad"]
    print((limpio[cols].isna().mean() * 100).round(1).to_string())


if __name__ == "__main__":
    main()
