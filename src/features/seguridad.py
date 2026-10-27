"""
Sub-paso 3b: índice de seguridad para cada anuncio.

ENTRADAS (todas ya extraídas con los scripts de src/scrapers):
    data/raw/seguridad/hurto_*_antioquia.csv           Policía/MinDefensa, por municipio, hasta 2026
    data/raw/poblacion/PPED-AreaMun-2018-2042_VP.xlsx  DANE, población por municipio
    data/raw/seguridad/criminalidad_comunas_anual.csv  Alcaldía, por comuna de Medellín, hasta 2023
    data/reference/poblacion_comunas.csv               Alcaldía/DANE, población por comuna
    data/interim/fincaraiz_ubicado_<fecha>.csv          anuncios con municipio y comuna

SALIDAS:
    data/interim/fincaraiz_seguridad_<fecha>.csv  anuncios + tasas de hurto + índice de seguridad
    reports/seguridad_unidades.csv                 tabla de las 30 unidades (9 municipios + 21 comunas)
    reports/figures/mapa_seguridad.png

MÉTODO (dos niveles)
  1. Municipio (los 10): tasa por 100.000 habitantes de homicidio intencional y de 4 hurtos,
     promedio 2024-2025 (últimos años completos), con UNA sola fuente (Policía) -> comparables.
     Por qué homicidios: los hurtos son DENUNCIAS (en estratos altos se denuncia mucho más:
     subregistro) y se inflan donde hay población flotante (Centro, Laureles, El Poblado).
     El homicidio casi siempre se registra y mide la violencia, que es lo que más pesa.
  2. Comunas de Medellín: factor relativo = tasa de la comuna / tasa de Medellín,
     con datos de la Alcaldía 2021-2022 (años completos post-pandemia).
     Tasa estimada de la comuna = tasa reciente de Medellín (nivel 1) x factor.
  3. Índices 0-100 (100 = más seguro), ordenando las 30 unidades (percentil):
       indice_violencia (homicidios), indice_hurto (promedio de los 4 hurtos)
       indice_seguridad = 50 % violencia + 50 % hurto

Ejecutar:  python -m src.features.seguridad
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.limpieza.fincaraiz import sin_tildes  # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]
RAW = RAIZ / "data" / "raw"
REF = RAIZ / "data" / "reference"
INTERIM = RAIZ / "data" / "interim"
REPORTS = RAIZ / "reports"

# Códigos DANE de los 10 municipios (5 dígitos: 2 del departamento + 3 del municipio)
DANE = {"05001": "Medellín", "05079": "Barbosa", "05088": "Bello", "05129": "Caldas", "05212": "Copacabana",
        "05266": "Envigado", "05308": "Girardota", "05360": "Itagüí", "05380": "La Estrella", "05631": "Sabaneta"}
ANIOS_MUNICIPIO = [2024, 2025]
ANIOS_COMUNA = [2021, 2022]
HURTOS = ["hurto_personas", "hurto_residencias", "hurto_motos", "hurto_carros"]
DELITOS = HURTOS + ["homicidios"]   # homicidio INTENCIONAL (sin los de tránsito)
PESO_VIOLENCIA = 0.5                # el índice = 50 % violencia (homicidios) + 50 % hurtos
CONDUCTAS_ALCALDIA = {"Homicidio": "homicidios", "Hurto a persona": "hurto_personas", "Hurto a residencia": "hurto_residencias",
                      "Hurto de moto": "hurto_motos", "Hurto de carro": "hurto_carros"}


# ---------------------------------------------------------------------------
# Lectura de fuentes
# ---------------------------------------------------------------------------
def anio_de(fechas: pd.Series) -> pd.Series:
    """Año de una fecha que puede venir como '2026-09-30T00:00:00' o como '15/07/2026'."""
    f = fechas.astype(str)
    iso = f.str.match(r"^\d{4}-")
    anio = pd.Series(np.nan, index=f.index)
    anio[iso] = pd.to_datetime(f[iso], errors="coerce").dt.year
    anio[~iso] = pd.to_datetime(f[~iso], errors="coerce", dayfirst=True).dt.year
    return anio


def leer_delitos_municipio() -> pd.DataFrame:
    """Devuelve casos por (codigo, delito, año) para los 10 municipios."""
    partes = []

    p = pd.read_csv(RAW / "seguridad" / "hurto_personas_antioquia.csv", dtype=str)
    partes.append(pd.DataFrame({"codigo": p["cod_muni"].str.zfill(5), "delito": "hurto_personas",
                                "anio": anio_de(p["fecha_hecho"]), "casos": pd.to_numeric(p["cantidad"])}))

    for archivo in ["hurto_residencias_comercio_antioquia.csv", "hurto_motos_carros_antioquia.csv"]:
        d = pd.read_csv(RAW / "seguridad" / archivo, dtype=str)
        tipo = d["tipo_de_hurto"].str.upper()
        delito = np.select([tipo.str.contains("RESIDENCIA"), tipo.str.contains("MOTOCICLETA"),
                            tipo.str.contains("AUTOMOTOR")],
                           ["hurto_residencias", "hurto_motos", "hurto_carros"], default="otro")
        # codigo_dane trae 8 dígitos (municipio + centro poblado): tomamos los 5 primeros
        partes.append(pd.DataFrame({"codigo": d["codigo_dane"].str.zfill(8).str[:5], "delito": delito,
                                    "anio": anio_de(d["fecha_hecho"]), "casos": pd.to_numeric(d["cantidad"])}))

    # Homicidios intencionales (archivos de la Policía, ya filtrados al Valle de Aburrá)
    for anio in ANIOS_MUNICIPIO:
        h = pd.read_csv(RAW / "seguridad" / f"homicidios_valle_{anio}.csv", dtype=str, encoding="utf-8-sig")
        h = h[h["DELITOS"].str.lower().str.contains("intencional")]
        partes.append(pd.DataFrame({"codigo": h["codigo5"].str.zfill(5), "delito": "homicidios",
                                    "anio": anio, "casos": pd.to_numeric(h["casos"])}))

    t = pd.concat(partes, ignore_index=True)
    t = t[t["codigo"].isin(DANE) & t["delito"].isin(DELITOS)]
    return t.groupby(["codigo", "delito", "anio"], as_index=False)["casos"].sum()


def leer_poblacion_dane() -> pd.DataFrame:
    """Población total por municipio y año desde el Excel del DANE (detecta columnas automáticamente)."""
    ruta = RAW / "poblacion" / "PPED-AreaMun-2018-2042_VP.xlsx"
    hoja = next(h for h in pd.ExcelFile(ruta).sheet_names if "mun" in sin_tildes(h))
    crudo = pd.read_excel(ruta, sheet_name=hoja, header=None, nrows=40)
    fila_enc = next(i for i, fila in crudo.iterrows()
                    if any(sin_tildes(v).startswith("ano") for v in fila.dropna().astype(str)))
    df = pd.read_excel(ruta, sheet_name=hoja, header=fila_enc)

    def col_codigo():
        for c in df.columns:
            s = df[c].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
            if (s == "05001").any():
                return c
    col_cod = col_codigo()
    col_anio = next(c for c in df.columns if sin_tildes(c).startswith("ano"))
    col_area = next((c for c in df.columns if "area" in sin_tildes(c)), None)
    col_pob = next(c for c in reversed(df.columns)
                   if pd.api.types.is_numeric_dtype(df[c]) and df[c].max() > 1_000_000 and c != col_anio)
    print(f"  Excel DANE -> hoja '{hoja}', encabezado en fila {fila_enc} | código: '{col_cod}', año: '{col_anio}', "
          f"área: '{col_area}', población: '{col_pob}'")
    if col_area is not None:
        print(f"  Valores de área: {df[col_area].dropna().unique().tolist()}")
        df = df[df[col_area].astype(str).str.lower().str.contains("total")]

    pob = pd.DataFrame({
        "codigo": df[col_cod].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5),
        "anio": pd.to_numeric(df[col_anio], errors="coerce"),
        "poblacion": pd.to_numeric(df[col_pob], errors="coerce"),
    })
    return pob[pob["codigo"].isin(DANE)].groupby(["codigo", "anio"], as_index=False)["poblacion"].sum()


def leer_comunas() -> pd.DataFrame:
    """Casos por comuna de Medellín (Alcaldía) y población por comuna, años 2021-2022."""
    c = pd.read_csv(RAW / "seguridad" / "criminalidad_comunas_anual.csv", encoding="utf-8-sig", dtype={"Codigo_comuna": str})
    c["comuna"] = pd.to_numeric(c["Codigo_comuna"].str.strip(), errors="coerce")   # '3' y '3.0' -> 3 ; 'SIN DATO' -> NaN
    c = c[c["comuna"].notna() & c["Fecha_hecho"].isin(ANIOS_COMUNA) & c["Conducta"].isin(CONDUCTAS_ALCALDIA)]
    c["delito"] = c["Conducta"].map(CONDUCTAS_ALCALDIA)
    c["comuna"] = c["comuna"].astype(int)
    dup = c.duplicated(["Fecha_hecho", "delito", "comuna"]).sum()
    print(f"  Alcaldía: filas repetidas (mismo año, delito y comuna) que se suman: {int(dup)}")
    casos = c.groupby(["comuna", "delito"], as_index=False)["Cantidad_casos"].sum()
    casos["casos_anuales"] = casos["Cantidad_casos"] / len(ANIOS_COMUNA)

    p = pd.read_csv(REF / "poblacion_comunas.csv", encoding="utf-8-sig")
    p["comuna"] = pd.to_numeric(p["codigo_num"], errors="coerce").astype("Int64")
    p["poblacion"] = p[[f"total_{a}" for a in ANIOS_COMUNA]].mean(axis=1)
    return casos.merge(p[["comuna", "nombre", "poblacion"]], on="comuna", how="left")


# ---------------------------------------------------------------------------
# Cálculo
# ---------------------------------------------------------------------------
def calcular_indices(tasas: pd.DataFrame, referencia: pd.DataFrame) -> pd.DataFrame:
    """
    Índices 0-100 (100 = más seguro) de cada fila de `tasas`, comparándola con las 30 unidades (`referencia`).
    Percentil de una tasa = fracción de unidades con una tasa menor o igual (0 = la más baja).
      indice_violencia = 100 x (1 - percentil de homicidios)
      indice_hurto     = 100 x (1 - promedio de los percentiles de los 4 hurtos)
      indice_seguridad = 50 % violencia + 50 % hurto
    """
    pct = pd.DataFrame({d: tasas[d].map(lambda v, d=d: (referencia[d] <= v).mean()) for d in DELITOS})
    out = pd.DataFrame(index=tasas.index)
    out["indice_violencia"] = (1 - pct["homicidios"]) * 100
    out["indice_hurto"] = (1 - pct[HURTOS].mean(axis=1)) * 100
    out["indice_seguridad"] = PESO_VIOLENCIA * out["indice_violencia"] + (1 - PESO_VIOLENCIA) * out["indice_hurto"]
    return out.round(1)


def main() -> None:
    print("1) Tasas por 100.000 habitantes por municipio (Policía, promedio 2024-2025)")
    casos = leer_delitos_municipio()
    pob = leer_poblacion_dane()
    print("  Años disponibles en los delitos:", sorted(casos["anio"].dropna().astype(int).unique())[-6:])

    rec = casos[casos["anio"].isin(ANIOS_MUNICIPIO)].groupby(["codigo", "delito"])["casos"].sum() / len(ANIOS_MUNICIPIO)
    pob_m = pob[pob["anio"].isin(ANIOS_MUNICIPIO)].groupby("codigo")["poblacion"].mean()
    tasa_mun = (rec.unstack("delito").reindex(columns=DELITOS).fillna(0)
                .div(pob_m, axis=0) * 100_000)
    tasa_mun.index = tasa_mun.index.map(DANE)
    tabla_m = tasa_mun.round(1).assign(poblacion=pob_m.rename(index=DANE).round(0))
    print(tabla_m.sort_values("hurto_personas", ascending=False).to_string())

    print("\n2) Factor relativo de cada comuna de Medellín (Alcaldía, 2021-2022)")
    com = leer_comunas()
    sin_pob = com[com["poblacion"].isna()]["comuna"].unique()
    if len(sin_pob):
        print(f"  AVISO: comunas sin población: {list(sin_pob)}")
    tasa_com = (com.assign(tasa=com["casos_anuales"] / com["poblacion"] * 100_000)
                .pivot(index="comuna", columns="delito", values="tasa").reindex(columns=DELITOS).fillna(0))
    total_med = com.groupby("delito")["casos_anuales"].sum() / com.drop_duplicates("comuna")["poblacion"].sum() * 100_000
    factor = tasa_com.div(total_med.reindex(DELITOS), axis=1)
    nombres = com.drop_duplicates("comuna").set_index("comuna")["nombre"]
    print(factor.round(2).assign(nombre=nombres).sort_values("hurto_personas", ascending=False).to_string())

    # Tasa estimada reciente de cada comuna = tasa Medellín (Policía 2024-25) x factor
    tasa_com_est = factor.mul(tasa_mun.loc["Medellín"], axis=1)

    # 30 unidades: 9 municipios + 21 comunas/corregimientos
    unidades = pd.concat([
        tasa_mun.drop(index="Medellín").assign(unidad_tipo="municipio", nombre=lambda d: d.index),
        tasa_com_est.assign(unidad_tipo="comuna", nombre=nombres.reindex(tasa_com_est.index).values),
    ])
    unidades.index = [f"{t}:{i}" for t, i in zip(unidades["unidad_tipo"], unidades.index)]
    # Índice: percentil de cada hurto (0 = la tasa más baja = más seguro) -> promedio -> 0-100 seguro
    unidades = unidades.join(calcular_indices(unidades[DELITOS], unidades[DELITOS]))
    print("\n3) Índice de seguridad (100 = más seguro) de las 30 unidades:")
    print(unidades[["nombre", "unidad_tipo", "indice_seguridad", "indice_violencia", "indice_hurto"] + DELITOS].round(1)
          .sort_values("indice_seguridad", ascending=False).to_string())
    REPORTS.mkdir(exist_ok=True)
    unidades.round(2).to_csv(REPORTS / "seguridad_unidades.csv", encoding="utf-8-sig")

    # --- Asignar a cada anuncio ------------------------------------------------
    entrada = sorted(INTERIM.glob("fincaraiz_ubicado_*.csv"))[-1]
    fecha = entrada.stem.replace("fincaraiz_ubicado_", "")
    df = pd.read_csv(entrada, dtype={"id": str, "comuna_codigo": str})
    comuna_num = pd.to_numeric(df["comuna_codigo"], errors="coerce")
    clave = np.where(df["municipio"].eq("Medellín") & comuna_num.notna(),
                     "comuna:" + comuna_num.astype("Int64").astype(str), "municipio:" + df["municipio"])
    # Medellín sin comuna (sin coordenadas): usa la tasa de Medellín completa
    medellin = tasa_mun.loc[["Medellín"]].assign(nombre="Medellín", unidad_tipo="municipio")
    # Su índice: en qué percentil quedarían las tasas de Medellín completa frente a las 30 unidades
    medellin = medellin.join(calcular_indices(medellin[DELITOS], unidades[DELITOS]))
    tabla = pd.concat([unidades, medellin.set_axis(["municipio:Medellín"])])
    indices = ["indice_seguridad", "indice_violencia", "indice_hurto"]
    tabla_cols = tabla[DELITOS].add_prefix("seg_").join(tabla[indices])
    df["unidad_seguridad"] = clave
    df = df.join(tabla_cols, on="unidad_seguridad")
    sin = df["indice_seguridad"].isna().sum()
    print(f"\nAnuncios con índice de seguridad: {df['indice_seguridad'].notna().sum():,} de {len(df):,} (sin índice: {sin})")

    # Sanity check: ¿el índice se relaciona con el estrato como esperaríamos?
    print("\nÍndice de seguridad mediano por estrato (sanity check):")
    print(df.groupby("estrato")[indices].median().round(1).to_string())

    salida = INTERIM / f"fincaraiz_seguridad_{fecha}.csv"
    df.to_csv(salida, index=False, encoding="utf-8-sig")
    print(f"\nGuardado: {salida}\n          {REPORTS / 'seguridad_unidades.csv'}")

    dibujar_mapa(unidades)


def dibujar_mapa(unidades: pd.DataFrame) -> None:
    mun = gpd.read_file(REF / "municipios_valle.geojson")
    comunas = gpd.read_file(REF / "comunas.geojson")
    comunas["clave"] = "comuna:" + pd.to_numeric(comunas["codigo"], errors="coerce").astype("Int64").astype(str)
    mun["clave"] = "municipio:" + mun["municipio"]
    capas = pd.concat([mun[mun["municipio"] != "Medellín"][["clave", "geometry"]],
                       comunas[comunas["clave"] != "comuna:<NA>"][["clave", "geometry"]]])
    capas = gpd.GeoDataFrame(capas, crs=mun.crs).merge(unidades[["indice_seguridad"]], left_on="clave", right_index=True, how="left")
    fig, ax = plt.subplots(figsize=(7, 10))
    capas.plot(ax=ax, column="indice_seguridad", cmap="RdYlGn", legend=True, edgecolor="black", linewidth=0.3,
               legend_kwds={"label": "Índice de seguridad (100 = más seguro)"}, missing_kwds={"color": "lightgray"})
    ax.set_title("Índice de seguridad: 50 % homicidios + 50 % hurtos (por habitante)")
    fig.savefig(REPORTS / "figures" / "mapa_seguridad.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Mapa: {REPORTS / 'figures' / 'mapa_seguridad.png'}")


if __name__ == "__main__":
    main()
